#!/usr/bin/env python3
"""Latency benchmark for CoreML model with synthetic inputs.

This benchmarks model runtime only. It does not compute WER.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any


def _resolve_compute_units(ct, raw_units: str):
    units_map = {
        "all": ct.ComputeUnit.ALL,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }
    key = raw_units.lower()
    if key not in units_map:
        raise ValueError(f"Unsupported compute units '{raw_units}'. Use one of: {', '.join(units_map)}")
    return units_map[key]


def _array_dtype_to_numpy(data_type_code: int):
    import numpy as np
    from coremltools.proto import FeatureTypes_pb2  # type: ignore

    mapping = {
        FeatureTypes_pb2.ArrayFeatureType.FLOAT32: np.float32,
        FeatureTypes_pb2.ArrayFeatureType.FLOAT16: np.float16,
        FeatureTypes_pb2.ArrayFeatureType.DOUBLE: np.float64,
        FeatureTypes_pb2.ArrayFeatureType.INT32: np.int32,
    }
    return mapping.get(data_type_code, np.float32)


def _build_inputs_from_spec(model, suggestions: dict[str, Any]):
    import numpy as np

    spec = model.get_spec()
    feeds: dict[str, Any] = {}
    input_shapes: dict[str, list[int]] = {}

    for entry in spec.description.input:
        if not entry.type.HasField("multiArrayType"):
            continue
        info = entry.type.multiArrayType
        shape = [int(dim) for dim in info.shape]
        dtype = _array_dtype_to_numpy(info.dataType)
        input_shapes[entry.name] = shape

        if not shape:
            shape = [1]

        if np.issubdtype(dtype, np.floating):
            values = np.array(
                [random.uniform(-0.2, 0.2) for _ in range(int(np.prod(shape)))],
                dtype=dtype,
            ).reshape(shape)
        elif np.issubdtype(dtype, np.integer):
            values = np.zeros(shape, dtype=dtype)
        else:
            values = np.zeros(shape, dtype=np.float32)

        feeds[entry.name] = values

    audio_name = suggestions.get("audio_input_name")
    length_name = suggestions.get("length_input_name")

    if audio_name and length_name and audio_name in feeds and length_name in feeds:
        audio_shape = input_shapes.get(audio_name, [])
        if audio_shape:
            inferred_len = int(audio_shape[-1])
            feeds[length_name] = np.array([inferred_len], dtype=feeds[length_name].dtype)

    if "targets" in feeds and "target_length" in feeds:
        token_len = int(input_shapes.get("targets", [1, 1])[-1])
        feeds["target_length"] = np.array([token_len], dtype=feeds["target_length"].dtype)

    return feeds, input_shapes


def _estimate_chunk_seconds(
    suggestions: dict[str, Any],
    input_shapes: dict[str, list[int]],
    sample_rate: int,
    frame_hop_ms: float,
) -> float | None:
    audio_name = suggestions.get("audio_input_name")
    if not audio_name or audio_name not in input_shapes:
        return None

    shape = input_shapes[audio_name]
    if len(shape) == 2:
        return shape[-1] / float(sample_rate)
    if len(shape) == 3:
        # Encoder-style log-mel (e.g. [B, 128, T]) or decoder encoder_outputs.
        return (shape[-1] * frame_hop_ms) / 1000.0
    return None


def run_benchmark(
    model_path: Path,
    manifest_path: Path,
    iterations: int,
    warmup: int,
    compute_units: str,
    sample_rate: int,
    frame_hop_ms: float,
) -> dict[str, Any]:
    import coremltools as ct  # type: ignore
    import numpy as np

    if not os.environ.get("TMPDIR"):
        tmpdir = (model_path.parent / ".tmp").resolve()
        tmpdir.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(tmpdir)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    suggestions = manifest.get("coreml_suggestions", {})

    compute_enum = _resolve_compute_units(ct, compute_units)
    model = ct.models.MLModel(str(model_path), compute_units=compute_enum)
    feeds, input_shapes = _build_inputs_from_spec(model=model, suggestions=suggestions)
    if not feeds:
        raise RuntimeError("No multi-array inputs found in model spec for benchmarking.")

    for _ in range(warmup):
        _ = model.predict(feeds)

    latencies_ms = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model.predict(feeds)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    chunk_sec = _estimate_chunk_seconds(
        suggestions=suggestions,
        input_shapes=input_shapes,
        sample_rate=sample_rate,
        frame_hop_ms=frame_hop_ms,
    )
    median_ms = statistics.median(latencies_ms)
    p95_ms = np.percentile(np.array(latencies_ms), 95).item()

    report = {
        "iterations": iterations,
        "warmup": warmup,
        "input_shapes": input_shapes,
        "median_latency_ms": median_ms,
        "p95_latency_ms": p95_ms,
    }
    if chunk_sec and chunk_sec > 0:
        rtf = (median_ms / 1000.0) / chunk_sec
        speed_x = 1.0 / rtf if rtf > 0 else float("inf")
        report["chunk_sec_estimate"] = chunk_sec
        report["rtf_median_estimate"] = rtf
        report["speed_x_median_estimate"] = speed_x
    else:
        report["chunk_sec_estimate"] = None

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CoreML model latency.")
    parser.add_argument("--model", type=Path, required=True, help="Path to .mlmodel/.mlpackage")
    parser.add_argument("--manifest", type=Path, required=True, help="ONNX manifest JSON from inspect_onnx.py")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--compute-units", default="all", help="all, cpu_only, cpu_and_gpu, cpu_and_ne")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--frame-hop-ms", type=float, default=10.0)
    args = parser.parse_args()

    report = run_benchmark(
        model_path=args.model,
        manifest_path=args.manifest,
        iterations=args.iterations,
        warmup=args.warmup,
        compute_units=args.compute_units,
        sample_rate=args.sample_rate,
        frame_hop_ms=args.frame_hop_ms,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
