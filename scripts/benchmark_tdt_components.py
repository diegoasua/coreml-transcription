#!/usr/bin/env python3
"""Benchmark TDT component latency (encoder + decoder loop) for CoreML models."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
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


def _ensure_tmpdir(anchor_path: Path) -> None:
    if os.environ.get("TMPDIR"):
        return
    tmpdir = (anchor_path.parent / ".tmp").resolve()
    tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmpdir)


@dataclass
class InputSpec:
    name: str
    shape: list[int]
    dtype: Any


def _get_input_specs(model) -> list[InputSpec]:
    specs: list[InputSpec] = []
    proto = model.get_spec()
    for entry in proto.description.input:
        if not entry.type.HasField("multiArrayType"):
            continue
        arr = entry.type.multiArrayType
        specs.append(
            InputSpec(
                name=entry.name,
                shape=[int(dim) for dim in arr.shape],
                dtype=_array_dtype_to_numpy(arr.dataType),
            )
        )
    return specs


def _random_tensor(shape: list[int], dtype):
    import numpy as np

    if not shape:
        shape = [1]
    if np.issubdtype(dtype, np.floating):
        return np.array(
            [random.uniform(-0.2, 0.2) for _ in range(int(np.prod(shape)))],
            dtype=dtype,
        ).reshape(shape)
    if np.issubdtype(dtype, np.integer):
        return np.zeros(shape, dtype=dtype)
    return np.zeros(shape, dtype=np.float32)


def _estimate_chunk_seconds(encoder_input_specs: list[InputSpec], frame_hop_ms: float) -> float | None:
    for spec in encoder_input_specs:
        if spec.name == "audio_signal" and len(spec.shape) == 3:
            return (spec.shape[-1] * frame_hop_ms) / 1000.0
        if spec.name == "audio_signal" and len(spec.shape) == 2:
            return spec.shape[-1] / 16_000.0
    return None


def _pick_encoder_output(outputs: dict[str, Any]) -> str:
    import numpy as np

    best_name = None
    best_score = None
    for name, value in outputs.items():
        array = np.asarray(value)
        if array.dtype.kind not in ("f", "i"):
            continue
        if array.ndim < 2:
            continue
        score = (array.ndim, array.shape[-1], array.size)
        if best_score is None or score > best_score:
            best_score = score
            best_name = name
    if best_name is None:
        raise RuntimeError("Could not infer encoder output tensor from model outputs.")
    return best_name


def _prepare_encoder_feed(encoder_input_specs: list[InputSpec]) -> dict[str, Any]:
    import numpy as np

    feeds = {spec.name: _random_tensor(spec.shape, spec.dtype) for spec in encoder_input_specs}
    if "audio_signal" in feeds and "length" in feeds:
        audio_len = int(np.asarray(feeds["audio_signal"]).shape[-1])
        feeds["length"] = np.array([audio_len], dtype=np.asarray(feeds["length"]).dtype)
    return feeds


def _prepare_decoder_feed(
    decoder_input_specs: list[InputSpec],
    encoder_outputs: Any,
    decoder_manifest: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    specs_by_name = {spec.name: spec for spec in decoder_input_specs}
    feeds = {}
    for spec in decoder_input_specs:
        feeds[spec.name] = _random_tensor(spec.shape, spec.dtype)

    # ONNX-derived "audio_input_name" suggestions can be wrong for decoder/joint
    # graphs (e.g. accidentally selecting input_states_*). Prefer explicit
    # encoder_outputs when present.
    enc_name = None
    if "encoder_outputs" in specs_by_name:
        enc_name = "encoder_outputs"
    else:
        suggested = decoder_manifest.get("coreml_suggestions", {}).get("audio_input_name")
        if suggested in specs_by_name:
            enc_name = suggested
        else:
            for candidate in specs_by_name:
                if "encoder" in candidate.lower() and "output" in candidate.lower():
                    enc_name = candidate
                    break

    if enc_name is None:
        raise RuntimeError("Could not determine decoder encoder-output input name.")

    if enc_name in specs_by_name:
        target_dtype = specs_by_name[enc_name].dtype
        target_shape = specs_by_name[enc_name].shape
        enc_value = np.asarray(encoder_outputs).astype(target_dtype, copy=False)
        if list(enc_value.shape) != list(target_shape):
            if enc_value.ndim != len(target_shape):
                raise RuntimeError(
                    f"Encoder output rank mismatch for '{enc_name}': "
                    f"got {list(enc_value.shape)} expected {target_shape}"
                )
            adjusted = np.zeros(target_shape, dtype=target_dtype)
            slices = tuple(slice(0, min(enc_value.shape[idx], target_shape[idx])) for idx in range(enc_value.ndim))
            adjusted[slices] = enc_value[slices]
            enc_value = adjusted
        feeds[enc_name] = enc_value

    if "targets" in feeds:
        target_shape = np.asarray(feeds["targets"]).shape
        feeds["targets"] = np.zeros(target_shape, dtype=np.asarray(feeds["targets"]).dtype)
    if "target_length" in feeds and "targets" in feeds:
        token_len = int(np.asarray(feeds["targets"]).shape[-1])
        feeds["target_length"] = np.array([token_len], dtype=np.asarray(feeds["target_length"]).dtype)
    return feeds


def _infer_decoder_roles(
    decoder_outputs: dict[str, Any],
    decoder_inputs: dict[str, Any],
    decoder_manifest: dict[str, Any],
) -> tuple[str, str | None, str | None, dict[str, str]]:
    import numpy as np

    suggestions = decoder_manifest.get("coreml_suggestions", {})
    logits_name = suggestions.get("logits_output_name")
    if logits_name not in decoder_outputs:
        logits_name = None
        best_vocab = -1
        for name, value in decoder_outputs.items():
            arr = np.asarray(value)
            if arr.dtype.kind != "f" or arr.ndim < 2:
                continue
            vocab = int(arr.shape[-1])
            if vocab > best_vocab:
                best_vocab = vocab
                logits_name = name
    if logits_name is None:
        raise RuntimeError("Could not infer decoder logits output.")

    target_input = "targets" if "targets" in decoder_inputs else None
    target_len_input = "target_length" if "target_length" in decoder_inputs else None

    state_input_names = [name for name in decoder_inputs if name.startswith("input_states")]
    state_map: dict[str, str] = {}
    if state_input_names:
        candidate_state_outputs = []
        for out_name, out_value in decoder_outputs.items():
            if out_name == logits_name:
                continue
            arr = np.asarray(out_value)
            if arr.dtype.kind != "f" or arr.ndim != 3:
                continue
            candidate_state_outputs.append(out_name)

        candidate_state_outputs = sorted(candidate_state_outputs)
        for in_name, out_name in zip(sorted(state_input_names), candidate_state_outputs):
            state_map[in_name] = out_name

    return logits_name, target_input, target_len_input, state_map


def _latency_stats(values_ms: list[float]) -> tuple[float, float]:
    import numpy as np

    if not values_ms:
        return 0.0, 0.0
    return statistics.median(values_ms), float(np.percentile(np.array(values_ms), 95).item())


def run_benchmark(
    encoder_model_path: Path,
    encoder_manifest_path: Path,
    decoder_model_path: Path,
    decoder_manifest_path: Path,
    iterations: int,
    warmup: int,
    decoder_steps: int,
    compute_units: str,
    frame_hop_ms: float,
) -> dict[str, Any]:
    import numpy as np
    import coremltools as ct  # type: ignore

    _ensure_tmpdir(encoder_model_path)

    encoder_manifest = json.loads(encoder_manifest_path.read_text(encoding="utf-8"))
    decoder_manifest = json.loads(decoder_manifest_path.read_text(encoding="utf-8"))

    compute_enum = _resolve_compute_units(ct, compute_units)
    encoder_model = ct.models.MLModel(str(encoder_model_path), compute_units=compute_enum)
    decoder_model = ct.models.MLModel(str(decoder_model_path), compute_units=compute_enum)

    encoder_input_specs = _get_input_specs(encoder_model)
    decoder_input_specs = _get_input_specs(decoder_model)

    encoder_latencies: list[float] = []
    decoder_step_latencies: list[float] = []
    decoder_total_latencies: list[float] = []
    end_to_end_latencies: list[float] = []

    total_rounds = warmup + iterations
    for round_idx in range(total_rounds):
        enc_feed = _prepare_encoder_feed(encoder_input_specs)

        t0 = time.perf_counter()
        enc_out = encoder_model.predict(enc_feed)
        t1 = time.perf_counter()
        encoder_ms = (t1 - t0) * 1000.0

        enc_out_name = _pick_encoder_output(enc_out)
        dec_feed = _prepare_decoder_feed(
            decoder_input_specs=decoder_input_specs,
            encoder_outputs=enc_out[enc_out_name],
            decoder_manifest=decoder_manifest,
        )

        logits_name = None
        target_input = None
        target_len_input = None
        state_map: dict[str, str] = {}

        decoder_iter_lat = []
        for _ in range(decoder_steps):
            t2 = time.perf_counter()
            dec_out = decoder_model.predict(dec_feed)
            t3 = time.perf_counter()
            step_ms = (t3 - t2) * 1000.0
            decoder_iter_lat.append(step_ms)

            if logits_name is None:
                logits_name, target_input, target_len_input, state_map = _infer_decoder_roles(
                    decoder_outputs=dec_out,
                    decoder_inputs=dec_feed,
                    decoder_manifest=decoder_manifest,
                )

            logits = np.asarray(dec_out[logits_name])
            vocab = int(logits.shape[-1])
            flattened = logits.reshape(-1, vocab)
            next_token = int(np.argmax(flattened[-1]))

            if target_input:
                target_array = np.asarray(dec_feed[target_input])
                dec_feed[target_input] = np.full(target_array.shape, next_token, dtype=target_array.dtype)
            if target_len_input and target_input:
                target_len_dtype = np.asarray(dec_feed[target_len_input]).dtype
                target_seq_len = int(np.asarray(dec_feed[target_input]).shape[-1])
                dec_feed[target_len_input] = np.array([target_seq_len], dtype=target_len_dtype)
            for in_name, out_name in state_map.items():
                dec_feed[in_name] = dec_out[out_name]

        decoder_total_ms = sum(decoder_iter_lat)
        end_to_end_ms = encoder_ms + decoder_total_ms

        if round_idx >= warmup:
            encoder_latencies.append(encoder_ms)
            decoder_step_latencies.extend(decoder_iter_lat)
            decoder_total_latencies.append(decoder_total_ms)
            end_to_end_latencies.append(end_to_end_ms)

    enc_median, enc_p95 = _latency_stats(encoder_latencies)
    dec_step_median, dec_step_p95 = _latency_stats(decoder_step_latencies)
    dec_total_median, dec_total_p95 = _latency_stats(decoder_total_latencies)
    e2e_median, e2e_p95 = _latency_stats(end_to_end_latencies)

    chunk_sec = _estimate_chunk_seconds(encoder_input_specs=encoder_input_specs, frame_hop_ms=frame_hop_ms)

    report: dict[str, Any] = {
        "iterations": iterations,
        "warmup": warmup,
        "decoder_steps": decoder_steps,
        "encoder_input_shapes": {spec.name: spec.shape for spec in encoder_input_specs},
        "decoder_input_shapes": {spec.name: spec.shape for spec in decoder_input_specs},
        "encoder_median_ms": enc_median,
        "encoder_p95_ms": enc_p95,
        "decoder_step_median_ms": dec_step_median,
        "decoder_step_p95_ms": dec_step_p95,
        "decoder_total_median_ms": dec_total_median,
        "decoder_total_p95_ms": dec_total_p95,
        "end_to_end_median_ms": e2e_median,
        "end_to_end_p95_ms": e2e_p95,
    }

    if dec_total_median > 0:
        report["decoder_tokens_per_sec_estimate"] = decoder_steps / (dec_total_median / 1000.0)

    if chunk_sec and chunk_sec > 0:
        rtf = (e2e_median / 1000.0) / chunk_sec
        report["chunk_sec_estimate"] = chunk_sec
        report["end_to_end_rtf_estimate"] = rtf
        report["end_to_end_speed_x_estimate"] = 1.0 / rtf if rtf > 0 else float("inf")
    else:
        report["chunk_sec_estimate"] = None

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TDT CoreML encoder+decoder components.")
    parser.add_argument("--encoder-model", type=Path, required=True)
    parser.add_argument("--encoder-manifest", type=Path, required=True)
    parser.add_argument("--decoder-model", type=Path, required=True)
    parser.add_argument("--decoder-manifest", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--decoder-steps", type=int, default=64)
    parser.add_argument("--compute-units", default="all", help="all, cpu_only, cpu_and_gpu, cpu_and_ne")
    parser.add_argument("--frame-hop-ms", type=float, default=10.0)
    args = parser.parse_args()

    report = run_benchmark(
        encoder_model_path=args.encoder_model,
        encoder_manifest_path=args.encoder_manifest,
        decoder_model_path=args.decoder_model,
        decoder_manifest_path=args.decoder_manifest,
        iterations=args.iterations,
        warmup=args.warmup,
        decoder_steps=args.decoder_steps,
        compute_units=args.compute_units,
        frame_hop_ms=args.frame_hop_ms,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
