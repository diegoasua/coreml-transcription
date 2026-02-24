#!/usr/bin/env python3
"""Compress CoreML weights with optimize API (and fallback to legacy quantization)."""

from __future__ import annotations

import argparse
from pathlib import Path


def compress_with_optimize_api(model, nbits: int, mode: str, group_size: int, granularity: str):
    import coremltools.optimize as cto  # type: ignore

    op_cfg = cto.coreml.OpPalettizerConfig(
        mode=mode,
        nbits=nbits,
        granularity=granularity,
        group_size=group_size,
    )
    cfg = cto.coreml.OptimizationConfig(global_config=op_cfg)
    return cto.coreml.palettize_weights(model, config=cfg)


def compress_with_legacy_api(model, nbits: int, mode: str):
    from coremltools.models.neural_network.quantization_utils import quantize_weights  # type: ignore

    quant_mode_map = {
        "kmeans": "kmeans_lut",
        "uniform": "linear",
    }
    mapped = quant_mode_map.get(mode, "kmeans_lut")
    return quantize_weights(model, nbits=nbits, quantization_mode=mapped)


def compress(model_path: Path, output_path: Path, nbits: int, mode: str, group_size: int):
    import coremltools as ct  # type: ignore

    model = ct.models.MLModel(str(model_path))
    model_type = model.get_spec().WhichOneof("Type")
    errors = []

    compressed = None

    # Try modern palettization first. Fall back from per_grouped_channel
    # to per_tensor for older deployment targets.
    for granularity in ["per_grouped_channel", "per_tensor"]:
        try:
            compressed = compress_with_optimize_api(
                model=model,
                nbits=nbits,
                mode=mode,
                group_size=group_size,
                granularity=granularity,
            )
            break
        except Exception as exc:
            errors.append(f"optimize API failed ({granularity}): {exc}")

    # Legacy API generally applies to neuralnetwork specs; skip it for mlProgram.
    if compressed is None and model_type != "mlProgram":
        try:
            compressed = compress_with_legacy_api(model=model, nbits=nbits, mode=mode)
        except Exception as exc:
            errors.append(f"legacy quantization failed: {exc}")

    if compressed is None:
        details = "\n".join(f"- {entry}" for entry in errors)
        raise RuntimeError(f"Compression failed for {model_path}.\nDetails:\n{details}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    compressed.save(str(output_path))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compress CoreML model weights.")
    parser.add_argument("--model", type=Path, required=True, help="Input CoreML model path.")
    parser.add_argument("--output", type=Path, required=True, help="Output compressed CoreML model path.")
    parser.add_argument("--nbits", type=int, default=4, choices=[2, 3, 4, 6, 8])
    parser.add_argument("--mode", default="kmeans", choices=["kmeans", "uniform"])
    parser.add_argument("--group-size", type=int, default=32, help="Grouped-channel size for palettization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = compress(
        model_path=args.model,
        output_path=args.output,
        nbits=args.nbits,
        mode=args.mode,
        group_size=args.group_size,
    )
    print(f"Compression complete: {output_path}")


if __name__ == "__main__":
    main()
