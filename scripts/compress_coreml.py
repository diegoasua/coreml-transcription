#!/usr/bin/env python3
"""Compress CoreML weights with optimize API (and fallback to legacy quantization)."""

from __future__ import annotations

import argparse
from pathlib import Path


def compress_with_optimize_api_palettize(
    model,
    nbits: int,
    mode: str,
    group_size: int,
    granularity: str,
    enable_per_channel_scale: bool,
    weight_threshold: int,
):
    import coremltools.optimize as cto  # type: ignore

    op_cfg = cto.coreml.OpPalettizerConfig(
        mode=mode,
        nbits=nbits,
        granularity=granularity,
        group_size=group_size,
        enable_per_channel_scale=enable_per_channel_scale,
        weight_threshold=weight_threshold,
    )
    cfg = cto.coreml.OptimizationConfig(global_config=op_cfg)
    return cto.coreml.palettize_weights(model, config=cfg)


def _linear_dtype_from_nbits(nbits: int) -> str:
    if nbits == 8:
        return "int8"
    if nbits == 4:
        return "int4"
    raise ValueError(f"Linear quantization supports nbits in {{4, 8}}. Got: {nbits}")


def compress_with_optimize_api_linear(
    model,
    nbits: int,
    mode: str,
    granularity: str,
    block_size: int,
    weight_threshold: int,
):
    import coremltools.optimize as cto  # type: ignore

    op_cfg = cto.coreml.OpLinearQuantizerConfig(
        mode=mode,
        dtype=_linear_dtype_from_nbits(nbits),
        granularity=granularity,
        block_size=block_size,
        weight_threshold=weight_threshold,
    )
    cfg = cto.coreml.OptimizationConfig(global_config=op_cfg)
    return cto.coreml.linear_quantize_weights(model, config=cfg)


def compress_with_legacy_api(model, nbits: int, mode: str):
    from coremltools.models.neural_network.quantization_utils import quantize_weights  # type: ignore

    quant_mode_map = {
        "kmeans": "kmeans_lut",
        "uniform": "linear",
    }
    mapped = quant_mode_map.get(mode, "kmeans_lut")
    return quantize_weights(model, nbits=nbits, quantization_mode=mapped)


def compress(
    model_path: Path,
    output_path: Path,
    nbits: int,
    algorithm: str,
    mode: str,
    group_size: int,
    block_size: int,
    granularity: str,
    enable_per_channel_scale: bool,
    weight_threshold: int,
):
    import coremltools as ct  # type: ignore

    model = ct.models.MLModel(str(model_path))
    model_type = model.get_spec().WhichOneof("Type")
    errors = []

    compressed = None

    if algorithm == "palettize":
        if mode not in {"kmeans", "uniform"}:
            raise ValueError(f"Unsupported palettize mode: {mode}")
        granularities = ["per_grouped_channel", "per_tensor"] if granularity == "auto" else [granularity]
    elif algorithm == "linear":
        if mode not in {"linear_symmetric", "linear"}:
            raise ValueError(f"Unsupported linear mode: {mode}")
        granularities = ["per_channel", "per_tensor"] if granularity == "auto" else [granularity]
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Try modern compression first. In "auto", fall back to a more
    # permissive granularity if needed.
    for granularity_item in granularities:
        try:
            if algorithm == "palettize":
                compressed = compress_with_optimize_api_palettize(
                    model=model,
                    nbits=nbits,
                    mode=mode,
                    group_size=group_size,
                    granularity=granularity_item,
                    enable_per_channel_scale=enable_per_channel_scale,
                    weight_threshold=weight_threshold,
                )
            else:
                compressed = compress_with_optimize_api_linear(
                    model=model,
                    nbits=nbits,
                    mode=mode,
                    granularity=granularity_item,
                    block_size=block_size,
                    weight_threshold=weight_threshold,
                )
            break
        except Exception as exc:
            errors.append(f"{algorithm} optimize API failed ({granularity_item}): {exc}")

    # Legacy API generally applies to neuralnetwork specs; skip it for mlProgram.
    # It only supports LUT-style palettization.
    if compressed is None and algorithm == "palettize" and model_type != "mlProgram":
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
    parser.add_argument("--algorithm", default="palettize", choices=["palettize", "linear"])
    parser.add_argument(
        "--mode",
        default=None,
        help=(
            "Compression mode. For palettize: kmeans|uniform. "
            "For linear: linear_symmetric|linear. "
            "Default depends on --algorithm."
        ),
    )
    parser.add_argument("--group-size", type=int, default=32, help="Grouped-channel size for palettization.")
    parser.add_argument("--block-size", type=int, default=32, help="Block size for linear quantization.")
    parser.add_argument(
        "--granularity",
        default="auto",
        choices=["auto", "per_grouped_channel", "per_tensor", "per_channel", "per_block"],
        help="Compression granularity. 'auto' picks algorithm-specific defaults.",
    )
    parser.add_argument(
        "--enable-per-channel-scale",
        action="store_true",
        help="Enable per-channel scaling before palettization.",
    )
    parser.add_argument(
        "--weight-threshold",
        type=int,
        default=2048,
        help="Skip compression for small weight tensors below this element count.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = args.mode
    if mode is None:
        mode = "kmeans" if args.algorithm == "palettize" else "linear_symmetric"

    output_path = compress(
        model_path=args.model,
        output_path=args.output,
        nbits=args.nbits,
        algorithm=args.algorithm,
        mode=mode,
        group_size=args.group_size,
        block_size=args.block_size,
        granularity=args.granularity,
        enable_per_channel_scale=args.enable_per_channel_scale,
        weight_threshold=args.weight_threshold,
    )
    print(f"Compression complete: {output_path}")


if __name__ == "__main__":
    main()
