#!/usr/bin/env python3
"""Compress CoreML weights with optimize API (and fallback to legacy quantization)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


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


def _sample_abs_values(arr: np.ndarray, sample_size: int) -> np.ndarray:
    flat = np.abs(arr.reshape(-1).astype(np.float32, copy=False))
    if flat.size <= sample_size:
        return flat
    step = max(1, flat.size // sample_size)
    sampled = flat[::step]
    if sampled.size > sample_size:
        sampled = sampled[:sample_size]
    return sampled


def _mixed_score(abs_values: np.ndarray, score_mode: str) -> float:
    eps = 1e-8
    if abs_values.size == 0:
        return 0.0
    if score_mode == "outlier_ratio":
        q99 = float(np.quantile(abs_values, 0.99))
        q999 = float(np.quantile(abs_values, 0.999))
        return q999 / (q99 + eps)
    if score_mode == "range_ratio":
        q95 = float(np.quantile(abs_values, 0.95))
        vmax = float(np.max(abs_values))
        return vmax / (q95 + eps)
    if score_mode == "std_ratio":
        return float(np.std(abs_values) / (np.mean(abs_values) + eps))
    raise ValueError(f"Unsupported mixed score mode: {score_mode}")


def compress_with_optimize_api_palettize_mixed(
    model,
    low_nbits: int,
    high_nbits: int,
    mode: str,
    group_size: int,
    granularity: str,
    enable_per_channel_scale: bool,
    weight_threshold: int,
    high_element_ratio: float,
    fp16_element_ratio: float,
    score_mode: str,
    sample_size: int,
):
    import coremltools.optimize as cto  # type: ignore

    if high_nbits <= low_nbits:
        return compress_with_optimize_api_palettize(
            model=model,
            nbits=low_nbits,
            mode=mode,
            group_size=group_size,
            granularity=granularity,
            enable_per_channel_scale=enable_per_channel_scale,
            weight_threshold=weight_threshold,
        )

    metadata = cto.coreml.get_weights_metadata(model, weight_threshold=weight_threshold)
    if not metadata:
        return compress_with_optimize_api_palettize(
            model=model,
            nbits=low_nbits,
            mode=mode,
            group_size=group_size,
            granularity=granularity,
            enable_per_channel_scale=enable_per_channel_scale,
            weight_threshold=weight_threshold,
        )

    stats = []
    total_elements = 0
    for name, info in metadata.items():
        vals = np.asarray(info.val)
        numel = int(vals.size)
        if numel <= 0:
            continue
        total_elements += numel
        abs_sample = _sample_abs_values(vals, sample_size=sample_size)
        score = _mixed_score(abs_sample, score_mode=score_mode)
        stats.append((name, numel, score))

    if not stats or total_elements <= 0:
        return compress_with_optimize_api_palettize(
            model=model,
            nbits=low_nbits,
            mode=mode,
            group_size=group_size,
            granularity=granularity,
            enable_per_channel_scale=enable_per_channel_scale,
            weight_threshold=weight_threshold,
        )

    stats.sort(key=lambda x: x[2], reverse=True)
    target_fp16_elements = int(max(0.0, min(1.0, fp16_element_ratio)) * total_elements)
    target_high_elements = int(max(0.0, min(1.0, high_element_ratio)) * total_elements)

    fp16_names = set()
    fp16_running = 0
    for name, numel, _ in stats:
        if fp16_running >= target_fp16_elements:
            break
        fp16_names.add(name)
        fp16_running += numel

    high_names = set()
    high_running = 0
    for name, numel, _ in stats:
        if name in fp16_names:
            continue
        if high_running >= target_high_elements:
            break
        high_names.add(name)
        high_running += numel

    low_cfg = cto.coreml.OpPalettizerConfig(
        mode=mode,
        nbits=low_nbits,
        granularity=granularity,
        group_size=group_size,
        enable_per_channel_scale=enable_per_channel_scale,
        weight_threshold=weight_threshold,
    )
    high_cfg = cto.coreml.OpPalettizerConfig(
        mode=mode,
        nbits=high_nbits,
        granularity=granularity,
        group_size=group_size,
        enable_per_channel_scale=enable_per_channel_scale,
        weight_threshold=weight_threshold,
    )
    op_name_configs = {name: high_cfg for name in high_names}
    for name in fp16_names:
        op_name_configs[name] = None
    cfg = cto.coreml.OptimizationConfig(global_config=low_cfg, op_name_configs=op_name_configs)

    achieved_high_ratio = (high_running / total_elements) if total_elements > 0 else 0.0
    achieved_fp16_ratio = (fp16_running / total_elements) if total_elements > 0 else 0.0
    print(
        "Mixed palettization selection: "
        f"{len(high_names)}/{len(stats)} weights high-bit, "
        f"{len(fp16_names)}/{len(stats)} weights fp16-skip, "
        f"target_high_element_ratio={high_element_ratio:.3f}, "
        f"achieved_high_element_ratio={achieved_high_ratio:.3f}, "
        f"target_fp16_element_ratio={fp16_element_ratio:.3f}, "
        f"achieved_fp16_element_ratio={achieved_fp16_ratio:.3f}, "
        f"low_nbits={low_nbits}, high_nbits={high_nbits}, score={score_mode}"
    )
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
    mixed_high_nbits: int,
    mixed_high_element_ratio: float,
    mixed_fp16_element_ratio: float,
    mixed_score_mode: str,
    mixed_sample_size: int,
):
    import coremltools as ct  # type: ignore

    model = ct.models.MLModel(str(model_path))
    model_type = model.get_spec().WhichOneof("Type")
    errors = []

    compressed = None

    if algorithm in {"palettize", "palettize_mixed"}:
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
            elif algorithm == "palettize_mixed":
                compressed = compress_with_optimize_api_palettize_mixed(
                    model=model,
                    low_nbits=nbits,
                    high_nbits=mixed_high_nbits,
                    mode=mode,
                    group_size=group_size,
                    granularity=granularity_item,
                    enable_per_channel_scale=enable_per_channel_scale,
                    weight_threshold=weight_threshold,
                    high_element_ratio=mixed_high_element_ratio,
                    fp16_element_ratio=mixed_fp16_element_ratio,
                    score_mode=mixed_score_mode,
                    sample_size=mixed_sample_size,
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
    if compressed is None and algorithm in {"palettize", "palettize_mixed"} and model_type != "mlProgram":
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
    parser.add_argument("--algorithm", default="palettize", choices=["palettize", "palettize_mixed", "linear"])
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
    parser.add_argument(
        "--mixed-high-nbits",
        type=int,
        default=8,
        choices=[2, 3, 4, 6, 8],
        help="High-bit setting for --algorithm palettize_mixed.",
    )
    parser.add_argument(
        "--mixed-high-element-ratio",
        type=float,
        default=0.5,
        help="Target fraction of weight elements assigned to high-bit for palettize_mixed.",
    )
    parser.add_argument(
        "--mixed-fp16-element-ratio",
        type=float,
        default=0.0,
        help="Target fraction of weight elements to keep uncompressed fp16 for palettize_mixed.",
    )
    parser.add_argument(
        "--mixed-score-mode",
        default="outlier_ratio",
        choices=["outlier_ratio", "range_ratio", "std_ratio"],
        help="Outlier scoring heuristic for palettize_mixed.",
    )
    parser.add_argument(
        "--mixed-sample-size",
        type=int,
        default=200000,
        help="Per-weight sample size for outlier scoring in palettize_mixed.",
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
        mixed_high_nbits=args.mixed_high_nbits,
        mixed_high_element_ratio=args.mixed_high_element_ratio,
        mixed_fp16_element_ratio=args.mixed_fp16_element_ratio,
        mixed_score_mode=args.mixed_score_mode,
        mixed_sample_size=args.mixed_sample_size,
    )
    print(f"Compression complete: {output_path}")


if __name__ == "__main__":
    main()
