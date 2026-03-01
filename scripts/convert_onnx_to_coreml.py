#!/usr/bin/env python3
"""Convert ONNX model(s) to CoreML with deployment and compute-unit controls."""

from __future__ import annotations

import argparse
from pathlib import Path


def _resolve_target(ct, raw_target: str):
    target_map = {
        "ios18": getattr(ct.target, "iOS18", None),
        "macos15": getattr(ct.target, "macOS15", None),
    }
    key = raw_target.lower()
    if key not in target_map:
        raise ValueError(f"Unsupported target '{raw_target}'. Use one of: {', '.join(target_map)}")
    resolved = target_map[key]
    if resolved is None:
        raise ValueError(
            f"Target '{raw_target}' is unavailable in this coremltools build. "
            "Use coremltools 8+/9+ with iOS18/macOS15 support."
        )
    return resolved


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


def convert(onnx_path: Path, output_path: Path, target: str, compute_units: str):
    try:
        import coremltools as ct  # type: ignore
    except ImportError as exc:
        raise SystemExit("coremltools is not installed. Run: pip install -r requirements-conversion.txt") from exc

    target_enum = _resolve_target(ct, target)
    compute_enum = _resolve_compute_units(ct, compute_units)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # The ONNX frontend has historically shifted APIs. Try current style first,
    # then fallback to legacy converter namespace.
    model = None
    conversion_errors = []
    try:
        model = ct.convert(
            str(onnx_path),
            source="onnx",
            minimum_deployment_target=target_enum,
            compute_units=compute_enum,
            convert_to="mlprogram",
        )
    except Exception as exc:
        conversion_errors.append(f"ct.convert(source='onnx') failed: {exc}")

    if model is None:
        try:
            model = ct.converters.onnx.convert(  # type: ignore[attr-defined]
                model=str(onnx_path),
                minimum_deployment_target=target_enum,
                compute_units=compute_enum,
            )
        except Exception as exc:
            conversion_errors.append(f"ct.converters.onnx.convert failed: {exc}")

    if model is None:
        joined = "\n".join(f"- {msg}" for msg in conversion_errors)
        raise RuntimeError(
            "Could not convert ONNX to CoreML.\n"
            "This usually means unsupported ONNX ops or a frontend mismatch.\n"
            "Details:\n"
            f"{joined}"
        )

    model.save(str(output_path))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ONNX model to CoreML.")
    parser.add_argument("--onnx", type=Path, required=True, help="Path to .onnx file.")
    parser.add_argument("--output", type=Path, required=True, help="Output .mlpackage or .mlmodel path.")
    parser.add_argument(
        "--target",
        default="macos15",
        help="Deployment target: ios18, macos15",
    )
    parser.add_argument(
        "--compute-units",
        default="all",
        help="Compute units: all, cpu_only, cpu_and_gpu, cpu_and_ne",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = convert(
        onnx_path=args.onnx,
        output_path=args.output,
        target=args.target,
        compute_units=args.compute_units,
    )
    print(f"CoreML conversion complete: {output_path}")


if __name__ == "__main__":
    main()
