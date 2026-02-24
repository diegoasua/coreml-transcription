#!/usr/bin/env python3
"""Convert TorchScript model(s) to CoreML using ONNX manifest input specs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _resolve_target(ct, raw_target: str):
    target_map = {
        "ios17": ct.target.iOS17,
        "ios18": getattr(ct.target, "iOS18", ct.target.iOS17),
        "macos14": getattr(ct.target, "macOS14", ct.target.iOS17),
        "macos15": getattr(ct.target, "macOS15", getattr(ct.target, "macOS14", ct.target.iOS18)),
    }
    key = raw_target.lower()
    if key not in target_map:
        raise ValueError(f"Unsupported target '{raw_target}'. Use one of: {', '.join(target_map)}")
    return target_map[key]


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


def _dtype_from_manifest(dtype: str):
    key = dtype.upper()
    if key == "FLOAT":
        return np.float32
    if key == "INT32":
        return np.int32
    if key == "INT64":
        return np.int64
    return np.float32


def _infer_component_name(torchscript_path: Path, component_name: str | None) -> str:
    if component_name:
        return component_name
    return torchscript_path.stem.lower()


def _fallback_dynamic_dim(name: str, dim_index: int, component_name: str, encoder_frames: int, decoder_frames: int) -> int:
    lname = name.lower()
    comp = component_name.lower()

    if "encoder" in comp and lname == "audio_signal":
        if dim_index == 0:
            return 1
        if dim_index == 2:
            return encoder_frames
    if "encoder" in comp and lname == "length":
        return 1

    if "decoder" in comp and lname == "encoder_outputs":
        if dim_index == 0:
            return 1
        if dim_index == 2:
            return decoder_frames
    if "decoder" in comp and lname in {"targets", "target_length"}:
        return 1
    if "decoder" in comp and lname.startswith("input_states_"):
        if dim_index == 1:
            return 1

    return 1


def _shape_from_manifest(
    shape: list[Any],
    input_name: str,
    component_name: str,
    encoder_frames: int,
    decoder_frames: int,
) -> tuple[Any, ...]:
    resolved = []
    for idx, dim in enumerate(shape):
        if isinstance(dim, int):
            resolved.append(dim)
        else:
            resolved.append(
                _fallback_dynamic_dim(
                    name=input_name,
                    dim_index=idx,
                    component_name=component_name,
                    encoder_frames=encoder_frames,
                    decoder_frames=decoder_frames,
                )
            )
    return tuple(resolved)


def _load_tensor_specs(
    ct,
    manifest_path: Path,
    component_name: str,
    encoder_frames: int,
    decoder_frames: int,
):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs = []
    for entry in manifest["inputs"]:
        name = entry["name"]
        shape = _shape_from_manifest(
            shape=entry["shape"],
            input_name=name,
            component_name=component_name,
            encoder_frames=encoder_frames,
            decoder_frames=decoder_frames,
        )
        dtype = _dtype_from_manifest(entry.get("dtype", "FLOAT"))
        specs.append(ct.TensorType(name=name, shape=shape, dtype=dtype))
    return specs


def _patch_coremltools_torch_cast_bug() -> None:
    """Patch coremltools torch frontend for size-1 ndarray int-cast constants.

    CoreMLTools 9 can fail on TorchScript graphs with:
    TypeError: only 0-dimensional arrays can be converted to Python scalars
    during torch `int` op conversion.
    """
    from coremltools.converters.mil.frontend.torch import ops as torch_ops  # type: ignore
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs  # type: ignore
    from coremltools.converters.mil.mil import Builder as mb  # type: ignore

    def patched_cast(context, node, dtype, dtype_name):
        inputs = _get_inputs(context, node, expected=1)
        x = inputs[0]
        if not (len(x.shape) == 0 or np.all([d == 1 for d in x.shape])):
            raise ValueError("input to cast must be either a scalar or a length 1 tensor")

        if x.can_be_folded_to_const():
            if not isinstance(x.val, dtype):
                value = x.val
                if isinstance(value, np.ndarray):
                    if value.size != 1:
                        raise ValueError("const cast expected size-1 array")
                    value = value.reshape(-1)[0]
                res = mb.const(val=dtype(value), name=node.name)
            else:
                res = x
        elif len(x.shape) > 0:
            x = mb.squeeze(x=x, name=node.name + "_item")
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        else:
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        context.add(res, node.name)

    torch_ops._cast = patched_cast


def convert(
    torchscript_path: Path,
    manifest_path: Path,
    output_path: Path,
    target: str,
    compute_units: str,
    component_name: str | None,
    encoder_frames: int,
    decoder_frames: int,
):
    import coremltools as ct  # type: ignore
    import torch

    target_enum = _resolve_target(ct, target)
    compute_enum = _resolve_compute_units(ct, compute_units)
    comp_name = _infer_component_name(torchscript_path=torchscript_path, component_name=component_name)

    _patch_coremltools_torch_cast_bug()

    inputs = _load_tensor_specs(
        ct=ct,
        manifest_path=manifest_path,
        component_name=comp_name,
        encoder_frames=encoder_frames,
        decoder_frames=decoder_frames,
    )

    model = torch.jit.load(str(torchscript_path), map_location="cpu")
    model.eval()

    converted = ct.convert(
        model,
        source="pytorch",
        inputs=inputs,
        minimum_deployment_target=target_enum,
        compute_units=compute_enum,
        convert_to="mlprogram",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(output_path))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert TorchScript model to CoreML.")
    parser.add_argument("--torchscript", type=Path, required=True, help="Path to .ts/.pt model.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to ONNX manifest JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Output .mlpackage path.")
    parser.add_argument(
        "--target",
        default="macos14",
        help="Deployment target: ios17, ios18, macos14, macos15",
    )
    parser.add_argument(
        "--compute-units",
        default="all",
        help="Compute units: all, cpu_only, cpu_and_gpu, cpu_and_ne",
    )
    parser.add_argument("--component-name", default=None, help="Optional component name override.")
    parser.add_argument("--encoder-frames", type=int, default=300, help="Fallback input time frames for encoder.")
    parser.add_argument("--decoder-frames", type=int, default=75, help="Fallback encoder-output frames for decoder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = convert(
        torchscript_path=args.torchscript,
        manifest_path=args.manifest,
        output_path=args.output,
        target=args.target,
        compute_units=args.compute_units,
        component_name=args.component_name,
        encoder_frames=args.encoder_frames,
        decoder_frames=args.decoder_frames,
    )
    print(f"CoreML conversion complete: {output_path}")


if __name__ == "__main__":
    main()
