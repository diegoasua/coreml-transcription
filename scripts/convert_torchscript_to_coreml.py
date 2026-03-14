#!/usr/bin/env python3
"""Convert TorchScript model(s) to CoreML using ONNX manifest input specs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _state_dtype_for(dtype: Any):
    # CoreML state tensors currently require fp16.
    if dtype in (np.float32, np.float64):
        return np.float16
    return dtype


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


def _dtype_from_manifest(dtype: str):
    key = dtype.upper()
    if key == "FLOAT":
        return np.float32
    if key in {"FLOAT16", "FP16", "HALF"}:
        return np.float16
    if key == "INT32":
        return np.int32
    if key == "INT64":
        return np.int64
    return np.float32


def _dtype_from_torch(torch_dtype):
    import torch

    if torch_dtype == torch.float16:
        return np.float16
    if torch_dtype == torch.float32:
        return np.float32
    if torch_dtype == torch.float64:
        return np.float64
    if torch_dtype == torch.int32:
        return np.int32
    if torch_dtype == torch.int64:
        return np.int64
    if torch_dtype == torch.int8:
        return np.int8
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
    from coremltools.converters.mil.frontend.torch.ops import (  # type: ignore
        _cast_to,
        _get_inputs,
        _get_kwinputs,
        NUMPY_DTYPE_TO_TORCH_NUM,
        NUM_TO_DTYPE_STRING,
        NUM_TO_TORCH_DTYPE,
        TorchFrontend,
        dtype_to_32bit,
        nptype_from_builtin,
    )
    from coremltools.converters.mil.mil import Builder as mb  # type: ignore
    from coremltools.converters.mil.mil import types  # type: ignore
    import torch

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
    try:
        # CoreMLTools 9 Torch frontend can receive string dtype tokens here.
        NUM_TO_DTYPE_STRING["fp32"] = "fp32"
        NUM_TO_DTYPE_STRING["float32"] = "fp32"
        NUM_TO_DTYPE_STRING["fp16"] = "fp16"
        NUM_TO_DTYPE_STRING["float16"] = "fp16"
    except Exception:
        pass

    def _dtype_str_from_to_arg(dtype_arg):
        mapping = {
            "fp16": "fp16",
            "float16": "fp16",
            "half": "fp16",
            "torch.float16": "fp16",
            "fp32": "fp32",
            "float32": "fp32",
            "float": "fp32",
            "torch.float32": "fp32",
            "int32": "int32",
            "torch.int32": "int32",
            "bool": "bool",
            "torch.bool": "bool",
        }

        if isinstance(dtype_arg, str):
            return mapping.get(dtype_arg.lower(), "fp32")

        try:
            if dtype_arg in NUM_TO_DTYPE_STRING:
                return NUM_TO_DTYPE_STRING[dtype_arg]
        except Exception:
            pass

        key = str(dtype_arg).lower()
        if key in mapping:
            return mapping[key]
        return "fp32"

    def patched_to(context, node):
        inputs = _get_inputs(
            context, node, expected={TorchFrontend.TORCHSCRIPT: (1, 2, 3, 4, 5, 6, 7, 8)}
        )
        nargs = len(inputs)
        _input = inputs[0]

        if context.frontend == TorchFrontend.TORCHSCRIPT:
            if nargs in (4, 5, 7, 8):
                dtype = inputs[1]
            elif nargs == 6:
                dtype = inputs[2]
            else:
                dtype = None

            if dtype is None:
                context.add(_input, torch_name=node.name)
                return
            if types.is_scalar(dtype.sym_type) and dtype.val is not None:
                dtype = dtype.val
            else:
                np_type = nptype_from_builtin(dtype.dtype)
                dtype = NUMPY_DTYPE_TO_TORCH_NUM[np_type]
        else:
            if node.kind in ("to.dtype", "_to_copy") and nargs > 1:
                dtype = inputs[1]
            else:
                dtype = None
            if dtype is None:
                context.add(_input, torch_name=node.name)
                return

        dtype = _get_kwinputs(context, node, "dtype", default=[dtype])[0]
        if isinstance(dtype, type(_input)):
            dtype = dtype.val

        if dtype is None:
            context.add(_input, torch_name=node.name)
            return

        if _input.can_be_folded_to_const() and not isinstance(dtype, str):
            torch_dtype = dtype_to_32bit(NUM_TO_TORCH_DTYPE[dtype])
            res = mb.const(val=torch.tensor(_input.val).type(torch_dtype).cpu().numpy())
        else:
            dtype_str = _dtype_str_from_to_arg(dtype)
            res = _cast_to(_input, dtype_str, node.name)
        context.add(res, node.name)

    torch_ops.to = patched_to
    try:
        torch_ops._TORCH_OPS_REGISTRY.set_func_by_name("to", patched_to)
        torch_ops._TORCH_OPS_REGISTRY.set_func_by_name("_to_copy", patched_to)
    except Exception:
        pass


def convert(
    torchscript_path: Path,
    manifest_path: Path,
    output_path: Path,
    target: str,
    compute_units: str,
    component_name: str | None,
    encoder_frames: int,
    decoder_frames: int,
    stateful_input_names: list[str] | None = None,
):
    import coremltools as ct  # type: ignore
    import torch

    target_enum = _resolve_target(ct, target)
    compute_enum = _resolve_compute_units(ct, compute_units)
    comp_name = _infer_component_name(torchscript_path=torchscript_path, component_name=component_name)

    _patch_coremltools_torch_cast_bug()

    model = torch.jit.load(str(torchscript_path), map_location="cpu")
    model.eval()

    inputs = _load_tensor_specs(
        ct=ct,
        manifest_path=manifest_path,
        component_name=comp_name,
        encoder_frames=encoder_frames,
        decoder_frames=decoder_frames,
    )

    inputs_for_convert = inputs
    states_for_convert = None
    if stateful_input_names:
        desired = {name.strip() for name in stateful_input_names if name.strip()}
        input_by_name = {spec.name: spec for spec in inputs}
        buffer_by_name = {name: buf for name, buf in model.named_buffers()}
        states_for_convert = []
        unavailable: list[str] = []
        for name in sorted(desired):
            if name in input_by_name:
                spec = input_by_name[name]
                wrapped = ct.TensorType(shape=spec.shape, dtype=_state_dtype_for(spec.dtype))
                states_for_convert.append(ct.StateType(wrapped_type=wrapped, name=name))
            elif name in buffer_by_name:
                buf = buffer_by_name[name]
                wrapped = ct.TensorType(
                    shape=tuple(int(x) for x in buf.shape),
                    dtype=_state_dtype_for(_dtype_from_torch(buf.dtype)),
                )
                states_for_convert.append(ct.StateType(wrapped_type=wrapped, name=name))
            else:
                unavailable.append(name)
        if unavailable:
            print(
                "[warning] requested stateful names not found in manifest inputs or TorchScript buffers; "
                f"stateful conversion disabled. missing: {sorted(unavailable)}; "
                f"available inputs: {sorted(input_by_name.keys())}; "
                f"available buffers: {sorted(buffer_by_name.keys())}"
            )
            states_for_convert = None

    convert_kwargs = dict(
        source="pytorch",
        inputs=inputs_for_convert,
        minimum_deployment_target=target_enum,
        compute_units=compute_enum,
        convert_to="mlprogram",
    )
    if states_for_convert:
        convert_kwargs["states"] = states_for_convert
    try:
        converted = ct.convert(model, **convert_kwargs)
    except Exception as exc:
        if states_for_convert and os.environ.get("COREML_STATEFUL_FALLBACK", "0") == "1":
            print(f"[warning] stateful conversion failed; retrying stateless conversion (COREML_STATEFUL_FALLBACK=1). reason: {exc}")
            convert_kwargs.pop("states", None)
            converted = ct.convert(model, **convert_kwargs)
        else:
            raise

    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(output_path))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    streaming_info = manifest.get("streaming_info")
    if isinstance(streaming_info, dict) and streaming_info:
        sidecar_path = output_path.with_suffix("").with_name(output_path.stem + "-streaming.json")
        sidecar_path.write_text(json.dumps(streaming_info, indent=2), encoding="utf-8")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert TorchScript model to CoreML.")
    parser.add_argument("--torchscript", type=Path, required=True, help="Path to .ts/.pt model.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to ONNX manifest JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Output .mlpackage path.")
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
    parser.add_argument("--component-name", default=None, help="Optional component name override.")
    parser.add_argument("--encoder-frames", type=int, default=300, help="Fallback input time frames for encoder.")
    parser.add_argument("--decoder-frames", type=int, default=75, help="Fallback encoder-output frames for decoder.")
    parser.add_argument(
        "--stateful-input-names",
        default="",
        help="Comma-separated input tensor names to convert into CoreML states (advanced).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        output_path = convert(
            torchscript_path=args.torchscript,
            manifest_path=args.manifest,
            output_path=args.output,
            target=args.target,
            compute_units=args.compute_units,
            component_name=args.component_name,
            encoder_frames=args.encoder_frames,
            decoder_frames=args.decoder_frames,
            stateful_input_names=[x for x in args.stateful_input_names.split(",") if x.strip()] if args.stateful_input_names else None,
        )
    except Exception as exc:
        print(f"[error] CoreML conversion failed for {args.torchscript}: {exc}")
        raise SystemExit(1) from exc
    print(f"CoreML conversion complete: {output_path}")


if __name__ == "__main__":
    main()
