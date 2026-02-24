#!/usr/bin/env python3
"""Inspect ONNX model I/O, graph ops, and write a Swift-friendly manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _shape_from_value_info(value_info: Any) -> list[Any]:
    tensor_type = value_info.type.tensor_type
    shape = []
    for dim in tensor_type.shape.dim:
        if dim.dim_param:
            shape.append(dim.dim_param)
        elif dim.dim_value:
            shape.append(int(dim.dim_value))
        else:
            shape.append("?")
    return shape


def _dtype_from_value_info(value_info: Any) -> str:
    import onnx

    tensor_type = value_info.type.tensor_type
    return onnx.TensorProto.DataType.Name(tensor_type.elem_type)


def _summarize_ios(value_infos: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for info in value_infos:
        rows.append(
            {
                "name": info.name,
                "dtype": _dtype_from_value_info(info),
                "shape": _shape_from_value_info(info),
            }
        )
    return rows


def _infer_candidate_names(inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, str | None]:
    input_names = [x["name"] for x in inputs]
    output_names = [x["name"] for x in outputs]

    def pick(candidates: list[str], pool: list[str]) -> str | None:
        lowered = {item.lower(): item for item in pool}
        for candidate in candidates:
            for key, original in lowered.items():
                if candidate in key:
                    return original
        return pool[0] if pool else None

    return {
        "audio_input_name": pick(["audio", "signal", "input"], input_names),
        "length_input_name": pick(["length", "len"], input_names),
        "logits_output_name": pick(["logits", "output", "probs"], output_names),
    }


def inspect(onnx_path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(onnx_path))
    graph = model.graph

    inputs = _summarize_ios(list(graph.input))
    outputs = _summarize_ios(list(graph.output))
    ops = Counter(node.op_type for node in graph.node)

    manifest = {
        "onnx_path": str(onnx_path),
        "inputs": inputs,
        "outputs": outputs,
        "opset_import": [
            {
                "domain": entry.domain or "ai.onnx",
                "version": int(entry.version),
            }
            for entry in model.opset_import
        ],
        "node_count": len(graph.node),
        "op_histogram": dict(sorted(ops.items(), key=lambda x: x[0])),
        "coreml_suggestions": _infer_candidate_names(inputs=inputs, outputs=outputs),
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ONNX model I/O and graph summary.")
    parser.add_argument("--onnx", type=Path, required=True, help="Path to ONNX model.")
    parser.add_argument(
        "--write-manifest",
        type=Path,
        help="Optional JSON output path to persist the inspection manifest.",
    )
    args = parser.parse_args()

    manifest = inspect(args.onnx)
    rendered = json.dumps(manifest, indent=2)
    print(rendered)

    if args.write_manifest:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nManifest written: {args.write_manifest}")


if __name__ == "__main__":
    main()
