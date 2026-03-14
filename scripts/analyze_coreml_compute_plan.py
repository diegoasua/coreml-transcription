#!/usr/bin/env python3
"""Summarize Core ML compute-plan device placement for a model."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
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


def _device_label(device: Any) -> str:
    if device is None:
        return "None"
    return type(device).__name__.replace("ML", "").replace("ComputeDevice", "")


def _top_entries(counter: Counter, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, count in counter.most_common(limit):
        if isinstance(key, tuple):
            row = {"key": list(key), "count": count}
        else:
            row = {"key": key, "count": count}
        rows.append(row)
    return rows


def _compile_model_if_needed(model_path: Path, cache_dir: Path) -> Path:
    if model_path.suffix.lower() == ".mlmodelc":
        return model_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    compiled_path = cache_dir / f"{model_path.stem}.mlmodelc"
    needs_compile = True
    if compiled_path.exists():
        needs_compile = compiled_path.stat().st_mtime < model_path.stat().st_mtime
    if needs_compile:
        if compiled_path.exists():
            subprocess.run(["rm", "-rf", str(compiled_path)], check=True)
        subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(model_path), str(cache_dir)],
            check=True,
        )
    return compiled_path


def _summarize_mlprogram(plan, block, top_k: int) -> dict[str, Any]:
    preferred_counts: Counter[str] = Counter()
    support_signature_counts: Counter[tuple[str, ...]] = Counter()
    op_preferred_counts: Counter[tuple[str, str]] = Counter()
    cpu_preferred_counts: Counter[str] = Counter()
    cpu_only_counts: Counter[str] = Counter()
    neural_preferred_counts: Counter[str] = Counter()
    estimated_cost_by_device: defaultdict[str, float] = defaultdict(float)

    for operation in block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        preferred_device = _device_label(usage.preferred_compute_device if usage else None)
        supported_devices = tuple(_device_label(d) for d in (usage.supported_compute_devices if usage else []))
        preferred_counts[preferred_device] += 1
        support_signature_counts[supported_devices] += 1
        op_preferred_counts[(operation.operator_name, preferred_device)] += 1
        if preferred_device == "CPU":
            cpu_preferred_counts[operation.operator_name] += 1
        if supported_devices == ("CPU",):
            cpu_only_counts[operation.operator_name] += 1
        if preferred_device == "NeuralEngine":
            neural_preferred_counts[operation.operator_name] += 1
        cost = plan.get_estimated_cost_for_mlprogram_operation(operation)
        if cost is not None:
            estimated_cost_by_device[preferred_device] += float(cost.weight)

    total_cost = sum(estimated_cost_by_device.values())
    cost_share = {
        device: (value / total_cost if total_cost > 0 else 0.0)
        for device, value in sorted(estimated_cost_by_device.items())
    }

    return {
        "representation": "mlprogram",
        "operation_count": len(block.operations),
        "preferred_device_counts": dict(preferred_counts),
        "supported_device_signature_counts": {",".join(sig) if sig else "": count for sig, count in support_signature_counts.items()},
        "estimated_cost_by_preferred_device": dict(estimated_cost_by_device),
        "estimated_cost_share_by_preferred_device": cost_share,
        "top_op_preferred_counts": _top_entries(op_preferred_counts, top_k),
        "top_cpu_preferred_ops": _top_entries(cpu_preferred_counts, top_k),
        "top_cpu_only_ops": _top_entries(cpu_only_counts, top_k),
        "top_neural_preferred_ops": _top_entries(neural_preferred_counts, top_k),
    }


def _summarize_neuralnetwork(plan, layers, top_k: int) -> dict[str, Any]:
    preferred_counts: Counter[str] = Counter()
    support_signature_counts: Counter[tuple[str, ...]] = Counter()
    layer_preferred_counts: Counter[tuple[str, str]] = Counter()
    cpu_preferred_counts: Counter[str] = Counter()
    cpu_only_counts: Counter[str] = Counter()
    neural_preferred_counts: Counter[str] = Counter()

    for layer in layers:
        usage = plan.get_compute_device_usage_for_neuralnetwork_layer(layer)
        preferred_device = _device_label(usage.preferred_compute_device if usage else None)
        supported_devices = tuple(_device_label(d) for d in (usage.supported_compute_devices if usage else []))
        preferred_counts[preferred_device] += 1
        support_signature_counts[supported_devices] += 1
        layer_preferred_counts[(layer.type, preferred_device)] += 1
        if preferred_device == "CPU":
            cpu_preferred_counts[layer.type] += 1
        if supported_devices == ("CPU",):
            cpu_only_counts[layer.type] += 1
        if preferred_device == "NeuralEngine":
            neural_preferred_counts[layer.type] += 1

    return {
        "representation": "neuralnetwork",
        "layer_count": len(layers),
        "preferred_device_counts": dict(preferred_counts),
        "supported_device_signature_counts": {",".join(sig) if sig else "": count for sig, count in support_signature_counts.items()},
        "top_layer_preferred_counts": _top_entries(layer_preferred_counts, top_k),
        "top_cpu_preferred_layers": _top_entries(cpu_preferred_counts, top_k),
        "top_cpu_only_layers": _top_entries(cpu_only_counts, top_k),
        "top_neural_preferred_layers": _top_entries(neural_preferred_counts, top_k),
    }


def analyze_model(model_path: Path, compute_units: str, top_k: int, cache_dir: Path | None) -> dict[str, Any]:
    import coremltools as ct  # type: ignore
    from coremltools.models.compute_plan import MLComputePlan  # type: ignore

    compiled_path = _compile_model_if_needed(
        model_path=model_path,
        cache_dir=cache_dir or model_path.parent / ".mlmodelc-analyze-cache",
    )
    plan = MLComputePlan.load_from_path(
        str(compiled_path),
        compute_units=_resolve_compute_units(ct, compute_units),
    )
    model_structure = plan.model_structure

    out: dict[str, Any] = {
        "model_path": str(model_path),
        "compiled_model_path": str(compiled_path),
        "compute_units": compute_units,
    }

    if model_structure.program is not None:
        if not model_structure.program.functions:
            raise RuntimeError("ML Program has no functions.")
        function_name, function = next(iter(model_structure.program.functions.items()))
        out["function_name"] = function_name
        out.update(_summarize_mlprogram(plan, function.block, top_k=top_k))
        return out

    if model_structure.neuralnetwork is not None:
        out.update(_summarize_neuralnetwork(plan, model_structure.neuralnetwork.layers, top_k=top_k))
        return out

    if model_structure.pipeline is not None:
        out["representation"] = "pipeline"
        out["sub_models"] = list(model_structure.pipeline.sub_models.keys())
        return out

    raise RuntimeError("Unsupported model structure: no program, neuralnetwork, or pipeline found.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Core ML compute-plan placement for a model.")
    parser.add_argument("--model", type=Path, required=True, help="Path to .mlpackage, .mlmodel, or .mlmodelc.")
    parser.add_argument(
        "--compute-units",
        default="cpu_and_ne",
        help="all, cpu_only, cpu_and_gpu, cpu_and_ne",
    )
    parser.add_argument("--top-k", type=int, default=12, help="How many top op summaries to include.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional compiled-model cache directory. Defaults beside the model.",
    )
    args = parser.parse_args()

    report = analyze_model(
        model_path=args.model,
        compute_units=args.compute_units,
        top_k=max(1, args.top_k),
        cache_dir=args.cache_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
