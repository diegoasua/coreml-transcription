#!/usr/bin/env python3
"""Export a NeMo ASR model (e.g. Parakeet) to ONNX/TorchScript with metadata."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _safe_getattr(obj: Any, path: str, default: Any = None) -> Any:
    current = obj
    for chunk in path.split("."):
        if not hasattr(current, chunk):
            return default
        current = getattr(current, chunk)
    return current


def _discover_onnx_artifacts(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("*.onnx"))


def _discover_torchscript_artifacts(output_dir: Path) -> list[Path]:
    files = sorted(output_dir.glob("*.ts"))
    files.extend(sorted(output_dir.glob("*.pt")))
    dedup: dict[str, Path] = {}
    for file_path in files:
        dedup[str(file_path)] = file_path
    return sorted(dedup.values())


def _write_metadata(
    model: Any,
    output_dir: Path,
    model_name: str,
    onnx_paths: list[Path],
    torchscript_paths: list[Path],
) -> None:
    tokenizer_vocab = []
    tokenizer = _safe_getattr(model, "tokenizer")
    if tokenizer is not None and hasattr(tokenizer, "vocab"):
        try:
            tokenizer_vocab = list(tokenizer.vocab)
        except Exception:
            tokenizer_vocab = []

    metadata = {
        "source_model": model_name,
        "onnx_paths": [str(path) for path in onnx_paths],
        "torchscript_paths": [str(path) for path in torchscript_paths],
        "sample_rate": _safe_getattr(model, "_cfg.sample_rate"),
        "model_type": type(model).__name__,
        "token_count": len(tokenizer_vocab),
        "has_tokenizer": tokenizer is not None,
        "has_encoder": hasattr(model, "encoder"),
        "has_decoder": hasattr(model, "decoder"),
        "has_joint": hasattr(model, "joint"),
    }

    (output_dir / "export-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if tokenizer_vocab:
        (output_dir / "vocab.txt").write_text("\n".join(tokenizer_vocab), encoding="utf-8")


@contextmanager
def _force_legacy_torch_onnx_export():
    """Force torch.onnx.export to use the legacy path (dynamo=False).

    PyTorch >= 2.10 defaults to dynamo=True, while current NeMo export code
    still passes dynamic_axes in a way that can fail in the compat bridge.
    """
    import torch

    original_export = torch.onnx.export

    def wrapped_export(*args, **kwargs):
        kwargs.setdefault("dynamo", False)
        kwargs.setdefault("external_data", True)
        return original_export(*args, **kwargs)

    torch.onnx.export = wrapped_export  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.onnx.export = original_export  # type: ignore[assignment]


def _run_nemo_export(model: Any, onnx_path: Path, opset: int) -> None:
    errors = []

    # Attempt 1: NeMo export with legacy torch.onnx path.
    try:
        with _force_legacy_torch_onnx_export():
            model.export(output=str(onnx_path), check_trace=False, onnx_opset_version=opset)
        return
    except TypeError:
        try:
            with _force_legacy_torch_onnx_export():
                model.export(str(onnx_path))
            return
        except Exception as exc:  # pragma: no cover - runtime-path fallback
            errors.append(f"legacy export fallback signature failed: {exc}")
    except Exception as exc:  # pragma: no cover - runtime-path fallback
        errors.append(f"legacy exporter path failed: {exc}")

    # Attempt 2: Explicitly disable dynamic axes (static-shape export fallback).
    try:
        with _force_legacy_torch_onnx_export():
            model.export(
                output=str(onnx_path),
                check_trace=False,
                onnx_opset_version=opset,
                dynamic_axes={},
            )
        return
    except Exception as exc:  # pragma: no cover - runtime-path fallback
        errors.append(f"static dynamic_axes={{}} fallback failed: {exc}")

    # Attempt 3: NeMo dynamo path.
    try:
        model.export(output=str(onnx_path), check_trace=False, onnx_opset_version=opset, use_dynamo=True)
        return
    except Exception as exc:  # pragma: no cover - runtime-path fallback
        errors.append(f"use_dynamo=True fallback failed: {exc}")

    detail = "\n".join(f"- {entry}" for entry in errors)
    raise RuntimeError(f"NeMo->ONNX export failed after compatibility fallbacks.\n{detail}")


def _run_nemo_torchscript_export(model: Any, torchscript_path: Path) -> None:
    errors = []
    try:
        model.export(output=str(torchscript_path), check_trace=False)
        return
    except TypeError:
        try:
            model.export(str(torchscript_path))
            return
        except Exception as exc:  # pragma: no cover - runtime-path fallback
            errors.append(f"fallback signature failed: {exc}")
    except Exception as exc:  # pragma: no cover - runtime-path fallback
        errors.append(str(exc))

    detail = "\n".join(f"- {entry}" for entry in errors)
    raise RuntimeError(f"NeMo->TorchScript export failed.\n{detail}")


def export_model(model_name: str, output_dir: Path, opset: int, formats: set[str]) -> Path:
    try:
        import onnxscript  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: onnxscript.\n"
            "Install with: pip install onnxscript\n"
            "or reinstall: pip install -r requirements-conversion.txt"
        ) from exc

    try:
        import nemo.collections.asr as nemo_asr  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "nemo_toolkit[asr] is not installed. Run: pip install -r requirements-conversion.txt"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    model.eval()
    model.to("cpu")

    onnx_paths = _discover_onnx_artifacts(output_dir)
    torchscript_paths = _discover_torchscript_artifacts(output_dir)

    if "onnx" in formats:
        _run_nemo_export(model=model, onnx_path=output_dir / "model.onnx", opset=opset)
        onnx_paths = _discover_onnx_artifacts(output_dir)

    if "ts" in formats:
        _run_nemo_torchscript_export(model=model, torchscript_path=output_dir / "model.ts")
        torchscript_paths = _discover_torchscript_artifacts(output_dir)

    if "onnx" in formats and not onnx_paths:
        raise RuntimeError(
            "Export completed without ONNX files in output directory. "
            "Expected at least one *.onnx artifact."
        )
    if "ts" in formats and not torchscript_paths:
        raise RuntimeError(
            "Export completed without TorchScript files in output directory. "
            "Expected at least one *.ts or *.pt artifact."
        )

    _write_metadata(
        model=model,
        output_dir=output_dir,
        model_name=model_name,
        onnx_paths=onnx_paths,
        torchscript_paths=torchscript_paths,
    )

    if onnx_paths:
        return onnx_paths[0]
    if torchscript_paths:
        return torchscript_paths[0]
    raise RuntimeError("No artifacts exported.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a NeMo ASR model to ONNX/TorchScript.")
    parser.add_argument(
        "--model",
        default="nvidia/parakeet-tdt-0.6b-v2",
        help="NeMo model name. Example: nvidia/parakeet-tdt-0.6b-v2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/parakeet-tdt-0.6b-v2"),
        help="Directory where model.onnx and metadata are written.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--formats",
        default="onnx,ts",
        help="Comma-separated export formats: onnx, ts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = {entry.strip().lower() for entry in args.formats.split(",") if entry.strip()}
    valid_formats = {"onnx", "ts"}
    invalid = sorted(formats - valid_formats)
    if invalid:
        raise SystemExit(f"Unsupported format(s): {', '.join(invalid)}. Allowed: onnx, ts")
    if not formats:
        raise SystemExit("No formats requested. Use --formats onnx,ts")

    _ = export_model(model_name=args.model, output_dir=args.output_dir, opset=args.opset, formats=formats)
    exported = sorted(args.output_dir.glob("*.onnx")) + sorted(args.output_dir.glob("*.ts")) + sorted(
        args.output_dir.glob("*.pt")
    )
    print("Export complete:")
    for path in exported:
        print(f"- {path}")


if __name__ == "__main__":
    main()
