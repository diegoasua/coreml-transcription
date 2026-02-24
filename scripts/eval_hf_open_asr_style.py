#!/usr/bin/env python3
"""Evaluate ASR with an Open ASR Leaderboard-style harness.

This runner is intentionally independent from OpenBench and follows the
evaluation pattern used in Hugging Face's open_asr_leaderboard:
  - load dataset via `datasets.load_dataset(...)`
  - normalize predictions with an English normalizer
  - compute WER via `evaluate.load("wer")`
  - report inverse real-time factor (RTFx)

It supports local command-based or python-module transcribers.
"""

from __future__ import annotations

import argparse
import io
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


def _load_english_spelling_mapping() -> dict[str, str]:
    # Prefer OpenBench's mapping if available locally.
    try:
        from openbench.metric.word_error_metrics.english_abbreviations import ABBR  # type: ignore

        if isinstance(ABBR, dict):
            return dict(ABBR)
    except Exception:
        pass

    # Fallback: load directly from cloned OpenBench tree if present.
    try:
        repo_root = Path(__file__).resolve().parents[1]
        abbr_path = repo_root / "external/OpenBench/src/openbench/metric/word_error_metrics/english_abbreviations.py"
        if abbr_path.exists():
            spec = importlib.util.spec_from_file_location("openbench_english_abbr", abbr_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                abbr = getattr(module, "ABBR", None)
                if isinstance(abbr, dict):
                    return dict(abbr)
    except Exception:
        pass

    return {}


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate ASR in Open ASR Leaderboard style.")

    parser.add_argument(
        "--dataset-path",
        type=str,
        default="hf-audio/esb-datasets-test-only-sorted",
        help=(
            "Dataset repository path. "
            "Recommended: hf-audio/esb-datasets-test-only-sorted (safe parquet)."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Dataset config name. If omitted, will auto-resolve from config names using "
            "--dataset-pattern."
        ),
    )
    parser.add_argument(
        "--dataset-pattern",
        type=str,
        default="earnings22",
        help="Pattern used to auto-pick dataset config when --dataset is omitted.",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split (default: test).")
    parser.add_argument("--streaming", action="store_true", help="Stream dataset from HF.")
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=0)

    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument(
        "--python-transcriber",
        type=Path,
        default=None,
        help=(
            "Path to python module exporting transcribe_file(audio_path, language=None, keywords=None)->str. "
            "Optional warmup() will be called if present."
        ),
    )
    backend.add_argument(
        "--transcribe-cmd",
        type=str,
        default=None,
        help="Command template including {audio_path}.",
    )

    parser.add_argument("--command-cwd", type=Path, default=repo_root)
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument(
        "--normalizer",
        choices=("open_asr", "basic", "none"),
        default="open_asr",
        help="Text normalization strategy for both references and predictions.",
    )
    parser.add_argument(
        "--reference-column",
        type=str,
        default=None,
        help="Explicit reference text column (default: auto from norm_text/text/transcription/sentence).",
    )
    parser.add_argument("--audio-column", type=str, default="audio")
    parser.add_argument("--id-column", type=str, default="id")

    parser.add_argument("--output-dir", type=Path, default=repo_root / "artifacts/hf-asr-eval")
    parser.add_argument("--run-name", type=str, default="parakeet-coreml-earnings22-hf")
    return parser.parse_args()


def _basic_normalize(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^a-z0-9'\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _open_asr_normalize(text: str) -> str:
    # Mirrors HF Open ASR leaderboard tooling:
    # normalizer = EnglishTextNormalizer()
    # ref = normalizer(text)
    try:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    except Exception:
        return _basic_normalize(text)

    try:
        # Newer transformers variants accept no args.
        normalizer = EnglishTextNormalizer()  # type: ignore[call-arg]
    except TypeError:
        # Some versions require british->american spelling map argument.
        normalizer = EnglishTextNormalizer(_load_english_spelling_mapping())  # type: ignore[call-arg]
    return normalizer(text)


def _normalize(text: str, mode: str) -> str:
    if mode == "none":
        return text.strip()
    if mode == "basic":
        return _basic_normalize(text)
    return _open_asr_normalize(text)


@dataclass
class EvalRow:
    sample_id: str
    reference: str
    prediction: str
    audio_sec: float
    infer_sec: float


def _resolve_dataset_config(dataset_path: str, dataset: str | None, pattern: str) -> str:
    if dataset:
        return dataset

    from datasets import get_dataset_config_names

    configs = list(get_dataset_config_names(dataset_path))
    if not configs:
        raise RuntimeError(f"No dataset configs found for {dataset_path!r}")
    lower_pattern = pattern.lower()
    ranked = sorted(configs, key=lambda name: (lower_pattern not in name.lower(), len(name)))
    return ranked[0]


def _resolve_split(dataset_path: str, dataset: str, split: str) -> str:
    from datasets import get_dataset_split_names

    splits = list(get_dataset_split_names(dataset_path, dataset))
    if split in splits:
        return split
    if "test" in splits:
        return "test"
    if splits:
        return splits[0]
    raise RuntimeError(f"No splits found for {dataset_path}:{dataset}")


def _build_backend(
    *,
    python_transcriber: Path | None,
    transcribe_cmd: str | None,
    command_cwd: Path,
    timeout_sec: float,
) -> Callable[[Path], str]:
    command_cwd.mkdir(parents=True, exist_ok=True)

    if python_transcriber is not None:
        path = python_transcriber
        if not path.is_absolute():
            path = (command_cwd / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Python transcriber module not found: {path}")

        module_name = f"hf_asr_transcriber_{abs(hash(str(path)))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load transcriber module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        warmup_fn = getattr(module, "warmup", None)
        if callable(warmup_fn):
            warmup_fn()

        transcribe_fn = getattr(module, "transcribe_file", None)
        if not callable(transcribe_fn):
            raise RuntimeError(
                f"Transcriber module '{path}' must export callable "
                "transcribe_file(audio_path, language=None, keywords=None)->str"
            )

        def _run_python(audio_path: Path) -> str:
            out = transcribe_fn(str(audio_path), language=None, keywords=None)
            return str(out)

        return _run_python

    assert transcribe_cmd is not None
    if "{audio_path}" not in transcribe_cmd:
        raise RuntimeError("--transcribe-cmd must include {audio_path}")

    def _run_cmd(audio_path: Path) -> str:
        cmd = transcribe_cmd.format(audio_path=shlex.quote(str(audio_path)))
        completed = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=command_cwd,
            check=False,
        )
        if completed.returncode != 0:
            err = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Command failed (code={completed.returncode}): {err[:500]}")
        return completed.stdout.strip()

    return _run_cmd


def _pick_reference_text(row: dict[str, Any], explicit_col: str | None) -> str:
    if explicit_col is not None:
        return str(row.get(explicit_col, ""))
    for candidate in ("norm_text", "text", "transcription", "sentence"):
        if candidate in row:
            value = row[candidate]
            if isinstance(value, list):
                return " ".join(str(v) for v in value)
            return str(value)
    return ""


def _iter_dataset(
    dataset_path: str,
    dataset_config: str,
    split: str,
    audio_column: str,
    streaming: bool,
    max_eval_samples: int | None,
) -> Iterable[dict[str, Any]]:
    from datasets import Audio, load_dataset

    ds = load_dataset(dataset_path, dataset_config, split=split, streaming=streaming)
    # Avoid datasets' runtime audio decoding dependency on torchcodec.
    try:
        ds = ds.cast_column(audio_column, Audio(decode=False))
    except Exception:
        # If casting is unsupported for this dataset layout, continue and let
        # the row decoder handle whichever shape is returned.
        pass

    if max_eval_samples is None or max_eval_samples <= 0:
        return ds
    if streaming:
        return ds.take(max_eval_samples)
    return ds.select(range(min(max_eval_samples, len(ds))))


def _decode_audio_field(payload: Any) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf  # type: ignore
    except Exception as exc:
        raise RuntimeError("soundfile is required for temporary WAV writing.") from exc

    if isinstance(payload, dict):
        # Already decoded path.
        if "array" in payload and "sampling_rate" in payload:
            arr = np.asarray(payload["array"], dtype=np.float32)
            sr = int(payload["sampling_rate"])
            if arr.ndim == 2:
                arr = np.mean(arr, axis=1, dtype=np.float32)
            return arr.astype(np.float32, copy=False), sr

        # Arrow storage shape with bytes blob.
        raw_bytes = payload.get("bytes")
        if raw_bytes is not None:
            if isinstance(raw_bytes, memoryview):
                raw_bytes = raw_bytes.tobytes()
            elif isinstance(raw_bytes, bytearray):
                raw_bytes = bytes(raw_bytes)
            if isinstance(raw_bytes, bytes) and len(raw_bytes) > 0:
                arr, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 2:
                    arr = np.mean(arr, axis=1, dtype=np.float32)
                return arr.astype(np.float32, copy=False), int(sr)

        # Arrow storage shape with lazy file path.
        if payload.get("path"):
            path = Path(str(payload["path"]))
            if not path.exists():
                # Some datasets store only file name; try common HF cache roots.
                cache_root = Path.home() / ".cache" / "huggingface" / "datasets"
                matches = list(cache_root.rglob(path.name))
                if matches:
                    path = matches[0]
            arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 2:
                arr = np.mean(arr, axis=1, dtype=np.float32)
            return arr.astype(np.float32, copy=False), int(sr)

    raise RuntimeError(
        "Unsupported audio payload format. Expected decoded array+sampling_rate or path/bytes dict. "
        f"Got: {type(payload)!r}"
    )


def main() -> None:
    args = _parse_args()

    dataset_config = _resolve_dataset_config(args.dataset_path, args.dataset, args.dataset_pattern)
    split = _resolve_split(args.dataset_path, dataset_config, args.split)
    backend = _build_backend(
        python_transcriber=args.python_transcriber,
        transcribe_cmd=args.transcribe_cmd,
        command_cwd=args.command_cwd,
        timeout_sec=args.timeout_sec,
    )

    out_dir = args.output_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    iterable = _iter_dataset(
        dataset_path=args.dataset_path,
        dataset_config=dataset_config,
        split=split,
        audio_column=args.audio_column,
        streaming=args.streaming,
        max_eval_samples=args.max_eval_samples,
    )

    # Warmup with first N samples (not counted) to reduce cold-start noise.
    warmup_buffer: list[dict[str, Any]] = []
    iterator = iter(iterable)
    for _ in range(max(0, int(args.warmup_steps))):
        try:
            warmup_buffer.append(next(iterator))
        except StopIteration:
            break

    with tempfile.TemporaryDirectory(prefix="hf_asr_eval_") as tmp:
        tmp_dir = Path(tmp)
        try:
            import soundfile as sf  # type: ignore
        except Exception as exc:
            raise RuntimeError("soundfile is required for temporary WAV writing.") from exc

        for idx, row in enumerate(warmup_buffer, start=1):
            audio_arr, sr = _decode_audio_field(row[args.audio_column])
            wav_path = tmp_dir / f"warmup-{idx}.wav"
            sf.write(wav_path, audio_arr, sr)
            _ = backend(wav_path)

        rows: list[EvalRow] = []
        total_audio_sec = 0.0
        total_infer_sec = 0.0

        for i, row in enumerate(iterator, start=1):
            audio_arr, sr = _decode_audio_field(row[args.audio_column])
            wav_path = tmp_dir / f"{i}.wav"
            sf.write(wav_path, audio_arr, sr)

            t0 = time.perf_counter()
            pred = backend(wav_path)
            infer_sec = time.perf_counter() - t0

            ref_text = _pick_reference_text(dict(row), args.reference_column)
            ref_norm = _normalize(ref_text, args.normalizer)
            pred_norm = _normalize(pred, args.normalizer)
            audio_sec = float(len(audio_arr) / max(sr, 1))

            sample_id = str(row.get(args.id_column, i))
            rows.append(
                EvalRow(
                    sample_id=sample_id,
                    reference=ref_norm,
                    prediction=pred_norm,
                    audio_sec=audio_sec,
                    infer_sec=infer_sec,
                )
            )
            total_audio_sec += audio_sec
            total_infer_sec += infer_sec

            if args.print_every > 0 and i % args.print_every == 0:
                speed_x = total_audio_sec / max(total_infer_sec, 1e-9)
                print(f"[progress] {i} samples | speed={speed_x:.2f}x")

    if not rows:
        raise SystemExit("No samples evaluated.")

    try:
        import evaluate  # type: ignore
    except Exception as exc:
        raise RuntimeError("evaluate is required. pip install evaluate") from exc
    wer_metric = evaluate.load("wer")
    eval_pairs = [(r.reference, r.prediction) for r in rows if r.reference.strip()]
    skipped_empty_references = len(rows) - len(eval_pairs)
    if not eval_pairs:
        raise RuntimeError(
            "All normalized references are empty; cannot compute WER. "
            "Try --normalizer basic or --normalizer none."
        )
    references = [ref for ref, _ in eval_pairs]
    predictions = [pred for _, pred in eval_pairs]
    if skipped_empty_references > 0:
        print(
            f"[warning] skipped {skipped_empty_references} sample(s) with empty normalized reference "
            "for WER computation."
        )
    wer = float(wer_metric.compute(references=references, predictions=predictions))

    rtfx = total_audio_sec / max(total_infer_sec, 1e-9)
    summary = {
        "run_name": args.run_name,
        "dataset_path": args.dataset_path,
        "dataset": dataset_config,
        "split": split,
        "normalizer": args.normalizer,
        "num_samples": len(rows),
        "num_samples_scored": len(eval_pairs),
        "num_samples_skipped_empty_reference": skipped_empty_references,
        "wer": wer,
        "wer_percent": round(100.0 * wer, 4),
        "total_audio_sec": total_audio_sec,
        "total_infer_sec": total_infer_sec,
        "rtf": total_infer_sec / max(total_audio_sec, 1e-9),
        "rtfx": rtfx,
        "output_dir": str(out_dir.resolve()),
    }

    pred_jsonl = out_dir / "predictions.jsonl"
    with pred_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {
                        "id": row.sample_id,
                        "reference": row.reference,
                        "prediction": row.prediction,
                        "audio_length_s": row.audio_sec,
                        "transcription_time_s": row.infer_sec,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
