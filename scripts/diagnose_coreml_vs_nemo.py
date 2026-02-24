#!/usr/bin/env python3
"""Paired diagnostic: CoreML transcriber vs NeMo transcriber on same HF samples.

Use this to isolate whether WER gap comes from decoder/runtime differences.
"""

from __future__ import annotations

import argparse
import io
import importlib.util
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Compare CoreML and NeMo outputs on identical dataset samples.")
    parser.add_argument("--dataset-path", type=str, default="hf-audio/esb-datasets-test-only-sorted")
    parser.add_argument("--dataset", type=str, default="earnings22")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-eval-samples", type=int, default=100)
    parser.add_argument("--audio-column", type=str, default="audio")
    parser.add_argument("--reference-column", type=str, default=None)
    parser.add_argument("--id-column", type=str, default="id")
    parser.add_argument("--normalizer", choices=("open_asr", "basic", "none"), default="open_asr")

    parser.add_argument(
        "--coreml-transcriber",
        type=Path,
        default=repo_root / "scripts/parakeet_coreml_rnnt_transcriber.py",
    )
    parser.add_argument(
        "--nemo-transcriber",
        type=Path,
        default=repo_root / "scripts/parakeet_nemo_transcriber.py",
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root / "artifacts/diagnostics")
    parser.add_argument("--run-name", type=str, default="coreml-vs-nemo-earnings22")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--top-k-delta", type=int, default=30)
    return parser.parse_args()


def _basic_normalize(text: str) -> str:
    import re

    lowered = text.lower().strip()
    lowered = re.sub(r"[^a-z0-9'\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _load_english_spelling_mapping() -> dict[str, str]:
    try:
        from openbench.metric.word_error_metrics.english_abbreviations import ABBR  # type: ignore

        if isinstance(ABBR, dict):
            return dict(ABBR)
    except Exception:
        pass

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


def _open_asr_normalize(text: str) -> str:
    try:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    except Exception:
        return _basic_normalize(text)

    try:
        normalizer = EnglishTextNormalizer()  # type: ignore[call-arg]
    except TypeError:
        normalizer = EnglishTextNormalizer(_load_english_spelling_mapping())  # type: ignore[call-arg]
    return normalizer(text)


def _normalize(text: str, mode: str) -> str:
    if mode == "none":
        return text.strip()
    if mode == "basic":
        return _basic_normalize(text)
    return _open_asr_normalize(text)


def _decode_audio_field(payload: Any) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf  # type: ignore
    except Exception as exc:
        raise RuntimeError("soundfile is required for audio decoding.") from exc

    if isinstance(payload, dict):
        if "array" in payload and "sampling_rate" in payload:
            arr = np.asarray(payload["array"], dtype=np.float32)
            sr = int(payload["sampling_rate"])
            if arr.ndim == 2:
                arr = np.mean(arr, axis=1, dtype=np.float32)
            return arr.astype(np.float32, copy=False), sr

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

        if payload.get("path"):
            path = Path(str(payload["path"]))
            if not path.exists():
                cache_root = Path.home() / ".cache" / "huggingface" / "datasets"
                matches = list(cache_root.rglob(path.name))
                if matches:
                    path = matches[0]
            arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 2:
                arr = np.mean(arr, axis=1, dtype=np.float32)
            return arr.astype(np.float32, copy=False), int(sr)

    raise RuntimeError(f"Unsupported audio payload: {type(payload)!r}")


def _load_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Transcriber module not found: {path}")
    name = f"diag_{path.stem}_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _make_transcribe_fn(path: Path) -> Callable[[str], str]:
    module = _load_module(path.resolve())
    warmup = getattr(module, "warmup", None)
    if callable(warmup):
        warmup()
    transcribe_file = getattr(module, "transcribe_file", None)
    if not callable(transcribe_file):
        raise RuntimeError(f"{path} must export transcribe_file(audio_path, language=None, keywords=None)")

    def _run(audio_path: str) -> str:
        out = transcribe_file(audio_path, language=None, keywords=None)
        return str(out)

    return _run


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


def _sample_wer(reference: str, hypothesis: str) -> float:
    ref = reference.strip()
    hyp = hypothesis.strip()
    if not ref:
        return 0.0
    try:
        import jiwer  # type: ignore

        return float(jiwer.wer(ref, hyp))
    except Exception:
        # fallback
        r = ref.split()
        h = hyp.split()
        m, n = len(r), len(h)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if r[i - 1] == h[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[m][n] / max(m, 1)


@dataclass
class DiagRow:
    sample_id: str
    ref: str
    coreml: str
    nemo: str
    audio_sec: float
    coreml_sec: float
    nemo_sec: float
    wer_coreml: float
    wer_nemo: float


def main() -> None:
    args = _parse_args()
    out_dir = args.output_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import Audio, load_dataset

    ds = load_dataset(args.dataset_path, args.dataset, split=args.split)
    ds = ds.cast_column(args.audio_column, Audio(decode=False))
    if args.max_eval_samples and args.max_eval_samples > 0:
        ds = ds.select(range(min(args.max_eval_samples, len(ds))))

    run_coreml = _make_transcribe_fn(args.coreml_transcriber)
    run_nemo = _make_transcribe_fn(args.nemo_transcriber)

    rows: list[DiagRow] = []
    with tempfile.TemporaryDirectory(prefix="diag_coreml_nemo_") as tmp:
        tmp_dir = Path(tmp)
        import soundfile as sf  # type: ignore

        for i, row in enumerate(ds, start=1):
            audio_arr, sr = _decode_audio_field(row[args.audio_column])
            wav_path = tmp_dir / f"{i}.wav"
            sf.write(wav_path, audio_arr, sr)

            ref = _normalize(_pick_reference_text(dict(row), args.reference_column), args.normalizer)
            sid = str(row.get(args.id_column, i))

            t0 = time.perf_counter()
            coreml_txt = _normalize(run_coreml(str(wav_path)), args.normalizer)
            coreml_sec = time.perf_counter() - t0

            t1 = time.perf_counter()
            nemo_txt = _normalize(run_nemo(str(wav_path)), args.normalizer)
            nemo_sec = time.perf_counter() - t1

            wer_coreml = _sample_wer(ref, coreml_txt)
            wer_nemo = _sample_wer(ref, nemo_txt)
            rows.append(
                DiagRow(
                    sample_id=sid,
                    ref=ref,
                    coreml=coreml_txt,
                    nemo=nemo_txt,
                    audio_sec=float(len(audio_arr) / max(sr, 1)),
                    coreml_sec=coreml_sec,
                    nemo_sec=nemo_sec,
                    wer_coreml=wer_coreml,
                    wer_nemo=wer_nemo,
                )
            )

            if args.print_every > 0 and i % args.print_every == 0:
                avg_delta = sum(r.wer_coreml - r.wer_nemo for r in rows) / max(len(rows), 1)
                print(f"[progress] {i} samples | avg_delta(coreml-nemo)={avg_delta:+.4f}")

    non_empty = [r for r in rows if r.ref.strip()]
    if not non_empty:
        raise RuntimeError("No non-empty references after normalization.")

    def _agg_wer(kind: str) -> float:
        try:
            import evaluate  # type: ignore

            metric = evaluate.load("wer")
            refs = [r.ref for r in non_empty]
            hyps = [getattr(r, kind) for r in non_empty]
            return float(metric.compute(references=refs, predictions=hyps))
        except Exception:
            # weighted per-sample fallback
            total = 0.0
            words = 0
            for r in non_empty:
                n = max(len(r.ref.split()), 1)
                total += (r.wer_coreml if kind == "coreml" else r.wer_nemo) * n
                words += n
            return total / max(words, 1)

    coreml_wer = _agg_wer("coreml")
    nemo_wer = _agg_wer("nemo")
    delta = coreml_wer - nemo_wer

    total_audio = sum(r.audio_sec for r in non_empty)
    coreml_time = sum(r.coreml_sec for r in non_empty)
    nemo_time = sum(r.nemo_sec for r in non_empty)

    worst = sorted(non_empty, key=lambda r: (r.wer_coreml - r.wer_nemo), reverse=True)[: max(1, args.top_k_delta)]
    worst_rows = [
        {
            "id": r.sample_id,
            "wer_coreml": r.wer_coreml,
            "wer_nemo": r.wer_nemo,
            "delta": r.wer_coreml - r.wer_nemo,
            "reference": r.ref,
            "coreml": r.coreml,
            "nemo": r.nemo,
        }
        for r in worst
    ]

    summary = {
        "run_name": args.run_name,
        "dataset_path": args.dataset_path,
        "dataset": args.dataset,
        "split": args.split,
        "normalizer": args.normalizer,
        "num_samples_total": len(rows),
        "num_samples_scored": len(non_empty),
        "wer_coreml": coreml_wer,
        "wer_nemo": nemo_wer,
        "wer_delta_coreml_minus_nemo": delta,
        "total_audio_sec": total_audio,
        "coreml_total_sec": coreml_time,
        "nemo_total_sec": nemo_time,
        "coreml_rtfx": total_audio / max(coreml_time, 1e-9),
        "nemo_rtfx": total_audio / max(nemo_time, 1e-9),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "worst_deltas.json").write_text(json.dumps(worst_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote: {(out_dir / 'worst_deltas.json').resolve()}")


if __name__ == "__main__":
    main()
