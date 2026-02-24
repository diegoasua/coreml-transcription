#!/usr/bin/env python3
"""Evaluate a local transcriber command on OpenBench-style datasets.

This script focuses on standardized ASR accuracy benchmarking:
- Loads a dataset from Hugging Face (for example, argmaxinc/librispeech-openbench),
  or a local JSONL manifest.
- Exports each utterance to a temporary WAV file.
- Invokes a user-provided transcription command template for each utterance.
- Computes WER and aggregate RTF.

The command template must include "{audio_path}".
Example:
  --transcribe-cmd "python my_transcriber.py --audio {audio_path}"
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import numpy as np


@dataclass
class EvalSample:
    sample_id: str
    reference_text: str
    audio: np.ndarray
    sample_rate: int
    audio_sec: float


def _simple_wer(reference: List[str], hypothesis: List[str]) -> float:
    m, n = len(reference), len(hypothesis)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    if m == 0:
        return 0.0 if n == 0 else 1.0
    return dp[m][n] / m


def compute_wer(reference_texts: List[str], hypothesis_texts: List[str]) -> float:
    try:
        from jiwer import wer  # type: ignore

        return float(wer(reference_texts, hypothesis_texts))
    except Exception:
        total_words = 0
        total_weighted_err = 0.0
        for ref, hyp in zip(reference_texts, hypothesis_texts):
            ref_words = ref.split()
            hyp_words = hyp.split()
            this_wer = _simple_wer(ref_words, hyp_words)
            total_words += max(len(ref_words), 1)
            total_weighted_err += this_wer * max(len(ref_words), 1)
        return total_weighted_err / max(total_words, 1)


def _normalize_text(text: str, mode: str) -> str:
    if mode == "none":
        return text.strip()
    lowered = text.lower().strip()
    stripped = re.sub(r"[^a-z0-9'\s]", " ", lowered)
    squashed = re.sub(r"\s+", " ", stripped).strip()
    return squashed


def _ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 0:
        return np.array([float(arr)], dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # Accept both [time, channels] and [channels, time].
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            return arr.mean(axis=0, dtype=np.float32)
        return arr.mean(axis=1, dtype=np.float32)
    return arr.reshape(-1).astype(np.float32)


def _read_audio_any(value: Any) -> Tuple[np.ndarray, int]:
    try:
        import soundfile as sf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("soundfile is required for audio decoding. pip install soundfile") from exc

    if isinstance(value, dict):
        if "array" in value and "sampling_rate" in value:
            arr = _ensure_mono_float32(np.asarray(value["array"], dtype=np.float32))
            sr = int(value["sampling_rate"])
            return arr, sr
        if "path" in value and value["path"]:
            arr, sr = sf.read(str(value["path"]), dtype="float32", always_2d=False)
            return _ensure_mono_float32(np.asarray(arr, dtype=np.float32)), int(sr)
        if "bytes" in value and value["bytes"] is not None:
            arr, sr = sf.read(io.BytesIO(value["bytes"]), dtype="float32", always_2d=False)
            return _ensure_mono_float32(np.asarray(arr, dtype=np.float32)), int(sr)

    if isinstance(value, (str, Path)):
        arr, sr = sf.read(str(value), dtype="float32", always_2d=False)
        return _ensure_mono_float32(np.asarray(arr, dtype=np.float32)), int(sr)

    raise ValueError(f"Unsupported audio payload type: {type(value)!r}")


def _pick_text_value(row: Dict[str, Any], preferred: str | None) -> str:
    if preferred:
        return str(row.get(preferred, ""))
    for key in ("text", "transcript", "sentence", "reference"):
        if key in row:
            value = row[key]
            if isinstance(value, list):
                return " ".join(str(v) for v in value)
            return str(value)
    return ""


def _iter_hf_samples(
    dataset_id: str,
    split: str,
    dataset_config: str | None,
    audio_column: str | None,
    text_column: str | None,
    id_column: str | None,
    limit: int | None,
) -> Iterator[EvalSample]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("datasets is required. pip install datasets") from exc

    ds = load_dataset(dataset_id, dataset_config, split=split)
    seen = 0
    for idx, row in enumerate(ds):
        row_dict = dict(row)
        audio_key = audio_column
        if audio_key is None:
            for candidate in ("audio", "wav", "file", "path"):
                if candidate in row_dict:
                    audio_key = candidate
                    break
        if audio_key is None:
            raise RuntimeError(
                "Could not infer audio column. Pass --audio-column. "
                f"Available columns: {sorted(row_dict.keys())}"
            )

        sample_id = str(row_dict[id_column]) if id_column and id_column in row_dict else str(idx)
        reference = _pick_text_value(row_dict, text_column)
        audio, sr = _read_audio_any(row_dict[audio_key])
        audio_sec = float(len(audio) / max(sr, 1))
        yield EvalSample(sample_id=sample_id, reference_text=reference, audio=audio, sample_rate=sr, audio_sec=audio_sec)

        seen += 1
        if limit is not None and seen >= limit:
            break


def _iter_jsonl_samples(
    manifest_jsonl: Path,
    audio_column: str,
    text_column: str,
    id_column: str | None,
    limit: int | None,
) -> Iterator[EvalSample]:
    seen = 0
    with manifest_jsonl.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row[id_column]) if id_column and id_column in row else str(idx)
            reference = str(row.get(text_column, ""))
            audio, sr = _read_audio_any(row[audio_column])
            audio_sec = float(len(audio) / max(sr, 1))
            yield EvalSample(sample_id=sample_id, reference_text=reference, audio=audio, sample_rate=sr, audio_sec=audio_sec)

            seen += 1
            if limit is not None and seen >= limit:
                break


def _run_transcriber(
    command_template: str,
    audio_path: Path,
    sample_id: str,
    timeout_sec: float,
    stdout_json_key: str | None,
) -> Tuple[str, float]:
    audio_quoted = shlex.quote(str(audio_path))
    cmd = command_template.format(audio_path=audio_quoted, id=sample_id)

    t0 = time.perf_counter()
    completed = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    infer_sec = time.perf_counter() - t0

    if completed.returncode != 0:
        err = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Transcriber command failed (code={completed.returncode}) for id={sample_id}: {err[:400]}")

    raw_out = completed.stdout.strip()
    if stdout_json_key:
        try:
            payload = json.loads(raw_out)
            text = str(payload.get(stdout_json_key, "")).strip()
            return text, infer_sec
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"stdout was not valid JSON for id={sample_id} but --stdout-json-key was provided"
            ) from exc

    return raw_out, infer_sec


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ASR on OpenBench-style datasets.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--dataset-id", type=str, help="Hugging Face dataset id (e.g., argmaxinc/librispeech-openbench).")
    input_group.add_argument("--manifest-jsonl", type=Path, help="Local JSONL with audio/text columns.")

    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--audio-column", type=str, default=None)
    parser.add_argument("--text-column", type=str, default=None)
    parser.add_argument("--id-column", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of utterances.")

    parser.add_argument("--transcribe-cmd", type=str, required=True, help="Command template. Must include {audio_path}.")
    parser.add_argument("--stdout-json-key", type=str, default=None, help="Parse stdout as JSON and read this key.")
    parser.add_argument("--timeout-sec", type=float, default=120.0)

    parser.add_argument("--normalize", choices=("basic", "none"), default="basic")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/openbench-eval"))
    parser.add_argument("--run-name", type=str, default="run")
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--keep-audio", action="store_true", help="Keep temporary WAV files for debugging.")
    args = parser.parse_args()

    if "{audio_path}" not in args.transcribe_cmd:
        raise SystemExit("--transcribe-cmd must contain '{audio_path}' placeholder.")

    out_dir = args.output_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest_jsonl:
        if args.audio_column is None or args.text_column is None:
            raise SystemExit("--manifest-jsonl requires --audio-column and --text-column.")
        sample_iter = _iter_jsonl_samples(
            manifest_jsonl=args.manifest_jsonl,
            audio_column=args.audio_column,
            text_column=args.text_column,
            id_column=args.id_column,
            limit=args.limit,
        )
        dataset_label = str(args.manifest_jsonl)
    else:
        sample_iter = _iter_hf_samples(
            dataset_id=args.dataset_id,
            split=args.split,
            dataset_config=args.dataset_config,
            audio_column=args.audio_column,
            text_column=args.text_column,
            id_column=args.id_column,
            limit=args.limit,
        )
        dataset_label = f"{args.dataset_id}:{args.split}"

    references: List[Dict[str, Any]] = []
    hypotheses: List[Dict[str, Any]] = []
    ref_texts: List[str] = []
    hyp_texts: List[str] = []
    total_audio_sec = 0.0
    total_infer_sec = 0.0
    total_items = 0

    with tempfile.TemporaryDirectory(prefix="openbench_eval_") as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)

        for sample in sample_iter:
            wav_path = tmp_dir / f"{sample.sample_id}.wav"
            try:
                import soundfile as sf  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("soundfile is required for writing temporary WAV files.") from exc

            sf.write(str(wav_path), sample.audio, sample.sample_rate)

            hyp_text, infer_sec = _run_transcriber(
                command_template=args.transcribe_cmd,
                audio_path=wav_path,
                sample_id=sample.sample_id,
                timeout_sec=args.timeout_sec,
                stdout_json_key=args.stdout_json_key,
            )

            ref_norm = _normalize_text(sample.reference_text, args.normalize)
            hyp_norm = _normalize_text(hyp_text, args.normalize)

            references.append(
                {
                    "id": sample.sample_id,
                    "text": ref_norm,
                    "audio_sec": sample.audio_sec,
                }
            )
            hypotheses.append(
                {
                    "id": sample.sample_id,
                    "text": hyp_norm,
                    "infer_sec": infer_sec,
                }
            )

            ref_texts.append(ref_norm)
            hyp_texts.append(hyp_norm)
            total_audio_sec += sample.audio_sec
            total_infer_sec += infer_sec
            total_items += 1

            if not args.keep_audio:
                wav_path.unlink(missing_ok=True)

            if args.print_every > 0 and total_items % args.print_every == 0:
                print(f"[progress] {total_items} utterances")

    if total_items == 0:
        raise SystemExit("No utterances processed.")

    wer_value = compute_wer(ref_texts, hyp_texts)
    rtf = total_infer_sec / max(total_audio_sec, 1e-9)
    speed_x = total_audio_sec / max(total_infer_sec, 1e-9)

    ref_path = out_dir / "reference.jsonl"
    hyp_path = out_dir / "hypothesis.jsonl"
    summary_path = out_dir / "summary.json"
    _write_jsonl(ref_path, references)
    _write_jsonl(hyp_path, hypotheses)

    summary = {
        "dataset": dataset_label,
        "num_utterances": total_items,
        "normalize": args.normalize,
        "wer": wer_value,
        "wer_percent": wer_value * 100.0,
        "total_audio_sec": total_audio_sec,
        "total_infer_sec": total_infer_sec,
        "rtf": rtf,
        "speed_x": speed_x,
        "reference_jsonl": str(ref_path),
        "hypothesis_jsonl": str(hyp_path),
        "command_template": args.transcribe_cmd,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
