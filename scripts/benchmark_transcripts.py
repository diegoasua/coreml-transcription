#!/usr/bin/env python3
"""Compute WER and real-time factor (RTF) from JSONL transcript logs.

Expected JSONL schema per line:
{
  "id": "utt-001",
  "text": "reference or hypothesis text",
  "audio_sec": 4.2,      # optional (reference file)
  "infer_sec": 0.21       # optional (hypothesis file)
}
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class Utterance:
    text: str
    audio_sec: float | None = None
    infer_sec: float | None = None


def _load_jsonl(path: Path) -> Dict[str, Utterance]:
    rows: Dict[str, Utterance] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            key = str(payload["id"])
            rows[key] = Utterance(
                text=str(payload.get("text", "")),
                audio_sec=float(payload["audio_sec"]) if "audio_sec" in payload else None,
                infer_sec=float(payload["infer_sec"]) if "infer_sec" in payload else None,
            )
    return rows


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark transcript outputs (WER + RTF).")
    parser.add_argument("--reference", type=Path, required=True, help="Reference JSONL file.")
    parser.add_argument("--hypothesis", type=Path, required=True, help="Hypothesis JSONL file.")
    args = parser.parse_args()

    refs = _load_jsonl(args.reference)
    hyps = _load_jsonl(args.hypothesis)

    shared_ids = sorted(set(refs) & set(hyps))
    if not shared_ids:
        raise SystemExit("No shared utterance ids found between reference and hypothesis files.")

    ref_texts = [refs[k].text for k in shared_ids]
    hyp_texts = [hyps[k].text for k in shared_ids]
    wer_value = compute_wer(ref_texts, hyp_texts)

    total_audio = 0.0
    total_infer = 0.0
    for key in shared_ids:
        ref = refs[key]
        hyp = hyps[key]
        if ref.audio_sec is not None:
            total_audio += ref.audio_sec
        if hyp.infer_sec is not None:
            total_infer += hyp.infer_sec

    if total_audio > 0 and total_infer > 0:
        rtf = total_infer / total_audio
        speed_x = total_audio / total_infer
        print(f"Matched utterances: {len(shared_ids)}")
        print(f"WER: {wer_value:.4f} ({wer_value * 100:.2f}%)")
        print(f"RTF: {rtf:.4f} ({speed_x:.2f}x realtime)")
    else:
        print(f"Matched utterances: {len(shared_ids)}")
        print(f"WER: {wer_value:.4f} ({wer_value * 100:.2f}%)")
        print("RTF: not available (missing audio_sec and/or infer_sec)")


if __name__ == "__main__":
    main()
