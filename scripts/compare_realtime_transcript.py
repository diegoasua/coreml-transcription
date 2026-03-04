#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


LINE_RE = re.compile(
    r"^\[(?P<start>\d+(?:\.\d+)?)\s*->\s*(?P<end>\d+(?:\.\d+)?)\]\s*Speaker\s+\d+:\s*(?P<text>.*)$"
)


def _load_reference(path: Path, max_seconds: float | None) -> str:
    parts: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE_RE.match(raw.strip())
        if not m:
            continue
        start = float(m.group("start"))
        if max_seconds is not None and start >= max_seconds:
            break
        text = m.group("text").strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _words(text: str) -> list[str]:
    return [w for w in text.split(" ") if w]


def _ngram_duplication_ratio(words: list[str], n: int = 3) -> float:
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(0, len(words) - n + 1)]
    unique = len(set(grams))
    return 1.0 - (unique / len(grams))


def _overlap_precision_recall(ref_words: list[str], hyp_words: list[str]) -> tuple[float, float]:
    if not ref_words and not hyp_words:
        return 1.0, 1.0
    if not ref_words:
        return 0.0, 1.0
    if not hyp_words:
        return 1.0, 0.0
    ref_c = Counter(ref_words)
    hyp_c = Counter(hyp_words)
    overlap = sum(min(ref_c[w], hyp_c[w]) for w in (set(ref_c) | set(hyp_c)))
    precision = overlap / max(1, len(hyp_words))
    recall = overlap / max(1, len(ref_words))
    return precision, recall


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare realtime confirmed transcript vs timestamped reference.")
    parser.add_argument("--reference", required=True, help="Timestamped reference transcript (e.g. transcription.txt).")
    parser.add_argument("--confirmed", required=True, help="Confirmed transcript file to evaluate.")
    parser.add_argument("--max-seconds", type=float, default=None, help="Use reference text only up to this time.")
    parser.add_argument("--preview-chars", type=int, default=420, help="Preview length.")
    args = parser.parse_args()

    ref_raw = _load_reference(Path(args.reference), args.max_seconds)
    hyp_raw = _load_text(Path(args.confirmed))
    ref = _normalize(ref_raw)
    hyp = _normalize(hyp_raw)

    ref_words = _words(ref)
    hyp_words = _words(hyp)
    precision, recall = _overlap_precision_recall(ref_words, hyp_words)
    dup_ratio = _ngram_duplication_ratio(hyp_words, n=3)
    # autojunk=True can collapse long repetitive ASR strings into unstable scores.
    # Disable it so score changes reflect transcript changes, not matcher heuristics.
    sm = SequenceMatcher(a=ref, b=hyp, autojunk=False)
    ratio = sm.ratio()
    word_sm = SequenceMatcher(a=ref_words, b=hyp_words, autojunk=False)
    word_ratio = word_sm.ratio()

    print("{")
    print(f'  "reference_words": {len(ref_words)},')
    print(f'  "hypothesis_words": {len(hyp_words)},')
    print(f'  "sequence_match_ratio": {ratio:.4f},')
    print(f'  "word_sequence_match_ratio": {word_ratio:.4f},')
    print(f'  "word_overlap_precision": {precision:.4f},')
    print(f'  "word_overlap_recall": {recall:.4f},')
    print(f'  "hypothesis_trigram_duplication_ratio": {dup_ratio:.4f}')
    print("}")
    print("\n[reference preview]")
    print(ref[: args.preview_chars])
    print("\n[hypothesis preview]")
    print(hyp[: args.preview_chars])


if __name__ == "__main__":
    main()
