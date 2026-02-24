#!/usr/bin/env python3
"""NeMo reference transcriber for Parakeet TDT.

Exports:
  transcribe_file(audio_path, language=None, keywords=None) -> str
  stream_transcribe_file(audio_path, language=None, keywords=None) -> dict
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

_MODEL = None


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    import torch
    from nemo.collections.asr.models import EncDecRNNTBPEModel

    model_name = os.environ.get("PARAKEET_NEMO_MODEL", "nvidia/parakeet-tdt-0.6b-v2")
    model = EncDecRNNTBPEModel.from_pretrained(model_name=model_name)
    model.eval()
    model.to(torch.device("cpu"))
    _MODEL = model
    return _MODEL


def _transcribe(audio_path: str) -> str:
    model = _load_model()
    output = model.transcribe([audio_path], batch_size=1, return_hypotheses=False)
    if isinstance(output, (list, tuple)) and output:
        return _normalize_text(str(output[0]))
    return _normalize_text(str(output))


def transcribe_file(audio_path: str, language: str | None = None, keywords: list[str] | None = None) -> str:
    del language, keywords
    return _transcribe(audio_path)


def stream_transcribe_file(
    audio_path: str,
    language: str | None = None,
    keywords: list[str] | None = None,
) -> dict[str, Any]:
    del language, keywords
    transcript = _transcribe(audio_path)

    import soundfile as sf

    audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    duration_sec = float(len(audio) / float(sr)) if sr > 0 else 0.0

    # Minimal streaming-compatible payload for OpenBench metrics.
    return {
        "transcript": transcript,
        "audio_cursor": [duration_sec] if transcript else [],
        "interim_results": [transcript] if transcript else [],
        "confirmed_audio_cursor": [duration_sec] if transcript else [],
        "confirmed_interim_results": [transcript] if transcript else [],
        "model_timestamps_hypothesis": None,
        "model_timestamps_confirmed": None,
    }


def warmup() -> None:
    model = _load_model()
    import soundfile as sf
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        silence = np.zeros(16_000 // 2, dtype=np.float32)
        sf.write(path, silence, 16_000)
        _ = model.transcribe([path], batch_size=1, return_hypotheses=False)
    finally:
        Path(path).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="NeMo Parakeet transcriber")
    parser.add_argument("--audio", type=Path, required=True)
    args = parser.parse_args()
    print(transcribe_file(str(args.audio)))


if __name__ == "__main__":
    main()

