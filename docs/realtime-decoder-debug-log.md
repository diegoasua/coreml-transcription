# Realtime Decoder Debug Log

Last updated: 2026-03-02

## Ground truth and evaluation method
- Do not rely on latency-only pass metrics for quality.
- Primary quality check is transcript text comparison. *DO READ* the text on both reference and output.
  - Realtime output: `artifacts/realtime-bench-runs/<run>/summary.confirmed.txt`
  - Reference: `/Users/diegoasua/Downloads/transcription.txt`
  - For 60s runs, compare only lines with `start < 60s`.

## Persistent conclusions (agreed)
- Packaging is not the root cause.
- Quantization packaging (`odmbp-approx` vs `odmbp-approx-stateful`) is not the main blocker for transcript correctness.
- The correctness bug is in realtime decoding behavior/policy, centered on decoder path.

## Added tooling
- `scripts/compare_realtime_transcript.py`
  - Compares confirmed realtime transcript vs timestamped reference (optionally capped by seconds).
  - Reports:
    - sequence match ratio
    - word overlap precision/recall
    - trigram duplication ratio
  - Use:
    - `python3 scripts/compare_realtime_transcript.py --reference /Users/diegoasua/Downloads/transcription.txt --confirmed <run>/summary.confirmed.txt --max-seconds 60`
- `scripts/compare_decoder_traces.py`
  - Compares per-step decoder traces (Swift vs Python) and reports the first divergence.
  - Default key set:
    - `chunk_index`, `t`, `token_id`, `duration_idx`, `skip`, `prev_token`
  - Use:
    - `python3 scripts/compare_decoder_traces.py --swift-trace artifacts/tmp/swift-decoder-trace.jsonl --python-trace artifacts/tmp/python-decoder-trace.jsonl`

## Decoder trace capture
- Swift trace env:
  - `PARAKEET_DECODER_TRACE_PATH=/abs/path/swift-decoder-trace.jsonl`
  - optional: `PARAKEET_DECODER_TRACE_MAX_EVENTS` (default `200000`)
- Python trace env:
  - `PARAKEET_PY_DECODER_TRACE_PATH=/abs/path/python-decoder-trace.jsonl`
  - optional: `PARAKEET_PY_DECODER_TRACE_MAX_EVENTS` (default `200000`)
- Reset behavior:
  - Swift: `PARAKEET_DECODER_TRACE_RESET=1` (default)
  - Python: `PARAKEET_PY_DECODER_TRACE_RESET=1` (default)

## Key run findings
- `rt-debug-finalize-draft-60s-v3`
  - Severe repetition explosion.
  - Example repeated 4-grams:
    - `so i have created`
    - `i have created here`
    - `that allows you to`
  - Duplication ratio (trigram) very high (~0.349).
- `rt-user-ref-1` and related runs:
  - Not random repetition only; also under-coverage and semantic drift.
- `rt-user-ref-1-nospeechgate` (`PARAKEET_DECODE_ONLY_WHEN_SPEECH=0`)
  - Did **not** fix quality; first-token latency worsened materially.
- `rt-fix-defaults-60s`
  - Realtime script defaults aligned with known offline-friendly decoder params:
    - `max_symbols_per_step=10`
    - `max_tokens_per_chunk=0`
  - Quality still incorrect.
- `rt-stateful-odmbp-60s`
  - Stateful model branch reduced some duplication, but transcript still semantically wrong.

## Code changes already made
- Realtime defaults aligned with offline decode settings:
  - `scripts/run_realtime_bench_cli.sh`
  - `scripts/run_transcribe_macos_release.sh`
  - `Sources/transcribe-macos/TranscribeMacOSApp.swift`
- Added robust token overlap merge heuristics:
  - `Sources/RealtimeTranscriptionCore/ParakeetCoreMLTDTTranscriptionModel.swift`
  - Goal: avoid repeated tail re-append.
- Decoder loop mismatch identified vs Python reference implementation:
  - Python reference (`scripts/parakeet_coreml_tdt_transcriber.py`) uses per-step decode loop semantics.
  - Swift previously used a custom optimized scan path across `t`.
  - Swift decoder loop was replaced to mirror Python greedy TDT loop semantics more directly.

## Critical technical finding
- Swift and Python decoder loop semantics were different.
- This is likely a major source of quality drift.
- Decoder is now being moved toward Python-reference semantics as the primary correction path.

## Next decoder-only steps
1. Re-run 60s benchmark after loop-alignment patch and compare text to reference.
2. If mismatch remains, add per-step decoder trace diff (Swift vs Python) for first divergence:
   - `t`, `token_id`, `duration_idx/skip`, `prev_token`.
3. Keep packaging untouched while iterating decoder correctness.
