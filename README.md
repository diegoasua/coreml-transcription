# coreml-transcription

Apple-first real-time transcription playground for Parakeet v2 (starting on macOS, then iOS).

This repo is set up to do three things:

1. Export Parakeet from NeMo into ONNX/TorchScript artifacts you control.
2. Convert/compress exported components into CoreML and iterate on quantization choices.
3. Run a streaming text pipeline in Swift with LocalAgreement-style stabilization and local VAD gating.

## Current Status

- Conversion scripts are included (`scripts/`), but full-model compatibility depends on the exact Parakeet export graph.
- Streaming runtime primitives and an inference engine are implemented in Swift (`Sources/RealtimeTranscriptionCore/`) with unit tests.
- Benchmark helper is included to track WER/RTF from your own transcript logs.
- OpenBench-style standardized dataset evaluation is available via `scripts/eval_openbench_dataset.py`.

## Quick Start

### 1) Python environment (conversion + benchmarking)

```bash
bash scripts/bootstrap_env.sh .venv
source .venv/bin/activate
```

By default, the scripts prefer `python3.11` (then `3.12`, `3.13`, then `python3`).
You can override with `PYTHON_BIN`, for example:

```bash
PYTHON_BIN=python3.11 bash scripts/bootstrap_env.sh .venv
```

If ONNX files are already exported and you want to rerun conversion/compression only:

```bash
SKIP_EXPORT=1 PYTHON_BIN=python3.11 bash scripts/run_conversion.sh
```

Locked baseline (matches `parakeet-coreml-ls200-v5`):

```bash
source configs/parakeet-coreml-v5.env
bash scripts/run_conversion.sh
```

FP16 baseline (no compression, for accuracy ceiling checks):

```bash
source configs/parakeet-coreml-fp16-baseline.env
```

High-accuracy mixed profile (keeps baseline artifacts, adds `mix84`):

```bash
source configs/parakeet-coreml-v5-hiacc.env
bash scripts/run_conversion.sh
```

OD-MBP-inspired profile (higher-fidelity encoder, adds `odmbp-lite`):

```bash
source configs/parakeet-coreml-v5-odmbp-lite.env
bash scripts/run_conversion.sh
```

Third profile, mixed 6/8-bit encoder palettization (adds `mix68`):

```bash
source configs/parakeet-coreml-v6-mixed68.env
bash scripts/run_conversion.sh
```

Fourth profile, OD-MBP approximation with fp16 escape tensors (adds `odmbp-approx`):

```bash
source configs/parakeet-coreml-v7-odmbp-approx.env
bash scripts/run_conversion.sh
```

If you need to regenerate both ONNX and TorchScript artifacts:

```bash
EXPORT_FORMATS=onnx,ts PYTHON_BIN=python3.11 bash scripts/run_conversion.sh
```

### 2) Export from NeMo

```bash
python scripts/export_nemo_to_onnx.py \
  --model nvidia/parakeet-tdt-0.6b-v2 \
  --output-dir artifacts/parakeet-tdt-0.6b-v2
```

### 3) Convert TorchScript -> CoreML (using ONNX manifest for input specs)

```bash
python scripts/convert_torchscript_to_coreml.py \
  --torchscript artifacts/parakeet-tdt-0.6b-v2/encoder-model.ts \
  --manifest artifacts/parakeet-tdt-0.6b-v2/encoder-model-manifest.json \
  --output artifacts/parakeet-tdt-0.6b-v2/encoder-model.mlpackage \
  --target macos14 \
  --compute-units all
```

### 4) Inspect ONNX I/O and create manifest

```bash
python scripts/inspect_onnx.py \
  --onnx artifacts/parakeet-tdt-0.6b-v2/encoder-model.onnx \
  --write-manifest artifacts/parakeet-tdt-0.6b-v2/encoder-model-manifest.json
```

### 5) Compression experiment (palettization / quant fallback)

```bash
python scripts/compress_coreml.py \
  --model artifacts/parakeet-tdt-0.6b-v2/encoder-model.mlpackage \
  --output artifacts/parakeet-tdt-0.6b-v2/encoder-model-int4.mlpackage \
  --nbits 4
```

### 6) CoreML latency benchmark (synthetic chunk input)

```bash
python scripts/benchmark_rnnt_components.py \
  --encoder-model artifacts/parakeet-tdt-0.6b-v2/encoder-model-int4.mlpackage \
  --encoder-manifest artifacts/parakeet-tdt-0.6b-v2/encoder-model-manifest.json \
  --decoder-model artifacts/parakeet-tdt-0.6b-v2/decoder_joint-model-int4.mlpackage \
  --decoder-manifest artifacts/parakeet-tdt-0.6b-v2/decoder_joint-model-manifest.json \
  --iterations 40 \
  --decoder-steps 64 \
  --compute-units cpu_and_ne
```

### 7) Swift runtime tests

```bash
swift test
```

### 8) Standardized WER benchmark (OpenBench datasets)

```bash
python scripts/eval_openbench_dataset.py \
  --dataset-id argmaxinc/librispeech-openbench \
  --split test \
  --transcribe-cmd "python your_transcriber.py --audio {audio_path}" \
  --run-name parakeet-coreml-ls-clean
```

Notes:
- `--transcribe-cmd` must print only the transcript text to stdout.
- Use `--stdout-json-key text` if your transcriber prints JSON.
- Output files are written under `artifacts/openbench-eval/<run-name>/`:
  - `reference.jsonl`
  - `hypothesis.jsonl`
  - `summary.json` (WER/RTF)

### 8b) Independent HF-style benchmark (no OpenBench runtime)

If you want a second harness independent of OpenBench internals, use:

```bash
bash scripts/run_hf_open_asr_eval.sh \
  --run-name parakeet-coreml-earnings22-hf \
  --python-transcriber scripts/parakeet_coreml_rnnt_transcriber.py
```

Defaults are aligned with Hugging Face ESB test-only datasets:
- `--dataset-path hf-audio/esb-datasets-test-only-sorted`
- auto-resolve a config containing `earnings22`
- `--split test`
- `--normalizer open_asr`

Outputs are written to:
- `artifacts/hf-asr-eval/<run-name>/summary.json`
- `artifacts/hf-asr-eval/<run-name>/predictions.jsonl`

FP16 long-form tuning sweep (context + max-symbols):

```bash
bash scripts/run_hf_earnings22_fp16_tuning.sh
```

### 9) Native OpenBench run (custom local pipeline)

If you cloned OpenBench at `external/OpenBench`, run this from repo root:

```bash
cd external/OpenBench
uv sync
uv run python ../../scripts/run_openbench_custom_transcription.py \
  --openbench-dir . \
  --dataset librispeech-200 \
  --python-transcriber scripts/parakeet_coreml_rnnt_transcriber.py \
  --run-name parakeet-coreml-ls200
```

This uses OpenBench's dataset/metric stack directly and writes:
- `artifacts/openbench-runs/<run-name>/custom-openbench-summary.json`

Shortcut wrapper:

```bash
bash scripts/run_openbench_eval.sh \
  --dataset librispeech-200 \
  --python-transcriber scripts/parakeet_coreml_rnnt_transcriber.py \
  --run-name parakeet-coreml-ls200
```

### 10) Native OpenBench real-time streaming run

Run OpenBench streaming metrics (`wer`, `streaming_latency`, `confirmed_streaming_latency`)
using the local CoreML transcriber:

```bash
source configs/parakeet-coreml-v5.env
bash scripts/run_openbench_streaming_eval.sh \
  --dataset timit-debug \
  --run-name parakeet-coreml-streaming-debug
```

Scale up to the full benchmark:

```bash
source configs/parakeet-coreml-v5.env
bash scripts/run_openbench_streaming_eval.sh \
  --dataset timit \
  --run-name parakeet-coreml-streaming-timit
```

If you have access to the stitched OpenBench dataset used in published tables,
swap `--dataset timit` for `--dataset timit-stitched` (`argmaxinc/timit_stitched`).

To benchmark the high-accuracy `mix84` profile:

```bash
source configs/parakeet-coreml-v5-hiacc.env
bash scripts/run_openbench_streaming_eval.sh \
  --dataset timit \
  --run-name parakeet-coreml-streaming-timit-mix84
```

To benchmark the OD-MBP-inspired `odmbp-lite` profile:

```bash
source configs/parakeet-coreml-v5-odmbp-lite.env
bash scripts/run_openbench_streaming_eval.sh \
  --dataset timit \
  --run-name parakeet-coreml-streaming-timit-odmbp-lite
```

To benchmark the mixed 6/8-bit profile:

```bash
source configs/parakeet-coreml-v6-mixed68.env
bash scripts/run_openbench_streaming_eval.sh \
  --dataset timit \
  --run-name parakeet-coreml-streaming-timit-mix68
```

To benchmark the OD-MBP approximation profile:

```bash
source configs/parakeet-coreml-v7-odmbp-approx.env
bash scripts/run_openbench_streaming_eval.sh \
  --dataset timit \
  --run-name parakeet-coreml-streaming-timit-odmbp-approx
```

To benchmark FP16 baseline:

```bash
source configs/parakeet-coreml-fp16-baseline.env
bash scripts/run_openbench_eval.sh \
  --dataset librispeech-200 \
  --python-transcriber scripts/parakeet_coreml_rnnt_transcriber.py \
  --run-name parakeet-coreml-ls200-fp16

bash scripts/run_openbench_streaming_eval.sh \
  --dataset timit \
  --run-name parakeet-coreml-streaming-timit-fp16
```

To run the same model with decoder beam search (quality-focused):

```bash
source configs/parakeet-coreml-decoder-beam2.env
bash scripts/run_openbench_streaming_eval.sh \
  --dataset timit \
  --run-name parakeet-coreml-streaming-timit-beam2
```

To measure a NeMo (PyTorch) reference ceiling on the same OpenBench setup:

```bash
OPENBENCH_PYTHON=python3.11 \
OPENBENCH_REBUILD_TEXTERRORS=0 \
bash scripts/run_openbench_eval.sh \
  --dataset timit \
  --num-samples 20 \
  --pipeline-kind transcription \
  --python-transcriber scripts/parakeet_nemo_transcriber.py \
  --metrics wer \
  --run-name parakeet-nemo-ref-smoke20
```

To run the English comparison matrix (default dataset: `earnings22`;
profiles: `fp16`, `odmbp-approx`, `odmbp-approx-beam2`):

```bash
OPENBENCH_PYTHON=python3.11 \
OPENBENCH_REBUILD_TEXTERRORS=0 \
bash scripts/run_openbench_matrix_english.sh
```

Optional quick smoke mode:

```bash
NUM_SAMPLES=20 \
OPENBENCH_PYTHON=python3.11 \
OPENBENCH_REBUILD_TEXTERRORS=0 \
bash scripts/run_openbench_matrix_english.sh
```

Outputs:
- `artifacts/openbench-runs/parakeet-coreml-english-matrix-manifest.tsv`
- `artifacts/openbench-runs/parakeet-coreml-english-matrix-summary.json`
- `artifacts/openbench-runs/parakeet-coreml-english-matrix-summary.csv`

Run names include mode suffixes automatically:
- `...-noseg` when `ENABLE_LONGFORM_SEGMENTED=0`
- `...-seg` when `ENABLE_LONGFORM_SEGMENTED=1`

For longform datasets (for example `earnings22`), you can enable segmentation
to avoid relying on one continuous decoder state over hour-long files:

```bash
source configs/parakeet-coreml-longform-segmented.env
```

Or via matrix script flag:

```bash
ENABLE_LONGFORM_SEGMENTED=1 \
OPENBENCH_PYTHON=python3.11 \
OPENBENCH_REBUILD_TEXTERRORS=0 \
bash scripts/run_openbench_matrix_english.sh
```

You can tune:
- `PARAKEET_LONGFORM_SEGMENT_SEC` (default suggested: `30`)
- `PARAKEET_LONGFORM_OVERLAP_SEC` (default suggested: `3`)

Optional long-form boundary tuning (fixed-shape encoder):

```bash
source configs/parakeet-coreml-decoder-longform-context.env
```

This sets `PARAKEET_ENCODER_LEFT_CONTEXT_FRAMES` (default in that profile: `240`)
to preserve encoder context across chunk boundaries.

Bidirectional context variant:

```bash
source configs/parakeet-coreml-decoder-longform-context-bidir.env
```

This sets `PARAKEET_ENCODER_LEFT_CONTEXT_FRAMES=240` and
`PARAKEET_ENCODER_RIGHT_CONTEXT_FRAMES=120`.

Note:
- `transcribe_file(...)` (offline) uses configured right-context.
- `stream_transcribe_file(...)` is causal by default (`PARAKEET_STREAM_ALLOW_RIGHT_CONTEXT=0`).

Optional: include AMI SDM in the same matrix:

```bash
DATASETS_CSV=earnings22,ami-sdm-openbench \
OPENBENCH_PYTHON=python3.11 \
OPENBENCH_REBUILD_TEXTERRORS=0 \
bash scripts/run_openbench_matrix_english.sh
```

Note:
- `--python-transcriber` keeps the model loaded in-process (recommended for speed).
- `--transcribe-cmd` is still supported for shell-command integration.
- Commands run from project root by default (`--command-cwd`).
- Decoder quality/speed knobs (CoreML RNNT transcriber):
  - `PARAKEET_RNNT_BEAM_WIDTH` (default `1`, greedy). Try `2` or `4` for lower WER.
  - `PARAKEET_RNNT_DURATION_BEAM_WIDTH` (default matches beam width).
  - `PARAKEET_RNNT_MAX_SYMBOLS_PER_STEP` (default `10`).

If OpenBench dependency import fails on macOS due `texterrors_align`:
- The runner auto-falls back to a safe stub for `texterrors` (keyword metrics disabled, WER unaffected).
- Optionally force a known Python: `OPENBENCH_PYTHON=python3.11`
- Optional rebuild from source: `OPENBENCH_REBUILD_TEXTERRORS=1`
- Wrapper exports:
  - `PARAKEET_COREML_MODEL_DIR` (default: `artifacts/parakeet-tdt-0.6b-v2`)
  - `TMPDIR` (default: `${PARAKEET_COREML_MODEL_DIR}/.tmp`)

## Repo Layout

- `scripts/export_nemo_to_onnx.py`: reproducible export from NeMo checkpoint to ONNX.
- `scripts/convert_torchscript_to_coreml.py`: CoreML conversion wrapper for TorchScript components.
- `scripts/compress_coreml.py`: compression experiment harness (optimize API with legacy fallback).
- `scripts/inspect_onnx.py`: ONNX graph/I-O inspector with candidate name suggestions.
- `scripts/benchmark_coreml_model.py`: runtime-only latency benchmark for CoreML models.
- `scripts/benchmark_rnnt_components.py`: encoder + decoder-loop benchmark with end-to-end RTF estimate.
- `scripts/benchmark_transcripts.py`: quick WER/RTF calculator from JSONL logs.
- `scripts/eval_openbench_dataset.py`: standardized dataset evaluation harness (HF OpenBench datasets + local transcriber command).
- `scripts/run_openbench_custom_transcription.py`: OpenBench-native benchmark runner with custom local pipeline support for both `transcription` and `streaming_transcription`.
- `scripts/parakeet_coreml_rnnt_transcriber.py`: local CoreML Parakeet RNNT/TDT module (`transcribe_file(...)`, `stream_transcribe_file(...)`).
- `scripts/run_openbench_eval.sh`: convenience wrapper to run OpenBench `uv sync` + custom benchmark script.
- `scripts/run_openbench_streaming_eval.sh`: convenience wrapper for OpenBench streaming transcription metrics.
- `scripts/run_openbench_matrix_english.sh`: matrix runner for English transcription benchmarks (`earnings22`, `ami-sdm-openbench`) across selected profiles.
- `scripts/build_openbench_matrix_report.py`: consolidates matrix run outputs into one JSON/CSV report.
- `configs/parakeet-coreml-v5.env`: locked conversion/runtime/benchmark environment settings for the current baseline.
- `configs/parakeet-coreml-fp16-baseline.env`: uncompressed fp16 runtime routing for accuracy ceiling measurement.
- `configs/parakeet-coreml-decoder-beam2.env`: decoder beam overlay (`beam=2`) for quality-focused runs.
- `configs/parakeet-coreml-decoder-longform-context.env`: long-form boundary overlay (`PARAKEET_ENCODER_LEFT_CONTEXT_FRAMES=240`).
- `configs/parakeet-coreml-v5-hiacc.env`: mixed precision profile (`encoder=int8`, `decoder=int4`) to trade model size for accuracy.
- `configs/parakeet-coreml-v5-odmbp-lite.env`: OD-MBP-inspired profile (`encoder=linear-int8`, `decoder=kmeans-int4`) for lower WER at larger size.
- `configs/parakeet-coreml-v6-mixed68.env`: mixed encoder palettization profile (`encoder=6/8-bit by outlier score`, `decoder=int4`) targeting Argmax-like model size.
- `configs/parakeet-coreml-v7-odmbp-approx.env`: OD-MBP approximation (`encoder=6/8-bit mixed + small fp16 outlier tensor escape`, `decoder=int4`).
- `Sources/RealtimeTranscriptionCore/`: LocalAgreement stabilizer, VAD, ring buffer, CTC decoder, streaming inference engine.
- `Sources/transcribe-cli/`: CLI demo that shows text stabilization behavior.
- `docs/runtime-integration.md`: minimal Swift wiring for CoreML model + decoder + streaming engine.

## Notes

- Parakeet model export shape/signatures vary by release. Inspect exported ONNX I/O before wiring production inference.
- RNNT/TDT exports are usually split into multiple ONNX files (for example `encoder-model.onnx` and `decoder_joint-model.onnx`).
- CoreMLTools 9 does not support ONNX frontend conversion; use TorchScript -> CoreML conversion.
- CoreML/ANE behavior is highly architecture-dependent; measure on target hardware for every quantization setting.
- `CoreMLCTCTranscriptionModel` is a baseline adapter; Parakeet TDT/RNNT graphs likely require custom decoder/state wiring.

## Troubleshooting

- If export fails with `ModuleNotFoundError: No module named 'onnxscript'`, run:
  - `pip install onnxscript`
  - or `pip install -r requirements-conversion.txt`
