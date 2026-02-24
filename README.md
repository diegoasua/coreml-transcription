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

Note:
- `--python-transcriber` keeps the model loaded in-process (recommended for speed).
- `--transcribe-cmd` is still supported for shell-command integration.
- Commands run from project root by default (`--command-cwd`).

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
- `configs/parakeet-coreml-v5.env`: locked conversion/runtime/benchmark environment settings for the current baseline.
- `configs/parakeet-coreml-v5-hiacc.env`: mixed precision profile (`encoder=int8`, `decoder=int4`) to trade model size for accuracy.
- `configs/parakeet-coreml-v5-odmbp-lite.env`: OD-MBP-inspired profile (`encoder=linear-int8`, `decoder=kmeans-int4`) for lower WER at larger size.
- `configs/parakeet-coreml-v6-mixed68.env`: mixed encoder palettization profile (`encoder=6/8-bit by outlier score`, `decoder=int4`) targeting Argmax-like model size.
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
