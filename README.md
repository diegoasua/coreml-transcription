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
  --target macos15 \
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
python scripts/benchmark_tdt_components.py \
  --encoder-model artifacts/parakeet-tdt-0.6b-v2/encoder-model-int4.mlpackage \
  --encoder-manifest artifacts/parakeet-tdt-0.6b-v2/encoder-model-manifest.json \
  --decoder-model artifacts/parakeet-tdt-0.6b-v2/decoder_joint-model-int4.mlpackage \
  --decoder-manifest artifacts/parakeet-tdt-0.6b-v2/decoder_joint-model-manifest.json \
  --iterations 40 \
  --decoder-steps 64 \
  --compute-units cpu_and_ne
```

To inspect per-op device placement with Core ML's compute plan:

```bash
./.venv/bin/python scripts/analyze_coreml_compute_plan.py \
  --model artifacts/parakeet-tdt-0.6b-v2/decoder_joint-model-odmbp-approx.mlpackage \
  --compute-units cpu_and_ne
```

### 7) Swift runtime tests

```bash
swift test
```

### 7b) Swift file transcription CLI (native CoreML TDT)

```bash
swift run transcribe-cli \
  --audio /path/to/audio.wav \
  --model-dir artifacts/parakeet-tdt-0.6b-v2 \
  --suffix odmbp-approx
```

If `--audio` is omitted, `transcribe-cli` falls back to the LocalAgreement text demo.

### 7c) Minimal macOS mic app (SwiftUI)

```bash
swift run transcribe-macos
```

Set `PARAKEET_COREML_MODEL_DIR` (or enter it in the app UI) to point to your model artifacts directory.

For realtime streaming on Apple Silicon, you can keep the `odmbp-approx` encoder and swap only the decoder to the stateful wrapper:

```bash
PARAKEET_COREML_MODEL_SUFFIX=odmbp-approx \
PARAKEET_COREML_DECODER_SUFFIX=odmbp-approx-stateful-v2 \
bash scripts/run_transcribe_macos_release.sh
```

### 7d) Minimal iPhone app (SwiftUI)

There are two iOS entry points:

- `transcribe-ios` in the Swift package for code-level validation.
- `Apps/TranscribeIOSApp/TranscribeIOSApp.xcodeproj` for an installable iPhone app target.

The Xcode project is checked in. No Ruby or project-generation step is required.

For the native iPhone app, stage an importable model folder:

```bash
bash scripts/stage_transcribe_ios_models.sh
```

This stages the active live runtime bundle into `artifacts/ios-model-import/parakeet-tdt-0.6b-v2`. By default it copies only:

- `encoder-model-odmbp-approx.mlpackage`
- `decoder_joint-model-odmbp-approx-stateful-v2.mlpackage`
- `vocab.txt`

Override with `PARAKEET_COREML_MODEL_SUFFIX`, `PARAKEET_COREML_DECODER_SUFFIX`, or `PARAKEET_IOS_IMPORT_DIR` if needed.

Then import the staged folder from the Files picker inside the native iPhone app with the `Import…` button.

The native Xcode iPhone target should currently use `Import…` instead of bundling raw `.mlpackage` files, because Xcode 26 device signing is unreliable for that layout in this repo.

To run on device:

1. Open `Apps/TranscribeIOSApp/TranscribeIOSApp.xcodeproj` in Xcode.
2. Select the `TranscribeIOSApp` scheme and your iPhone.
3. If the phone does not appear, unlock it, keep it on the same Wi-Fi as the Mac, or connect it once over USB so Xcode can pair with it.
4. The app already includes `NSMicrophoneUsageDescription` via project build settings.

For command-line builds, this works once the device is available:

```bash
xcodebuild -project Apps/TranscribeIOSApp/TranscribeIOSApp.xcodeproj \
  -scheme TranscribeIOSApp \
  -destination 'id=<YOUR_DEVICE_ID>' \
  build
```

The iOS app looks for models in this order:
- `PARAKEET_COREML_MODEL_DIR`, if set
- imported app Documents folder `parakeet-tdt-0.6b-v2`
- bundled app resources `parakeet-tdt-0.6b-v2` if you explicitly add them yourself

To specialize rewrite-prefix stride for one device using repeated realtime-bench runs, exact confirmed-transcript matching, and `ingest->ready` latency as the ranking target while preserving headroom when latency differences are negligible:

```bash
PARAKEET_COREML_MODEL_SUFFIX=odmbp-approx \
PARAKEET_COREML_DECODER_SUFFIX=odmbp-approx-stateful-v2 \
PARAKEET_STREAM_MODE=rewrite-prefix \
PARAKEET_STREAM_PREFIX_LEFT_CONTEXT_FRAMES=160 \
PARAKEET_STREAM_PREFIX_RIGHT_CONTEXT_FRAMES=0 \
PARAKEET_STREAM_PREFIX_ALLOW_RIGHT_CONTEXT=0 \
PARAKEET_STREAM_PREFIX_ADAPTIVE=0 \
PARAKEET_STREAM_PREFIX_ENCODER_CACHE=1 \
PARAKEET_STREAM_LATEST_FIRST=1 \
PARAKEET_STREAM_CHUNK_MS=500 \
PARAKEET_STREAM_HOP_MS=250 \
PARAKEET_STREAM_MAX_BATCH_MS=500 \
PARAKEET_STREAM_BACKLOG_SOFT_SEC=5.0 \
PARAKEET_STREAM_BACKLOG_TARGET_SEC=1.5 \
PARAKEET_DECODE_ONLY_WHEN_SPEECH=0 \
PARAKEET_STREAM_FLUSH_ON_SPEECH_END=0 \
python3 scripts/specialize_realtime_prefix_stride.py \
  --audio artifacts/tmp/recording_60s.wav \
  --repetitions 3 \
  --latency-equivalence-ms 5 \
  --stride 0.60 \
  --stride 0.55 \
  --stride 0.50 \
  --stride 0.48 \
  --stride 0.45 \
  --stride 0.40
```

The script writes a JSON report, candidate summary CSV, and `recommended.env` under `artifacts/realtime-specialization-runs/<run-name>/`. By default, candidates within `5ms` of the best latency mean are treated as equivalent and the recommendation then prefers the one with the highest minimum `infer_rtfx`.

Realtime metric contract:
- Track this in CLI benches: `draft_ready_latency_ms_avg` and `draft_ready_latency_ms_p95`. These are audio ingress -> draft/hypothesis update ready, and are the primary benchmark latency KPIs.
- Track this in the app for UX: `ingest->screen(d)` avg and p95. This is audio ingress -> published draft text, including app/UI overhead.
- `ingest->ready(d)` in the app is the app-side equivalent of CLI `draft_ready_latency_ms_*`. Use it when comparing app vs bench without UI overhead mixed in.
- `ingest->screen(f)` / `final_confirmed_*` mean true final segment finalization only. They are informational, can be sparse, and are not the primary live latency KPI.
- `draft_onset_latency_ms_*`: speech-start ingress -> first visible draft text for that utterance.
- Legacy `confirmed_*` keys in `summary.json` are retained as aliases, but now mean true `final_confirmed_*`, not preview-stable text. Do not use them as the primary KPI.

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
  --python-transcriber scripts/parakeet_coreml_tdt_transcriber.py
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
  --python-transcriber scripts/parakeet_coreml_tdt_transcriber.py \
  --run-name parakeet-coreml-ls200
```

This uses OpenBench's dataset/metric stack directly and writes:
- `artifacts/openbench-runs/<run-name>/custom-openbench-summary.json`

Shortcut wrapper:

```bash
bash scripts/run_openbench_eval.sh \
  --dataset librispeech-200 \
  --python-transcriber scripts/parakeet_coreml_tdt_transcriber.py \
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
  --python-transcriber scripts/parakeet_coreml_tdt_transcriber.py \
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
- Decoder quality/speed knobs (CoreML TDT transcriber):
  - `PARAKEET_TDT_BEAM_WIDTH` (default `1`, greedy). Try `2` or `4` for lower WER.
  - `PARAKEET_TDT_DURATION_BEAM_WIDTH` (default matches beam width).
  - `PARAKEET_TDT_MAX_SYMBOLS_PER_STEP` (default `10`).

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
- `scripts/benchmark_tdt_components.py`: encoder + decoder-loop benchmark with end-to-end RTF estimate.
- `scripts/benchmark_transcripts.py`: quick WER/RTF calculator from JSONL logs.
- `scripts/eval_openbench_dataset.py`: standardized dataset evaluation harness (HF OpenBench datasets + local transcriber command).
- `scripts/run_openbench_custom_transcription.py`: OpenBench-native benchmark runner with custom local pipeline support for both `transcription` and `streaming_transcription`.
- `scripts/parakeet_coreml_tdt_transcriber.py`: local CoreML Parakeet TDT module (`transcribe_file(...)`, `stream_transcribe_file(...)`).
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
- TDT exports are usually split into multiple ONNX files (for example `encoder-model.onnx` and `decoder_joint-model.onnx`).
- CoreMLTools 9 does not support ONNX frontend conversion; use TorchScript -> CoreML conversion.
- CoreML/ANE behavior is highly architecture-dependent; measure on target hardware for every quantization setting.
- `CoreMLCTCTranscriptionModel` is a baseline adapter; Parakeet TDT/TDT graphs likely require custom decoder/state wiring.

## Troubleshooting

- If export fails with `ModuleNotFoundError: No module named 'onnxscript'`, run:
  - `pip install onnxscript`
  - or `pip install -r requirements-conversion.txt`
