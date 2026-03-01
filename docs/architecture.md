# Architecture (macOS-first, then iOS)

## Goals

- Reproduce a transparent Parakeet v2 -> CoreML conversion pipeline.
- Maximize Apple Neural Engine usage without opaque vendor SDK lock-in.
- Build a streaming pipeline with:
  - local VAD for segmentation,
  - rolling hypothesis updates,
  - confirmed text stabilization using LocalAgreement-style logic.

## High-Level Pipeline

1. `Audio Capture`
2. `Feature Extraction` (in Swift/Accelerate or inside model graph)
3. `ASR Inference` (CoreML model on ANE/GPU/CPU)
4. `Partial Decode`
5. `LocalAgreement Stabilizer`
6. `Confirmed + Hypothesis output streams`

## Conversion Track

1. Export from NeMo (`scripts/export_nemo_to_onnx.py`).
2. Inspect ONNX input/output signatures.
3. Convert TorchScript -> CoreML (`scripts/convert_torchscript_to_coreml.py`) using ONNX manifests for shapes.
4. Run compression experiments (`scripts/compress_coreml.py`).
5. Measure component latency + RTF estimate (`scripts/benchmark_tdt_components.py`) and WER (`scripts/benchmark_transcripts.py`, `scripts/eval_openbench_dataset.py`, or OpenBench-native `scripts/run_openbench_custom_transcription.py`).

## Streaming Track

- `VoiceActivityDetector.swift`: local gating of speech/silence windows.
- `LocalAgreement.swift`: confirms only stable prefix text.
- `AudioRingBuffer.swift`: maintains chunk/hop audio flow.
- `StreamingPolicy.swift`: one place for chunk/hop/sample-rate constants.
- `StreamingInferenceEngine.swift`: orchestrates chunking + VAD + model calls + segment flush.
- `CoreMLCTCTranscriptionModel.swift`: baseline CoreML adapter for CTC-style output heads.

## Why this split

- Model and streaming logic should evolve independently.
- You can swap model variants (full precision vs int4 vs sparse+int4) without touching text stabilization.
- You can benchmark each layer separately (model latency vs end-to-end UX/WER).

## References

- Argmax/WhisperKit paper: https://arxiv.org/pdf/2507.10860
- CoreML conversion docs: https://apple.github.io/coremltools/docs-guides/source/convert-models.html
- CoreML optimization docs: https://apple.github.io/coremltools/docs-guides/source/opt-conversion.html
- NeMo ASR docs: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html
