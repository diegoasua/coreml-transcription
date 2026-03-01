# Runtime Integration (CoreML -> Swift)

This repo includes:

- `CoreMLCTCTranscriptionModel`: baseline CTC adapter.
- `ParakeetCoreMLTDTTranscriptionModel`: native TDT adapter for Parakeet v2 encoder+decoder CoreML packages.

1. Export and inspect ONNX:
   - `scripts/export_nemo_to_onnx.py`
   - `scripts/inspect_onnx.py`
2. Convert/compress CoreML:
   - `scripts/convert_onnx_to_coreml.py`
   - `scripts/compress_coreml.py`
3. Load model in Swift with either `CoreMLCTCTranscriptionModel` (CTC) or `ParakeetCoreMLTDTTranscriptionModel` (Parakeet TDT).
4. Drive chunked inference with `StreamingInferenceEngine`.

## Minimal Swift Wiring

```swift
import RealtimeTranscriptionCore
import CoreML

let vocab = try String(contentsOfFile: "vocab.txt")
    .split(separator: "\n")
    .map(String.init)

let decoder = CTCGreedyDecoder(vocabulary: vocab, blankTokenID: 0)
let config = CoreMLCTCConfig(
    audioInputName: "audio_signal",
    lengthInputName: "audio_signal_length",
    logitsOutputName: "logits",
    expectedSampleRate: 16_000,
    computeUnits: .all
)

let modelURL = URL(fileURLWithPath: "parakeet-int4.mlpackage")
let model = try CoreMLCTCTranscriptionModel(modelURL: modelURL, config: config, decoder: decoder)
let vad = EnergyVAD()
var engine = StreamingInferenceEngine(model: model, vad: vad)
```

## Important

- Parakeet adapter expects split CoreML packages (`encoder-model-*.mlpackage`, `decoder_joint-model-*.mlpackage`) and `vocab.txt`.
- Streaming engine supports both policies:
  - `flushOnSpeechEnd=true`: reset on VAD speech->silence transitions.
  - `flushOnSpeechEnd=false` (current low-latency default): preserve decoder/model state across short pauses.
- Use `scripts/inspect_onnx.py` output to validate input/output names when changing exported models.

## Realtime Defaults (Current)

The current low-latency profile used by `transcribe-macos` and realtime bench:

- `chunk=160ms`, `hop=80ms`
- `decodeOnlyWhenSpeech=true`
- `flushOnSpeechEnd=false`
- `VAD start=-50 dBFS`, `VAD end=-58 dBFS`
- `maxSymbolsPerStep=4`, `maxTokensPerChunk=64`

Run realtime bench:

```bash
AUDIO_PATH=/path/to/audio.wav \
RUN_NAME=rt-bench \
bash scripts/run_realtime_bench_cli.sh
```
