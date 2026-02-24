# Runtime Integration (CoreML -> Swift)

This repo now includes a baseline CTC runtime path:

1. Export and inspect ONNX:
   - `scripts/export_nemo_to_onnx.py`
   - `scripts/inspect_onnx.py`
2. Convert/compress CoreML:
   - `scripts/convert_onnx_to_coreml.py`
   - `scripts/compress_coreml.py`
3. Load model in Swift with `CoreMLCTCTranscriptionModel`.
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

- The included adapter is for CTC-style logits output.
- Parakeet TDT/RNNT variants typically need additional decoder state management and custom post-processing.
- Use `scripts/inspect_onnx.py` output to map real input/output names before wiring.
