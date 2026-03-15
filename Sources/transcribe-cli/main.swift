import Foundation
import RealtimeTranscriptionCore

struct CLIArgs {
    private static let env = ProcessInfo.processInfo.environment
    private static let defaultStreamMode = (env["PARAKEET_STREAM_MODE"] ?? "rewrite-prefix").lowercased()

    var audioPath: String?
    var modelDir: String?
    var modelSuffix: String = ProcessInfo.processInfo.environment["PARAKEET_COREML_MODEL_SUFFIX"] ?? "odmbp-approx"
    var encoderModelSuffix: String? = ProcessInfo.processInfo.environment["PARAKEET_COREML_ENCODER_SUFFIX"]
    var decoderModelSuffix: String? = ProcessInfo.processInfo.environment["PARAKEET_COREML_DECODER_SUFFIX"]
    var chunkSec: Double = Double(ProcessInfo.processInfo.environment["PARAKEET_CLI_CHUNK_SEC"] ?? "") ?? 3.0
    var longformSegmentSec: Double = Double(ProcessInfo.processInfo.environment["PARAKEET_LONGFORM_SEGMENT_SEC"] ?? "") ?? 0.0
    var longformOverlapSec: Double = Double(ProcessInfo.processInfo.environment["PARAKEET_LONGFORM_OVERLAP_SEC"] ?? "") ?? 0.0
    var leftContextFrames: Int = Int(ProcessInfo.processInfo.environment["PARAKEET_ENCODER_LEFT_CONTEXT_FRAMES"] ?? "") ?? 300
    var rightContextFrames: Int = Int(ProcessInfo.processInfo.environment["PARAKEET_ENCODER_RIGHT_CONTEXT_FRAMES"] ?? "") ?? 120
    var allowRightContext: Bool = (ProcessInfo.processInfo.environment["PARAKEET_CLI_ALLOW_RIGHT_CONTEXT"] ?? "1") != "0"
    var maxSymbolsPerStep: Int = Int(ProcessInfo.processInfo.environment["PARAKEET_TDT_MAX_SYMBOLS_PER_STEP"] ?? "") ?? 10
    var maxTokensPerChunk: Int = Int(ProcessInfo.processInfo.environment["PARAKEET_TDT_MAX_TOKENS_PER_CHUNK"] ?? "") ?? 0
    var streamingHistoryFrames: Int = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_HISTORY_FRAMES"] ?? "") ?? 300
    var streamingMinTailFrames: Int = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_MIN_TAIL_FRAMES"] ?? "") ?? 8
    var realtimeBench: Bool = false
    var streamChunkMs: Int = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_CHUNK_MS"] ?? "") ??
        (CLIArgs.defaultStreamMode == "rewrite-prefix" ? 500 : 160)
    var streamHopMs: Int = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_HOP_MS"] ?? "") ??
        (CLIArgs.defaultStreamMode == "rewrite-prefix" ? 250 : 80)
    var streamAgreement: Int = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_AGREEMENT"] ?? "") ?? 2
    var streamDraftAgreement: Int = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_DRAFT_AGREEMENT"] ?? "") ?? 1
    var reportEveryMs: Int = Int(ProcessInfo.processInfo.environment["PARAKEET_BENCH_REPORT_MS"] ?? "") ?? 200
    var maxBatchMs: Int = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_MAX_BATCH_MS"] ?? "") ??
        (CLIArgs.defaultStreamMode == "rewrite-prefix" ? 500 : 200)
    var latestFirst: Bool = (ProcessInfo.processInfo.environment["PARAKEET_STREAM_LATEST_FIRST"] ?? "1") != "0"
    var backlogSoftSec: Double = Double(ProcessInfo.processInfo.environment["PARAKEET_STREAM_BACKLOG_SOFT_SEC"] ?? "") ??
        (CLIArgs.defaultStreamMode == "rewrite-prefix" ? 5.0 : 1.5)
    var backlogTargetSec: Double = Double(ProcessInfo.processInfo.environment["PARAKEET_STREAM_BACKLOG_TARGET_SEC"] ?? "") ??
        (CLIArgs.defaultStreamMode == "rewrite-prefix" ? 1.5 : 0.25)
    var queuePassSec: Double = Double(ProcessInfo.processInfo.environment["PARAKEET_BENCH_QUEUE_PASS_SEC"] ?? "") ?? 0.5
    var firstTokenPassMs: Double = Double(ProcessInfo.processInfo.environment["PARAKEET_BENCH_FIRST_TOKEN_PASS_MS"] ?? "") ?? 300
    var confirmedPassMs: Double = Double(ProcessInfo.processInfo.environment["PARAKEET_BENCH_CONFIRMED_PASS_MS"] ?? "") ?? 1700
    var metricsOutputPath: String?
    var traceOutputPath: String? = ProcessInfo.processInfo.environment["PARAKEET_STREAM_TRACE_PATH"]
}

func parseArgs() -> CLIArgs {
    var args = CLIArgs()
    var i = 1
    let argv = CommandLine.arguments
    while i < argv.count {
        switch argv[i] {
        case "--audio":
            if i + 1 < argv.count {
                args.audioPath = argv[i + 1]
                i += 1
            }
        case "--model-dir":
            if i + 1 < argv.count {
                args.modelDir = argv[i + 1]
                i += 1
            }
        case "--suffix":
            if i + 1 < argv.count {
                args.modelSuffix = argv[i + 1]
                i += 1
            }
        case "--encoder-suffix":
            if i + 1 < argv.count {
                args.encoderModelSuffix = argv[i + 1]
                i += 1
            }
        case "--decoder-suffix":
            if i + 1 < argv.count {
                args.decoderModelSuffix = argv[i + 1]
                i += 1
            }
        case "--chunk-sec":
            if i + 1 < argv.count {
                args.chunkSec = Double(argv[i + 1]) ?? args.chunkSec
                i += 1
            }
        case "--longform-segment-sec":
            if i + 1 < argv.count {
                args.longformSegmentSec = Double(argv[i + 1]) ?? args.longformSegmentSec
                i += 1
            }
        case "--longform-overlap-sec":
            if i + 1 < argv.count {
                args.longformOverlapSec = Double(argv[i + 1]) ?? args.longformOverlapSec
                i += 1
            }
        case "--left-context-frames":
            if i + 1 < argv.count {
                args.leftContextFrames = Int(argv[i + 1]) ?? args.leftContextFrames
                i += 1
            }
        case "--right-context-frames":
            if i + 1 < argv.count {
                args.rightContextFrames = Int(argv[i + 1]) ?? args.rightContextFrames
                i += 1
            }
        case "--no-right-context":
            args.allowRightContext = false
        case "--max-symbols-per-step":
            if i + 1 < argv.count {
                args.maxSymbolsPerStep = Int(argv[i + 1]) ?? args.maxSymbolsPerStep
                i += 1
            }
        case "--max-tokens-per-chunk":
            if i + 1 < argv.count {
                args.maxTokensPerChunk = Int(argv[i + 1]) ?? args.maxTokensPerChunk
                i += 1
            }
        case "--realtime-bench":
            args.realtimeBench = true
        case "--stream-chunk-ms":
            if i + 1 < argv.count {
                args.streamChunkMs = Int(argv[i + 1]) ?? args.streamChunkMs
                i += 1
            }
        case "--stream-hop-ms":
            if i + 1 < argv.count {
                args.streamHopMs = Int(argv[i + 1]) ?? args.streamHopMs
                i += 1
            }
        case "--stream-agreement":
            if i + 1 < argv.count {
                args.streamAgreement = Int(argv[i + 1]) ?? args.streamAgreement
                i += 1
            }
        case "--stream-draft-agreement":
            if i + 1 < argv.count {
                args.streamDraftAgreement = Int(argv[i + 1]) ?? args.streamDraftAgreement
                i += 1
            }
        case "--report-every-ms":
            if i + 1 < argv.count {
                args.reportEveryMs = Int(argv[i + 1]) ?? args.reportEveryMs
                i += 1
            }
        case "--max-batch-ms":
            if i + 1 < argv.count {
                args.maxBatchMs = Int(argv[i + 1]) ?? args.maxBatchMs
                i += 1
            }
        case "--latest-first":
            args.latestFirst = true
        case "--oldest-first":
            args.latestFirst = false
        case "--backlog-soft-sec":
            if i + 1 < argv.count {
                args.backlogSoftSec = Double(argv[i + 1]) ?? args.backlogSoftSec
                i += 1
            }
        case "--backlog-target-sec":
            if i + 1 < argv.count {
                args.backlogTargetSec = Double(argv[i + 1]) ?? args.backlogTargetSec
                i += 1
            }
        case "--queue-pass-sec":
            if i + 1 < argv.count {
                args.queuePassSec = Double(argv[i + 1]) ?? args.queuePassSec
                i += 1
            }
        case "--first-token-pass-ms":
            if i + 1 < argv.count {
                args.firstTokenPassMs = Double(argv[i + 1]) ?? args.firstTokenPassMs
                i += 1
            }
        case "--confirmed-pass-ms":
            if i + 1 < argv.count {
                args.confirmedPassMs = Double(argv[i + 1]) ?? args.confirmedPassMs
                i += 1
            }
        case "--metrics-output":
            if i + 1 < argv.count {
                args.metricsOutputPath = argv[i + 1]
                i += 1
            }
        case "--trace-output":
            if i + 1 < argv.count {
                args.traceOutputPath = argv[i + 1]
                i += 1
            }
        default:
            break
        }
        i += 1
    }
    return args
}

func runInteractiveDemo() {
    print("LocalAgreement CLI demo")
    print("Enter partial hypotheses line by line. Empty line flushes segment. Ctrl-D exits.")

    var controller = StreamingTextController(requiredAgreementCount: 2)
    while let line = readLine() {
        if line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let state = controller.endSegment()
            print("CONFIRMED: \(state.confirmed)")
            print("HYPOTHESIS: \(state.hypothesis)")
            continue
        }

        let state = controller.update(partialText: line)
        print("CONFIRMED: \(state.confirmed)")
        print("HYPOTHESIS: \(state.hypothesis)")
    }

    let final = controller.endSegment()
    if !final.confirmed.isEmpty || !final.hypothesis.isEmpty {
        print("FINAL CONFIRMED: \(final.confirmed)")
        print("FINAL HYPOTHESIS: \(final.hypothesis)")
    }
}

#if canImport(CoreML) && canImport(AVFoundation)
import CoreML
@preconcurrency import AVFoundation

private final class ConversionInputState: @unchecked Sendable {
    var consumed = false
}

func logProgress(_ message: String) {
    fputs("[transcribe-cli] \(message)\n", stderr)
    fflush(stderr)
}

private final class JSONLTraceWriter {
    private let fileHandle: FileHandle

    init?(path: String) {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let url = URL(fileURLWithPath: trimmed)
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            if FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)
            }
            FileManager.default.createFile(atPath: url.path, contents: nil)
            self.fileHandle = try FileHandle(forWritingTo: url)
        } catch {
            return nil
        }
    }

    deinit {
        try? fileHandle.close()
    }

    func write(_ object: [String: Any]) {
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(withJSONObject: object, options: []) else {
            return
        }
        var line = data
        line.append(0x0A)
        try? fileHandle.write(contentsOf: line)
    }
}

func defaultModelDirectory() -> URL {
    if let env = ProcessInfo.processInfo.environment["PARAKEET_COREML_MODEL_DIR"], !env.isEmpty {
        return URL(fileURLWithPath: env)
    }
    return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appendingPathComponent("artifacts/parakeet-tdt-0.6b-v2")
}

func readAudioFileMonoFloat(url: URL) throws -> ([Float], Int) {
    let file = try AVAudioFile(forReading: url)
    let inputFormat = file.processingFormat
    let totalFrames = AVAudioFrameCount(file.length)
    guard let buffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: totalFrames) else {
        throw NSError(domain: "transcribe-cli", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate audio buffer"])
    }
    try file.read(into: buffer)

    let channels = Int(inputFormat.channelCount)
    let frameCount = Int(buffer.frameLength)
    guard let channelData = buffer.floatChannelData else {
        throw NSError(domain: "transcribe-cli", code: 2, userInfo: [NSLocalizedDescriptionKey: "Unsupported audio format"])
    }

    var mono = Array(repeating: Float(0), count: frameCount)
    if channels == 1 {
        mono = Array(UnsafeBufferPointer(start: channelData[0], count: frameCount))
    } else {
        for i in 0..<frameCount {
            var sum: Float = 0
            for c in 0..<channels {
                sum += channelData[c][i]
            }
            mono[i] = sum / Float(channels)
        }
    }

    return (mono, Int(inputFormat.sampleRate))
}

func avResampleMonoFloat(_ samples: [Float], from sourceRate: Int, to targetRate: Int) throws -> [Float] {
    guard sourceRate != targetRate, !samples.isEmpty else { return samples }

    guard let srcFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(sourceRate),
        channels: 1,
        interleaved: false
    ), let dstFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(targetRate),
        channels: 1,
        interleaved: false
    ), let converter = AVAudioConverter(from: srcFormat, to: dstFormat) else {
        throw NSError(domain: "transcribe-cli", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create AVAudioConverter"])
    }

    guard let srcBuffer = AVAudioPCMBuffer(
        pcmFormat: srcFormat,
        frameCapacity: AVAudioFrameCount(samples.count)
    ) else {
        throw NSError(domain: "transcribe-cli", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate source conversion buffer"])
    }
    srcBuffer.frameLength = AVAudioFrameCount(samples.count)
    if let srcData = srcBuffer.floatChannelData?[0] {
        samples.withUnsafeBufferPointer { src in
            srcData.initialize(from: src.baseAddress!, count: src.count)
        }
    }

    let estimatedOutFrames = max(1, Int(ceil(Double(samples.count) * Double(targetRate) / Double(sourceRate))) + 256)
    guard let dstBuffer = AVAudioPCMBuffer(
        pcmFormat: dstFormat,
        frameCapacity: AVAudioFrameCount(estimatedOutFrames)
    ) else {
        throw NSError(domain: "transcribe-cli", code: 5, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate destination conversion buffer"])
    }

    let inputState = ConversionInputState()
    var convError: NSError?
    let status = converter.convert(to: dstBuffer, error: &convError) { _, outStatus in
        if inputState.consumed {
            outStatus.pointee = .noDataNow
            return nil
        }
        inputState.consumed = true
        outStatus.pointee = .haveData
        return srcBuffer
    }

    if status == .error || convError != nil {
        throw convError ?? NSError(domain: "transcribe-cli", code: 6, userInfo: [NSLocalizedDescriptionKey: "AVAudioConverter conversion failed"])
    }

    let outLen = Int(dstBuffer.frameLength)
    guard let dstData = dstBuffer.floatChannelData?[0], outLen > 0 else {
        return []
    }
    return Array(UnsafeBufferPointer(start: dstData, count: outLen))
}

func mergeTextByOverlap(_ chunks: [String]) -> String {
    var merged: [String] = []
    for text in chunks {
        let tokens = text.split(whereSeparator: \.isWhitespace).map(String.init)
        if tokens.isEmpty { continue }
        if merged.isEmpty {
            merged.append(contentsOf: tokens)
            continue
        }
        let maxOverlap = min(64, min(merged.count, tokens.count))
        var overlap = 0
        if maxOverlap > 0 {
            for candidate in stride(from: maxOverlap, through: 1, by: -1) {
                let lhs = Array(merged.suffix(candidate))
                let rhs = Array(tokens.prefix(candidate))
                if lhs == rhs {
                    overlap = candidate
                    break
                }
            }
        }
        merged.append(contentsOf: tokens.dropFirst(overlap))
    }
    return merged.joined(separator: " ")
}

func transcribeSamplesByChunks(
    model: ParakeetCoreMLTDTTranscriptionModel,
    samples: [Float],
    sampleRate: Int,
    chunkSamples: Int,
    progressPrefix: String
) throws -> String {
    if samples.isEmpty { return "" }
    model.resetState()
    var start = 0
    var latest = ""
    let totalChunks = max(1, Int(ceil(Double(samples.count) / Double(chunkSamples))))
    var chunkIndex = 0
    while start < samples.count {
        let end = min(samples.count, start + chunkSamples)
        chunkIndex += 1
        logProgress("\(progressPrefix) chunk \(chunkIndex)/\(totalChunks)")
        latest = try model.transcribeChunk(Array(samples[start..<end]), sampleRate: sampleRate)
        start = end
    }
    return latest
}

func mergedTranscript(confirmed: String, hypothesis: String) -> String {
    let c = confirmed.trimmingCharacters(in: .whitespacesAndNewlines)
    let h = hypothesis.trimmingCharacters(in: .whitespacesAndNewlines)
    if c.isEmpty { return h }
    if h.isEmpty { return c }
    return c + " " + h
}

func percentile(_ values: [Double], q: Double) -> Double? {
    guard !values.isEmpty else { return nil }
    let sorted = values.sorted()
    if sorted.count == 1 { return sorted[0] }
    let index = max(0, min(Double(sorted.count - 1), q * Double(sorted.count - 1)))
    let lo = Int(floor(index))
    let hi = Int(ceil(index))
    if lo == hi { return sorted[lo] }
    let frac = index - Double(lo)
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

private struct BenchModelProfileAccumulator {
    var modelCallCount = 0
    var featureExtractMs = 0.0
    var encoderPrepareMs = 0.0
    var encoderPredictMs = 0.0
    var decodeMs = 0.0
    var mergeMs = 0.0
    var totalModelMs = 0.0
    var encoderWindowCount = 0
    var encoderWindowReuseCount = 0
    var encoderWindowComputeCount = 0
    var encoderReuseStepCount = 0
    var encoderComputeStepCount = 0
    var decoderWindowCount = 0
    var decoderResumeWindowCount = 0
    var decoderResumeTokenCount = 0
    var decoderPredictCallCount = 0
    var decoderAdvanceCallCount = 0
    var decoderEmittedTokenCount = 0
    var decoderBlankStepCount = 0
    var decodeRequestedFrames = 0
    var decodeCopiedFrames = 0

    mutating func add(_ diagnostic: ParakeetStreamingDiagnostic) {
        modelCallCount += 1
        featureExtractMs += diagnostic.featureExtractMs
        encoderPrepareMs += diagnostic.encoderPrepareMs
        encoderPredictMs += diagnostic.encoderPredictMs
        decodeMs += diagnostic.decodeMs
        mergeMs += diagnostic.mergeMs
        totalModelMs += diagnostic.totalModelMs
        encoderWindowCount += diagnostic.encoderWindowCount
        encoderWindowReuseCount += diagnostic.encoderWindowReuseCount
        encoderWindowComputeCount += diagnostic.encoderWindowComputeCount
        encoderReuseStepCount += diagnostic.encoderReuseStepCount
        encoderComputeStepCount += diagnostic.encoderComputeStepCount
        decoderWindowCount += diagnostic.decoderWindowCount
        decoderResumeWindowCount += diagnostic.decoderResumeWindowCount
        decoderResumeTokenCount += diagnostic.decoderResumeTokenCount
        decoderPredictCallCount += diagnostic.decoderPredictCallCount
        decoderAdvanceCallCount += diagnostic.decoderAdvanceCallCount
        decoderEmittedTokenCount += diagnostic.decoderEmittedTokenCount
        decoderBlankStepCount += diagnostic.decoderBlankStepCount
        decodeRequestedFrames += diagnostic.decodeRequestedFrames
        decodeCopiedFrames += diagnostic.decodeCopiedFrames
    }

    var jsonObject: [String: Any] {
        guard modelCallCount > 0 else {
            return [
                "model_call_count": 0
            ]
        }
        let callCount = Double(modelCallCount)
        let totalEncoderSteps = max(0, encoderReuseStepCount + encoderComputeStepCount)
        return [
            "model_call_count": modelCallCount,
            "avg_total_model_ms": totalModelMs / callCount,
            "avg_feature_extract_ms": featureExtractMs / callCount,
            "avg_encoder_prepare_ms": encoderPrepareMs / callCount,
            "avg_encoder_predict_ms": encoderPredictMs / callCount,
            "avg_decode_ms": decodeMs / callCount,
            "avg_merge_ms": mergeMs / callCount,
            "share_feature_extract": featureExtractMs / max(totalModelMs, 1e-9),
            "share_encoder_prepare": encoderPrepareMs / max(totalModelMs, 1e-9),
            "share_encoder_predict": encoderPredictMs / max(totalModelMs, 1e-9),
            "share_decode": decodeMs / max(totalModelMs, 1e-9),
            "share_merge": mergeMs / max(totalModelMs, 1e-9),
            "encoder_window_count_total": encoderWindowCount,
            "encoder_window_reuse_count_total": encoderWindowReuseCount,
            "encoder_window_compute_count_total": encoderWindowComputeCount,
            "encoder_window_reuse_ratio": Double(encoderWindowReuseCount) / max(1.0, Double(encoderWindowCount)),
            "encoder_reuse_step_count_total": encoderReuseStepCount,
            "encoder_compute_step_count_total": encoderComputeStepCount,
            "encoder_step_reuse_ratio": Double(encoderReuseStepCount) / max(1.0, Double(totalEncoderSteps)),
            "avg_decode_requested_frames": Double(decodeRequestedFrames) / callCount,
            "avg_decode_copied_frames": Double(decodeCopiedFrames) / callCount,
            "decoder_window_count_total": decoderWindowCount,
            "decoder_resume_window_count_total": decoderResumeWindowCount,
            "decoder_resume_token_count_total": decoderResumeTokenCount,
            "decoder_predict_call_count_total": decoderPredictCallCount,
            "decoder_advance_call_count_total": decoderAdvanceCallCount,
            "decoder_emitted_token_count_total": decoderEmittedTokenCount,
            "decoder_blank_step_count_total": decoderBlankStepCount,
            "decoder_windows_per_model_call": Double(decoderWindowCount) / callCount,
            "decoder_predict_calls_per_model_call": Double(decoderPredictCallCount) / callCount,
            "decoder_predict_calls_per_window": Double(decoderPredictCallCount) / max(1.0, Double(decoderWindowCount)),
            "decoder_predict_calls_per_emitted_token": Double(decoderPredictCallCount) / max(1.0, Double(decoderEmittedTokenCount)),
            "decoder_advance_call_ratio": Double(decoderAdvanceCallCount) / max(1.0, Double(decoderPredictCallCount)),
        ]
    }
}

func runRealtimeBenchmark(
    model: ParakeetCoreMLTDTTranscriptionModel,
    samples: [Float],
    sampleRate: Int,
    args: CLIArgs
) throws {
    typealias BenchEngine = StreamingInferenceEngine<ParakeetCoreMLTDTTranscriptionModel, EnergyVAD>

    let chunkSamples = max(1, sampleRate * max(20, args.streamChunkMs) / 1000)
    let hopSamples = max(1, min(chunkSamples, sampleRate * max(10, args.streamHopMs) / 1000))
    let maxBatchSamples = max(hopSamples, sampleRate * max(10, args.maxBatchMs) / 1000)
    let maxBatchChunks = max(1, Int(ceil(Double(maxBatchSamples) / Double(hopSamples))))
    let reportEverySec = Double(max(50, args.reportEveryMs)) / 1000.0
    let totalAudioSec = Double(samples.count) / Double(sampleRate)
    let backlogSoftSamples = max(hopSamples, Int(args.backlogSoftSec * Double(sampleRate)))
    let backlogTargetSamples = max(hopSamples, Int(args.backlogTargetSec * Double(sampleRate)))
    let latestResyncContextSec = max(
        0.0,
        Double(ProcessInfo.processInfo.environment["PARAKEET_STREAM_LATEST_RESYNC_CONTEXT_SEC"] ?? "") ?? 0.48
    )
    let latestResyncContextSamples = max(
        maxBatchSamples,
        Int(round(latestResyncContextSec * Double(sampleRate)))
    )
    let latestResyncContextChunks = max(1, Int(ceil(Double(latestResyncContextSamples) / Double(hopSamples))))
    let latestOnsetProtectSec = max(
        0.0,
        Double(ProcessInfo.processInfo.environment["PARAKEET_STREAM_ONSET_PROTECT_SEC"] ?? "") ?? 0.8
    )
    let latestOnsetMinConfirmedWords = max(
        1,
        Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_ONSET_MIN_CONFIRMED_WORDS"] ?? "") ?? 2
    )
    let debugText = (ProcessInfo.processInfo.environment["PARAKEET_BENCH_DEBUG_TEXT"] ?? "0") == "1"
    let benchVADStart = Float(ProcessInfo.processInfo.environment["PARAKEET_VAD_START_DBFS"] ?? "") ?? -50
    let benchVADEnd = Float(ProcessInfo.processInfo.environment["PARAKEET_VAD_END_DBFS"] ?? "") ?? -58
    let benchVADMinSpeech = Int(ProcessInfo.processInfo.environment["PARAKEET_VAD_MIN_SPEECH_MS"] ?? "") ?? 60
    let benchVADMinSilence = Int(ProcessInfo.processInfo.environment["PARAKEET_VAD_MIN_SILENCE_MS"] ?? "") ?? 400
    let benchDecodeOnlyWhenSpeech = (ProcessInfo.processInfo.environment["PARAKEET_DECODE_ONLY_WHEN_SPEECH"] ?? "0") != "0"
    let benchFlushOnSpeechEnd = (ProcessInfo.processInfo.environment["PARAKEET_STREAM_FLUSH_ON_SPEECH_END"] ?? "0") != "0"
    let benchMaxSpeechChunks = Int(
        ProcessInfo.processInfo.environment["PARAKEET_STREAM_MAX_SPEECH_CHUNKS"] ?? ""
    ) ?? 80
    let benchMaxStagnantChunks = Int(
        ProcessInfo.processInfo.environment["PARAKEET_STREAM_MAX_STAGNANT_CHUNKS"] ?? ""
    ) ?? 24
    let traceWriter = args.traceOutputPath.flatMap(JSONLTraceWriter.init(path:))

    var chunks: [[Float]] = []
    chunks.reserveCapacity(max(1, Int(ceil(Double(samples.count) / Double(hopSamples)))))
    var s = 0
    while s < samples.count {
        let e = min(samples.count, s + hopSamples)
        chunks.append(Array(samples[s..<e]))
        s = e
    }
    let totalChunks = chunks.count
    let hopSec = Double(hopSamples) / Double(sampleRate)

    var engine = BenchEngine(
        model: model,
        vad: EnergyVAD(config: .init(
            startThresholdDBFS: benchVADStart,
            endThresholdDBFS: benchVADEnd,
            minSpeechMs: benchVADMinSpeech,
            minSilenceMs: benchVADMinSilence
        )),
        policy: .init(sampleRate: sampleRate, chunkMs: args.streamChunkMs, hopMs: args.streamHopMs),
        requiredAgreementCount: max(1, args.streamAgreement),
        draftAgreementCount: max(1, args.streamDraftAgreement),
        decodeOnlyWhenSpeech: benchDecodeOnlyWhenSpeech,
        flushOnSpeechEnd: benchFlushOnSpeechEnd,
        maxSpeechChunkRunBeforeReset: benchMaxSpeechChunks > 0 ? benchMaxSpeechChunks : nil,
        maxStagnantSpeechChunks: benchMaxStagnantChunks > 0 ? benchMaxStagnantChunks : nil,
        ringBufferCapacity: sampleRate * 12
    )

    var currentTimeSec = 0.0
    var nextReportSec = 0.0
    var arrivalIndex = 0
    var queue: [Int] = []
    var processedChunks = 0
    var totalInferSec = 0.0
    var maxQueueSecObserved = 0.0
    var droppedChunks = 0

    var prevIsSpeech = false
    var speechStartSampleCursor: Int?
    var speechBaselineDraft = ""
    var speechBaselineMerged = ""
    var speechBaselineConfirmed = ""
    var speechDraftFirstTokenRecorded = false
    var speechMergedFirstTokenRecorded = false
    var speechConfirmedRecorded = false
    var draftFirstTokenMs: [Double] = []
    var mergedFirstTokenMs: [Double] = []
    var confirmedMs: [Double] = []
    var draftReadyLatencyMs: [Double] = []
    var confirmedReadyLatencyMs: [Double] = []
    var corrections = 0
    var lastConfirmed = ""
    var lastHypothesis = ""
    var lastMerged = ""
    var transcriptCursorSec = 0.0
    var traceBatchIndex = 0
    var processedAudioCursorSec = 0.0
    var ingressTimeline = AudioIngressTimeline(sampleRate: sampleRate)
    var engineAudioCursorBaseSampleCursor = 0
    let tailHalfStartSec = totalAudioSec * 0.5
    var modelProfileOverall = BenchModelProfileAccumulator()
    var modelProfileTailHalf = BenchModelProfileAccumulator()
    var lastModelDiagnosticCallIndexSeen = -1

    let startWall = CFAbsoluteTimeGetCurrent()
    while arrivalIndex < totalChunks || !queue.isEmpty {
        while arrivalIndex < totalChunks {
            let arrivalTime = Double(arrivalIndex) * hopSec
            if arrivalTime <= currentTimeSec {
                _ = ingressTimeline.recordIngress(sampleCount: chunks[arrivalIndex].count, receivedAtSec: arrivalTime)
                queue.append(arrivalIndex)
                arrivalIndex += 1
            } else {
                break
            }
        }

        if queue.isEmpty {
            if arrivalIndex < totalChunks {
                currentTimeSec = Double(arrivalIndex) * hopSec
            }
            continue
        }

        let protectLatestFirstOnset = StreamingSchedulerSupport.shouldProtectLatestFirstOnset(
            elapsedSec: speechStartSampleCursor.map {
                max(0.0, Double(Int(round(processedAudioCursorSec * Double(sampleRate))) - $0) / Double(sampleRate))
            },
            currentConfirmed: lastConfirmed,
            baselineConfirmed: speechBaselineConfirmed,
            minConfirmedWords: latestOnsetMinConfirmedWords,
            maxOnsetSec: latestOnsetProtectSec
        )

        // Low-latency policy: drop stale queue beyond configured budget.
        var shouldResyncBeforeBatch = false
        var preserveHypothesisOnResync = false
        let queueSamples = queue.count * hopSamples
        if !protectLatestFirstOnset && queueSamples > backlogSoftSamples && queue.count > 1 {
            var targetChunkCount = max(1, Int(ceil(Double(backlogTargetSamples) / Double(hopSamples))))
            if args.latestFirst {
                targetChunkCount = max(targetChunkCount, latestResyncContextChunks)
            }
            if queue.count > targetChunkCount {
                let keep: [Int]
                if args.latestFirst {
                    keep = Array(queue.suffix(targetChunkCount))
                } else {
                    keep = Array(queue.prefix(targetChunkCount))
                }
                droppedChunks += max(0, queue.count - keep.count)
                queue = keep
                shouldResyncBeforeBatch = true
                preserveHypothesisOnResync = args.latestFirst
            }
        }

        let shouldUseLatestFirstCatchUp = StreamingSchedulerSupport.shouldUseLatestFirstCatchUp(
            latestFirstEnabled: args.latestFirst,
            protectOnset: protectLatestFirstOnset,
            queuedSampleCount: queueSamples,
            catchUpTriggerSamples: backlogTargetSamples
        )
        let batchIndexes: [Int]
        if shouldUseLatestFirstCatchUp {
            let targetChunkCount = max(1, Int(ceil(Double(backlogTargetSamples) / Double(hopSamples))))
            let batchChunkCount = StreamingSchedulerSupport.catchUpBatchCount(
                queuedUnitCount: queue.count,
                baseBatchCount: maxBatchChunks,
                catchUpTargetUnitCount: targetChunkCount,
                minContextUnitCount: latestResyncContextChunks
            )
            if queueSamples > backlogSoftSamples {
                let dropped = max(0, queue.count - batchChunkCount)
                if dropped > 0 {
                    droppedChunks += dropped
                    shouldResyncBeforeBatch = true
                    preserveHypothesisOnResync = args.latestFirst
                }
                batchIndexes = Array(queue.suffix(batchChunkCount))
                queue.removeAll(keepingCapacity: true)
            } else {
                batchIndexes = Array(queue.prefix(batchChunkCount))
                queue.removeFirst(batchIndexes.count)
            }
        } else {
            batchIndexes = Array(queue.prefix(maxBatchChunks))
            queue.removeFirst(batchIndexes.count)
        }

        var batch: [Float] = []
        batch.reserveCapacity(batchIndexes.reduce(0) { $0 + chunks[$1].count })
        for idx in batchIndexes {
            batch.append(contentsOf: chunks[idx])
        }
        let batchStartSampleCursor = (batchIndexes.first ?? 0) * hopSamples

        if shouldResyncBeforeBatch {
            _ = engine.discardStream(preserveHypothesis: preserveHypothesisOnResync)
            engineAudioCursorBaseSampleCursor = batchStartSampleCursor
            prevIsSpeech = false
            speechStartSampleCursor = nil
            speechBaselineDraft = ""
            speechBaselineMerged = ""
            speechBaselineConfirmed = ""
            speechDraftFirstTokenRecorded = false
            speechMergedFirstTokenRecorded = false
            speechConfirmedRecorded = false
        }

        let batchStartTimeSec = currentTimeSec
        let before = CFAbsoluteTimeGetCurrent()
        let events = try engine.process(samples: batch)
        let elapsed = CFAbsoluteTimeGetCurrent() - before
        let batchEndTimeSec = batchStartTimeSec + elapsed
        totalInferSec += elapsed
        currentTimeSec = batchEndTimeSec
        processedChunks += batchIndexes.count
        traceBatchIndex += 1

        let latestChunk = batchIndexes.last ?? 0
        let latestAudioCursorSec = min(totalAudioSec, Double(latestChunk + 1) * hopSec)
        processedAudioCursorSec = latestAudioCursorSec

        for event in events {
            let eventSampleCursor = engineAudioCursorBaseSampleCursor + Int(
                round(event.audioCursorSec * Double(sampleRate))
            )
            let eventAudioCursorSec = min(totalAudioSec, Double(eventSampleCursor) / Double(sampleRate))
            let eventIngressTimeSec = ingressTimeline.ingressTime(forSampleCursor: eventSampleCursor)
            let preEventConfirmed = lastConfirmed
            let preEventDraft = lastHypothesis
            let preEventMerged = lastMerged
            let merged = mergedTranscript(confirmed: event.transcript.confirmed, hypothesis: event.transcript.hypothesis)
            if debugText, merged != lastMerged {
                let snippet = String(merged.prefix(160)).replacingOccurrences(of: "\n", with: " ")
                logProgress("realtime-bench text-change: \(snippet)")
            }
            if event.isSpeech, !prevIsSpeech {
                speechStartSampleCursor = eventSampleCursor
                speechBaselineDraft = preEventDraft
                speechBaselineMerged = preEventMerged
                speechBaselineConfirmed = preEventConfirmed
                speechDraftFirstTokenRecorded = false
                speechMergedFirstTokenRecorded = false
                speechConfirmedRecorded = false
            }
            if event.isSpeech,
               let startSampleCursor = speechStartSampleCursor,
               !speechDraftFirstTokenRecorded,
               event.transcript.hypothesis != speechBaselineDraft,
               let startIngressTimeSec = ingressTimeline.ingressTime(forSampleCursor: startSampleCursor) {
                draftFirstTokenMs.append(max(0, (batchEndTimeSec - startIngressTimeSec) * 1000.0))
                speechDraftFirstTokenRecorded = true
            }
            if event.isSpeech,
               let startSampleCursor = speechStartSampleCursor,
               !speechMergedFirstTokenRecorded,
               merged != speechBaselineMerged,
               let startIngressTimeSec = ingressTimeline.ingressTime(forSampleCursor: startSampleCursor) {
                mergedFirstTokenMs.append(max(0, (batchEndTimeSec - startIngressTimeSec) * 1000.0))
                speechMergedFirstTokenRecorded = true
            }
            if event.isSpeech,
               let startSampleCursor = speechStartSampleCursor,
               !speechConfirmedRecorded,
               event.transcript.confirmed != speechBaselineConfirmed,
               let startIngressTimeSec = ingressTimeline.ingressTime(forSampleCursor: startSampleCursor) {
                confirmedMs.append(max(0, (batchEndTimeSec - startIngressTimeSec) * 1000.0))
                speechConfirmedRecorded = true
            }
            if event.transcript.hypothesis != lastHypothesis,
               let eventIngressTimeSec {
                draftReadyLatencyMs.append(max(0, (batchEndTimeSec - eventIngressTimeSec) * 1000.0))
            }
            if event.transcript.confirmed != lastConfirmed,
               let eventIngressTimeSec {
                confirmedReadyLatencyMs.append(max(0, (batchEndTimeSec - eventIngressTimeSec) * 1000.0))
            }
            if merged != lastMerged || event.transcript.confirmed != lastConfirmed {
                transcriptCursorSec = max(transcriptCursorSec, eventAudioCursorSec)
            }
            if event.didFlushSegment, let startSampleCursor = speechStartSampleCursor {
                if !speechConfirmedRecorded,
                   let startIngressTimeSec = ingressTimeline.ingressTime(forSampleCursor: startSampleCursor) {
                    confirmedMs.append(max(0, (batchEndTimeSec - startIngressTimeSec) * 1000.0))
                    speechConfirmedRecorded = true
                }
                transcriptCursorSec = max(transcriptCursorSec, eventAudioCursorSec)
                speechStartSampleCursor = nil
                speechBaselineDraft = ""
                speechBaselineMerged = ""
                speechBaselineConfirmed = ""
                speechDraftFirstTokenRecorded = false
                speechMergedFirstTokenRecorded = false
                speechConfirmedRecorded = false
            }
            if event.transcript.confirmed == lastConfirmed,
               event.transcript.hypothesis != lastHypothesis,
               !lastHypothesis.isEmpty {
                corrections += 1
            }
            lastConfirmed = event.transcript.confirmed
            lastHypothesis = event.transcript.hypothesis
            lastMerged = merged
            prevIsSpeech = event.isSpeech
        }

        let queueSec = Double(queue.count * hopSamples) / Double(sampleRate)
        maxQueueSecObserved = max(maxQueueSecObserved, queueSec)
        if let diagnostic = model.lastStreamingDiagnostic,
           diagnostic.callIndex != lastModelDiagnosticCallIndexSeen {
            lastModelDiagnosticCallIndexSeen = diagnostic.callIndex
            modelProfileOverall.add(diagnostic)
            if latestAudioCursorSec >= tailHalfStartSec {
                modelProfileTailHalf.add(diagnostic)
            }
        }
        if let traceWriter {
            var traceObject: [String: Any] = [
                "kind": "realtime_batch",
                "trace_batch_index": traceBatchIndex,
                "arrival_index": arrivalIndex,
                "batch_chunk_count": batchIndexes.count,
                "batch_first_chunk": batchIndexes.first ?? -1,
                "batch_last_chunk": latestChunk,
                "latest_audio_cursor_sec": latestAudioCursorSec,
                "current_time_sec": currentTimeSec,
                "infer_elapsed_sec": elapsed,
                "queue_sec": queueSec,
                "queue_chunk_count": queue.count,
                "dropped_chunks_total": droppedChunks,
                "resync_before_batch": shouldResyncBeforeBatch,
                "events_count": events.count,
                "transcript_cursor_sec": transcriptCursorSec,
                "finalized_confirmed_len": lastConfirmed.count,
                "finalized_hypothesis_len": lastHypothesis.count,
                "merged_preview": String(lastMerged.prefix(120)),
            ]
            if let event = events.last {
                traceObject["event_is_speech"] = event.isSpeech
                traceObject["event_did_flush_segment"] = event.didFlushSegment
                traceObject["event_audio_cursor_sec"] = event.audioCursorSec
                traceObject["event_energy_dbfs"] = event.energyDBFS
                traceObject["event_revision"] = event.revision
                traceObject["event_raw_partial_preview"] = String((event.rawPartialText ?? "").prefix(160))
                traceObject["event_raw_partial_len"] = event.rawPartialText?.count ?? 0
            } else {
                traceObject["event_is_speech"] = NSNull()
                traceObject["event_did_flush_segment"] = NSNull()
                traceObject["event_audio_cursor_sec"] = NSNull()
                traceObject["event_energy_dbfs"] = NSNull()
                traceObject["event_revision"] = NSNull()
                traceObject["event_raw_partial_preview"] = ""
                traceObject["event_raw_partial_len"] = 0
            }
            if let diagnostic = model.lastStreamingDiagnostic {
                traceObject["model"] = diagnostic.jsonObject
            }
            traceWriter.write(traceObject)
        }
        if currentTimeSec >= nextReportSec {
            let ingestedAudioSec = min(totalAudioSec, Double(arrivalIndex) * hopSec)
            let inferRTFx = ingestedAudioSec / max(totalInferSec, 1e-9)
            let wallRTFx = ingestedAudioSec / max(currentTimeSec, 1e-9)
            let onsetSeries = draftFirstTokenMs.isEmpty ? mergedFirstTokenMs : draftFirstTokenMs
            let onsetNow = onsetSeries.last ?? -1
            let onsetAvg = onsetSeries.isEmpty ? -1 : (onsetSeries.reduce(0, +) / Double(onsetSeries.count))
            let onsetWorst = onsetSeries.max() ?? -1
            let readySeries = draftReadyLatencyMs.isEmpty ? confirmedReadyLatencyMs : draftReadyLatencyMs
            let readyNow = readySeries.last ?? -1
            let readyAvg = readySeries.isEmpty ? -1 : (readySeries.reduce(0, +) / Double(readySeries.count))
            logProgress(
                String(
                    format: "realtime-bench t=%.2fs audio=%.2fs transcript=%.2fs queue=%.2fs inf=%.2fx wall=%.2fx onset=%.0fms avg=%.0f worst=%.0f ready=%.0fms avg=%.0f drops=%d",
                    currentTimeSec,
                    ingestedAudioSec,
                    transcriptCursorSec,
                    queueSec,
                    inferRTFx,
                    wallRTFx,
                    onsetNow,
                    onsetAvg,
                    onsetWorst,
                    readyNow,
                    readyAvg,
                    droppedChunks
                )
            )
            nextReportSec += reportEverySec
        }
    }

    let rawFinalState = engine.finishStream()
    let finalState: TranscriptState
    if rawFinalState.confirmed.isEmpty,
       rawFinalState.hypothesis.isEmpty,
       (!lastConfirmed.isEmpty || !lastHypothesis.isEmpty) {
        finalState = TranscriptState(confirmed: lastConfirmed, hypothesis: lastHypothesis)
    } else {
        finalState = rawFinalState
    }
    if let pendingStartSampleCursor = speechStartSampleCursor,
       let pendingStartIngressTimeSec = ingressTimeline.ingressTime(forSampleCursor: pendingStartSampleCursor) {
        let anyText = !finalState.confirmed.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ||
            !finalState.hypothesis.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        if anyText, !speechConfirmedRecorded {
            confirmedMs.append(max(0, (currentTimeSec - pendingStartIngressTimeSec) * 1000.0))
        }
    }
    if !finalState.confirmed.isEmpty {
        transcriptCursorSec = totalAudioSec
    }
    let wallElapsed = CFAbsoluteTimeGetCurrent() - startWall
    let inferRTFx = totalAudioSec / max(totalInferSec, 1e-9)
    let wallRTFx = totalAudioSec / max(currentTimeSec, 1e-9)
    let actualWallRTFx = totalAudioSec / max(wallElapsed, 1e-9)

    let firstTokenSeries = draftFirstTokenMs.isEmpty ? mergedFirstTokenMs : draftFirstTokenMs
    let firstP95 = percentile(firstTokenSeries, q: 0.95) ?? .infinity
    let draftFirstP95 = percentile(draftFirstTokenMs, q: 0.95)
    let mergedFirstP95 = percentile(mergedFirstTokenMs, q: 0.95)
    let confirmedP95 = percentile(confirmedMs, q: 0.95) ?? .infinity
    let draftReadyP95 = percentile(draftReadyLatencyMs, q: 0.95)
    let confirmedReadyP95 = percentile(confirmedReadyLatencyMs, q: 0.95)
    let correctionsPerMin = totalAudioSec > 0 ? (Double(corrections) / totalAudioSec) * 60.0 : 0.0

    let passQueue = maxQueueSecObserved <= args.queuePassSec
    let passFirst = firstP95 <= args.firstTokenPassMs
    let passConfirmed = confirmedP95 <= args.confirmedPassMs
    let overallPass = passQueue && passFirst && passConfirmed

    let summary: [String: Any] = [
        "mode": "realtime_bench",
        "audio_sec": totalAudioSec,
        "stream_chunk_ms": args.streamChunkMs,
        "stream_hop_ms": args.streamHopMs,
        "agreement": args.streamAgreement,
        "draft_agreement": args.streamDraftAgreement,
        "latest_first": args.latestFirst,
        "max_batch_ms": args.maxBatchMs,
        "backlog_soft_sec": args.backlogSoftSec,
        "backlog_target_sec": args.backlogTargetSec,
        "processed_chunks": processedChunks,
        "dropped_chunks": droppedChunks,
        "max_queue_sec": maxQueueSecObserved,
        "first_token_ms_p95": firstP95.isFinite ? firstP95 : NSNull(),
        "draft_first_token_ms_p95": (draftFirstP95 != nil && draftFirstP95!.isFinite) ? draftFirstP95! : NSNull(),
        "merged_first_token_ms_p95": (mergedFirstP95 != nil && mergedFirstP95!.isFinite) ? mergedFirstP95! : NSNull(),
        "confirmed_latency_ms_p95": confirmedP95.isFinite ? confirmedP95 : NSNull(),
        "draft_onset_latency_ms_p95": (draftFirstP95 != nil && draftFirstP95!.isFinite) ? draftFirstP95! : NSNull(),
        "confirmed_onset_latency_ms_p95": confirmedP95.isFinite ? confirmedP95 : NSNull(),
        "draft_ready_latency_ms_p95": (draftReadyP95 != nil && draftReadyP95!.isFinite) ? draftReadyP95! : NSNull(),
        "confirmed_ready_latency_ms_p95": (confirmedReadyP95 != nil && confirmedReadyP95!.isFinite) ? confirmedReadyP95! : NSNull(),
        "first_token_ms_avg": firstTokenSeries.isEmpty ? NSNull() : (firstTokenSeries.reduce(0, +) / Double(firstTokenSeries.count)),
        "draft_first_token_ms_avg": draftFirstTokenMs.isEmpty ? NSNull() : (draftFirstTokenMs.reduce(0, +) / Double(draftFirstTokenMs.count)),
        "merged_first_token_ms_avg": mergedFirstTokenMs.isEmpty ? NSNull() : (mergedFirstTokenMs.reduce(0, +) / Double(mergedFirstTokenMs.count)),
        "confirmed_latency_ms_avg": confirmedMs.isEmpty ? NSNull() : (confirmedMs.reduce(0, +) / Double(confirmedMs.count)),
        "draft_onset_latency_ms_avg": draftFirstTokenMs.isEmpty ? NSNull() : (draftFirstTokenMs.reduce(0, +) / Double(draftFirstTokenMs.count)),
        "confirmed_onset_latency_ms_avg": confirmedMs.isEmpty ? NSNull() : (confirmedMs.reduce(0, +) / Double(confirmedMs.count)),
        "draft_ready_latency_ms_avg": draftReadyLatencyMs.isEmpty ? NSNull() : (draftReadyLatencyMs.reduce(0, +) / Double(draftReadyLatencyMs.count)),
        "confirmed_ready_latency_ms_avg": confirmedReadyLatencyMs.isEmpty ? NSNull() : (confirmedReadyLatencyMs.reduce(0, +) / Double(confirmedReadyLatencyMs.count)),
        "draft_word_vis_ms_p95": (draftReadyP95 != nil && draftReadyP95!.isFinite) ? draftReadyP95! : NSNull(),
        "confirmed_word_vis_ms_p95": (confirmedReadyP95 != nil && confirmedReadyP95!.isFinite) ? confirmedReadyP95! : NSNull(),
        "draft_word_vis_ms_avg": draftReadyLatencyMs.isEmpty ? NSNull() : (draftReadyLatencyMs.reduce(0, +) / Double(draftReadyLatencyMs.count)),
        "confirmed_word_vis_ms_avg": confirmedReadyLatencyMs.isEmpty ? NSNull() : (confirmedReadyLatencyMs.reduce(0, +) / Double(confirmedReadyLatencyMs.count)),
        "corrections": corrections,
        "corrections_per_min": correctionsPerMin,
        "infer_rtfx": inferRTFx,
        "wall_rtfx_virtual": wallRTFx,
        "wall_rtfx_actual": actualWallRTFx,
        "wall_elapsed_sec": wallElapsed,
        "model_profile_overall": modelProfileOverall.jsonObject,
        "model_profile_tail_half_start_sec": tailHalfStartSec,
        "model_profile_tail_half": modelProfileTailHalf.jsonObject,
        "thresholds": [
            "queue_pass_sec": args.queuePassSec,
            "first_token_pass_ms": args.firstTokenPassMs,
            "confirmed_pass_ms": args.confirmedPassMs
        ],
        "pass": overallPass,
        "pass_breakdown": [
            "queue": passQueue,
            "first_token": passFirst,
            "confirmed": passConfirmed
        ],
        "final_confirmed_len": finalState.confirmed.count,
        "final_hypothesis_len": finalState.hypothesis.count,
        "final_confirmed_preview": String(finalState.confirmed.prefix(200)),
        "final_hypothesis_preview": String(finalState.hypothesis.prefix(200))
    ]

    let summaryData = try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
    if let outPath = args.metricsOutputPath, !outPath.isEmpty {
        let url = URL(fileURLWithPath: outPath)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try summaryData.write(to: url)
        let confirmedURL = url.deletingPathExtension().appendingPathExtension("confirmed.txt")
        let hypothesisURL = url.deletingPathExtension().appendingPathExtension("hypothesis.txt")
        try finalState.confirmed.write(to: confirmedURL, atomically: true, encoding: .utf8)
        try finalState.hypothesis.write(to: hypothesisURL, atomically: true, encoding: .utf8)
        logProgress("wrote realtime benchmark summary: \(url.path)")
        logProgress("wrote realtime benchmark confirmed: \(confirmedURL.path)")
        logProgress("wrote realtime benchmark hypothesis: \(hypothesisURL.path)")
    }
    print(String(decoding: summaryData, as: UTF8.self))
}

func runAudioTranscription(args: CLIArgs) throws {
    guard let audioPath = args.audioPath else {
        runInteractiveDemo()
        return
    }
    let audioURL = URL(fileURLWithPath: audioPath)
    let modelDir = args.modelDir.map { URL(fileURLWithPath: $0) } ?? defaultModelDirectory()
    let startTime = Date()

    let resolvedEncoderSuffix = args.encoderModelSuffix ?? args.modelSuffix
    let resolvedDecoderSuffix = args.decoderModelSuffix ?? args.modelSuffix
    logProgress(
        "loading models from \(modelDir.path) (encoder_suffix=\(resolvedEncoderSuffix), decoder_suffix=\(resolvedDecoderSuffix))"
    )
    let model = try ParakeetCoreMLTDTTranscriptionModel(
        modelDirectory: modelDir,
        modelSuffix: args.modelSuffix,
        encoderModelSuffix: args.encoderModelSuffix,
        decoderModelSuffix: args.decoderModelSuffix,
        config: .init(
            maxSymbolsPerStep: args.maxSymbolsPerStep,
            maxTokensPerChunk: args.maxTokensPerChunk,
            streamingHistoryFrames: args.streamingHistoryFrames,
            streamingMinTailDecodeFrames: args.streamingMinTailFrames
        )
    )
    logProgress("models loaded")

    logProgress("reading audio \(audioURL.path)")
    let (rawSamples, rawSR) = try readAudioFileMonoFloat(url: audioURL)
    let sampleRate = 16_000
    logProgress("input sample rate: \(rawSR) Hz")
    let samples = try avResampleMonoFloat(rawSamples, from: rawSR, to: sampleRate)
    let audioSeconds = Double(samples.count) / Double(sampleRate)
    logProgress(String(format: "audio ready: %.2fs at %d Hz", audioSeconds, sampleRate))

    let modelVar = model
    if args.realtimeBench {
        logProgress(
            "realtime-bench chunk=\(args.streamChunkMs)ms hop=\(args.streamHopMs)ms agreement=\(args.streamAgreement)/\(args.streamDraftAgreement) batch=\(args.maxBatchMs)ms latest_first=\(args.latestFirst)"
        )
        try runRealtimeBenchmark(
            model: modelVar,
            samples: samples,
            sampleRate: sampleRate,
            args: args
        )
        return
    }

    let latest: String
    if args.longformSegmentSec <= 0 {
        logProgress("longform decode left=\(args.leftContextFrames) right=\(args.allowRightContext ? args.rightContextFrames : 0)")
        latest = try modelVar.transcribeLongform(
            samples,
            sampleRate: sampleRate,
            leftContextFrames: args.leftContextFrames,
            rightContextFrames: args.rightContextFrames,
            allowRightContext: args.allowRightContext
        )
    } else {
        let segmentSamples = max(1, Int(round(args.longformSegmentSec * Double(sampleRate))))
        var overlapSamples = max(0, Int(round(max(0, args.longformOverlapSec) * Double(sampleRate))))
        if overlapSamples >= segmentSamples {
            overlapSamples = max(0, segmentSamples / 4)
        }
        let stepSamples = max(1, segmentSamples - overlapSamples)

        var chunkTexts: [String] = []
        var segmentStart = 0
        var segmentIndex = 0
        let totalSegments = max(1, Int(ceil(Double(samples.count) / Double(stepSamples))))
        while segmentStart < samples.count {
            let end = min(samples.count, segmentStart + segmentSamples)
            segmentIndex += 1
            logProgress("segment \(segmentIndex)/\(totalSegments) [\(segmentStart):\(end)]")
            let segmentAudio = Array(samples[segmentStart..<end])
            let segmentText = try modelVar.transcribeLongform(
                segmentAudio,
                sampleRate: sampleRate,
                leftContextFrames: args.leftContextFrames,
                rightContextFrames: args.rightContextFrames,
                allowRightContext: args.allowRightContext
            )
            if !segmentText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                chunkTexts.append(segmentText)
            }
            if end >= samples.count { break }
            segmentStart += stepSamples
        }
        latest = mergeTextByOverlap(chunkTexts)
    }

    let elapsed = Date().timeIntervalSince(startTime)
    logProgress(String(format: "done in %.2fs", elapsed))
    print(latest)
}
#endif

let parsed = parseArgs()
#if canImport(CoreML) && canImport(AVFoundation)
do {
    try runAudioTranscription(args: parsed)
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}
#else
runInteractiveDemo()
#endif
