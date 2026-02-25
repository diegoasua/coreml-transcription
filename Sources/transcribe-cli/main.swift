import Foundation
import RealtimeTranscriptionCore

struct CLIArgs {
    var audioPath: String?
    var modelDir: String?
    var modelSuffix: String = ProcessInfo.processInfo.environment["PARAKEET_COREML_MODEL_SUFFIX"] ?? "odmbp-approx"
    var chunkSec: Double = Double(ProcessInfo.processInfo.environment["PARAKEET_CLI_CHUNK_SEC"] ?? "") ?? 3.0
    var longformSegmentSec: Double = Double(ProcessInfo.processInfo.environment["PARAKEET_LONGFORM_SEGMENT_SEC"] ?? "") ?? 0.0
    var longformOverlapSec: Double = Double(ProcessInfo.processInfo.environment["PARAKEET_LONGFORM_OVERLAP_SEC"] ?? "") ?? 0.0
    var leftContextFrames: Int = Int(ProcessInfo.processInfo.environment["PARAKEET_ENCODER_LEFT_CONTEXT_FRAMES"] ?? "") ?? 300
    var rightContextFrames: Int = Int(ProcessInfo.processInfo.environment["PARAKEET_ENCODER_RIGHT_CONTEXT_FRAMES"] ?? "") ?? 120
    var allowRightContext: Bool = (ProcessInfo.processInfo.environment["PARAKEET_CLI_ALLOW_RIGHT_CONTEXT"] ?? "1") != "0"
    var maxSymbolsPerStep: Int = Int(ProcessInfo.processInfo.environment["PARAKEET_RNNT_MAX_SYMBOLS_PER_STEP"] ?? "") ?? 10
    var maxTokensPerChunk: Int = Int(ProcessInfo.processInfo.environment["PARAKEET_RNNT_MAX_TOKENS_PER_CHUNK"] ?? "") ?? 0
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
import AVFoundation

func logProgress(_ message: String) {
    fputs("[transcribe-cli] \(message)\n", stderr)
    fflush(stderr)
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

    var provided = false
    var convError: NSError?
    let status = converter.convert(to: dstBuffer, error: &convError) { _, outStatus in
        if provided {
            outStatus.pointee = .noDataNow
            return nil
        }
        provided = true
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
    model: ParakeetCoreMLRNNTTranscriptionModel,
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

func runAudioTranscription(args: CLIArgs) throws {
    guard let audioPath = args.audioPath else {
        runInteractiveDemo()
        return
    }
    let audioURL = URL(fileURLWithPath: audioPath)
    let modelDir = args.modelDir.map { URL(fileURLWithPath: $0) } ?? defaultModelDirectory()
    let startTime = Date()

    logProgress("loading models from \(modelDir.path) (suffix=\(args.modelSuffix))")
    let model = try ParakeetCoreMLRNNTTranscriptionModel(
        modelDirectory: modelDir,
        modelSuffix: args.modelSuffix,
        config: .init(
            maxSymbolsPerStep: args.maxSymbolsPerStep,
            maxTokensPerChunk: args.maxTokensPerChunk
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
