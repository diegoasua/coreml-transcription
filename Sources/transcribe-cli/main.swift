import Foundation
import RealtimeTranscriptionCore

struct CLIArgs {
    var audioPath: String?
    var modelDir: String?
    var modelSuffix: String = ProcessInfo.processInfo.environment["PARAKEET_COREML_MODEL_SUFFIX"] ?? "odmbp-approx"
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
        modelSuffix: args.modelSuffix
    )
    logProgress("models loaded")

    logProgress("reading audio \(audioURL.path)")
    let (rawSamples, rawSR) = try readAudioFileMonoFloat(url: audioURL)
    let sampleRate = 16_000
    logProgress("input sample rate: \(rawSR) Hz")
    let samples = try avResampleMonoFloat(rawSamples, from: rawSR, to: sampleRate)
    let audioSeconds = Double(samples.count) / Double(sampleRate)
    logProgress(String(format: "audio ready: %.2fs at %d Hz", audioSeconds, sampleRate))

    let chunkSamples = sampleRate * 3
    let modelVar = model
    var start = 0
    var latest = ""
    let totalChunks = max(1, Int(ceil(Double(samples.count) / Double(chunkSamples))))
    var chunkIndex = 0
    while start < samples.count {
        let end = min(samples.count, start + chunkSamples)
        chunkIndex += 1
        logProgress("transcribing chunk \(chunkIndex)/\(totalChunks)")
        latest = try modelVar.transcribeChunk(Array(samples[start..<end]), sampleRate: sampleRate)
        start = end
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
