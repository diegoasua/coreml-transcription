import SwiftUI
import AVFoundation
import RealtimeTranscriptionCore
#if canImport(AppKit)
import AppKit
#endif

private func parakeetModelDirectoryCandidates() -> [URL] {
    let env = ProcessInfo.processInfo.environment

    let fm = FileManager.default
    var candidates: [URL] = []
    if let configured = env["PARAKEET_COREML_MODEL_DIR"], !configured.isEmpty {
        candidates.append(URL(fileURLWithPath: configured))
    }
    candidates.append(
        URL(fileURLWithPath: fm.currentDirectoryPath)
            .appendingPathComponent("artifacts/parakeet-tdt-0.6b-v2")
    )

    if let srcRoot = env["SRCROOT"], !srcRoot.isEmpty {
        candidates.append(URL(fileURLWithPath: srcRoot).appendingPathComponent("artifacts/parakeet-tdt-0.6b-v2"))
    }
    if let projectDir = env["PROJECT_DIR"], !projectDir.isEmpty {
        candidates.append(URL(fileURLWithPath: projectDir).appendingPathComponent("artifacts/parakeet-tdt-0.6b-v2"))
    }

    // Works when built via SwiftPM/Xcode from this repo checkout.
    let sourceURL = URL(fileURLWithPath: #filePath)
    let repoRoot = sourceURL
        .deletingLastPathComponent() // .../Sources/transcribe-macos
        .deletingLastPathComponent() // .../Sources
        .deletingLastPathComponent() // repo root
    candidates.append(repoRoot.appendingPathComponent("artifacts/parakeet-tdt-0.6b-v2"))
    return candidates
}

private func hasParakeetArtifacts(at directory: URL, suffix: String) -> Bool {
    let fm = FileManager.default
    let encoder = directory.appendingPathComponent("encoder-model-\(suffix).mlpackage").path
    let decoder = directory.appendingPathComponent("decoder_joint-model-\(suffix).mlpackage").path
    let vocab = directory.appendingPathComponent("vocab.txt").path
    return fm.fileExists(atPath: encoder) && fm.fileExists(atPath: decoder) && fm.fileExists(atPath: vocab)
}

private func resolveParakeetModelDirectory(preferred: URL?, suffix: String) -> URL? {
    let fm = FileManager.default
    var ordered: [URL] = []
    if let preferred {
        ordered.append(preferred)
    }
    ordered.append(contentsOf: parakeetModelDirectoryCandidates())
    var seen: Set<String> = []
    for candidate in ordered {
        let normalized = candidate.standardizedFileURL.path
        if seen.contains(normalized) { continue }
        seen.insert(normalized)
        guard fm.fileExists(atPath: normalized) else { continue }
        if hasParakeetArtifacts(at: URL(fileURLWithPath: normalized), suffix: suffix) {
            return URL(fileURLWithPath: normalized)
        }
    }
    return nil
}

private func defaultParakeetModelDirectoryPath() -> String {
    let defaultSuffix = ProcessInfo.processInfo.environment["PARAKEET_COREML_MODEL_SUFFIX"] ?? "odmbp-approx"
    if let resolved = resolveParakeetModelDirectory(preferred: nil, suffix: defaultSuffix) {
        return resolved.path
    }
    return parakeetModelDirectoryCandidates().last?.path ?? ""
}

@main
struct TranscribeMacOSApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    @StateObject private var vm = MicTranscriptionViewModel()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Model Directory")
                TextField("Path to artifacts/parakeet-tdt-0.6b-v2", text: $vm.modelDirectory)
                    .textFieldStyle(.roundedBorder)
                Button("Browse…") {
                    vm.chooseModelDirectory()
                }
            }

            HStack {
                Text("Model Suffix")
                TextField("odmbp-approx", text: $vm.modelSuffix)
                    .frame(width: 220)
                    .textFieldStyle(.roundedBorder)
                Spacer()
                Button(vm.isRunning ? "Stop" : (vm.isStarting ? "Starting…" : "Start")) {
                    vm.toggle()
                }
                .keyboardShortcut(.space, modifiers: [])
                .disabled(vm.isStarting)
            }

            Text("Status: \(vm.status)")
                .font(.footnote)
                .foregroundStyle(.secondary)

            GroupBox("Confirmed") {
                ScrollView {
                    Text(vm.confirmedText.isEmpty ? "..." : vm.confirmedText)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(minHeight: 140)
            }

            GroupBox("Hypothesis") {
                ScrollView {
                    Text(vm.hypothesisText.isEmpty ? "..." : vm.hypothesisText)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(minHeight: 80)
            }
        }
        .padding(16)
        .frame(minWidth: 760, minHeight: 420)
        .onAppear {
            vm.activateAppWindow()
        }
    }
}

final class MicTranscriptionViewModel: ObservableObject {
    @Published var confirmedText: String = ""
    @Published var hypothesisText: String = ""
    @Published var status: String = "Idle"
    @Published var isRunning: Bool = false
    @Published var isStarting: Bool = false
    @Published var modelDirectory: String = defaultParakeetModelDirectoryPath()
    @Published var modelSuffix: String = ProcessInfo.processInfo.environment["PARAKEET_COREML_MODEL_SUFFIX"] ?? "odmbp-approx"

    private typealias InferenceEngine = StreamingInferenceEngine<ParakeetCoreMLRNNTTranscriptionModel, EnergyVAD>
    private var inferenceEngine: InferenceEngine?

    private let queue = DispatchQueue(label: "transcribe.macos.inference")
    private let audioEngine = AVAudioEngine()
    private var audioConverter: AVAudioConverter?
    private let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!
    private let chunkMs = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_CHUNK_MS"] ?? "") ?? 960
    private let hopMs = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_HOP_MS"] ?? "") ?? 960
    private let agreementCount = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_AGREEMENT"] ?? "") ?? 2
    private let maxSymbolsPerStep = Int(ProcessInfo.processInfo.environment["PARAKEET_RNNT_MAX_SYMBOLS_PER_STEP"] ?? "") ?? 10
    private let maxTokensPerChunk = Int(ProcessInfo.processInfo.environment["PARAKEET_RNNT_MAX_TOKENS_PER_CHUNK"] ?? "") ?? 0
    private let vadStartThresholdDBFS = Float(ProcessInfo.processInfo.environment["PARAKEET_VAD_START_DBFS"] ?? "") ?? -42
    private let vadEndThresholdDBFS = Float(ProcessInfo.processInfo.environment["PARAKEET_VAD_END_DBFS"] ?? "") ?? -52
    private let vadMinSpeechMs = Int(ProcessInfo.processInfo.environment["PARAKEET_VAD_MIN_SPEECH_MS"] ?? "") ?? 120
    private let vadMinSilenceMs = Int(ProcessInfo.processInfo.environment["PARAKEET_VAD_MIN_SILENCE_MS"] ?? "") ?? 800
    private let pendingLock = NSLock()
    private var pendingSamples: [Float] = []
    private var processingScheduled = false
    private let maxBufferedSamples = (Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_MAX_BUFFER_SEC"] ?? "") ?? 0) * 16_000
    private let maxProcessingBatchSamples = (Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_MAX_BATCH_SEC"] ?? "") ?? 2) * 16_000
    private var totalProcessedAudioSec: Double = 0
    private var totalInferenceSec: Double = 0
    private let maxStagnantSpeechChunks = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_MAX_STAGNANT_CHUNKS"] ?? "") ?? 8
    private var stagnantSpeechChunkCount = 0
    private var lastSpeechTranscriptFingerprint = ""

    func toggle() {
        if isRunning {
            stop()
        } else {
            start()
        }
    }

    func activateAppWindow() {
#if canImport(AppKit)
        NSApp.activate(ignoringOtherApps: true)
#endif
    }

    func chooseModelDirectory() {
#if canImport(AppKit)
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Use Folder"
        panel.message = "Select the Parakeet artifact directory that contains encoder/decoder .mlpackage files."
        if panel.runModal() == .OK, let url = panel.url {
            modelDirectory = url.path
        }
#endif
    }

    private func start() {
        if isRunning || isStarting { return }

        let preferredURL = URL(fileURLWithPath: modelDirectory)
        guard let dirURL = resolveParakeetModelDirectory(preferred: preferredURL, suffix: modelSuffix) else {
            status = "Model artifacts not found for suffix '\(modelSuffix)'. Expected encoder/decoder/vocab under: \(preferredURL.path)"
            return
        }
        if modelDirectory != dirURL.path {
            modelDirectory = dirURL.path
        }

        isStarting = true
        status = "Requesting microphone access..."
        requestMicrophoneAccess { [weak self] granted in
            guard let self else { return }
            if !granted {
                Task { @MainActor in
                    self.status = "Microphone access denied. Enable it in System Settings > Privacy & Security > Microphone."
                    self.isStarting = false
                }
                return
            }

            self.queue.async { [weak self] in
                guard let self else { return }
                do {
                    let model = try ParakeetCoreMLRNNTTranscriptionModel(
                        modelDirectory: dirURL,
                        modelSuffix: self.modelSuffix,
                        config: .init(maxSymbolsPerStep: self.maxSymbolsPerStep, maxTokensPerChunk: self.maxTokensPerChunk)
                    )
                    let engine = InferenceEngine(
                        model: model,
                        vad: EnergyVAD(
                            config: .init(
                                startThresholdDBFS: self.vadStartThresholdDBFS,
                                endThresholdDBFS: self.vadEndThresholdDBFS,
                                minSpeechMs: self.vadMinSpeechMs,
                                minSilenceMs: self.vadMinSilenceMs
                            )
                        ),
                        policy: .init(sampleRate: 16_000, chunkMs: self.chunkMs, hopMs: self.hopMs),
                        requiredAgreementCount: self.agreementCount,
                        ringBufferCapacity: 16_000 * 12
                    )

                    Task { @MainActor in
                        self.configureAndStartAudio(engine: engine)
                    }
                } catch {
                    Task { @MainActor in
                        self.status = "Failed to load model: \(error)"
                        self.isStarting = false
                    }
                }
            }
        }
    }

    private func stop() {
        if !isRunning && !isStarting { return }
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        isRunning = false
        isStarting = false
        pendingLock.lock()
        pendingSamples.removeAll(keepingCapacity: false)
        pendingLock.unlock()
        totalProcessedAudioSec = 0
        totalInferenceSec = 0
        stagnantSpeechChunkCount = 0
        lastSpeechTranscriptFingerprint = ""

        queue.sync {
            if var engine = inferenceEngine {
                let finalState = engine.finishStream()
                inferenceEngine = engine
                Task { @MainActor in
                    self.confirmedText = finalState.confirmed
                    self.hypothesisText = finalState.hypothesis
                }
            }
        }
        status = "Stopped"
    }

    @MainActor
    private func configureAndStartAudio(engine: InferenceEngine) {
        inferenceEngine = engine
        totalProcessedAudioSec = 0
        totalInferenceSec = 0
        pendingLock.lock()
        pendingSamples.removeAll(keepingCapacity: false)
        processingScheduled = false
        pendingLock.unlock()

        do {
            let inputNode = audioEngine.inputNode
            let inputFormat = inputNode.inputFormat(forBus: 0)
            guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
                throw NSError(
                    domain: "transcribe-macos",
                    code: 10,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to create AVAudioConverter"]
                )
            }
            audioConverter = converter

            inputNode.removeTap(onBus: 0)
            inputNode.installTap(onBus: 0, bufferSize: 2048, format: inputFormat) { [weak self] buffer, _ in
                self?.handleAudioBuffer(buffer)
            }

            try audioEngine.start()
            isRunning = true
            isStarting = false
#if DEBUG
            status = "Listening... (DEBUG build; slower) chunk \(chunkMs)ms / hop \(hopMs)ms"
#else
            status = "Listening... (chunk \(chunkMs)ms / hop \(hopMs)ms)"
#endif
            activateAppWindow()
        } catch {
            inferenceEngine = nil
            isRunning = false
            isStarting = false
            audioEngine.stop()
            status = "Audio start failed: \(error)"
        }
    }

    private func handleAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let converter = audioConverter else { return }
        let targetFrames = AVAudioFrameCount(Double(buffer.frameLength) * targetFormat.sampleRate / buffer.format.sampleRate + 64)
        guard let converted = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: targetFrames) else { return }

        var error: NSError?
        let sourceConsumed = UnsafeMutablePointer<Bool>.allocate(capacity: 1)
        sourceConsumed.initialize(to: false)
        defer {
            sourceConsumed.deinitialize(count: 1)
            sourceConsumed.deallocate()
        }
        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            if sourceConsumed.pointee {
                outStatus.pointee = .noDataNow
                return nil
            }
            sourceConsumed.pointee = true
            outStatus.pointee = .haveData
            return buffer
        }

        let status = converter.convert(to: converted, error: &error, withInputFrom: inputBlock)
        if status == .error || error != nil { return }
        guard status == .haveData || status == .inputRanDry else { return }
        guard let channel = converted.floatChannelData?.pointee else { return }
        let frameCount = Int(converted.frameLength)
        if frameCount == 0 { return }

        let samples = Array(UnsafeBufferPointer(start: channel, count: frameCount))
        enqueueSamplesForProcessing(samples)
    }

    private func enqueueSamplesForProcessing(_ samples: [Float]) {
        var shouldSchedule = false
        var droppedCount = 0

        pendingLock.lock()
        pendingSamples.append(contentsOf: samples)
        if maxBufferedSamples > 0, pendingSamples.count > maxBufferedSamples {
            droppedCount = pendingSamples.count - maxBufferedSamples
            pendingSamples.removeFirst(droppedCount)
        }
        if !processingScheduled {
            processingScheduled = true
            shouldSchedule = true
        }
        let queuedSeconds = Double(pendingSamples.count) / 16_000.0
        pendingLock.unlock()

        if droppedCount > 0 {
            Task { @MainActor in
                self.status = String(format: "Catching up (dropped %.2fs)", Double(droppedCount) / 16_000.0)
            }
        } else if queuedSeconds > 1.0 {
            Task { @MainActor in
                self.status = String(format: "Processing backlog %.1fs", queuedSeconds)
            }
        }

        if shouldSchedule {
            queue.async { [weak self] in
                self?.processPendingAudioLoop()
            }
        }
    }

    private func processPendingAudioLoop() {
        while true {
            guard isRunning else {
                pendingLock.lock()
                processingScheduled = false
                pendingSamples.removeAll(keepingCapacity: false)
                pendingLock.unlock()
                return
            }

            let batch: [Float]
            let queuedAfterPopSec: Double
            pendingLock.lock()
            if pendingSamples.isEmpty {
                processingScheduled = false
                pendingLock.unlock()
                return
            }
            if maxProcessingBatchSamples > 0, pendingSamples.count > maxProcessingBatchSamples {
                batch = Array(pendingSamples.prefix(maxProcessingBatchSamples))
                pendingSamples.removeFirst(maxProcessingBatchSamples)
            } else {
                batch = pendingSamples
                pendingSamples.removeAll(keepingCapacity: true)
            }
            queuedAfterPopSec = Double(pendingSamples.count) / 16_000.0
            pendingLock.unlock()

            guard var engine = inferenceEngine else { return }
            let started = CFAbsoluteTimeGetCurrent()
            do {
                let events = try engine.process(samples: batch)
                let elapsed = CFAbsoluteTimeGetCurrent() - started
                let batchAudioSec = Double(batch.count) / 16_000.0
                totalProcessedAudioSec += batchAudioSec
                totalInferenceSec += elapsed
                let inferRTFx = totalProcessedAudioSec / max(totalInferenceSec, 1e-6)
                inferenceEngine = engine

                if let latest = events.last {
                    if latest.didFlushSegment {
                        stagnantSpeechChunkCount = 0
                        lastSpeechTranscriptFingerprint = ""
                    } else if latest.isSpeech {
                        let fingerprint = "\(latest.transcript.confirmed)|\(latest.transcript.hypothesis)"
                        if fingerprint == lastSpeechTranscriptFingerprint {
                            stagnantSpeechChunkCount += 1
                        } else {
                            stagnantSpeechChunkCount = 0
                            lastSpeechTranscriptFingerprint = fingerprint
                        }

                        if stagnantSpeechChunkCount >= maxStagnantSpeechChunks {
                            let flushed = engine.finishStream()
                            inferenceEngine = engine
                            stagnantSpeechChunkCount = 0
                            lastSpeechTranscriptFingerprint = ""
                            Task { @MainActor in
                                self.confirmedText = flushed.confirmed
                                self.hypothesisText = flushed.hypothesis
                                self.status = "Stream resynced after stagnant decode."
                            }
                            continue
                        }
                    } else {
                        stagnantSpeechChunkCount = 0
                        lastSpeechTranscriptFingerprint = ""
                    }

                    Task { @MainActor in
                        self.confirmedText = latest.transcript.confirmed
                        self.hypothesisText = latest.transcript.hypothesis
                        self.status = String(
                            format: "%@ | %.1fx realtime | queue %.1fs",
                            latest.isSpeech ? "Listening..." : "Silence",
                            inferRTFx,
                            queuedAfterPopSec
                        )
                    }
                } else {
                    Task { @MainActor in
                        self.status = String(format: "Listening... | %.1fx realtime | queue %.1fs", inferRTFx, queuedAfterPopSec)
                    }
                }
            } catch {
                Task { @MainActor in
                    self.status = "Inference error: \(error)"
                }
                return
            }
        }
    }

    private func requestMicrophoneAccess(completion: @escaping (Bool) -> Void) {
        let status = AVCaptureDevice.authorizationStatus(for: .audio)
        switch status {
        case .authorized:
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                completion(granted)
            }
        case .denied, .restricted:
            completion(false)
        @unknown default:
            completion(false)
        }
    }

}
