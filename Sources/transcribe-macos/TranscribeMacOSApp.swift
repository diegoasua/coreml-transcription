import SwiftUI
@preconcurrency import AVFoundation
import RealtimeTranscriptionCore
#if canImport(AppKit)
import AppKit
#endif
#if canImport(CoreAudio)
import CoreAudio
#endif

private final class ConversionInputState: @unchecked Sendable {
    var consumed = false
}

#if canImport(CoreAudio)
private func defaultInputDeviceName() -> String? {
    var deviceID = AudioDeviceID(0)
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultInputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    let status = AudioObjectGetPropertyData(
        AudioObjectID(kAudioObjectSystemObject),
        &address,
        0,
        nil,
        &size,
        &deviceID
    )
    guard status == noErr, deviceID != 0 else { return nil }

    var nameAddress = AudioObjectPropertyAddress(
        mSelector: kAudioObjectPropertyName,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var cfName: Unmanaged<CFString>?
    var nameSize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    let nameStatus = AudioObjectGetPropertyData(
        deviceID,
        &nameAddress,
        0,
        nil,
        &nameSize,
        &cfName
    )
    guard nameStatus == noErr, let cfName else { return nil }
    return cfName.takeUnretainedValue() as String
}
#else
private func defaultInputDeviceName() -> String? { nil }
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
            Text("Metrics: \(vm.metrics)")
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

final class MicTranscriptionViewModel: ObservableObject, @unchecked Sendable {
    @Published var confirmedText: String = ""
    @Published var hypothesisText: String = ""
    @Published var status: String = "Idle"
    @Published var metrics: String = "RTFx(inf/wall)=0.0/0.0 | first-token=n/a"
    @Published var isRunning: Bool = false
    @Published var isStarting: Bool = false
    @Published var modelDirectory: String = defaultParakeetModelDirectoryPath()
    @Published var modelSuffix: String = ProcessInfo.processInfo.environment["PARAKEET_COREML_MODEL_SUFFIX"] ?? "odmbp-approx"

    private typealias InferenceEngine = StreamingInferenceEngine<ParakeetCoreMLTDTTranscriptionModel, EnergyVAD>
    private var inferenceEngine: InferenceEngine?

    private let queue = DispatchQueue(label: "transcribe.macos.inference")
    private let startupQueue = DispatchQueue(label: "transcribe.macos.startup", qos: .userInitiated)
    private let audioEngine = AVAudioEngine()
    private var audioConverter: AVAudioConverter?
    private let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!
    private let chunkMs = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_CHUNK_MS"] ?? "") ?? 160
    private let hopMs = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_HOP_MS"] ?? "") ?? 80
    private let agreementCount = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_AGREEMENT"] ?? "") ?? 2
    private let decodeOnlyWhenSpeech = (ProcessInfo.processInfo.environment["PARAKEET_DECODE_ONLY_WHEN_SPEECH"] ?? "1") != "0"
    private let flushOnSpeechEnd: Bool = {
        let env = ProcessInfo.processInfo.environment
        if let explicit = env["PARAKEET_STREAM_FLUSH_ON_SPEECH_END"], !explicit.isEmpty {
            return explicit != "0"
        }
        // Default low-latency mode: keep decoder/model state across silence
        // boundaries unless explicitly requested otherwise.
        return false
    }()
    private let maxSpeechChunkRunBeforeReset = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_MAX_SPEECH_CHUNKS"] ?? "") ?? 240
    private let maxStagnantSpeechChunks = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_MAX_STAGNANT_CHUNKS"] ?? "") ?? 24
    private let maxSymbolsPerStep = Int(ProcessInfo.processInfo.environment["PARAKEET_TDT_MAX_SYMBOLS_PER_STEP"] ?? "") ?? 4
    private let maxTokensPerChunk = Int(ProcessInfo.processInfo.environment["PARAKEET_TDT_MAX_TOKENS_PER_CHUNK"] ?? "") ?? 64
    private let streamingHistoryFrames = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_HISTORY_FRAMES"] ?? "") ?? 300
    private let streamingMinTailDecodeFrames = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_MIN_TAIL_FRAMES"] ?? "") ?? 8
    private let vadStartThresholdDBFS = Float(ProcessInfo.processInfo.environment["PARAKEET_VAD_START_DBFS"] ?? "") ?? -50
    private let vadEndThresholdDBFS = Float(ProcessInfo.processInfo.environment["PARAKEET_VAD_END_DBFS"] ?? "") ?? -58
    private let vadMinSpeechMs = Int(ProcessInfo.processInfo.environment["PARAKEET_VAD_MIN_SPEECH_MS"] ?? "") ?? 60
    private let vadMinSilenceMs = Int(ProcessInfo.processInfo.environment["PARAKEET_VAD_MIN_SILENCE_MS"] ?? "") ?? 400
    private let enableVoiceProcessing = (ProcessInfo.processInfo.environment["PARAKEET_AUDIO_VOICE_PROCESSING"] ?? "1") != "0"
    private let enableVoiceProcessingAGC = (ProcessInfo.processInfo.environment["PARAKEET_AUDIO_VOICE_PROCESSING_AGC"] ?? "1") != "0"
    private let pendingLock = NSLock()
    private var pendingSamples: [Float] = []
    private var processingScheduled = false
    private let maxBufferedSamples = (Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_MAX_BUFFER_SEC"] ?? "") ?? 4) * 16_000
    private let latestFirstScheduling = (ProcessInfo.processInfo.environment["PARAKEET_STREAM_LATEST_FIRST"] ?? "1") != "0"
    private let backlogSoftLimitSamples: Int = {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["PARAKEET_STREAM_BACKLOG_SOFT_SEC"], let sec = Double(raw), sec > 0 {
            return Int(sec * 16_000.0)
        }
        return Int(1.5 * 16_000.0)
    }()
    private let backlogTargetSamples: Int = {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["PARAKEET_STREAM_BACKLOG_TARGET_SEC"], let sec = Double(raw), sec > 0 {
            return Int(sec * 16_000.0)
        }
        return Int(0.4 * 16_000.0)
    }()
    private var resyncAfterDrop = false
    private let maxProcessingBatchSamples: Int = {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["PARAKEET_STREAM_MAX_BATCH_SEC"], let sec = Double(raw), sec > 0 {
            return max(1, Int(sec * 16_000.0))
        }
        // Keep UI responsive and avoid bursty "all-at-once" transcript updates.
        return Int(0.25 * 16_000.0)
    }()
    private var totalProcessedAudioSec: Double = 0
    private var totalInferenceSec: Double = 0
    private var runStartedAtSec: Double = 0
    private var currentSpeechStartedAtSec: Double?
    private var currentSpeechBaselineText: String = ""
    private var currentSpeechFirstTokenLatencyMs: Double?
    private var firstTokenLatencyMsSamples: [Double] = []
    private var modelLoadStartedAt: Date?
    private var modelLoadTimer: Timer?
    private let metricsLogEnabled = (ProcessInfo.processInfo.environment["PARAKEET_METRICS_LOG"] ?? "0") != "0"
    private var inputDeviceName: String = "unknown"

    func toggle() {
        if isRunning {
            stop()
        } else {
            start()
        }
    }

    func activateAppWindow() {
#if canImport(AppKit)
        DispatchQueue.main.async {
            NSApp.activate(ignoringOtherApps: true)
        }
#endif
    }

    @MainActor
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
                DispatchQueue.main.async {
                    self.status = "Microphone access denied. Enable it in System Settings > Privacy & Security > Microphone."
                    self.isStarting = false
                }
                return
            }

            DispatchQueue.main.async {
                self.status = "Microphone granted. Loading model..."
                self.startModelLoadTicker()
            }

            self.startupQueue.async { [weak self] in
                guard let self else { return }
                do {
                    let model = try ParakeetCoreMLTDTTranscriptionModel(
                        modelDirectory: dirURL,
                        modelSuffix: self.modelSuffix,
                        config: .init(
                            maxSymbolsPerStep: self.maxSymbolsPerStep,
                            maxTokensPerChunk: self.maxTokensPerChunk,
                            streamingHistoryFrames: self.streamingHistoryFrames,
                            streamingMinTailDecodeFrames: self.streamingMinTailDecodeFrames
                        ),
                        progress: { [weak self] message in
                            guard let self else { return }
                            DispatchQueue.main.async {
                                self.status = self.decorateModelLoadStatus(message)
                            }
                        }
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
                        decodeOnlyWhenSpeech: self.decodeOnlyWhenSpeech,
                        flushOnSpeechEnd: self.flushOnSpeechEnd,
                        maxSpeechChunkRunBeforeReset: self.maxSpeechChunkRunBeforeReset,
                        maxStagnantSpeechChunks: self.maxStagnantSpeechChunks,
                        ringBufferCapacity: 16_000 * 12
                    )

                    DispatchQueue.main.async {
                        self.stopModelLoadTicker()
                        self.status = "Model loaded. Starting audio engine..."
                        self.configureAndStartAudio(engine: engine)
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.stopModelLoadTicker()
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
        stopModelLoadTicker()
        runStartedAtSec = 0
        currentSpeechStartedAtSec = nil
        currentSpeechBaselineText = ""
        currentSpeechFirstTokenLatencyMs = nil
        firstTokenLatencyMsSamples.removeAll(keepingCapacity: false)

        queue.sync {
            if var engine = inferenceEngine {
                let finalState = engine.finishStream()
                inferenceEngine = engine
                let finalConfirmed = finalState.confirmed
                let finalHypothesis = finalState.hypothesis
                DispatchQueue.main.async {
                    self.confirmedText = finalConfirmed
                    self.hypothesisText = finalHypothesis
                    self.metrics = "RTFx(inf/wall)=0.0/0.0 | first-token=n/a"
                }
            }
        }
        status = "Stopped"
    }

    private func configureAndStartAudio(engine: InferenceEngine) {
        inferenceEngine = engine
        totalProcessedAudioSec = 0
        totalInferenceSec = 0
        stopModelLoadTicker()
        runStartedAtSec = 0
        currentSpeechStartedAtSec = nil
        currentSpeechBaselineText = ""
        currentSpeechFirstTokenLatencyMs = nil
        firstTokenLatencyMsSamples.removeAll(keepingCapacity: false)
        pendingLock.lock()
        pendingSamples.removeAll(keepingCapacity: false)
        processingScheduled = false
        pendingLock.unlock()

        do {
            let inputNode = audioEngine.inputNode
            let inputFormat = inputNode.inputFormat(forBus: 0)
            inputDeviceName = defaultInputDeviceName() ?? "unknown"
            fputs("[transcribe-macos] input device: \(inputDeviceName)\n", stderr)
            fputs("[transcribe-macos] input format: \(Int(inputFormat.sampleRate)) Hz, channels \(inputFormat.channelCount)\n", stderr)

            if enableVoiceProcessing {
                do {
                    try inputNode.setVoiceProcessingEnabled(true)
                    inputNode.isVoiceProcessingBypassed = false
                    inputNode.isVoiceProcessingAGCEnabled = enableVoiceProcessingAGC
                } catch {
                    fputs("[transcribe-macos] voice processing unavailable: \(error)\n", stderr)
                }
            }

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
            status = "Listening... (DEBUG build; slower) chunk \(chunkMs)ms / hop \(hopMs)ms | mic \(inputDeviceName)"
#else
            status = "Listening... (chunk \(chunkMs)ms / hop \(hopMs)ms) | mic \(inputDeviceName)"
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
        let inputState = ConversionInputState()
        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            if inputState.consumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            inputState.consumed = true
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
            resyncAfterDrop = true
        }
        if backlogSoftLimitSamples > 0,
           pendingSamples.count > backlogSoftLimitSamples {
            let target = max(backlogTargetSamples, maxProcessingBatchSamples)
            if pendingSamples.count > target {
                let extraDrop = pendingSamples.count - target
                droppedCount += extraDrop
                pendingSamples.removeFirst(extraDrop)
                resyncAfterDrop = true
            }
        }
        if !processingScheduled {
            processingScheduled = true
            shouldSchedule = true
        }
        let queuedSeconds = Double(pendingSamples.count) / 16_000.0
        pendingLock.unlock()

        if droppedCount > 0 {
            let statusLine = String(format: "Catching up (dropped %.2fs)", Double(droppedCount) / 16_000.0)
            DispatchQueue.main.async {
                self.status = statusLine
            }
        } else if queuedSeconds > 1.0 {
            let statusLine = String(format: "Processing backlog %.1fs", queuedSeconds)
            DispatchQueue.main.async {
                self.status = statusLine
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
            let shouldResyncBeforeBatch: Bool
            var droppedForRealtime = 0
            pendingLock.lock()
            if pendingSamples.isEmpty {
                processingScheduled = false
                pendingLock.unlock()
                return
            }
            if maxProcessingBatchSamples > 0, pendingSamples.count > maxProcessingBatchSamples {
                if latestFirstScheduling {
                    // Realtime mode: decode the most recent audio and drop stale backlog.
                    batch = Array(pendingSamples.suffix(maxProcessingBatchSamples))
                    droppedForRealtime = pendingSamples.count - batch.count
                    pendingSamples.removeAll(keepingCapacity: true)
                    if droppedForRealtime > 0 {
                        resyncAfterDrop = true
                    }
                } else {
                    batch = Array(pendingSamples.prefix(maxProcessingBatchSamples))
                    pendingSamples.removeFirst(maxProcessingBatchSamples)
                }
            } else {
                batch = pendingSamples
                pendingSamples.removeAll(keepingCapacity: true)
            }
            shouldResyncBeforeBatch = resyncAfterDrop
            resyncAfterDrop = false
            queuedAfterPopSec = Double(pendingSamples.count) / 16_000.0
            pendingLock.unlock()

            if droppedForRealtime > 0 {
                let droppedSec = Double(droppedForRealtime) / 16_000.0
                let statusLine = String(format: "Realtime catch-up (dropped %.2fs stale audio)", droppedSec)
                DispatchQueue.main.async {
                    self.status = statusLine
                }
            }

            guard var engine = inferenceEngine else { return }
            if shouldResyncBeforeBatch {
                let flushed = engine.finishStream()
                let flushedConfirmed = flushed.confirmed
                let flushedHypothesis = flushed.hypothesis
                inferenceEngine = engine
                currentSpeechStartedAtSec = nil
                currentSpeechBaselineText = ""
                currentSpeechFirstTokenLatencyMs = nil
                DispatchQueue.main.async {
                    self.confirmedText = flushedConfirmed
                    self.hypothesisText = flushedHypothesis
                }
            }
            let started = CFAbsoluteTimeGetCurrent()
            do {
                let events = try engine.process(samples: batch)
                let elapsed = CFAbsoluteTimeGetCurrent() - started
                let batchAudioSec = Double(batch.count) / 16_000.0
                totalProcessedAudioSec += batchAudioSec
                totalInferenceSec += elapsed
                let inferRTFx = totalProcessedAudioSec / max(totalInferenceSec, 1e-6)
                if runStartedAtSec == 0 {
                    runStartedAtSec = started
                }
                let wallElapsed = max(CFAbsoluteTimeGetCurrent() - runStartedAtSec, 1e-6)
                let wallRTFx = totalProcessedAudioSec / wallElapsed
                inferenceEngine = engine

                if let latest = events.last {
                    let now = CFAbsoluteTimeGetCurrent()
                    if latest.didFlushSegment {
                        if let observed = currentSpeechFirstTokenLatencyMs {
                            firstTokenLatencyMsSamples.append(observed)
                        }
                        currentSpeechStartedAtSec = nil
                        currentSpeechBaselineText = ""
                        currentSpeechFirstTokenLatencyMs = nil
                    } else if latest.isSpeech {
                        let merged = mergedTranscript(confirmed: latest.transcript.confirmed, hypothesis: latest.transcript.hypothesis)
                        if currentSpeechStartedAtSec == nil {
                            currentSpeechStartedAtSec = now
                            currentSpeechBaselineText = merged
                            currentSpeechFirstTokenLatencyMs = nil
                        } else if currentSpeechFirstTokenLatencyMs == nil, merged != currentSpeechBaselineText {
                            let latencyMs = (now - (currentSpeechStartedAtSec ?? now)) * 1000.0
                            currentSpeechFirstTokenLatencyMs = latencyMs
                            firstTokenLatencyMsSamples.append(latencyMs)
                            if metricsLogEnabled {
                                fputs(String(format: "[metrics] first-token-latency=%.1f ms\n", latencyMs), stderr)
                            }
                        }
                    }

                    let firstTokenCurrent = currentSpeechFirstTokenLatencyMs ?? firstTokenLatencyMsSamples.last
                    let firstTokenAvg = firstTokenLatencyMsSamples.isEmpty ? nil : firstTokenLatencyMsSamples.reduce(0, +) / Double(firstTokenLatencyMsSamples.count)
                    let firstTokenWorst = firstTokenLatencyMsSamples.max()
                    let firstTokenSummary: String
                    if let current = firstTokenCurrent {
                        if let avg = firstTokenAvg, let worst = firstTokenWorst {
                            firstTokenSummary = String(format: "first-token %.0fms (avg %.0f / worst %.0f)", current, avg, worst)
                        } else {
                            firstTokenSummary = String(format: "first-token %.0fms", current)
                        }
                    } else {
                        firstTokenSummary = "first-token n/a"
                    }

                    // For UI responsiveness, prefer the newest non-flush event for
                    // display; flush events are still useful for finalization but
                    // otherwise make the transcript appear as abrupt bursts.
                    let displayEvent = events.last(where: { !$0.didFlushSegment }) ?? latest
                    let latestConfirmed = displayEvent.transcript.confirmed
                    let latestHypothesis = displayEvent.transcript.hypothesis
                    let statusLine = String(
                        format: "Listening... | mic %@ | vad=%@ %.0fdBFS | inf %.1fx | wall %.1fx | queue %.1fs",
                        self.inputDeviceName,
                        displayEvent.isSpeech ? "speech" : "silence",
                        displayEvent.energyDBFS,
                        inferRTFx,
                        wallRTFx,
                        queuedAfterPopSec
                    )
                    DispatchQueue.main.async {
                        self.confirmedText = latestConfirmed
                        self.hypothesisText = latestHypothesis
                        self.status = statusLine
                        self.metrics = firstTokenSummary
                    }
                } else {
                    let firstTokenAvg = firstTokenLatencyMsSamples.isEmpty ? nil : firstTokenLatencyMsSamples.reduce(0, +) / Double(firstTokenLatencyMsSamples.count)
                    let firstTokenWorst = firstTokenLatencyMsSamples.max()
                    let firstTokenSummary: String
                    if let avg = firstTokenAvg, let worst = firstTokenWorst {
                        firstTokenSummary = String(format: "first-token avg %.0fms / worst %.0fms", avg, worst)
                    } else {
                        firstTokenSummary = "first-token n/a"
                    }
                    let statusLine = String(format: "Listening... | inf %.1fx | wall %.1fx | queue %.1fs", inferRTFx, wallRTFx, queuedAfterPopSec)
                    DispatchQueue.main.async {
                        self.status = statusLine
                        self.metrics = firstTokenSummary
                    }
                }
            } catch {
                let errorLine = "Inference error: \(error)"
                DispatchQueue.main.async {
                    self.status = errorLine
                }
                return
            }
        }
    }

    private func mergedTranscript(confirmed: String, hypothesis: String) -> String {
        let c = confirmed.trimmingCharacters(in: .whitespacesAndNewlines)
        let h = hypothesis.trimmingCharacters(in: .whitespacesAndNewlines)
        if c.isEmpty { return h }
        if h.isEmpty { return c }
        return c + " " + h
    }

    private func startModelLoadTicker() {
        stopModelLoadTicker()
        modelLoadStartedAt = Date()
        modelLoadTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            guard let self else { return }
            guard self.isStarting, let started = self.modelLoadStartedAt else { return }
            let elapsed = Date().timeIntervalSince(started)
            if self.status.contains("Loading") || self.status.contains("Microphone granted") || self.status.contains("Model ready") {
                self.status = self.decorateModelLoadStatus(
                    self.status.components(separatedBy: " | ").first ?? self.status,
                    elapsedOverride: elapsed
                )
            }
        }
    }

    private func stopModelLoadTicker() {
        modelLoadTimer?.invalidate()
        modelLoadTimer = nil
        modelLoadStartedAt = nil
    }

    private func decorateModelLoadStatus(_ base: String, elapsedOverride: TimeInterval? = nil) -> String {
        let elapsed = elapsedOverride ?? modelLoadStartedAt.map { Date().timeIntervalSince($0) } ?? 0
        if elapsed <= 0 { return base }
        return String(format: "%@ | %.1fs", base, elapsed)
    }

    private func requestMicrophoneAccess(completion: @escaping @Sendable (Bool) -> Void) {
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
