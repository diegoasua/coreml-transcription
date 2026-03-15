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

private func hasParakeetArtifacts(
    at directory: URL,
    suffix: String,
    encoderSuffix: String? = nil,
    decoderSuffix: String? = nil
) -> Bool {
    let fm = FileManager.default
    let resolvedEncoderSuffix = (encoderSuffix?.isEmpty == false) ? encoderSuffix! : suffix
    let resolvedDecoderSuffix = preferredParakeetDecoderSuffix(
        at: directory,
        suffix: suffix,
        decoderSuffix: decoderSuffix
    )
    let encoder = directory.appendingPathComponent("encoder-model-\(resolvedEncoderSuffix).mlpackage").path
    let decoder = directory.appendingPathComponent("decoder_joint-model-\(resolvedDecoderSuffix).mlpackage").path
    let vocab = directory.appendingPathComponent("vocab.txt").path
    return fm.fileExists(atPath: encoder) && fm.fileExists(atPath: decoder) && fm.fileExists(atPath: vocab)
}

private func preferredParakeetDecoderSuffix(
    at directory: URL,
    suffix: String,
    decoderSuffix: String? = nil
) -> String {
    if let decoderSuffix, !decoderSuffix.isEmpty {
        return decoderSuffix
    }
    let preferred = "\(suffix)-stateful-v2"
    let preferredPath = directory.appendingPathComponent("decoder_joint-model-\(preferred).mlpackage").path
    if FileManager.default.fileExists(atPath: preferredPath) {
        return preferred
    }
    return suffix
}

private func resolveParakeetModelDirectory(
    preferred: URL?,
    suffix: String,
    encoderSuffix: String? = nil,
    decoderSuffix: String? = nil
) -> URL? {
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
        if hasParakeetArtifacts(
            at: URL(fileURLWithPath: normalized),
            suffix: suffix,
            encoderSuffix: encoderSuffix,
            decoderSuffix: decoderSuffix
        ) {
            return URL(fileURLWithPath: normalized)
        }
    }
    return nil
}

private func defaultParakeetModelDirectoryPath() -> String {
    let defaultSuffix = ProcessInfo.processInfo.environment["PARAKEET_COREML_MODEL_SUFFIX"] ?? "odmbp-approx"
    let encoderSuffix = ProcessInfo.processInfo.environment["PARAKEET_COREML_ENCODER_SUFFIX"]
    let decoderSuffix = ProcessInfo.processInfo.environment["PARAKEET_COREML_DECODER_SUFFIX"]
    if let resolved = resolveParakeetModelDirectory(
        preferred: nil,
        suffix: defaultSuffix,
        encoderSuffix: encoderSuffix,
        decoderSuffix: decoderSuffix
    ) {
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
            Text("Metrics:\n\(vm.metrics)")
                .font(.footnote)
                .foregroundStyle(.secondary)

            GroupBox("Transcript") {
                ScrollView {
                    if vm.confirmedText.isEmpty && vm.hypothesisText.isEmpty {
                        Text("...")
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    } else {
                        let draftTail = vm.hypothesisText.isEmpty
                            ? ""
                            : (vm.confirmedText.isEmpty ? vm.hypothesisText : " \(vm.hypothesisText)")
                        Text(
                            "\(Text(verbatim: vm.confirmedText))\(Text(verbatim: draftTail).foregroundStyle(.secondary))"
                        )
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                    }
                }
                .frame(minHeight: 220)
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
    @Published var metrics: String = MicTranscriptionViewModel.emptyMetricsLine
    @Published var isRunning: Bool = false
    @Published var isStarting: Bool = false
    @Published var modelDirectory: String = defaultParakeetModelDirectoryPath()
    @Published var modelSuffix: String = ProcessInfo.processInfo.environment["PARAKEET_COREML_MODEL_SUFFIX"] ?? "odmbp-approx"
    private let encoderModelSuffix = ProcessInfo.processInfo.environment["PARAKEET_COREML_ENCODER_SUFFIX"]
    private let decoderModelSuffix = ProcessInfo.processInfo.environment["PARAKEET_COREML_DECODER_SUFFIX"]

    private static let streamModeDefault = (ProcessInfo.processInfo.environment["PARAKEET_STREAM_MODE"] ?? "rewrite-prefix").lowercased()

    private typealias InferenceEngine = StreamingInferenceEngine<ParakeetCoreMLTDTTranscriptionModel, EnergyVAD>
    private struct LatencyStats {
        let current: Double?
        let avg: Double?
        let p95: Double?
    }

    private struct MetricsSnapshot {
        let inferRTFx: Double
        let wallRTFx: Double
        let draftReady: LatencyStats
        let confirmedReady: LatencyStats
        let draftOnset: LatencyStats
        let confirmedOnset: LatencyStats
    }

    private var inferenceEngine: InferenceEngine?

    private let queue = DispatchQueue(label: "transcribe.macos.inference")
    private let startupQueue = DispatchQueue(label: "transcribe.macos.startup", qos: .userInitiated)
    private let audioEngine = AVAudioEngine()
    private var audioConverter: AVAudioConverter?
    private let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!
    private let chunkMs = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_CHUNK_MS"] ?? "") ??
        (MicTranscriptionViewModel.streamModeDefault == "rewrite-prefix" ? 500 : 160)
    private let hopMs = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_HOP_MS"] ?? "") ??
        (MicTranscriptionViewModel.streamModeDefault == "rewrite-prefix" ? 250 : 80)
    private var processingHopSamples: Int {
        max(1, Int(round(Double(hopMs) * 16_000.0 / 1000.0)))
    }
    private let agreementCount = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_AGREEMENT"] ?? "") ?? 2
    private let draftAgreementCount = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_DRAFT_AGREEMENT"] ?? "") ?? 1
    private let decodeOnlyWhenSpeech = (ProcessInfo.processInfo.environment["PARAKEET_DECODE_ONLY_WHEN_SPEECH"] ?? "0") != "0"
    private let flushOnSpeechEnd: Bool = {
        let env = ProcessInfo.processInfo.environment
        if let explicit = env["PARAKEET_STREAM_FLUSH_ON_SPEECH_END"], !explicit.isEmpty {
            return explicit != "0"
        }
        // Default low-latency mode: keep decoder/model state across silence
        // boundaries unless explicitly requested otherwise.
        return false
    }()
    private let maxSpeechChunkRunBeforeReset = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_MAX_SPEECH_CHUNKS"] ?? "") ?? 80
    private let maxStagnantSpeechChunks = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_MAX_STAGNANT_CHUNKS"] ?? "") ?? 24
    private let maxSymbolsPerStep = Int(ProcessInfo.processInfo.environment["PARAKEET_TDT_MAX_SYMBOLS_PER_STEP"] ?? "") ?? 10
    private let maxTokensPerChunk = Int(ProcessInfo.processInfo.environment["PARAKEET_TDT_MAX_TOKENS_PER_CHUNK"] ?? "") ?? 0
    private let streamingHistoryFrames = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_HISTORY_FRAMES"] ?? "") ?? 300
    private let streamingMinTailDecodeFrames = Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_MIN_TAIL_FRAMES"] ?? "") ?? 8
    private let vadStartThresholdDBFS = Float(ProcessInfo.processInfo.environment["PARAKEET_VAD_START_DBFS"] ?? "") ?? -50
    private let vadEndThresholdDBFS = Float(ProcessInfo.processInfo.environment["PARAKEET_VAD_END_DBFS"] ?? "") ?? -58
    private let vadMinSpeechMs = Int(ProcessInfo.processInfo.environment["PARAKEET_VAD_MIN_SPEECH_MS"] ?? "") ?? 60
    private let vadMinSilenceMs = Int(ProcessInfo.processInfo.environment["PARAKEET_VAD_MIN_SILENCE_MS"] ?? "") ?? 400
    private let enableVoiceProcessing = (ProcessInfo.processInfo.environment["PARAKEET_AUDIO_VOICE_PROCESSING"] ?? "0") != "0"
    private let enableVoiceProcessingAGC = (ProcessInfo.processInfo.environment["PARAKEET_AUDIO_VOICE_PROCESSING_AGC"] ?? "0") != "0"
    private let forcedInputChannelIndex: Int? = {
        let env = ProcessInfo.processInfo.environment
        guard let raw = env["PARAKEET_AUDIO_INPUT_CHANNEL"], let parsed = Int(raw) else { return nil }
        return max(0, parsed - 1)
    }()
    private let pendingLock = NSLock()
    private var pendingSamples: [Float] = []
    private var pendingBaseSampleCursor = 0
    private var audioIngressTimeline = AudioIngressTimeline(sampleRate: 16_000)
    private var processingScheduled = false
    private let maxBufferedSamples = (Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_MAX_BUFFER_SEC"] ?? "") ?? 4) * 16_000
    private let latestFirstScheduling = (ProcessInfo.processInfo.environment["PARAKEET_STREAM_LATEST_FIRST"] ?? "1") != "0"
    private let backlogSoftLimitSamples: Int = {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["PARAKEET_STREAM_BACKLOG_SOFT_SEC"], let sec = Double(raw), sec > 0 {
            return Int(sec * 16_000.0)
        }
        if MicTranscriptionViewModel.streamModeDefault == "rewrite-prefix" {
            return Int(5.0 * 16_000.0)
        }
        return Int(1.5 * 16_000.0)
    }()
    private let backlogTargetSamples: Int = {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["PARAKEET_STREAM_BACKLOG_TARGET_SEC"], let sec = Double(raw), sec > 0 {
            return Int(sec * 16_000.0)
        }
        if MicTranscriptionViewModel.streamModeDefault == "rewrite-prefix" {
            return Int(1.5 * 16_000.0)
        }
        return Int(0.4 * 16_000.0)
    }()
    private let latestResyncContextSamples: Int = {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["PARAKEET_STREAM_LATEST_RESYNC_CONTEXT_SEC"], let sec = Double(raw), sec > 0 {
            return max(1, Int(sec * 16_000.0))
        }
        if MicTranscriptionViewModel.streamModeDefault == "rewrite-prefix" {
            return Int(0.75 * 16_000.0)
        }
        return Int(0.48 * 16_000.0)
    }()
    private let latestOnsetProtectSec: Double = {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["PARAKEET_STREAM_ONSET_PROTECT_SEC"], let sec = Double(raw), sec >= 0 {
            return sec
        }
        return 0.8
    }()
    private let latestOnsetMinConfirmedWords: Int = {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["PARAKEET_STREAM_ONSET_MIN_CONFIRMED_WORDS"], let words = Int(raw), words > 0 {
            return words
        }
        return 2
    }()
    private var resyncAfterDrop = false
    private let maxProcessingBatchSamples: Int = {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["PARAKEET_STREAM_MAX_BATCH_SEC"], let sec = Double(raw), sec > 0 {
            return max(1, Int(sec * 16_000.0))
        }
        if MicTranscriptionViewModel.streamModeDefault == "rewrite-prefix" {
            return Int(0.5 * 16_000.0)
        }
        // Keep UI responsive and avoid bursty "all-at-once" transcript updates.
        return Int(0.25 * 16_000.0)
    }()
    private var captureStartedAtSec: Double = 0
    private var totalProcessedAudioSec: Double = 0
    private var totalProcessedSampleCursor: Int {
        Int(round(totalProcessedAudioSec * 16_000.0))
    }
    private var engineAudioCursorBaseSampleCursor = 0
    private var totalInferenceSec: Double = 0
    private var runStartedAtSec: Double = 0
    private var currentSpeechStartedAtSampleCursor: Int?
    private var currentSpeechBaselineDraft: String = ""
    private var currentSpeechBaselineConfirmed: String = ""
    private var currentSpeechDraftLatencyMs: Double?
    private var currentSpeechConfirmedLatencyMs: Double?
    private var draftLatencyMsSamples: [Double] = []
    private var confirmedLatencyMsSamples: [Double] = []
    private var currentDraftDisplayLatencyMs: Double?
    private var currentConfirmedDisplayLatencyMs: Double?
    private var draftDisplayLatencyMsSamples: [Double] = []
    private var confirmedDisplayLatencyMsSamples: [Double] = []
    // Screen-visible latency is tracked on the main thread so it includes app queue/UI publication overhead.
    private var currentDraftScreenLatencyMs: Double?
    private var currentConfirmedScreenLatencyMs: Double?
    private var draftScreenLatencyMsSamples: [Double] = []
    private var confirmedScreenLatencyMsSamples: [Double] = []
    private let latencyDisplaySampleWindow = 256
    private var audioCallbackCount = 0
    private var audioCallbackSampleTotal = 0
    private var engineBatchCount = 0
    private var emptyEngineBatchCount = 0
    private var previousEventWasSpeech = false
    private var lastObservedConfirmedText = ""
    private var lastObservedHypothesisText = ""
    private var modelLoadStartedAt: Date?
    private var modelLoadTimer: Timer?
    private let metricsLogEnabled = (ProcessInfo.processInfo.environment["PARAKEET_METRICS_LOG"] ?? "0") != "0"
    private var inputDeviceName: String = "unknown"
    private var selectedInputChannel: Int?
    private static let emptyMetricsLine =
        "RTFx(inf/live)=0.0/0.0\ningest->ready(d/c) n/a / n/a\ningest->screen(d/c) n/a / n/a\nonset(d/c) n/a / n/a"

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
        let resolvedEncoderSuffix = (encoderModelSuffix?.isEmpty == false) ? encoderModelSuffix! : modelSuffix
        guard let dirURL = resolveParakeetModelDirectory(
            preferred: preferredURL,
            suffix: modelSuffix,
            encoderSuffix: encoderModelSuffix,
            decoderSuffix: decoderModelSuffix
        ) else {
            let resolvedDecoderSuffix = preferredParakeetDecoderSuffix(
                at: preferredURL,
                suffix: modelSuffix,
                decoderSuffix: decoderModelSuffix
            )
            status = "Model artifacts not found for encoder '\(resolvedEncoderSuffix)' and decoder '\(resolvedDecoderSuffix)' under: \(preferredURL.path)"
            return
        }
        let resolvedDecoderSuffix = preferredParakeetDecoderSuffix(
            at: dirURL,
            suffix: modelSuffix,
            decoderSuffix: decoderModelSuffix
        )
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
                        encoderModelSuffix: self.encoderModelSuffix,
                        decoderModelSuffix: resolvedDecoderSuffix,
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
                        draftAgreementCount: self.draftAgreementCount,
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
        pendingBaseSampleCursor = 0
        audioIngressTimeline.reset()
        pendingLock.unlock()
        captureStartedAtSec = 0
        totalProcessedAudioSec = 0
        engineAudioCursorBaseSampleCursor = 0
        totalInferenceSec = 0
        stopModelLoadTicker()
        runStartedAtSec = 0
        currentSpeechStartedAtSampleCursor = nil
        currentSpeechBaselineDraft = ""
        currentSpeechBaselineConfirmed = ""
        currentSpeechDraftLatencyMs = nil
        currentSpeechConfirmedLatencyMs = nil
        draftLatencyMsSamples.removeAll(keepingCapacity: false)
        confirmedLatencyMsSamples.removeAll(keepingCapacity: false)
        currentDraftDisplayLatencyMs = nil
        currentConfirmedDisplayLatencyMs = nil
        draftDisplayLatencyMsSamples.removeAll(keepingCapacity: false)
        confirmedDisplayLatencyMsSamples.removeAll(keepingCapacity: false)
        currentDraftScreenLatencyMs = nil
        currentConfirmedScreenLatencyMs = nil
        draftScreenLatencyMsSamples.removeAll(keepingCapacity: false)
        confirmedScreenLatencyMsSamples.removeAll(keepingCapacity: false)
        audioCallbackCount = 0
        audioCallbackSampleTotal = 0
        engineBatchCount = 0
        emptyEngineBatchCount = 0
        previousEventWasSpeech = false
        lastObservedConfirmedText = ""
        lastObservedHypothesisText = ""
        selectedInputChannel = nil

        queue.sync {
            if var engine = inferenceEngine {
                let finalState = engine.finishStream()
                inferenceEngine = engine
                let finalConfirmed = finalState.confirmed
                let finalHypothesis = finalState.hypothesis
                DispatchQueue.main.async {
                    self.confirmedText = finalConfirmed
                    self.hypothesisText = finalHypothesis
                    self.metrics = Self.emptyMetricsLine
                }
            }
        }
        status = "Stopped"
    }

    private func configureAndStartAudio(engine: InferenceEngine) {
        inferenceEngine = engine
        captureStartedAtSec = 0
        totalProcessedAudioSec = 0
        engineAudioCursorBaseSampleCursor = 0
        totalInferenceSec = 0
        stopModelLoadTicker()
        runStartedAtSec = 0
        currentSpeechStartedAtSampleCursor = nil
        currentSpeechBaselineDraft = ""
        currentSpeechBaselineConfirmed = ""
        currentSpeechDraftLatencyMs = nil
        currentSpeechConfirmedLatencyMs = nil
        draftLatencyMsSamples.removeAll(keepingCapacity: false)
        confirmedLatencyMsSamples.removeAll(keepingCapacity: false)
        currentDraftDisplayLatencyMs = nil
        currentConfirmedDisplayLatencyMs = nil
        draftDisplayLatencyMsSamples.removeAll(keepingCapacity: false)
        confirmedDisplayLatencyMsSamples.removeAll(keepingCapacity: false)
        currentDraftScreenLatencyMs = nil
        currentConfirmedScreenLatencyMs = nil
        draftScreenLatencyMsSamples.removeAll(keepingCapacity: false)
        confirmedScreenLatencyMsSamples.removeAll(keepingCapacity: false)
        audioCallbackCount = 0
        audioCallbackSampleTotal = 0
        engineBatchCount = 0
        emptyEngineBatchCount = 0
        previousEventWasSpeech = false
        lastObservedConfirmedText = ""
        lastObservedHypothesisText = ""
        selectedInputChannel = nil
        pendingLock.lock()
        pendingSamples.removeAll(keepingCapacity: false)
        pendingBaseSampleCursor = 0
        audioIngressTimeline.reset()
        processingScheduled = false
        pendingLock.unlock()

        do {
            let inputNode = audioEngine.inputNode
            inputDeviceName = defaultInputDeviceName() ?? "unknown"
            fputs("[transcribe-macos] input device: \(inputDeviceName)\n", stderr)

            var activeVoiceProcessing = false

            func configureInputPath(voiceProcessing: Bool) throws -> AVAudioFormat {
                if voiceProcessing {
                    try inputNode.setVoiceProcessingEnabled(true)
                    inputNode.isVoiceProcessingBypassed = false
                    inputNode.isVoiceProcessingAGCEnabled = enableVoiceProcessingAGC
                    activeVoiceProcessing = true
                } else {
                    if enableVoiceProcessing {
                        do {
                            try inputNode.setVoiceProcessingEnabled(false)
                        } catch {
                            fputs("[transcribe-macos] unable to disable voice processing cleanly: \(error)\n", stderr)
                        }
                    }
                    activeVoiceProcessing = false
                }

                let inputFormat = inputNode.inputFormat(forBus: 0)
                fputs(
                    "[transcribe-macos] input format: \(Int(inputFormat.sampleRate)) Hz, channels \(inputFormat.channelCount), vp=\(activeVoiceProcessing ? "on" : "off")\n",
                    stderr
                )

                if voiceProcessing, inputFormat.channelCount > 2 {
                    throw NSError(
                        domain: "transcribe-macos",
                        code: 11,
                        userInfo: [
                            NSLocalizedDescriptionKey:
                                "Voice processing yielded \(inputFormat.channelCount) channels; falling back to vp=off for compatibility"
                        ]
                    )
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
                return inputFormat
            }

            if enableVoiceProcessing {
                do {
                    _ = try configureInputPath(voiceProcessing: true)
                } catch {
                    fputs("[transcribe-macos] voice processing unavailable: \(error)\n", stderr)
                    _ = try configureInputPath(voiceProcessing: false)
                }
            } else {
                _ = try configureInputPath(voiceProcessing: false)
            }

            do {
                try audioEngine.start()
            } catch {
                if activeVoiceProcessing {
                    fputs("[transcribe-macos] audio start failed with voice processing (\(error)); retrying without voice processing\n", stderr)
                    audioEngine.stop()
                    _ = try configureInputPath(voiceProcessing: false)
                    try audioEngine.start()
                } else {
                    throw error
                }
            }

            isRunning = true
            isStarting = false
            captureStartedAtSec = CFAbsoluteTimeGetCurrent()
            runStartedAtSec = captureStartedAtSec
#if DEBUG
            status = "Listening... (DEBUG build; slower) chunk \(chunkMs)ms / hop \(hopMs)ms | mic \(inputDeviceName) | vp \(activeVoiceProcessing ? "on" : "off")"
#else
            status = "Listening... (chunk \(chunkMs)ms / hop \(hopMs)ms) | mic \(inputDeviceName) | vp \(activeVoiceProcessing ? "on" : "off")"
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
        let receivedAtSec = CFAbsoluteTimeGetCurrent()
        if Int(buffer.format.channelCount) > 1,
           let monoSamples = extractPreferredMonoSamples(from: buffer) {
            if !monoSamples.isEmpty {
                enqueueSamplesForProcessing(monoSamples, receivedAtSec: receivedAtSec)
            }
            return
        }

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
        enqueueSamplesForProcessing(samples, receivedAtSec: receivedAtSec)
    }

    private func extractPreferredMonoSamples(from buffer: AVAudioPCMBuffer) -> [Float]? {
        guard let channels = buffer.floatChannelData else { return nil }
        let channelCount = Int(buffer.format.channelCount)
        let frameCount = Int(buffer.frameLength)
        guard channelCount > 1, frameCount > 0 else { return nil }

        let pickedChannel: Int
        if let forced = forcedInputChannelIndex, forced < channelCount {
            pickedChannel = forced
            if selectedInputChannel != forced {
                selectedInputChannel = forced
                if metricsLogEnabled {
                    fputs(
                        "[transcribe-macos] forcing input channel \(forced + 1)/\(channelCount)\n",
                        stderr
                    )
                }
            }
        } else if let selected = selectedInputChannel, selected >= 0, selected < channelCount {
            pickedChannel = selected
        } else {
            var bestChannel = 0
            var bestEnergy: Float = -.infinity
            for idx in 0 ..< channelCount {
                let ptr = channels[idx]
                var energy: Float = 0
                var i = 0
                while i < frameCount {
                    let v = ptr[i]
                    energy += v * v
                    i += 1
                }
                if energy > bestEnergy {
                    bestEnergy = energy
                    bestChannel = idx
                }
            }
            selectedInputChannel = bestChannel
            pickedChannel = bestChannel
            if metricsLogEnabled {
                fputs(
                    "[transcribe-macos] selected input channel \(bestChannel + 1)/\(channelCount)\n",
                    stderr
                )
            }
        }

        let src = channels[pickedChannel]
        var mono = Array(UnsafeBufferPointer(start: src, count: frameCount))
        let inputRate = buffer.format.sampleRate
        let outputRate = targetFormat.sampleRate
        if abs(inputRate - outputRate) > 1e-6 {
            mono = Self.resampleLinear(mono, from: inputRate, to: outputRate)
        }
        return mono
    }

    private static func resampleLinear(_ input: [Float], from inputRate: Double, to outputRate: Double) -> [Float] {
        guard !input.isEmpty else { return [] }
        guard inputRate > 0, outputRate > 0 else { return input }
        if abs(inputRate - outputRate) < 1e-6 { return input }

        let ratio = outputRate / inputRate
        let outputCount = max(1, Int((Double(input.count) * ratio).rounded()))
        if input.count == 1 {
            return Array(repeating: input[0], count: outputCount)
        }

        var output = Array(repeating: Float(0), count: outputCount)
        for outIdx in 0 ..< outputCount {
            let srcPos = Double(outIdx) / ratio
            let lo = min(max(Int(floor(srcPos)), 0), input.count - 1)
            let hi = min(lo + 1, input.count - 1)
            let frac = Float(srcPos - Double(lo))
            output[outIdx] = input[lo] * (1 - frac) + input[hi] * frac
        }
        return output
    }

    private func enqueueSamplesForProcessing(_ samples: [Float], receivedAtSec: Double) {
        var shouldSchedule = false
        var droppedCount = 0

        pendingLock.lock()
        audioCallbackCount += 1
        audioCallbackSampleTotal += samples.count
        _ = audioIngressTimeline.recordIngress(sampleCount: samples.count, receivedAtSec: receivedAtSec)
        pendingSamples.append(contentsOf: samples)
        if maxBufferedSamples > 0, pendingSamples.count > maxBufferedSamples {
            droppedCount = pendingSamples.count - maxBufferedSamples
            pendingSamples.removeFirst(droppedCount)
            pendingBaseSampleCursor += droppedCount
            resyncAfterDrop = true
        }
        if !latestFirstScheduling,
           backlogSoftLimitSamples > 0,
           pendingSamples.count > backlogSoftLimitSamples {
            var target = max(backlogTargetSamples, maxProcessingBatchSamples)
            if latestFirstScheduling {
                target = max(target, latestResyncContextSamples)
            }
            if pendingSamples.count > target {
                let extraDrop = pendingSamples.count - target
                droppedCount += extraDrop
                pendingSamples.removeFirst(extraDrop)
                pendingBaseSampleCursor += extraDrop
                resyncAfterDrop = true
            }
        }
        if !processingScheduled {
            processingScheduled = true
            shouldSchedule = true
        }
        let queuedSeconds = Double(pendingSamples.count) / 16_000.0
        let callbackCount = audioCallbackCount
        let callbackSampleTotal = audioCallbackSampleTotal
        let queuedSampleCount = pendingSamples.count
        pendingLock.unlock()

        if metricsLogEnabled, (callbackCount <= 3 || callbackCount % 20 == 0) {
            fputs(
                String(
                    format: "[transcribe-macos] audio callbacks=%d total_samples=%d last_batch=%d queued_samples=%d\n",
                    callbackCount,
                    callbackSampleTotal,
                    samples.count,
                    queuedSampleCount
                ),
                stderr
            )
        }

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
            let batchStartSampleCursor: Int
            let queuedAfterPopSec: Double
            let shouldResyncBeforeBatch: Bool
            var droppedForRealtime = 0
            let protectLatestFirstOnset = shouldProtectLatestFirstOnset()
            pendingLock.lock()
            if pendingSamples.isEmpty {
                processingScheduled = false
                pendingLock.unlock()
                return
            }
            let shouldUseLatestFirstCatchUp = StreamingSchedulerSupport.shouldUseLatestFirstCatchUp(
                latestFirstEnabled: latestFirstScheduling,
                protectOnset: protectLatestFirstOnset,
                queuedSampleCount: pendingSamples.count,
                catchUpTriggerSamples: backlogTargetSamples
            )
            if maxProcessingBatchSamples > 0, pendingSamples.count > maxProcessingBatchSamples {
                if shouldUseLatestFirstCatchUp {
                    let batchSampleCount = StreamingSchedulerSupport.catchUpBatchCount(
                        queuedUnitCount: pendingSamples.count,
                        baseBatchCount: maxProcessingBatchSamples,
                        catchUpTargetUnitCount: backlogTargetSamples,
                        minContextUnitCount: latestResyncContextSamples
                    )
                    if backlogSoftLimitSamples > 0, pendingSamples.count > backlogSoftLimitSamples {
                        // Under real overflow, restart from a bounded contiguous
                        // tail instead of feeding a sparse queue into the model.
                        batchStartSampleCursor = pendingBaseSampleCursor + pendingSamples.count - batchSampleCount
                        batch = Array(pendingSamples.suffix(batchSampleCount))
                        droppedForRealtime = pendingSamples.count - batch.count
                        pendingBaseSampleCursor += pendingSamples.count
                        pendingSamples.removeAll(keepingCapacity: true)
                        if droppedForRealtime > 0 {
                            resyncAfterDrop = true
                        }
                    } else {
                        batchStartSampleCursor = pendingBaseSampleCursor
                        batch = Array(pendingSamples.prefix(batchSampleCount))
                        pendingBaseSampleCursor += batch.count
                        pendingSamples.removeFirst(batch.count)
                    }
                } else {
                    batchStartSampleCursor = pendingBaseSampleCursor
                    batch = Array(pendingSamples.prefix(maxProcessingBatchSamples))
                    pendingBaseSampleCursor += maxProcessingBatchSamples
                    pendingSamples.removeFirst(maxProcessingBatchSamples)
                }
            } else {
                batchStartSampleCursor = pendingBaseSampleCursor
                batch = pendingSamples
                pendingBaseSampleCursor += pendingSamples.count
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
                let snapshot = engine.discardStream(preserveHypothesis: latestFirstScheduling)
                let confirmed = snapshot.confirmed
                let hypothesis = snapshot.hypothesis
                inferenceEngine = engine
                engineAudioCursorBaseSampleCursor = batchStartSampleCursor
                currentSpeechStartedAtSampleCursor = nil
                currentSpeechBaselineDraft = ""
                currentSpeechBaselineConfirmed = ""
                currentSpeechDraftLatencyMs = nil
                currentSpeechConfirmedLatencyMs = nil
                currentDraftDisplayLatencyMs = nil
                currentConfirmedDisplayLatencyMs = nil
                currentDraftScreenLatencyMs = nil
                currentConfirmedScreenLatencyMs = nil
                previousEventWasSpeech = false
                lastObservedConfirmedText = confirmed
                lastObservedHypothesisText = hypothesis
                DispatchQueue.main.async {
                    self.confirmedText = confirmed
                    self.hypothesisText = hypothesis
                }
            }
            let started = CFAbsoluteTimeGetCurrent()
            do {
                let events = try engine.process(samples: batch)
                let elapsed = CFAbsoluteTimeGetCurrent() - started
                let batchAudioSec = Double(batch.count) / 16_000.0
                let batchCompletedAtSec = started + elapsed
                totalProcessedAudioSec += batchAudioSec
                totalInferenceSec += elapsed
                let inferRTFx = totalProcessedAudioSec / max(totalInferenceSec, 1e-6)
                if runStartedAtSec == 0 {
                    runStartedAtSec = captureStartedAtSec > 0 ? captureStartedAtSec : started
                }
                let wallElapsed = max(batchCompletedAtSec - runStartedAtSec, 1e-6)
                let wallRTFx = totalProcessedAudioSec / wallElapsed
                inferenceEngine = engine
                engineBatchCount += 1
                if metricsLogEnabled {
                    if events.isEmpty {
                        emptyEngineBatchCount += 1
                        if emptyEngineBatchCount <= 3 || emptyEngineBatchCount % 20 == 0 {
                            fputs(
                                String(
                                    format: "[transcribe-macos] engine batch=%d samples=%d produced no events (empty_batches=%d)\n",
                                    engineBatchCount,
                                    batch.count,
                                    emptyEngineBatchCount
                                ),
                                stderr
                            )
                        }
                    } else if engineBatchCount <= 3 || engineBatchCount % 20 == 0 {
                        fputs(
                            String(
                                format: "[transcribe-macos] engine batch=%d samples=%d events=%d latest_cursor=%.3fs\n",
                                engineBatchCount,
                                batch.count,
                                events.count,
                                events.last?.audioCursorSec ?? 0
                            ),
                            stderr
                        )
                    }
                }

                if let latest = events.last {
                    var latestDraftScreenSampleCursor: Int?
                    var latestConfirmedScreenSampleCursor: Int?
                    for event in events {
                        let eventSampleCursor = engineAudioCursorBaseSampleCursor + Int(
                            round(event.audioCursorSec * 16_000.0)
                        )
                        if event.isSpeech, !previousEventWasSpeech {
                            currentSpeechStartedAtSampleCursor = eventSampleCursor
                            currentSpeechBaselineDraft = lastObservedHypothesisText
                            currentSpeechBaselineConfirmed = lastObservedConfirmedText
                            currentSpeechDraftLatencyMs = nil
                            currentSpeechConfirmedLatencyMs = nil
                        }
                        if event.isSpeech,
                           let startedAtSampleCursor = currentSpeechStartedAtSampleCursor,
                           currentSpeechDraftLatencyMs == nil,
                           event.transcript.hypothesis != currentSpeechBaselineDraft {
                            guard var latencyMs = ingestLatencyMs(
                                observedAtSec: batchCompletedAtSec,
                                sampleCursor: startedAtSampleCursor
                            ) else {
                                continue
                            }
                            if latencyMs < 1.0 {
                                latencyMs = Double(self.hopMs)
                            }
                            currentSpeechDraftLatencyMs = latencyMs
                            draftLatencyMsSamples.append(latencyMs)
                            if metricsLogEnabled {
                                fputs(String(format: "[metrics] draft-onset-latency=%.1f ms\n", latencyMs), stderr)
                            }
                        }
                        if event.transcript.hypothesis != lastObservedHypothesisText {
                            guard let latencyMs = ingestLatencyMs(
                                observedAtSec: batchCompletedAtSec,
                                sampleCursor: eventSampleCursor
                            ) else {
                                continue
                            }
                            currentDraftDisplayLatencyMs = latencyMs
                            Self.appendRollingSample(
                                latencyMs,
                                into: &draftDisplayLatencyMsSamples,
                                maxCount: latencyDisplaySampleWindow
                            )
                            latestDraftScreenSampleCursor = eventSampleCursor
                            if metricsLogEnabled {
                                fputs(String(format: "[metrics] draft-ingest-ready-latency=%.1f ms\n", latencyMs), stderr)
                            }
                        }
                        if event.isSpeech,
                           let startedAtSampleCursor = currentSpeechStartedAtSampleCursor,
                           currentSpeechConfirmedLatencyMs == nil,
                           event.transcript.confirmed != currentSpeechBaselineConfirmed {
                            guard let latencyMs = ingestLatencyMs(
                                observedAtSec: batchCompletedAtSec,
                                sampleCursor: startedAtSampleCursor
                            ) else {
                                continue
                            }
                            currentSpeechConfirmedLatencyMs = latencyMs
                            confirmedLatencyMsSamples.append(latencyMs)
                            if metricsLogEnabled {
                                fputs(String(format: "[metrics] confirmed-onset-latency=%.1f ms\n", latencyMs), stderr)
                            }
                        }
                        if event.transcript.confirmed != lastObservedConfirmedText {
                            guard let latencyMs = ingestLatencyMs(
                                observedAtSec: batchCompletedAtSec,
                                sampleCursor: eventSampleCursor
                            ) else {
                                continue
                            }
                            currentConfirmedDisplayLatencyMs = latencyMs
                            Self.appendRollingSample(
                                latencyMs,
                                into: &confirmedDisplayLatencyMsSamples,
                                maxCount: latencyDisplaySampleWindow
                            )
                            latestConfirmedScreenSampleCursor = eventSampleCursor
                            if metricsLogEnabled {
                                fputs(String(format: "[metrics] confirmed-ingest-ready-latency=%.1f ms\n", latencyMs), stderr)
                            }
                        }
                        if event.didFlushSegment, let startedAtSampleCursor = currentSpeechStartedAtSampleCursor {
                            if currentSpeechConfirmedLatencyMs == nil {
                                if let latencyMs = ingestLatencyMs(
                                    observedAtSec: batchCompletedAtSec,
                                    sampleCursor: startedAtSampleCursor
                                ) {
                                    currentSpeechConfirmedLatencyMs = latencyMs
                                    confirmedLatencyMsSamples.append(latencyMs)
                                    if metricsLogEnabled {
                                        fputs(String(format: "[metrics] confirmed-onset-latency-flush=%.1f ms\n", latencyMs), stderr)
                                    }
                                }
                            }
                            currentSpeechStartedAtSampleCursor = nil
                            currentSpeechBaselineDraft = ""
                            currentSpeechBaselineConfirmed = ""
                            currentSpeechDraftLatencyMs = nil
                            currentSpeechConfirmedLatencyMs = nil
                        }
                        previousEventWasSpeech = event.isSpeech
                        lastObservedConfirmedText = event.transcript.confirmed
                        lastObservedHypothesisText = event.transcript.hypothesis
                    }

                    let metricsSnapshot = metricsSnapshot(inferRTFx: inferRTFx, wallRTFx: wallRTFx)
                    let latestDraftScreenSampleCursorLocal = latestDraftScreenSampleCursor
                    let latestConfirmedScreenSampleCursorLocal = latestConfirmedScreenSampleCursor

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
                        let screenObservedAtSec = CFAbsoluteTimeGetCurrent()
                        if let sampleCursor = latestDraftScreenSampleCursorLocal,
                           let latencyMs = self.ingestLatencyMs(
                               observedAtSec: screenObservedAtSec,
                               sampleCursor: sampleCursor
                           ) {
                            self.currentDraftScreenLatencyMs = latencyMs
                            Self.appendRollingSample(
                                latencyMs,
                                into: &self.draftScreenLatencyMsSamples,
                                maxCount: self.latencyDisplaySampleWindow
                            )
                            if self.metricsLogEnabled {
                                fputs(String(format: "[metrics] draft-ingest-screen-latency=%.1f ms\n", latencyMs), stderr)
                            }
                        }
                        if let sampleCursor = latestConfirmedScreenSampleCursorLocal,
                           let latencyMs = self.ingestLatencyMs(
                               observedAtSec: screenObservedAtSec,
                               sampleCursor: sampleCursor
                           ) {
                            self.currentConfirmedScreenLatencyMs = latencyMs
                            Self.appendRollingSample(
                                latencyMs,
                                into: &self.confirmedScreenLatencyMsSamples,
                                maxCount: self.latencyDisplaySampleWindow
                            )
                            if self.metricsLogEnabled {
                                fputs(String(format: "[metrics] confirmed-ingest-screen-latency=%.1f ms\n", latencyMs), stderr)
                            }
                        }
                        self.confirmedText = latestConfirmed
                        self.hypothesisText = latestHypothesis
                        self.status = statusLine
                        self.metrics = self.metricsLine(snapshot: metricsSnapshot)
                    }
                } else {
                    let metricsSnapshot = metricsSnapshot(inferRTFx: inferRTFx, wallRTFx: wallRTFx)
                    let statusLine = String(format: "Listening... | inf %.1fx | wall %.1fx | queue %.1fs", inferRTFx, wallRTFx, queuedAfterPopSec)
                    DispatchQueue.main.async {
                        self.status = statusLine
                        self.metrics = self.metricsLine(snapshot: metricsSnapshot)
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

    private static func percentile(_ values: [Double], q: Double) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        if sorted.count == 1 { return sorted[0] }
        let idx = max(0, min(Double(sorted.count - 1), q * Double(sorted.count - 1)))
        let lo = Int(floor(idx))
        let hi = Int(ceil(idx))
        if lo == hi { return sorted[lo] }
        let frac = idx - Double(lo)
        return sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }

    private static func appendRollingSample(_ value: Double, into values: inout [Double], maxCount: Int) {
        values.append(value)
        let overflow = values.count - max(1, maxCount)
        if overflow > 0 {
            values.removeFirst(overflow)
        }
    }

    private func ingestLatencyMs(observedAtSec: Double, sampleCursor: Int) -> Double? {
        pendingLock.lock()
        let ingressTimeSec = audioIngressTimeline.ingressTime(forSampleCursor: sampleCursor)
        pendingLock.unlock()
        guard let ingressTimeSec else { return nil }
        return max(0, (observedAtSec - ingressTimeSec) * 1000.0)
    }

    private func metricsSnapshot(inferRTFx: Double, wallRTFx: Double) -> MetricsSnapshot {
        MetricsSnapshot(
            inferRTFx: inferRTFx,
            wallRTFx: wallRTFx,
            draftReady: Self.latencyStats(
                current: currentDraftDisplayLatencyMs ?? draftDisplayLatencyMsSamples.last,
                samples: draftDisplayLatencyMsSamples
            ),
            confirmedReady: Self.latencyStats(
                current: currentConfirmedDisplayLatencyMs ?? confirmedDisplayLatencyMsSamples.last,
                samples: confirmedDisplayLatencyMsSamples
            ),
            draftOnset: Self.latencyStats(
                current: currentSpeechDraftLatencyMs ?? draftLatencyMsSamples.last,
                samples: draftLatencyMsSamples
            ),
            confirmedOnset: Self.latencyStats(
                current: currentSpeechConfirmedLatencyMs ?? confirmedLatencyMsSamples.last,
                samples: confirmedLatencyMsSamples
            )
        )
    }

    private static func latencyStats(current: Double?, samples: [Double]) -> LatencyStats {
        LatencyStats(
            current: current,
            avg: average(samples),
            p95: percentile(samples, q: 0.95)
        )
    }

    private func metricsLine(snapshot: MetricsSnapshot) -> String {
        let draftScreen = Self.latencyStats(
            current: currentDraftScreenLatencyMs ?? draftScreenLatencyMsSamples.last,
            samples: draftScreenLatencyMsSamples
        )
        let confirmedScreen = Self.latencyStats(
            current: currentConfirmedScreenLatencyMs ?? confirmedScreenLatencyMsSamples.last,
            samples: confirmedScreenLatencyMsSamples
        )

        return String(
            format: "RTFx(inf/live)=%.1f/%.1f\ningest->ready(d/c) %@ / %@\ningest->screen(d/c) %@ / %@\nonset(d/c) %@ / %@",
            snapshot.inferRTFx,
            snapshot.wallRTFx,
            Self.formatLatency(
                current: snapshot.draftReady.current,
                avg: snapshot.draftReady.avg,
                p95: snapshot.draftReady.p95
            ),
            Self.formatLatency(
                current: snapshot.confirmedReady.current,
                avg: snapshot.confirmedReady.avg,
                p95: snapshot.confirmedReady.p95
            ),
            Self.formatLatency(
                current: draftScreen.current,
                avg: draftScreen.avg,
                p95: draftScreen.p95
            ),
            Self.formatLatency(
                current: confirmedScreen.current,
                avg: confirmedScreen.avg,
                p95: confirmedScreen.p95
            ),
            Self.formatLatency(
                current: snapshot.draftOnset.current,
                avg: snapshot.draftOnset.avg,
                p95: snapshot.draftOnset.p95
            ),
            Self.formatLatency(
                current: snapshot.confirmedOnset.current,
                avg: snapshot.confirmedOnset.avg,
                p95: snapshot.confirmedOnset.p95
            )
        )
    }

    private static func average(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        return values.reduce(0, +) / Double(values.count)
    }

    private static func formatLatency(current: Double?, avg: Double?, p95: Double?) -> String {
        if let current, let avg, let p95 {
            return String(format: "%.0fms (avg %.0f p95 %.0f)", current, avg, p95)
        }
        if let avg, let p95 {
            return String(format: "n/a (avg %.0f p95 %.0f)", avg, p95)
        }
        if let current {
            return String(format: "%.0fms", current)
        }
        return "n/a"
    }

    private func shouldProtectLatestFirstOnset() -> Bool {
        StreamingSchedulerSupport.shouldProtectLatestFirstOnset(
            elapsedSec: currentSpeechStartedAtSampleCursor.map {
                max(0.0, Double(totalProcessedSampleCursor - $0) / 16_000.0)
            },
            currentConfirmed: lastObservedConfirmedText,
            baselineConfirmed: currentSpeechBaselineConfirmed,
            minConfirmedWords: latestOnsetMinConfirmedWords,
            maxOnsetSec: latestOnsetProtectSec
        )
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
