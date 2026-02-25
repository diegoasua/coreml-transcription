import SwiftUI
import AVFoundation
import RealtimeTranscriptionCore

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
            }

            HStack {
                Text("Model Suffix")
                TextField("odmbp-approx", text: $vm.modelSuffix)
                    .frame(width: 220)
                    .textFieldStyle(.roundedBorder)
                Spacer()
                Button(vm.isRunning ? "Stop" : "Start") {
                    vm.toggle()
                }
                .keyboardShortcut(.space, modifiers: [])
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
    }
}

final class MicTranscriptionViewModel: ObservableObject {
    @Published var confirmedText: String = ""
    @Published var hypothesisText: String = ""
    @Published var status: String = "Idle"
    @Published var isRunning: Bool = false
    @Published var modelDirectory: String = {
        if let env = ProcessInfo.processInfo.environment["PARAKEET_COREML_MODEL_DIR"], !env.isEmpty {
            return env
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("artifacts/parakeet-tdt-0.6b-v2")
            .path
    }()
    @Published var modelSuffix: String = ProcessInfo.processInfo.environment["PARAKEET_COREML_MODEL_SUFFIX"] ?? "odmbp-approx"

    private typealias InferenceEngine = StreamingInferenceEngine<ParakeetCoreMLRNNTTranscriptionModel, EnergyVAD>
    private var inferenceEngine: InferenceEngine?

    private let queue = DispatchQueue(label: "transcribe.macos.inference")
    private let audioEngine = AVAudioEngine()
    private var audioConverter: AVAudioConverter?
    private let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!

    func toggle() {
        if isRunning {
            stop()
        } else {
            start()
        }
    }

    private func start() {
        if isRunning { return }

        do {
            let dirURL = URL(fileURLWithPath: modelDirectory)
            let model = try ParakeetCoreMLRNNTTranscriptionModel(
                modelDirectory: dirURL,
                modelSuffix: modelSuffix,
                config: .init(maxSymbolsPerStep: 4, maxTokensPerChunk: 192)
            )
            inferenceEngine = InferenceEngine(
                model: model,
                vad: EnergyVAD(),
                policy: .init(sampleRate: 16_000, chunkMs: 960, hopMs: 320),
                requiredAgreementCount: 2,
                ringBufferCapacity: 16_000 * 8
            )
        } catch {
            status = "Failed to load model: \(error)"
            return
        }

        do {
            let inputNode = audioEngine.inputNode
            let inputFormat = inputNode.inputFormat(forBus: 0)
            audioConverter = AVAudioConverter(from: inputFormat, to: targetFormat)

            inputNode.removeTap(onBus: 0)
            inputNode.installTap(onBus: 0, bufferSize: 2048, format: inputFormat) { [weak self] buffer, _ in
                self?.handleAudioBuffer(buffer)
            }

            try audioEngine.start()
            status = "Listening..."
            isRunning = true
        } catch {
            status = "Audio start failed: \(error)"
            isRunning = false
            audioEngine.stop()
        }
    }

    private func stop() {
        guard isRunning else { return }
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        isRunning = false

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

        queue.async { [weak self] in
            guard let self else { return }
            guard var engine = self.inferenceEngine else { return }
            do {
                let events = try engine.process(samples: samples)
                self.inferenceEngine = engine
                guard let latest = events.last else { return }
                Task { @MainActor in
                    self.confirmedText = latest.transcript.confirmed
                    self.hypothesisText = latest.transcript.hypothesis
                    self.status = latest.isSpeech ? "Listening..." : "Silence"
                }
            } catch {
                Task { @MainActor in
                    self.status = "Inference error: \(error)"
                }
            }
        }
    }
}
