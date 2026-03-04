import Foundation

#if canImport(CoreML)
import CoreML

public enum ParakeetCoreMLTDTError: Error {
    case sampleRateMismatch(expected: Int, actual: Int)
    case missingRequiredInput(String)
    case invalidModelIO(String)
    case missingOutput(String)
    case unsupportedArrayRank(Int)
}

public struct ParakeetCoreMLTDTConfig {
    public let expectedSampleRate: Int
    public let durations: [Int]
    public let maxSymbolsPerStep: Int
    public let maxTokensPerChunk: Int
    public let computeUnits: MLComputeUnits
    public let melBins: Int
    public let nFFT: Int
    public let windowLength: Int
    public let hopLength: Int
    public let streamingHistoryFrames: Int
    public let streamingMinTailDecodeFrames: Int

    public init(
        expectedSampleRate: Int = 16_000,
        durations: [Int] = [0, 1, 2, 3, 4],
        maxSymbolsPerStep: Int = 4,
        maxTokensPerChunk: Int = 192,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        melBins: Int = 128,
        nFFT: Int = 512,
        windowLength: Int = 400,
        hopLength: Int = 160,
        streamingHistoryFrames: Int = 300,
        streamingMinTailDecodeFrames: Int = 8
    ) {
        self.expectedSampleRate = expectedSampleRate
        self.durations = durations
        self.maxSymbolsPerStep = max(1, maxSymbolsPerStep)
        self.maxTokensPerChunk = max(0, maxTokensPerChunk)
        self.computeUnits = computeUnits
        self.melBins = melBins
        self.nFFT = nFFT
        self.windowLength = windowLength
        self.hopLength = hopLength
        self.streamingHistoryFrames = max(64, streamingHistoryFrames)
        self.streamingMinTailDecodeFrames = max(1, streamingMinTailDecodeFrames)
    }
}

public final class ParakeetCoreMLTDTTranscriptionModel: TranscriptionModel {
    private enum StreamingMode {
        case rewritePrefix
        case incremental
    }

    private let encoder: MLModel
    private let decoder: MLModel
    private let config: ParakeetCoreMLTDTConfig
    private let vocab: [String]
    private let blankID: Int

    private let encoderInputSpecs: [String: ModelInputSpec]
    private let decoderInputSpecs: [String: ModelInputSpec]
    private let encoderAudioInputName: String
    private let encoderLengthInputName: String?
    private let encoderLengthDataType: MLMultiArrayDataType?
    private let encoderAudioDataType: MLMultiArrayDataType
    private let encoderFrameCount: Int

    private let decoderEncoderInputName: String
    private let decoderStateInputNames: [String]
    private let useStatefulDecoderRuntime: Bool
    private var decoderLogitsOutputName: String?
    private var decoderStateOutputByInputName: [String: String]
    private let decoderEncoderInputShape: [Int]
    private let decoderEncoderInputDataType: MLMultiArrayDataType
    private let decoderFrameCount: Int
    private let decoderStateUpdateGateInputName: String?
    private let streamingFinalizeDraftEnabled: Bool
    private let streamingEmitDraftEnabled: Bool
    private let streamingDeterministicCursorEnabled: Bool
    private let streamingRollingStatelessEnabled: Bool
    private let streamingRollingDecodeFrames: Int
    private let streamingMode: StreamingMode
    private let streamingPrefixDecodeStrideSamples: Int
    private let streamingPrefixAdaptiveEnabled: Bool
    private let streamingPrefixAdaptiveTargetUtilization: Double
    private let streamingPrefixAdaptiveMinStrideSamples: Int
    private let streamingPrefixAdaptiveMaxStrideSamples: Int
    private let streamingPrefixAdaptiveEWMAAlpha: Double
    private let streamingPrefixMaxSamples: Int
    private let streamingPrefixLeftContextFrames: Int
    private let streamingPrefixRightContextFrames: Int
    private let streamingPrefixAllowRightContext: Bool

    private var state1: MLMultiArray
    private var state2: MLMultiArray
    private var encoderAudioBuffer: MLMultiArray
    private var encoderLengthBuffer: MLMultiArray?
    private var decoderEncoderBuffer: MLMultiArray
    private var decoderTargetsBuffer: MLMultiArray
    private var decoderTargetLengthBuffer: MLMultiArray
    private var decoderStateUpdateGateBuffer: MLMultiArray?
    private var decoderState: MLState?
    private var previousToken: Int
    private var committedTokenIDs: [Int]
    private var draftTokenIDs: [Int]
    private var streamingSampleHistory: [Float]
    private let streamingHistorySampleLimit: Int
    private var streamingPrefixSamples: [Float]
    private var streamingPrefixLastDecodeSampleCount: Int
    private var streamingPrefixLastText: String
    private var streamingPrefixAdaptiveDecodeSecEWMA: Double
    private var streamingPrefixAdaptiveRequiredSamples: Int
    private var streamingDecodedFrameCursor: Int
    private let decoderTraceWriter: DecoderTraceWriter?
    private var decoderTraceChunkIndex: Int
    private let debugLoggingEnabled: Bool

    private let featureExtractor: MelFeatureExtractor
    private let minNonZeroDuration: Int

    public init(
        modelDirectory: URL,
        encoderModelName: String,
        decoderModelName: String,
        vocabFileName: String = "vocab.txt",
        config: ParakeetCoreMLTDTConfig = .init(),
        progress: ((String) -> Void)? = nil
    ) throws {
        self.config = config
        self.decoderStateOutputByInputName = [:]
        self.committedTokenIDs = []
        self.draftTokenIDs = []
        self.debugLoggingEnabled = ProcessInfo.processInfo.environment["PARAKEET_SWIFT_DEBUG"] == "1"
        let tracePath = ProcessInfo.processInfo.environment["PARAKEET_DECODER_TRACE_PATH"] ?? ""
        let traceMaxEvents = Int(ProcessInfo.processInfo.environment["PARAKEET_DECODER_TRACE_MAX_EVENTS"] ?? "") ?? 200_000
        let traceReset = (ProcessInfo.processInfo.environment["PARAKEET_DECODER_TRACE_RESET"] ?? "1") != "0"
        self.decoderTraceWriter = DecoderTraceWriter(
            path: tracePath,
            maxEvents: max(1, traceMaxEvents),
            reset: traceReset
        )
        self.decoderTraceChunkIndex = 0

        let encoderURL = modelDirectory.appendingPathComponent(encoderModelName)
        let decoderURL = modelDirectory.appendingPathComponent(decoderModelName)
        let vocabURL = modelDirectory.appendingPathComponent(vocabFileName)

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = config.computeUnits
        progress?("Loading encoder CoreML model...")
        self.encoder = try Self.loadModel(at: encoderURL, configuration: mlConfig)
        progress?("Loading decoder CoreML model...")
        self.decoder = try Self.loadModel(at: decoderURL, configuration: mlConfig)

        progress?("Loading vocabulary...")
        self.vocab = try String(contentsOf: vocabURL, encoding: .utf8).split(separator: "\n").map(String.init)
        self.blankID = vocab.count

        let encoderInputs = Self.inputSpecs(model: encoder)
        let decoderInputs = Self.inputSpecs(model: decoder)
        self.encoderInputSpecs = encoderInputs
        self.decoderInputSpecs = decoderInputs
        guard let audioInput = encoderInputs["audio_signal"] ?? encoderInputs.values.first else {
            throw ParakeetCoreMLTDTError.invalidModelIO("Could not find encoder input.")
        }
        self.encoderAudioInputName = audioInput.name
        self.encoderAudioDataType = .float32
        self.encoderLengthInputName = encoderInputs["length"]?.name
        self.encoderLengthDataType = self.encoderLengthInputName.flatMap { encoderInputs[$0]?.dataType }
        self.encoderFrameCount = max(1, audioInput.shape.last ?? 1200)

        if let encoderInput = decoderInputs["encoder_outputs"] {
            self.decoderEncoderInputName = encoderInput.name
        } else if let inferred = decoderInputs.values.first(where: { $0.name.lowercased().contains("encoder") }) {
            self.decoderEncoderInputName = inferred.name
        } else {
            throw ParakeetCoreMLTDTError.invalidModelIO("Could not find decoder encoder input.")
        }

        self.decoderStateInputNames = decoderInputs.values
            .filter { $0.name.hasPrefix("input_states") }
            .map(\.name)
            .sorted()
        let statefulRuntimeEnabled = (ProcessInfo.processInfo.environment["PARAKEET_USE_STATEFUL_DECODER"] ?? "1") != "0"
        let hasCoreMLStateDescriptions: Bool
        if #available(macOS 15.0, iOS 18.0, *) {
            hasCoreMLStateDescriptions = !decoder.modelDescription.stateDescriptionsByName.isEmpty
        } else {
            hasCoreMLStateDescriptions = false
        }
        self.useStatefulDecoderRuntime = statefulRuntimeEnabled && hasCoreMLStateDescriptions
        if !useStatefulDecoderRuntime && decoderStateInputNames.count < 2 {
            throw ParakeetCoreMLTDTError.invalidModelIO(
                "Decoder missing recurrent state tensors (input_states_*) and no CoreML state descriptions detected."
            )
        }
        self.decoderFrameCount = max(1, decoderInputs[decoderEncoderInputName]?.shape.last ?? 300)
        self.decoderEncoderInputShape = decoderInputs[decoderEncoderInputName]?.shape ?? [1, 1024, decoderFrameCount]
        self.decoderEncoderInputDataType = .float32
        self.streamingFinalizeDraftEnabled = (ProcessInfo.processInfo.environment["PARAKEET_STREAM_FINALIZE_DRAFT"] ?? "0") == "1"
        self.streamingEmitDraftEnabled = (ProcessInfo.processInfo.environment["PARAKEET_STREAM_EMIT_DRAFT"] ?? "0") == "1"
        self.streamingDeterministicCursorEnabled =
            (ProcessInfo.processInfo.environment["PARAKEET_STREAM_DETERMINISTIC_CURSOR"] ?? "1") != "0"
        self.streamingRollingStatelessEnabled =
            (ProcessInfo.processInfo.environment["PARAKEET_STREAM_ROLLING_STATELESS"] ?? "0") != "0"
        self.streamingRollingDecodeFrames = max(
            config.streamingMinTailDecodeFrames,
            Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_ROLLING_DECODE_FRAMES"] ?? "") ?? 48
        )
        let modeRaw = (ProcessInfo.processInfo.environment["PARAKEET_STREAM_MODE"] ?? "rewrite-prefix").lowercased()
        self.streamingMode = modeRaw == "incremental" ? .incremental : .rewritePrefix
        let strideSec = max(
            0.01,
            Double(ProcessInfo.processInfo.environment["PARAKEET_STREAM_PREFIX_DECODE_STRIDE_SEC"] ?? "") ?? 0.25
        )
        self.streamingPrefixDecodeStrideSamples = max(1, Int(round(strideSec * Double(config.expectedSampleRate))))
        self.streamingPrefixAdaptiveEnabled =
            (ProcessInfo.processInfo.environment["PARAKEET_STREAM_PREFIX_ADAPTIVE"] ?? "1") != "0"
        self.streamingPrefixAdaptiveTargetUtilization = min(
            0.98,
            max(0.10, Double(ProcessInfo.processInfo.environment["PARAKEET_STREAM_PREFIX_TARGET_UTILIZATION"] ?? "") ?? 0.90)
        )
        let minStrideSec = max(
            0.01,
            Double(ProcessInfo.processInfo.environment["PARAKEET_STREAM_PREFIX_MIN_STRIDE_SEC"] ?? "") ?? 0.25
        )
        self.streamingPrefixAdaptiveMinStrideSamples = max(
            1,
            Int(round(minStrideSec * Double(config.expectedSampleRate)))
        )
        let maxStrideSec = max(
            minStrideSec,
            Double(ProcessInfo.processInfo.environment["PARAKEET_STREAM_PREFIX_MAX_STRIDE_SEC"] ?? "") ?? 0.75
        )
        self.streamingPrefixAdaptiveMaxStrideSamples = max(
            streamingPrefixAdaptiveMinStrideSamples,
            Int(round(maxStrideSec * Double(config.expectedSampleRate)))
        )
        self.streamingPrefixAdaptiveEWMAAlpha = min(
            1.0,
            max(0.01, Double(ProcessInfo.processInfo.environment["PARAKEET_STREAM_PREFIX_ADAPTIVE_EWMA_ALPHA"] ?? "") ?? 0.15)
        )
        let maxSec = Double(ProcessInfo.processInfo.environment["PARAKEET_STREAM_PREFIX_MAX_SEC"] ?? "") ?? 0.0
        self.streamingPrefixMaxSamples = maxSec <= 0 ? 0 : max(1, Int(round(maxSec * Double(config.expectedSampleRate))))
        self.streamingPrefixLeftContextFrames = max(
            0,
            Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_PREFIX_LEFT_CONTEXT_FRAMES"] ?? "") ?? max(300, config.streamingHistoryFrames)
        )
        self.streamingPrefixRightContextFrames = max(
            0,
            Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_PREFIX_RIGHT_CONTEXT_FRAMES"] ?? "") ?? 0
        )
        self.streamingPrefixAllowRightContext =
            (ProcessInfo.processInfo.environment["PARAKEET_STREAM_PREFIX_ALLOW_RIGHT_CONTEXT"] ?? "0") == "1"
        if let gateInput = decoderInputs["state_update_gate"] ??
            decoderInputs.values.first(where: { $0.name.lowercased().contains("state_update_gate") }) {
            self.decoderStateUpdateGateInputName = gateInput.name
        } else {
            self.decoderStateUpdateGateInputName = nil
        }

        let stateSpec1 = decoderStateInputNames.indices.contains(0) ? decoderInputs[decoderStateInputNames[0]] : nil
        let stateSpec2 = decoderStateInputNames.indices.contains(1) ? decoderInputs[decoderStateInputNames[1]] : nil
        self.state1 = try Self.makeArray(
            shape: stateSpec1?.shape ?? [2, 1, 640],
            dataType: stateSpec1?.dataType ?? .float32
        )
        self.state2 = try Self.makeArray(
            shape: stateSpec2?.shape ?? [2, 1, 640],
            dataType: stateSpec2?.dataType ?? .float32
        )
        self.encoderAudioBuffer = try Self.makeArray(
            shape: [1, config.melBins, encoderFrameCount],
            dataType: encoderAudioDataType
        )
        if let lengthType = encoderLengthDataType {
            self.encoderLengthBuffer = try Self.makeArray(shape: [1], dataType: lengthType)
        } else {
            self.encoderLengthBuffer = nil
        }
        self.decoderEncoderBuffer = try Self.makeArray(
            shape: decoderEncoderInputShape,
            dataType: decoderEncoderInputDataType
        )
        self.decoderTargetsBuffer = try Self.makeArray(
            shape: decoderInputs["targets"]?.shape ?? [1, 1],
            dataType: decoderInputs["targets"]?.dataType ?? .int32
        )
        self.decoderTargetLengthBuffer = try Self.makeArray(
            shape: decoderInputs["target_length"]?.shape ?? [1],
            dataType: decoderInputs["target_length"]?.dataType ?? .int32
        )
        if let gateName = decoderStateUpdateGateInputName, let gateSpec = decoderInputs[gateName] {
            self.decoderStateUpdateGateBuffer = try Self.makeArray(
                shape: gateSpec.shape,
                dataType: gateSpec.dataType
            )
            try Self.setScalarGate(self.decoderStateUpdateGateBuffer!, value: 0)
        } else {
            self.decoderStateUpdateGateBuffer = nil
        }
        if #available(macOS 15.0, iOS 18.0, *), useStatefulDecoderRuntime {
            self.decoderState = decoder.makeState()
        } else {
            self.decoderState = nil
        }
        try Self.setScalarInt(decoderTargetLengthBuffer, value: 1)
        self.previousToken = blankID
        self.streamingSampleHistory = []
        self.streamingHistorySampleLimit = max(
            config.windowLength + config.hopLength,
            config.streamingHistoryFrames * config.hopLength + config.windowLength
        )
        self.streamingPrefixSamples = []
        self.streamingPrefixLastDecodeSampleCount = 0
        self.streamingPrefixLastText = ""
        self.streamingPrefixAdaptiveDecodeSecEWMA = 0
        self.streamingPrefixAdaptiveRequiredSamples = self.streamingPrefixDecodeStrideSamples
        self.streamingDecodedFrameCursor = 0

        self.featureExtractor = MelFeatureExtractor(
            sampleRate: config.expectedSampleRate,
            nFFT: config.nFFT,
            windowLength: config.windowLength,
            hopLength: config.hopLength,
            melBins: config.melBins
        )
        self.minNonZeroDuration = config.durations.filter { $0 > 0 }.min() ?? 1
        if debugLoggingEnabled {
            let mode = useStatefulDecoderRuntime ? "stateful" : "stateless"
            fputs("[ParakeetSwift] decoder runtime mode=\(mode)\n", stderr)
            let streamMode = streamingMode == .rewritePrefix ? "rewrite-prefix" : "incremental"
            fputs("[ParakeetSwift] streaming mode=\(streamMode)\n", stderr)
        }
        progress?("Model ready.")
    }

    public convenience init(
        modelDirectory: URL,
        modelSuffix: String = "odmbp-approx",
        vocabFileName: String = "vocab.txt",
        config: ParakeetCoreMLTDTConfig = .init(),
        progress: ((String) -> Void)? = nil
    ) throws {
        try self.init(
            modelDirectory: modelDirectory,
            encoderModelName: "encoder-model-\(modelSuffix).mlpackage",
            decoderModelName: "decoder_joint-model-\(modelSuffix).mlpackage",
            vocabFileName: vocabFileName,
            config: config,
            progress: progress
        )
    }

    public func transcribeChunk(_ samples: [Float], sampleRate: Int) throws -> String {
        guard sampleRate == config.expectedSampleRate else {
            throw ParakeetCoreMLTDTError.sampleRateMismatch(expected: config.expectedSampleRate, actual: sampleRate)
        }
        if streamingMode == .rewritePrefix {
            return try transcribeChunkRewritePrefix(samples, sampleRate: sampleRate)
        }
        if samples.isEmpty {
            return Self.decodePieces(from: committedTokenIDs + draftTokenIDs, vocab: vocab)
        }

        // Streaming path: maintain a rolling causal context so tiny realtime chunks
        // are decoded with sufficient left context.
        streamingSampleHistory.append(contentsOf: samples)
        if streamingSampleHistory.count > streamingHistorySampleLimit {
            streamingSampleHistory.removeFirst(streamingSampleHistory.count - streamingHistorySampleLimit)
        }

        let (featureMatrix, frameCount) = featureExtractor.extract(samples: streamingSampleHistory)
        let historyFrames = min(frameCount, encoderFrameCount)
        let windowFrames = min(historyFrames, config.streamingHistoryFrames)
        let clampedFrames = max(1, windowFrames)
        let startFrame = max(0, frameCount - clampedFrames)
        let approxNewFrames = max(1, Int(round(Double(samples.count) / Double(config.hopLength))))
        if debugLoggingEnabled {
            let decodeWindowEstimate = max(config.streamingMinTailDecodeFrames * 4, approxNewFrames * 6)
            let preview = featureMatrix.prefix(10).map { String(format: "%.6f", $0) }.joined(separator: ", ")
            fputs(
                "[ParakeetSwift] feature frames=\(frameCount) window=\(clampedFrames) start=\(startFrame) decodeWindowEst=\(decodeWindowEstimate) first10=[\(preview)]\n",
                stderr
            )
        }

        try Self.fill(array: encoderAudioBuffer, with: 0)
        try Self.copyFeatureWindowToEncoderInput(
            featureMatrix: featureMatrix,
            melBins: config.melBins,
            totalFrames: frameCount,
            startFrame: startFrame,
            sourceFrames: clampedFrames,
            destination: encoderAudioBuffer,
            copyFrames: clampedFrames
        )

        var encoderFeed: [String: Any] = [encoderAudioInputName: encoderAudioBuffer]
        if let lengthName = encoderLengthInputName {
            if let lengthArray = encoderLengthBuffer {
                try Self.setScalarInt(lengthArray, value: clampedFrames)
                encoderFeed[lengthName] = lengthArray
            } else if let lengthType = encoderLengthDataType {
                let lengthArray = try Self.makeArray(shape: [1], dataType: lengthType)
                try Self.setScalarInt(lengthArray, value: clampedFrames)
                encoderFeed[lengthName] = lengthArray
            }
        }

        let encoderProvider = try MLDictionaryFeatureProvider(dictionary: encoderFeed)
        let encoderOutput = try encoder.prediction(from: encoderProvider)
        let (encoderTensor, encoderLength) = try Self.pickEncoderTensorAndLength(
            from: encoderOutput,
            debug: debugLoggingEnabled
        )
        let steps = max(0, min(encoderLength ?? clampedFrames, decoderFrameCount))
        if steps == 0 {
            return Self.decodePieces(from: committedTokenIDs + draftTokenIDs, vocab: vocab)
        }

        let minNewEncoderSteps = max(
            1,
            Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_MIN_NEW_ENCODER_STEPS"] ?? "") ?? 1
        )
        let useEncoderStepAdvance = (ProcessInfo.processInfo.environment["PARAKEET_STREAM_USE_ENCODER_STEP_ADVANCE"] ?? "0") == "1"
        let approxNewSteps: Int
        if useEncoderStepAdvance {
            // Experimental: convert newly arrived feature-frame progress into
            // encoder-step progress. Keep opt-in until streaming state/cursor
            // interactions are fully validated.
            approxNewSteps = max(
                minNewEncoderSteps,
                Int(round(Double(approxNewFrames) * Double(steps) / Double(max(1, clampedFrames))))
            )
        } else {
            // Legacy default: historically tuned in feature-frame units.
            approxNewSteps = max(minNewEncoderSteps, approxNewFrames)
        }
        let decodeWindowSteps = min(
            steps,
            max(config.streamingMinTailDecodeFrames * 4, approxNewSteps * 6)
        )

        if streamingRollingStatelessEnabled {
            // Streaming quality mode: decode a rolling tail window from a fresh decoder
            // state each hop to avoid recurrent-state drift from tiny incremental slices.
            let decodeFrames = min(steps, max(streamingRollingDecodeFrames, approxNewSteps))
            if decodeFrames > 0 {
                try Self.fill(array: decoderEncoderBuffer, with: 0)
                let copiedFrames = try Self.copyTensorWindowToDecoderInput(
                    source: encoderTensor,
                    sourceStartFrame: max(0, steps - decodeFrames),
                    sourceFrameCount: decodeFrames,
                    destination: decoderEncoderBuffer
                )
                if copiedFrames > 0 {
                    try resetDecoderContextForStreamingWindow()
                    let chunkTokenIDs = try decodeChunk(
                        decoderEncoderInput: decoderEncoderBuffer,
                        encoderSteps: copiedFrames,
                        commitState: true
                    )
                    committedTokenIDs = Self.appendWithOverlap(base: committedTokenIDs, next: chunkTokenIDs)
                }
            }
            draftTokenIDs.removeAll(keepingCapacity: true)
            return Self.decodePieces(from: committedTokenIDs, vocab: vocab)
        }

        if !streamingFinalizeDraftEnabled {
            if streamingDeterministicCursorEnabled {
                // Decode each encoder frame once using a global cursor.
                let windowGlobalEnd = max(steps, streamingDecodedFrameCursor + approxNewSteps)
                let windowGlobalStart = max(0, windowGlobalEnd - steps)
                var decodeGlobalStart = max(streamingDecodedFrameCursor, windowGlobalStart)
                decodeGlobalStart = min(decodeGlobalStart, windowGlobalEnd)
                let decodeFrames = max(0, windowGlobalEnd - decodeGlobalStart)

                if decodeFrames > 0 {
                    try Self.fill(array: decoderEncoderBuffer, with: 0)
                    let sourceStart = max(0, decodeGlobalStart - windowGlobalStart)
                    let copiedFrames = try Self.copyTensorWindowToDecoderInput(
                        source: encoderTensor,
                        sourceStartFrame: sourceStart,
                        sourceFrameCount: decodeFrames,
                        destination: decoderEncoderBuffer
                    )
                    if copiedFrames > 0 {
                        let chunkTokenIDs = try decodeChunk(
                            decoderEncoderInput: decoderEncoderBuffer,
                            encoderSteps: copiedFrames,
                            commitState: true
                        )
                        committedTokenIDs.append(contentsOf: chunkTokenIDs)
                        streamingDecodedFrameCursor = decodeGlobalStart + copiedFrames
                    } else {
                        streamingDecodedFrameCursor = windowGlobalEnd
                    }
                } else {
                    streamingDecodedFrameCursor = windowGlobalEnd
                }
            } else {
                // Legacy mode: decode recent tail and overlap-merge token chunks.
                let tailDecodeFrames = min(steps, max(config.streamingMinTailDecodeFrames, approxNewSteps))
                try Self.fill(array: decoderEncoderBuffer, with: 0)
                let copiedFrames = try Self.copyTensorWindowToDecoderInput(
                    source: encoderTensor,
                    sourceStartFrame: max(0, steps - tailDecodeFrames),
                    sourceFrameCount: tailDecodeFrames,
                    destination: decoderEncoderBuffer
                )
                if copiedFrames > 0 {
                    let chunkTokenIDs = try decodeChunk(
                        decoderEncoderInput: decoderEncoderBuffer,
                        encoderSteps: copiedFrames,
                        commitState: true
                    )
                    committedTokenIDs = Self.appendWithOverlap(base: committedTokenIDs, next: chunkTokenIDs)
                }
            }

            draftTokenIDs.removeAll(keepingCapacity: true)
            return Self.decodePieces(from: committedTokenIDs, vocab: vocab)
        }

        // Strict-causal finalized path: commit only newly advanced tail frames.
        // The previous overlap-based finalized window caused repeated commits in
        // realtime because old context frames were being re-committed.
        let finalizedSteps = min(steps, max(config.streamingMinTailDecodeFrames, approxNewSteps))
        let finalizedStart = max(0, steps - finalizedSteps)

        if finalizedSteps > 0 {
            try Self.fill(array: decoderEncoderBuffer, with: 0)
            let finalCopied = try Self.copyTensorWindowToDecoderInput(
                source: encoderTensor,
                sourceStartFrame: finalizedStart,
                sourceFrameCount: finalizedSteps,
                destination: decoderEncoderBuffer
            )
            if finalCopied > 0 {
                let finalizedTokenIDs = try decodeChunk(
                    decoderEncoderInput: decoderEncoderBuffer,
                    encoderSteps: finalCopied,
                    commitState: true
                )
                committedTokenIDs = Self.appendWithOverlap(base: committedTokenIDs, next: finalizedTokenIDs)
            }
        }

        draftTokenIDs.removeAll(keepingCapacity: true)
        if streamingEmitDraftEnabled {
            // Optional draft-only decode on a wider tail window. This remains
            // non-committing and is disabled by default until transcript APIs
            // expose separate confirmed vs hypothesis channels end-to-end.
            let draftSteps = min(steps, decodeWindowSteps)
            let draftStart = max(0, steps - draftSteps)
            if draftSteps > 0 {
                try Self.fill(array: decoderEncoderBuffer, with: 0)
                let draftCopied = try Self.copyTensorWindowToDecoderInput(
                    source: encoderTensor,
                    sourceStartFrame: draftStart,
                    sourceFrameCount: draftSteps,
                    destination: decoderEncoderBuffer
                )
                if draftCopied > 0 {
                    draftTokenIDs = try decodeChunk(
                        decoderEncoderInput: decoderEncoderBuffer,
                        encoderSteps: draftCopied,
                        commitState: false
                    )
                }
            }
        }

        if streamingEmitDraftEnabled {
            return Self.decodePieces(from: committedTokenIDs + draftTokenIDs, vocab: vocab)
        }
        return Self.decodePieces(from: committedTokenIDs, vocab: vocab)
    }

    private func transcribeChunkRewritePrefix(_ samples: [Float], sampleRate: Int) throws -> String {
        if !samples.isEmpty {
            streamingPrefixSamples.append(contentsOf: samples)
            if streamingPrefixMaxSamples > 0, streamingPrefixSamples.count > streamingPrefixMaxSamples {
                let drop = streamingPrefixSamples.count - streamingPrefixMaxSamples
                streamingPrefixSamples.removeFirst(drop)
                streamingPrefixLastDecodeSampleCount = min(streamingPrefixLastDecodeSampleCount, streamingPrefixSamples.count)
            }
        }
        if streamingPrefixSamples.isEmpty {
            return streamingPrefixLastText
        }

        let newSamples = streamingPrefixSamples.count - streamingPrefixLastDecodeSampleCount
        let requiredSamples = max(streamingPrefixDecodeStrideSamples, streamingPrefixAdaptiveRequiredSamples)
        let shouldDecode = streamingPrefixLastText.isEmpty || newSamples >= requiredSamples
        if !shouldDecode {
            return streamingPrefixLastText
        }

        // `transcribeLongform` performs a model-state reset internally.
        // Preserve stream prefix buffers around that internal reset.
        let prefixCopy = streamingPrefixSamples
        let started = CFAbsoluteTimeGetCurrent()
        let text = try transcribeLongform(
            prefixCopy,
            sampleRate: sampleRate,
            leftContextFrames: streamingPrefixLeftContextFrames,
            rightContextFrames: streamingPrefixRightContextFrames,
            allowRightContext: streamingPrefixAllowRightContext
        )
        let decodeSec = max(0, CFAbsoluteTimeGetCurrent() - started)
        streamingPrefixSamples = prefixCopy
        streamingPrefixLastDecodeSampleCount = streamingPrefixSamples.count
        streamingPrefixLastText = text
        updatePrefixAdaptiveStride(decodeSec: decodeSec)
        return text
    }

    private func updatePrefixAdaptiveStride(decodeSec: Double) {
        guard streamingPrefixAdaptiveEnabled else {
            streamingPrefixAdaptiveRequiredSamples = streamingPrefixDecodeStrideSamples
            return
        }
        guard decodeSec.isFinite, decodeSec > 0 else { return }

        if streamingPrefixAdaptiveDecodeSecEWMA <= 0 {
            streamingPrefixAdaptiveDecodeSecEWMA = decodeSec
        } else {
            let alpha = streamingPrefixAdaptiveEWMAAlpha
            streamingPrefixAdaptiveDecodeSecEWMA =
                alpha * decodeSec + (1.0 - alpha) * streamingPrefixAdaptiveDecodeSecEWMA
        }

        let required = Int(
            ceil(
                streamingPrefixAdaptiveDecodeSecEWMA * Double(config.expectedSampleRate) /
                    streamingPrefixAdaptiveTargetUtilization
            )
        )
        let clamped = min(
            streamingPrefixAdaptiveMaxStrideSamples,
            max(streamingPrefixAdaptiveMinStrideSamples, required)
        )
        streamingPrefixAdaptiveRequiredSamples = max(streamingPrefixDecodeStrideSamples, clamped)
    }

    public func transcribeLongform(
        _ samples: [Float],
        sampleRate: Int,
        leftContextFrames: Int,
        rightContextFrames: Int,
        allowRightContext: Bool = true
    ) throws -> String {
        guard sampleRate == config.expectedSampleRate else {
            throw ParakeetCoreMLTDTError.sampleRateMismatch(expected: config.expectedSampleRate, actual: sampleRate)
        }
        if samples.isEmpty { return "" }

        resetState()
        let (featureMatrix, totalFrames) = featureExtractor.extract(samples: samples)
        if totalFrames <= 0 { return "" }

        let maxContextTotal = max(0, encoderFrameCount - 1)
        let leftCtx = min(max(0, leftContextFrames), maxContextTotal)
        let configuredRightCtx = allowRightContext ? max(0, rightContextFrames) : 0
        let rightCtx = min(configuredRightCtx, max(0, maxContextTotal - leftCtx))
        let hopFrames = max(1, encoderFrameCount - leftCtx - rightCtx)

        var allTokenIDs: [Int] = []
        for centerStart in stride(from: 0, to: totalFrames, by: hopFrames) {
            let inputStart = max(0, centerStart - leftCtx)
            let actualFrames = max(0, min(encoderFrameCount, totalFrames - inputStart))
            if actualFrames <= 0 { continue }

            try Self.fill(array: encoderAudioBuffer, with: 0)
            try Self.copyFeatureWindowToEncoderInput(
                featureMatrix: featureMatrix,
                melBins: config.melBins,
                totalFrames: totalFrames,
                startFrame: inputStart,
                sourceFrames: actualFrames,
                destination: encoderAudioBuffer,
                copyFrames: actualFrames
            )

            var encoderFeed: [String: Any] = [encoderAudioInputName: encoderAudioBuffer]
            if let lengthName = encoderLengthInputName {
                if let lengthArray = encoderLengthBuffer {
                    try Self.setScalarInt(lengthArray, value: actualFrames)
                    encoderFeed[lengthName] = lengthArray
                } else if let lengthType = encoderLengthDataType {
                    let lengthArray = try Self.makeArray(shape: [1], dataType: lengthType)
                    try Self.setScalarInt(lengthArray, value: actualFrames)
                    encoderFeed[lengthName] = lengthArray
                }
            }

            let encoderProvider = try MLDictionaryFeatureProvider(dictionary: encoderFeed)
            let encoderOutput = try encoder.prediction(from: encoderProvider)
            let (rawEncoderTensor, rawEncoderLength) = try Self.pickEncoderTensorAndLength(from: encoderOutput, debug: false)
            let rawSteps = max(0, min(rawEncoderLength ?? actualFrames, decoderFrameCount))
            if rawSteps <= 0 { continue }
            var sourceStartOut = 0
            var sourceCopyFrames = rawSteps

            if (leftCtx > 0 || rightCtx > 0), actualFrames > 0 {
                let leftIn = centerStart - inputStart
                let centerIn = min(hopFrames, totalFrames - centerStart)
                let scale = Double(rawSteps) / Double(actualFrames)
                var leftOut = Int((Double(leftIn) * scale).rounded())
                var centerOut = Int((Double(centerIn) * scale).rounded())

                leftOut = max(0, min(leftOut, rawSteps))
                centerOut = max(1, centerOut)

                let endOut: Int
                if centerStart + centerIn >= totalFrames {
                    endOut = rawSteps
                } else {
                    endOut = max(leftOut, min(rawSteps, leftOut + centerOut))
                }
                sourceStartOut = leftOut
                sourceCopyFrames = max(0, endOut - leftOut)
            }

            if sourceCopyFrames <= 0 { continue }

            try Self.fill(array: decoderEncoderBuffer, with: 0)
            let encoderSteps = try Self.copyTensorWindowToDecoderInput(
                source: rawEncoderTensor,
                sourceStartFrame: sourceStartOut,
                sourceFrameCount: sourceCopyFrames,
                destination: decoderEncoderBuffer
            )
            if encoderSteps <= 0 { continue }

            let chunkTokenIDs = try decodeChunk(
                decoderEncoderInput: decoderEncoderBuffer,
                encoderSteps: encoderSteps,
                commitState: true
            )
            allTokenIDs.append(contentsOf: chunkTokenIDs)
        }

        committedTokenIDs = allTokenIDs
        draftTokenIDs.removeAll(keepingCapacity: true)
        return Self.decodePieces(from: allTokenIDs, vocab: vocab)
    }

    public func resetState() {
        if #available(macOS 15.0, iOS 18.0, *), useStatefulDecoderRuntime {
            decoderState = decoder.makeState()
        } else {
            try? Self.fill(array: state1, with: 0)
            try? Self.fill(array: state2, with: 0)
        }
        previousToken = blankID
        committedTokenIDs.removeAll(keepingCapacity: true)
        draftTokenIDs.removeAll(keepingCapacity: true)
        streamingSampleHistory.removeAll(keepingCapacity: true)
        streamingPrefixSamples.removeAll(keepingCapacity: true)
        streamingPrefixLastDecodeSampleCount = 0
        streamingPrefixLastText = ""
        streamingPrefixAdaptiveDecodeSecEWMA = 0
        streamingPrefixAdaptiveRequiredSamples = streamingPrefixDecodeStrideSamples
        streamingDecodedFrameCursor = 0
        decoderTraceChunkIndex = 0
    }

    private func resetDecoderContextForStreamingWindow() throws {
        if #available(macOS 15.0, iOS 18.0, *), useStatefulDecoderRuntime {
            decoderState = decoder.makeState()
        } else {
            try Self.fill(array: state1, with: 0)
            try Self.fill(array: state2, with: 0)
        }
        previousToken = blankID
    }

    private func decodeChunk(
        decoderEncoderInput: MLMultiArray,
        encoderSteps: Int,
        commitState: Bool
    ) throws -> [Int] {
        let targetInputName = "targets"
        let targetLengthName = "target_length"
        guard decoderInputSpecs[targetInputName] != nil else {
            throw ParakeetCoreMLTDTError.missingRequiredInput(targetInputName)
        }
        guard decoderInputSpecs[targetLengthName] != nil else {
            throw ParakeetCoreMLTDTError.missingRequiredInput(targetLengthName)
        }

        let maxTokens = maxTokensForChunk(encoderSteps: encoderSteps)
        var tokenIDs: [Int] = []
        tokenIDs.reserveCapacity(min(maxTokens, encoderSteps * 4))
        let targets = decoderTargetsBuffer
        let targetLength = decoderTargetLengthBuffer
        try Self.setScalarInt(targetLength, value: 1)
        var localPreviousToken = previousToken
        var localState1 = state1
        var localState2 = state2
        let chunkIndex = decoderTraceChunkIndex
        decoderTraceChunkIndex += 1
        var traceStepIndex = 0

        var t = 0
        while t < encoderSteps && tokenIDs.count < maxTokens {
            var symbolsAdded = 0
            var needLoop = true

            while needLoop && symbolsAdded < config.maxSymbolsPerStep && tokenIDs.count < maxTokens {
                let tBefore = t
                let prevTokenBefore = localPreviousToken
                try Self.setScalarInt(targets, value: localPreviousToken)
                var feed: [String: Any] = [
                    decoderEncoderInputName: decoderEncoderInput,
                    targetInputName: targets,
                    targetLengthName: targetLength,
                ]
                if useStatefulDecoderRuntime,
                   let gateName = decoderStateUpdateGateInputName,
                   let gateArray = decoderStateUpdateGateBuffer {
                    try Self.setScalarGate(gateArray, value: 0)
                    feed[gateName] = gateArray
                }
                if !useStatefulDecoderRuntime {
                    feed[decoderStateInputNames[0]] = localState1
                    feed[decoderStateInputNames[1]] = localState2
                }

                let provider = try MLDictionaryFeatureProvider(dictionary: feed)
                let output: MLFeatureProvider
                if #available(macOS 15.0, iOS 18.0, *), useStatefulDecoderRuntime, let state = decoderState {
                    output = try decoder.prediction(from: provider, using: state)
                } else {
                    output = try decoder.prediction(from: provider)
                }
                try inferDecoderOutputRolesIfNeeded(output: output, feed: feed)

                guard let logitsName = decoderLogitsOutputName,
                      let logits = output.featureValue(for: logitsName)?.multiArrayValue else {
                    throw ParakeetCoreMLTDTError.missingOutput("Decoder logits output not found.")
                }
                if debugLoggingEnabled && t == 0 && tokenIDs.isEmpty {
                    Self.debugLogitsCandidates(logits: logits, blankID: blankID)
                }

                let stepLogits = try Self.extractStepLogits(logits: logits, tIndex: t)
                let tokenVocabSize = blankID + 1
                if stepLogits.count < tokenVocabSize {
                    throw ParakeetCoreMLTDTError.invalidModelIO("Decoder logits smaller than vocab+blank.")
                }

                let tokenScores = Self.logSoftmax(Array(stepLogits[0..<tokenVocabSize]))
                let tokenID = Self.argmax(tokenScores)
                let durationPart = Array(stepLogits[tokenVocabSize..<stepLogits.count])
                let durationScores = durationPart.isEmpty ? [0] : Self.logSoftmax(durationPart)
                let durationIdx = Self.argmax(durationScores)
                var skip = config.durations.indices.contains(durationIdx) ? config.durations[durationIdx] : 1
                if tokenID == blankID && skip == 0 {
                    skip = minNonZeroDuration
                }
                if skip < 0 { skip = 0 }

                if tokenID != blankID {
                    tokenIDs.append(tokenID)

                    if useStatefulDecoderRuntime {
                        if commitState,
                           let gateName = decoderStateUpdateGateInputName,
                           let gateArray = decoderStateUpdateGateBuffer,
                           #available(macOS 15.0, iOS 18.0, *),
                           let state = decoderState {
                            try Self.setScalarInt(targets, value: tokenID)
                            try Self.setScalarGate(gateArray, value: 1)
                            var advanceFeed = feed
                            advanceFeed[targetInputName] = targets
                            advanceFeed[gateName] = gateArray
                            let advanceProvider = try MLDictionaryFeatureProvider(dictionary: advanceFeed)
                            _ = try decoder.prediction(from: advanceProvider, using: state)
                            try Self.setScalarGate(gateArray, value: 0)
                        }
                    } else {
                        if let stateOutput1 = decoderStateOutputByInputName[decoderStateInputNames[0]],
                           let stateOutput2 = decoderStateOutputByInputName[decoderStateInputNames[1]],
                           let next1 = output.featureValue(for: stateOutput1)?.multiArrayValue,
                           let next2 = output.featureValue(for: stateOutput2)?.multiArrayValue {
                            localState1 = next1
                            localState2 = next2
                        } else {
                            throw ParakeetCoreMLTDTError.missingOutput("Decoder recurrent state outputs not found.")
                        }
                    }

                    localPreviousToken = tokenID
                }

                if debugLoggingEnabled && tokenIDs.count <= 24 {
                    fputs(
                        "[ParakeetSwift] t=\(t) token=\(tokenID) skip=\(skip) blank=\(blankID)\n",
                        stderr
                    )
                }
                decoderTraceWriter?.write(event: [
                    "source": "swift",
                    "kind": "step",
                    "chunk_index": chunkIndex,
                    "step_index": traceStepIndex,
                    "encoder_steps": encoderSteps,
                    "t": tBefore,
                    "token_id": tokenID,
                    "duration_idx": durationIdx,
                    "skip": skip,
                    "prev_token": prevTokenBefore,
                    "emitted": tokenID != blankID,
                    "commit_state": commitState ? 1 : 0
                ])
                traceStepIndex += 1

                symbolsAdded += 1
                t += skip
                needLoop = (skip == 0)
            }

            if symbolsAdded >= config.maxSymbolsPerStep {
                t += 1
            }
        }

        if commitState {
            previousToken = localPreviousToken
            if !useStatefulDecoderRuntime {
                state1 = localState1
                state2 = localState2
            }
        }

        return tokenIDs
    }

    private func maxTokensForChunk(encoderSteps: Int) -> Int {
        // Prevent over-generation on tiny realtime hops (e.g. 8 encoder frames per decode).
        // Keep configured cap as an upper bound, but apply an adaptive cap tied to current
        // encoder steps to reduce unstable token bursts.
        let adaptiveCap = max(16, encoderSteps * 3)
        if config.maxTokensPerChunk > 0 {
            return min(config.maxTokensPerChunk, adaptiveCap)
        }
        return max(256, encoderSteps * 4)
    }

    private func inferDecoderOutputRolesIfNeeded(output: MLFeatureProvider, feed: [String: Any]) throws {
        if decoderLogitsOutputName == nil {
            var best: (name: String, rank: Int, lastDim: Int, count: Int)?
            for name in output.featureNames {
                guard let arr = output.featureValue(for: name)?.multiArrayValue else { continue }
                let shape = arr.intShape
                guard arr.isFloatLike, shape.count >= 2 else { continue }
                let candidate: (name: String, rank: Int, lastDim: Int, count: Int) = (
                    name: name,
                    rank: shape.count,
                    lastDim: shape.last ?? 0,
                    count: arr.count
                )
                if let b = best {
                    if candidate.rank > b.rank ||
                        (candidate.rank == b.rank && candidate.lastDim > b.lastDim) ||
                        (candidate.rank == b.rank && candidate.lastDim == b.lastDim && candidate.count > b.count) {
                        best = candidate
                    }
                } else {
                    best = candidate
                }
            }
            decoderLogitsOutputName = best?.name
        }

        if decoderStateOutputByInputName.isEmpty, !useStatefulDecoderRuntime {
            let stateInputs = decoderStateInputNames
            var unusedOutputStateNames: [String] = []
            let orderedOutputNames = Array(output.featureNames).sorted()
            for name in orderedOutputNames {
                guard name != decoderLogitsOutputName else { continue }
                guard let out = output.featureValue(for: name)?.multiArrayValue else { continue }
                guard out.isFloatLike, out.intShape.count == 3 else { continue }
                unusedOutputStateNames.append(name)
            }

            for inName in stateInputs {
                // Prefer explicit name-pair mapping when available.
                let preferredOutName = inName.replacingOccurrences(of: "input_", with: "output_")
                if let preferredIdx = unusedOutputStateNames.firstIndex(of: preferredOutName),
                   let inState = feed[inName] as? MLMultiArray,
                   let out = output.featureValue(for: preferredOutName)?.multiArrayValue,
                   out.intShape == inState.intShape {
                    decoderStateOutputByInputName[inName] = preferredOutName
                    unusedOutputStateNames.remove(at: preferredIdx)
                    continue
                }

                guard let inState = feed[inName] as? MLMultiArray else { continue }
                let inShape = inState.intShape
                if let matched = unusedOutputStateNames.first(where: {
                    guard let out = output.featureValue(for: $0)?.multiArrayValue else { return false }
                    return out.intShape == inShape
                }) {
                    decoderStateOutputByInputName[inName] = matched
                    unusedOutputStateNames.removeAll(where: { $0 == matched })
                }
            }

            if debugLoggingEnabled {
                let pairs = decoderStateInputNames.compactMap { inName -> String? in
                    guard let outName = decoderStateOutputByInputName[inName] else { return nil }
                    return "\(inName)->\(outName)"
                }.joined(separator: ", ")
                fputs("[ParakeetSwift] decoder logits output=\(decoderLogitsOutputName ?? "nil")\n", stderr)
                fputs("[ParakeetSwift] decoder state map: \(pairs)\n", stderr)
            }
        }
    }
}

private struct ModelInputSpec {
    let name: String
    let shape: [Int]
    let dataType: MLMultiArrayDataType
}

private final class DecoderTraceWriter {
    private let fileHandle: FileHandle
    private let maxEvents: Int
    private var eventsWritten: Int

    init?(path: String, maxEvents: Int, reset: Bool) {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let url = URL(fileURLWithPath: trimmed)
        let dir = url.deletingLastPathComponent()
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            if reset, FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)
            }
            if !FileManager.default.fileExists(atPath: url.path) {
                FileManager.default.createFile(atPath: url.path, contents: nil)
            }
            self.fileHandle = try FileHandle(forWritingTo: url)
            try self.fileHandle.seekToEnd()
            self.maxEvents = max(1, maxEvents)
            self.eventsWritten = 0
        } catch {
            return nil
        }
    }

    deinit {
        try? fileHandle.close()
    }

    func write(event: [String: Any]) {
        guard eventsWritten < maxEvents else { return }
        guard JSONSerialization.isValidJSONObject(event) else { return }
        guard let data = try? JSONSerialization.data(withJSONObject: event, options: []) else { return }
        var line = data
        line.append(0x0A)
        do {
            try fileHandle.write(contentsOf: line)
            eventsWritten += 1
        } catch {
            // Best-effort tracing; ignore write failures.
        }
    }
}

private extension ParakeetCoreMLTDTTranscriptionModel {
    static func loadModel(at url: URL, configuration: MLModelConfiguration) throws -> MLModel {
        let ext = url.pathExtension.lowercased()
        if ext == "mlmodelc" {
            return try MLModel(contentsOf: url, configuration: configuration)
        }
        if ext == "mlpackage" || ext == "mlmodel" {
            let compiledURL = try compiledModelURL(for: url)
            return try MLModel(contentsOf: compiledURL, configuration: configuration)
        }
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    static func compiledModelURL(for sourceURL: URL) throws -> URL {
        let cacheDir = sourceURL.deletingLastPathComponent().appendingPathComponent(".mlmodelc-cache", isDirectory: true)
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        let cacheName = sourceURL.lastPathComponent.replacingOccurrences(of: ".", with: "_") + ".mlmodelc"
        let cachedURL = cacheDir.appendingPathComponent(cacheName, isDirectory: true)

        if FileManager.default.fileExists(atPath: cachedURL.path), try isCachedCompiledModelCurrent(cachedURL: cachedURL, sourceURL: sourceURL) {
            return cachedURL
        }

        let compiledTempURL = try MLModel.compileModel(at: sourceURL)
        if FileManager.default.fileExists(atPath: cachedURL.path) {
            try FileManager.default.removeItem(at: cachedURL)
        }
        try FileManager.default.copyItem(at: compiledTempURL, to: cachedURL)
        return cachedURL
    }

    static func isCachedCompiledModelCurrent(cachedURL: URL, sourceURL: URL) throws -> Bool {
        let fm = FileManager.default
        let sourceAttrs = try fm.attributesOfItem(atPath: sourceURL.path)
        let cachedAttrs = try fm.attributesOfItem(atPath: cachedURL.path)
        let sourceDate = sourceAttrs[.modificationDate] as? Date ?? .distantPast
        let cachedDate = cachedAttrs[.modificationDate] as? Date ?? .distantPast
        return cachedDate >= sourceDate
    }

    static func inputSpecs(model: MLModel) -> [String: ModelInputSpec] {
        var out: [String: ModelInputSpec] = [:]
        for (name, desc) in model.modelDescription.inputDescriptionsByName {
            guard let constraint = desc.multiArrayConstraint else { continue }
            out[name] = ModelInputSpec(
                name: name,
                shape: constraint.shape.map(\.intValue),
                dataType: constraint.dataType
            )
        }
        return out
    }

    static func makeArray(shape: [Int], dataType: MLMultiArrayDataType) throws -> MLMultiArray {
        let nsShape = shape.map { NSNumber(value: $0) }
        return try MLMultiArray(shape: nsShape, dataType: dataType)
    }

    static func fill(array: MLMultiArray, with value: Float) throws {
        if value == 0 {
            let bytesPerElement: Int
            switch array.dataType {
            case .float16:
                bytesPerElement = MemoryLayout<UInt16>.stride
            case .float32:
                bytesPerElement = MemoryLayout<Float>.stride
            case .double:
                bytesPerElement = MemoryLayout<Double>.stride
            case .int32:
                bytesPerElement = MemoryLayout<Int32>.stride
            case .int8:
                bytesPerElement = MemoryLayout<Int8>.stride
            default:
                bytesPerElement = 0
            }
            if bytesPerElement > 0 {
                memset(array.dataPointer, 0, array.count * bytesPerElement)
                return
            }
        }
        for idx in 0..<array.count {
            array.setFloatValue(value, flatIndex: idx)
        }
    }

    static func setScalarInt(_ array: MLMultiArray, value: Int) throws {
        guard array.count > 0 else { return }
        switch array.dataType {
        case .int32:
            let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: 1)
            ptr[0] = Int32(value)
        default:
            array.setFloatValue(Float(value), flatIndex: 0)
        }
    }

    static func setScalarFloat(_ array: MLMultiArray, value: Float) throws {
        guard array.count > 0 else { return }
        array.setFloatValue(value, flatIndex: 0)
    }

    static func setScalarGate(_ array: MLMultiArray, value: Int) throws {
        guard array.count > 0 else { return }
        switch array.dataType {
        case .int32:
            try setScalarInt(array, value: value)
        case .int8:
            let ptr = array.dataPointer.bindMemory(to: Int8.self, capacity: 1)
            ptr[0] = Int8(max(-128, min(127, value)))
        default:
            try setScalarFloat(array, value: Float(value))
        }
    }

    static func copyFeatureWindowToEncoderInput(
        featureMatrix: [Float],
        melBins: Int,
        totalFrames: Int,
        startFrame: Int,
        sourceFrames: Int,
        destination: MLMultiArray,
        copyFrames: Int
    ) throws {
        guard destination.intShape.count == 3 else {
            throw ParakeetCoreMLTDTError.unsupportedArrayRank(destination.intShape.count)
        }
        let shape = destination.intShape
        guard shape[0] >= 1, shape[1] >= melBins, shape[2] >= copyFrames else {
            throw ParakeetCoreMLTDTError.invalidModelIO("Encoder input shape mismatch.")
        }
        guard totalFrames > 0 else { return }
        let strides = destination.intStrides
        let dstMelStride = strides[1]
        let dstFrameStride = strides[2]

        if destination.dataType == .float32 {
            let dst = destination.dataPointer.bindMemory(to: Float.self, capacity: destination.count)
            for m in 0..<melBins {
                let srcBase = m * totalFrames + startFrame
                let dstBase = m * dstMelStride
                if dstFrameStride == 1 {
                    for t in 0..<copyFrames {
                        dst[dstBase + t] = featureMatrix[srcBase + t]
                    }
                } else {
                    for t in 0..<copyFrames {
                        dst[dstBase + t * dstFrameStride] = featureMatrix[srcBase + t]
                    }
                }
            }
            return
        }

        for m in 0..<melBins {
            let base = m * totalFrames
            for t in 0..<copyFrames {
                let srcT = startFrame + t
                let value: Float
                if srcT >= 0, srcT < sourceFrames + startFrame, base + srcT < featureMatrix.count {
                    value = featureMatrix[base + srcT]
                } else {
                    value = 0
                }
                destination.setFloatValue(value, indices: [0, m, t])
            }
        }
    }

    static func pickEncoderTensorAndLength(from provider: MLFeatureProvider, debug: Bool = false) throws -> (MLMultiArray, Int?) {
        var length: Int?
        var bestName: String?
        var bestScore: (Int, Int, Int)?

        let orderedNames = Array(provider.featureNames).sorted()
        for name in orderedNames {
            guard let arr = provider.featureValue(for: name)?.multiArrayValue else { continue }
            let shape = arr.intShape
            if debug {
                fputs("[ParakeetSwift] encoder output \(name): dtype=\(arr.dataType.rawValue) shape=\(shape)\n", stderr)
            }
            if arr.isIntegerLike, shape.count == 1, arr.count == 1 {
                length = Int(arr.floatValue(flatIndex: 0))
                continue
            }
            guard arr.isFloatLike, shape.count >= 2 else { continue }
            let score = (shape.count, shape.last ?? 0, arr.count)
            if let b = bestScore {
                if score.0 > b.0 || (score.0 == b.0 && score.1 > b.1) || (score.0 == b.0 && score.1 == b.1 && score.2 > b.2) {
                    bestName = name
                    bestScore = score
                }
            } else {
                bestName = name
                bestScore = score
            }
        }

        guard let tensorName = bestName,
              let tensor = provider.featureValue(for: tensorName)?.multiArrayValue else {
            throw ParakeetCoreMLTDTError.missingOutput("Could not infer encoder tensor output.")
        }
        if debug {
            fputs("[ParakeetSwift] selected encoder tensor \(tensorName) shape=\(tensor.intShape) length=\(length.map(String.init) ?? "nil")\n", stderr)
            if tensor.intShape.count == 3, tensor.intShape[0] > 0, tensor.intShape[1] > 0 {
                let limit = min(10, tensor.intShape[2])
                let preview = (0..<limit).map { String(format: "%.6f", tensor.floatValue(indices: [0, 0, $0])) }.joined(separator: ", ")
                fputs("[ParakeetSwift] encoder[0,0,0:\(limit)] = [\(preview)]\n", stderr)
            }
        }
        return (tensor, length)
    }

    static func copyTensorWindowToDecoderInput(
        source: MLMultiArray,
        sourceStartFrame: Int,
        sourceFrameCount: Int,
        destination: MLMultiArray
    ) throws -> Int {
        let srcShape = source.intShape
        let dstShape = destination.intShape
        guard srcShape.count == 3, dstShape.count == 3 else {
            throw ParakeetCoreMLTDTError.invalidModelIO("Expected rank-3 tensors for decoder copy.")
        }
        let srcStart = max(0, min(sourceStartFrame, srcShape[2]))
        let srcAvailable = max(0, srcShape[2] - srcStart)
        let frames = max(0, min(sourceFrameCount, min(srcAvailable, dstShape[2])))
        if frames == 0 {
            return 0
        }

        let batch = min(srcShape[0], dstShape[0])
        let channels = min(srcShape[1], dstShape[1])
        if batch == 0 || channels == 0 {
            return 0
        }

        if source.dataType == .float32, destination.dataType == .float32 {
            let src = source.dataPointer.bindMemory(to: Float.self, capacity: source.count)
            let dst = destination.dataPointer.bindMemory(to: Float.self, capacity: destination.count)
            let srcStrides = source.intStrides
            let dstStrides = destination.intStrides
            for b in 0..<batch {
                for c in 0..<channels {
                    let srcBase = b * srcStrides[0] + c * srcStrides[1] + srcStart * srcStrides[2]
                    let dstBase = b * dstStrides[0] + c * dstStrides[1]
                    if srcStrides[2] == 1, dstStrides[2] == 1 {
                        dst.advanced(by: dstBase).update(from: src.advanced(by: srcBase), count: frames)
                    } else {
                        for t in 0..<frames {
                            dst[dstBase + t * dstStrides[2]] = src[srcBase + t * srcStrides[2]]
                        }
                    }
                }
            }
            return frames
        }

        for b in 0..<batch {
            for c in 0..<channels {
                for t in 0..<frames {
                    destination.setFloatValue(
                        source.floatValue(indices: [b, c, srcStart + t]),
                        indices: [b, c, t]
                    )
                }
            }
        }
        return frames
    }

    static func slice3DTensor(_ source: MLMultiArray, startFrame: Int, endFrame: Int) throws -> MLMultiArray {
        let shape = source.intShape
        guard shape.count == 3 else {
            throw ParakeetCoreMLTDTError.unsupportedArrayRank(shape.count)
        }
        let start = max(0, min(startFrame, shape[2]))
        let end = max(start, min(endFrame, shape[2]))
        let outFrames = max(0, end - start)
        let out = try makeArray(shape: [shape[0], shape[1], outFrames], dataType: source.dataType)
        if outFrames == 0 { return out }
        for b in 0..<shape[0] {
            for c in 0..<shape[1] {
                for t in 0..<outFrames {
                    out.setFloatValue(source.floatValue(indices: [b, c, start + t]), indices: [b, c, t])
                }
            }
        }
        return out
    }

    static func extractStepLogits(logits: MLMultiArray, tIndex: Int) throws -> [Float] {
        let shape = logits.intShape
        switch shape.count {
        case 4:
            guard shape[0] > 0 else { return [] }
            return try extractStepLogitsRank3(
                logits: logits,
                dims: [shape[1], shape[2], shape[3]],
                tIndex: tIndex,
                batchIndex: 0
            )
        case 3:
            return try extractStepLogitsRank3(
                logits: logits,
                dims: shape,
                tIndex: tIndex,
                batchIndex: nil
            )
        case 2:
            let t = min(max(tIndex, 0), max(0, shape[0] - 1))
            return (0..<shape[1]).map { v in logits.floatValue(indices: [t, v]) }
        case 1:
            return (0..<shape[0]).map { v in logits.floatValue(indices: [v]) }
        default:
            throw ParakeetCoreMLTDTError.unsupportedArrayRank(shape.count)
        }
    }

    static func extractStepLogitsRank3(
        logits: MLMultiArray,
        dims: [Int],
        tIndex: Int,
        batchIndex: Int?
    ) throws -> [Float] {
        guard dims.count == 3 else {
            throw ParakeetCoreMLTDTError.unsupportedArrayRank(dims.count)
        }

        // Mirror Python reference logic:
        // 1) treat largest dim as vocab axis,
        // 2) move vocab axis to the last position,
        // 3) collapse U dimension via [:, 0, :] (or [0] when A==1),
        // 4) pick time row t.
        let vocabAxis = dims.enumerated().max(by: { $0.element < $1.element })?.offset ?? 2
        let movedAxes: [Int]
        switch vocabAxis {
        case 0:
            movedAxes = [1, 2, 0]
        case 1:
            movedAxes = [0, 2, 1]
        default:
            movedAxes = [0, 1, 2]
        }

        let aAxis = movedAxes[0]
        let bAxis = movedAxes[1]
        let vAxis = movedAxes[2]

        let aSize = dims[aAxis]
        let bSize = dims[bAxis]
        let vocabSize = dims[vAxis]
        if vocabSize <= 0 {
            return []
        }

        let clampedT = max(0, tIndex)
        let aIndex: Int
        let bIndex: Int
        if aSize == 1 {
            aIndex = 0
            bIndex = min(clampedT, max(0, bSize - 1))
        } else {
            aIndex = min(clampedT, max(0, aSize - 1))
            bIndex = 0
        }

        var out: [Float] = []
        out.reserveCapacity(vocabSize)
        for v in 0..<vocabSize {
            var idx3 = [0, 0, 0]
            idx3[aAxis] = aIndex
            idx3[bAxis] = bIndex
            idx3[vAxis] = v

            if let b = batchIndex {
                out.append(logits.floatValue(indices: [b, idx3[0], idx3[1], idx3[2]]))
            } else {
                out.append(logits.floatValue(indices: [idx3[0], idx3[1], idx3[2]]))
            }
        }
        return out
    }

    static func argmax<T: Collection>(_ values: T) -> Int where T.Element == Float {
        var bestIndex = 0
        var bestValue = -Float.infinity
        var i = 0
        for value in values {
            if value > bestValue {
                bestValue = value
                bestIndex = i
            }
            i += 1
        }
        return bestIndex
    }

    static func logSoftmax(_ values: [Float]) -> [Float] {
        guard !values.isEmpty else { return [] }
        let maxValue = values.max() ?? 0
        var expSum: Double = 0
        for value in values {
            expSum += Foundation.exp(Double(value - maxValue))
        }
        let logDenom = Foundation.log(expSum)
        return values.map { value in
            Float(Double(value - maxValue) - logDenom)
        }
    }

    static func decodePieces(from tokenIDs: [Int], vocab: [String]) -> String {
        let pieces = tokenIDs.compactMap { token -> String? in
            guard token >= 0 && token < vocab.count else { return nil }
            return vocab[token]
        }
        let merged = pieces.joined()
            .replacingOccurrences(of: "▁", with: " ")
        return merged.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    static func appendWithOverlap(base: [Int], next: [Int], maxOverlap: Int = 512) -> [Int] {
        guard !base.isEmpty else { return next }
        guard !next.isEmpty else { return base }

        // Fast path: if the incoming token sequence already appears near the
        // tail, treat it as a full replay and do not append.
        if next.count >= 8 {
            let searchWindow = min(base.count, 4096)
            let tail = Array(base.suffix(searchWindow))
            if containsSubsequence(haystack: tail, needle: next) {
                return base
            }
        }

        let overlapLimit = min(maxOverlap, min(base.count, next.count))
        var overlap = 0
        if overlapLimit > 0 {
            for candidate in stride(from: overlapLimit, through: 1, by: -1) {
                let lhs = base.suffix(candidate)
                let rhs = next.prefix(candidate)
                if lhs.elementsEqual(rhs) {
                    overlap = candidate
                    break
                }
            }
        }
        if overlap > 0 {
            return base + next.dropFirst(overlap)
        }

        // Rewind-tolerant merge: if a prefix of the new sequence appears in the
        // recent tail, only append the non-overlapping suffix.
        if next.count >= 8 {
            let searchWindow = min(base.count, 4096)
            let tail = Array(base.suffix(searchWindow))
            let maxPrefix = min(next.count, 512)
            for prefixLen in stride(from: maxPrefix, through: 8, by: -1) {
                let prefix = Array(next.prefix(prefixLen))
                if containsSubsequence(haystack: tail, needle: prefix) {
                    return base + next.dropFirst(prefixLen)
                }
            }
        }

        return base + next
    }

    static func containsSubsequence(haystack: [Int], needle: [Int]) -> Bool {
        guard !needle.isEmpty else { return true }
        guard needle.count <= haystack.count else { return false }
        let limit = haystack.count - needle.count
        for i in 0...limit {
            var matched = true
            for j in 0..<needle.count where haystack[i + j] != needle[j] {
                matched = false
                break
            }
            if matched {
                return true
            }
        }
        return false
    }

    static func debugLogitsCandidates(logits: MLMultiArray, blankID: Int) {
        let shape = logits.intShape
        fputs("[ParakeetSwift] logits shape=\(shape) dtype=\(logits.dataType.rawValue)\n", stderr)
        guard shape.count == 4, shape[0] > 0 else { return }

        let axes = [1, 2, 3]
        for vocabAxis in axes {
            let remaining = axes.filter { $0 != vocabAxis }
            for tAxis in remaining {
                let uAxis = remaining.first { $0 != tAxis }!
                let vocabSize = shape[vocabAxis]
                let tokenLimit = min(vocabSize, blankID + 1)
                if tokenLimit <= 0 { continue }

                var bestToken = 0
                var bestValue = -Float.infinity
                for v in 0..<tokenLimit {
                    var idx = [0, 0, 0, 0]
                    idx[0] = 0
                    idx[vocabAxis] = v
                    idx[tAxis] = 0
                    idx[uAxis] = 0
                    let val = logits.floatValue(indices: idx)
                    if val > bestValue {
                        bestValue = val
                        bestToken = v
                    }
                }
                fputs(
                    "[ParakeetSwift] candidate vocabAxis=\(vocabAxis) tAxis=\(tAxis) uAxis=\(uAxis) -> token=\(bestToken)\n",
                    stderr
                )
            }
        }
    }
}

private extension MLMultiArray {
    var intShape: [Int] {
        shape.map(\.intValue)
    }

    var intStrides: [Int] {
        strides.map(\.intValue)
    }

    var isFloatLike: Bool {
        switch dataType {
        case .float16, .float32, .double:
            return true
        default:
            return false
        }
    }

    var isIntegerLike: Bool {
        switch dataType {
        case .int32:
            return true
        default:
            return false
        }
    }

    func flatIndex(for indices: [Int]) -> Int {
        zip(indices, intStrides).reduce(0) { $0 + $1.0 * $1.1 }
    }

    func floatValue(indices: [Int]) -> Float {
        if dataType == .float16 {
            return self[indices.map { NSNumber(value: $0) }].floatValue
        }
        return floatValue(flatIndex: flatIndex(for: indices))
    }

    func setFloatValue(_ value: Float, indices: [Int]) {
        if dataType == .float16 {
            self[indices.map { NSNumber(value: $0) }] = NSNumber(value: value)
            return
        }
        setFloatValue(value, flatIndex: flatIndex(for: indices))
    }

    func floatValue(flatIndex idx: Int) -> Float {
        switch dataType {
        case .float32:
            return dataPointer.bindMemory(to: Float.self, capacity: count)[idx]
        case .double:
            return Float(dataPointer.bindMemory(to: Double.self, capacity: count)[idx])
        case .int32:
            return Float(dataPointer.bindMemory(to: Int32.self, capacity: count)[idx])
        case .int8:
            return Float(dataPointer.bindMemory(to: Int8.self, capacity: count)[idx])
        case .float16:
            let bits = dataPointer.bindMemory(to: UInt16.self, capacity: count)[idx]
            return Self.float16ToFloat32(bits)
        @unknown default:
            return 0
        }
    }

    func setFloatValue(_ value: Float, flatIndex idx: Int) {
        switch dataType {
        case .float32:
            dataPointer.bindMemory(to: Float.self, capacity: count)[idx] = value
        case .double:
            dataPointer.bindMemory(to: Double.self, capacity: count)[idx] = Double(value)
        case .int32:
            dataPointer.bindMemory(to: Int32.self, capacity: count)[idx] = Int32(value)
        case .int8:
            dataPointer.bindMemory(to: Int8.self, capacity: count)[idx] = Int8(value)
        case .float16:
            dataPointer.bindMemory(to: UInt16.self, capacity: count)[idx] = Self.float32ToFloat16(value)
        @unknown default:
            break
        }
    }

    static func float16ToFloat32(_ bits: UInt16) -> Float {
        let sign = UInt32(bits & 0x8000) << 16
        var exponent = Int((bits & 0x7C00) >> 10)
        var mantissa = UInt32(bits & 0x03FF)

        if exponent == 0 {
            if mantissa == 0 { return Float(bitPattern: sign) }
            exponent = 1
            while (mantissa & 0x0400) == 0 {
                mantissa <<= 1
                exponent -= 1
            }
            mantissa &= 0x03FF
        } else if exponent == 0x1F {
            return Float(bitPattern: sign | 0x7F80_0000 | (mantissa << 13))
        }

        let exp32 = UInt32(exponent + (127 - 15)) << 23
        let mant32 = mantissa << 13
        return Float(bitPattern: sign | exp32 | mant32)
    }

    static func float32ToFloat16(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let sign = UInt16((bits >> 16) & 0x8000)
        let exponent = Int((bits >> 23) & 0xFF) - 127 + 15
        let mantissa = bits & 0x7F_FFFF

        if exponent <= 0 {
            if exponent < -10 { return sign }
            let mant = mantissa | 0x80_0000
            let shift = UInt32(14 - exponent)
            var halfMant = UInt16(mant >> shift)
            if (mant >> (shift - 1)) & 1 == 1 {
                halfMant &+= 1
            }
            return sign | halfMant
        } else if exponent >= 0x1F {
            return sign | 0x7C00
        } else {
            let halfExp = UInt16(exponent) << 10
            var halfMant = UInt16(mantissa >> 13)
            if (mantissa >> 12) & 1 == 1 {
                halfMant &+= 1
            }
            return sign | halfExp | halfMant
        }
    }
}

private struct MelFeatureExtractor {
    let sampleRate: Int
    let nFFT: Int
    let windowLength: Int
    let hopLength: Int
    let melBins: Int

    private let window: [Float]
    private let melFilters: [[Float]]
    private let cosTable: [[Float]]
    private let sinTable: [[Float]]
    private let spectrumBins: Int

    init(sampleRate: Int, nFFT: Int, windowLength: Int, hopLength: Int, melBins: Int) {
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.windowLength = windowLength
        self.hopLength = hopLength
        self.melBins = melBins
        self.spectrumBins = nFFT / 2 + 1

        var w: [Float] = []
        w.reserveCapacity(windowLength)
        if windowLength <= 1 {
            w = [1.0]
        } else {
            for n in 0..<windowLength {
                // Match FFT-style periodic Hann window (librosa/scipy fftbins=True).
                let value = 0.5 - 0.5 * cos(2.0 * Float.pi * Float(n) / Float(windowLength))
                w.append(value)
            }
        }
        self.window = w
        self.melFilters = Self.buildMelFilters(
            sampleRate: sampleRate,
            nFFT: nFFT,
            melBins: melBins,
            lowHz: 0,
            highHz: Float(sampleRate) / 2
        )

        var cosMatrix: [[Float]] = Array(repeating: Array(repeating: 0, count: nFFT), count: spectrumBins)
        var sinMatrix: [[Float]] = Array(repeating: Array(repeating: 0, count: nFFT), count: spectrumBins)
        for k in 0..<spectrumBins {
            for n in 0..<nFFT {
                let phase = 2 * Float.pi * Float(k * n) / Float(nFFT)
                cosMatrix[k][n] = cos(phase)
                sinMatrix[k][n] = sin(phase)
            }
        }
        self.cosTable = cosMatrix
        self.sinTable = sinMatrix
    }

    func extract(samples: [Float]) -> ([Float], Int) {
        if samples.isEmpty {
            return (Array(repeating: 0, count: melBins), 1)
        }
        let centerPad = nFFT / 2
        var padded = Array(repeating: Float(0), count: centerPad + samples.count + centerPad)
        for i in 0..<samples.count {
            padded[centerPad + i] = samples[i]
        }

        let frameCount: Int
        if padded.count <= nFFT {
            frameCount = 1
        } else {
            frameCount = 1 + ((padded.count - nFFT) / hopLength)
        }

        var logMel = Array(repeating: Float(0), count: melBins * frameCount)
        var fftBuffer = Array(repeating: Float(0), count: nFFT)
        var powerSpectrum = Array(repeating: Float(0), count: spectrumBins)
        let windowOffset = max(0, (nFFT - windowLength) / 2)

        for frameIdx in 0..<frameCount {
            let start = frameIdx * hopLength
            for i in 0..<nFFT {
                fftBuffer[i] = 0
            }
            for i in 0..<windowLength {
                let src = start + i
                let sample = src < padded.count ? padded[src] : 0
                let dst = windowOffset + i
                if dst < nFFT {
                    fftBuffer[dst] = sample * window[i]
                }
            }

            for k in 0..<spectrumBins {
                var real: Float = 0
                var imag: Float = 0
                let cosRow = cosTable[k]
                let sinRow = sinTable[k]
                for n in 0..<nFFT {
                    let x = fftBuffer[n]
                    real += x * cosRow[n]
                    imag -= x * sinRow[n]
                }
                powerSpectrum[k] = real * real + imag * imag
            }

            for m in 0..<melBins {
                var energy: Float = 0
                let weights = melFilters[m]
                for k in 0..<spectrumBins {
                    energy += weights[k] * powerSpectrum[k]
                }
                logMel[m * frameCount + frameIdx] = log(max(energy, 1e-6))
            }
        }

        // Per-feature normalization.
        for m in 0..<melBins {
            let base = m * frameCount
            var mean: Float = 0
            for t in 0..<frameCount {
                mean += logMel[base + t]
            }
            mean /= Float(frameCount)

            var variance: Float = 0
            for t in 0..<frameCount {
                let d = logMel[base + t] - mean
                variance += d * d
            }
            variance /= Float(frameCount)
            let std = sqrt(variance) + 1e-5

            for t in 0..<frameCount {
                logMel[base + t] = (logMel[base + t] - mean) / std
            }
        }

        return (logMel, frameCount)
    }

    private static func hzToMel(_ hz: Float) -> Float {
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1000.0
        let minLogMel: Float = minLogHz / fSp
        let logStep: Float = log(6.4) / 27.0
        if hz >= minLogHz {
            return minLogMel + log(hz / minLogHz) / logStep
        }
        return hz / fSp
    }

    private static func melToHz(_ mel: Float) -> Float {
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1000.0
        let minLogMel: Float = minLogHz / fSp
        let logStep: Float = log(6.4) / 27.0
        if mel >= minLogMel {
            return minLogHz * exp(logStep * (mel - minLogMel))
        }
        return mel * fSp
    }

    private static func buildMelFilters(
        sampleRate: Int,
        nFFT: Int,
        melBins: Int,
        lowHz: Float,
        highHz: Float
    ) -> [[Float]] {
        let spectrumBins = nFFT / 2 + 1
        let lowMel = hzToMel(lowHz)
        let highMel = hzToMel(highHz)
        let melStep = (highMel - lowMel) / Float(melBins + 1)
        let melPoints = (0..<(melBins + 2)).map { lowMel + Float($0) * melStep }
        let hzPoints = melPoints.map(melToHz)
        let fftFreqs = (0..<spectrumBins).map { Float($0) * Float(sampleRate) / Float(nFFT) }

        var filters = Array(repeating: Array(repeating: Float(0), count: spectrumBins), count: melBins)
        for m in 0..<melBins {
            let leftHz = hzPoints[m]
            let centerHz = hzPoints[m + 1]
            let rightHz = hzPoints[m + 2]

            for k in 0..<spectrumBins {
                let f = fftFreqs[k]
                if f >= leftHz, f <= centerHz, centerHz > leftHz {
                    filters[m][k] = (f - leftHz) / (centerHz - leftHz)
                } else if f > centerHz, f <= rightHz, rightHz > centerHz {
                    filters[m][k] = (rightHz - f) / (rightHz - centerHz)
                }
            }
            // Slaney-style area normalization (librosa default).
            let enorm = rightHz > leftHz ? 2.0 / (rightHz - leftHz) : 0
            if enorm > 0 {
                for k in 0..<spectrumBins {
                    filters[m][k] *= enorm
                }
            }
        }
        return filters
    }
}

#endif
