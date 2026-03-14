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

private struct EncoderStreamingSidecar {
    let kind: String
    let inputFeatureFrames: Int
    let shiftFeatureFrames: Int
    let preEncodeCacheFrames: Int
    let validOutputSteps: Int
    let stateInputName: String
    let timeStateInputName: String?
    let stateLengthInputName: String
    let stateOutputName: String
    let timeStateOutputName: String?
    let stateLengthOutputName: String

    static func load(from modelURL: URL) -> EncoderStreamingSidecar? {
        let sidecarURL = modelURL.deletingLastPathComponent()
            .appendingPathComponent(modelURL.deletingPathExtension().lastPathComponent + "-streaming.json")
        guard let data = try? Data(contentsOf: sidecarURL),
              let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        guard let kind = raw["kind"] as? String,
              let inputFeatureFrames = raw["input_feature_frames"] as? Int,
              let shiftFeatureFrames = raw["shift_feature_frames"] as? Int,
              let preEncodeCacheFrames = raw["pre_encode_cache_frames"] as? Int,
              let validOutputSteps = raw["valid_output_steps"] as? Int,
              let stateInputName = raw["state_input_name"] as? String,
              let stateLengthInputName = raw["state_length_input_name"] as? String,
              let stateOutputName = raw["state_output_name"] as? String,
              let stateLengthOutputName = raw["state_length_output_name"] as? String else {
            return nil
        }
        let timeStateInputName = raw["time_state_input_name"] as? String
        let timeStateOutputName = raw["time_state_output_name"] as? String
        return .init(
            kind: kind,
            inputFeatureFrames: inputFeatureFrames,
            shiftFeatureFrames: shiftFeatureFrames,
            preEncodeCacheFrames: preEncodeCacheFrames,
            validOutputSteps: validOutputSteps,
            stateInputName: stateInputName,
            timeStateInputName: timeStateInputName,
            stateLengthInputName: stateLengthInputName,
            stateOutputName: stateOutputName,
            timeStateOutputName: timeStateOutputName,
            stateLengthOutputName: stateLengthOutputName
        )
    }
}

public struct ParakeetStreamingDiagnostic: Sendable {
    public struct MergeDiagnostic: Sendable {
        public let strategy: String
        public let alignmentStart: Int?
        public let matchLength: Int
        public let baseTokenCount: Int
        public let nextTokenCount: Int
        public let mergedTokenCount: Int
        public let baseTailTokenIDs: [Int]
        public let nextTokenIDs: [Int]
        public let mergedTailTokenIDs: [Int]
        public let baseTailText: String
        public let nextText: String
        public let mergedTailText: String

        public var jsonObject: [String: Any] {
            [
                "strategy": strategy,
                "alignment_start": alignmentStart as Any,
                "match_length": matchLength,
                "base_token_count": baseTokenCount,
                "next_token_count": nextTokenCount,
                "merged_token_count": mergedTokenCount,
                "base_tail_token_ids": baseTailTokenIDs,
                "next_token_ids": nextTokenIDs,
                "merged_tail_token_ids": mergedTailTokenIDs,
                "base_tail_text": baseTailText,
                "next_text": nextText,
                "merged_tail_text": mergedTailText,
            ]
        }
    }

    public let callIndex: Int
    public let inputSamples: Int
    public let historySamples: Int
    public let droppedSamples: Int
    public let frameCount: Int
    public let windowFrames: Int
    public let windowStartFrame: Int
    public let encoderSteps: Int
    public let approxNewFrames: Int
    public let approxNewSteps: Int
    public let decodeStrategy: String
    public let decodeSourceStartStep: Int
    public let decodeRequestedFrames: Int
    public let decodeCopiedFrames: Int
    public let decodedCursorBefore: Int
    public let decodedCursorAfter: Int
    public let featureExtractMs: Double
    public let encoderPrepareMs: Double
    public let encoderPredictMs: Double
    public let decodeMs: Double
    public let mergeMs: Double
    public let totalModelMs: Double
    public let committedTokenCountBefore: Int
    public let committedTokenCountAfter: Int
    public let draftTokenCountAfter: Int
    public let schedulerPendingStepBudget: Double
    public let useStatefulDecoderRuntime: Bool
    public let streamingDeterministicCursorEnabled: Bool
    public let streamingRollingStatelessEnabled: Bool
    public let streamingFinalizeDraftEnabled: Bool
    public let mergeDiagnostic: MergeDiagnostic?

    public var jsonObject: [String: Any] {
        [
            "call_index": callIndex,
            "input_samples": inputSamples,
            "history_samples": historySamples,
            "dropped_samples": droppedSamples,
            "frame_count": frameCount,
            "window_frames": windowFrames,
            "window_start_frame": windowStartFrame,
            "encoder_steps": encoderSteps,
            "approx_new_frames": approxNewFrames,
            "approx_new_steps": approxNewSteps,
            "decode_strategy": decodeStrategy,
            "decode_source_start_step": decodeSourceStartStep,
            "decode_requested_frames": decodeRequestedFrames,
            "decode_copied_frames": decodeCopiedFrames,
            "decoded_cursor_before": decodedCursorBefore,
            "decoded_cursor_after": decodedCursorAfter,
            "feature_extract_ms": featureExtractMs,
            "encoder_prepare_ms": encoderPrepareMs,
            "encoder_predict_ms": encoderPredictMs,
            "decode_ms": decodeMs,
            "merge_ms": mergeMs,
            "total_model_ms": totalModelMs,
            "committed_token_count_before": committedTokenCountBefore,
            "committed_token_count_after": committedTokenCountAfter,
            "draft_token_count_after": draftTokenCountAfter,
            "scheduler_pending_step_budget": schedulerPendingStepBudget,
            "use_stateful_decoder_runtime": useStatefulDecoderRuntime,
            "streaming_deterministic_cursor_enabled": streamingDeterministicCursorEnabled,
            "streaming_rolling_stateless_enabled": streamingRollingStatelessEnabled,
            "streaming_finalize_draft_enabled": streamingFinalizeDraftEnabled,
            "merge": mergeDiagnostic?.jsonObject as Any,
        ]
    }
}

public final class ParakeetCoreMLTDTTranscriptionModel: TranscriptionModel {
    enum StreamingMode {
        case rewritePrefix
        case incremental
    }

    struct LongformWindowCacheKey: Equatable {
        let centerStart: Int
        let inputStart: Int
        let actualFrames: Int
        let centerFrames: Int
    }

    enum StatefulCommitPolicy {
        case always
        case emitPrevToken
        case emitToken
        case autoNonBlank

        init(raw: String) {
            switch raw {
            case "always":
                self = .always
            case "emit_token":
                self = .emitToken
            case "auto_nonblank":
                self = .autoNonBlank
            default:
                self = .emitPrevToken
            }
        }

        func firstPassGateValue(commitState: Bool, canAutoCommitNonBlank: Bool) -> Int {
            guard commitState else { return 0 }
            switch self {
            case .always:
                return 1
            case .autoNonBlank:
                return 2
            case .emitPrevToken:
                return canAutoCommitNonBlank ? 2 : 0
            case .emitToken:
                return 0
            }
        }

        func requiresExplicitAdvanceAfterEmit(canAutoCommitNonBlank: Bool) -> Bool {
            switch self {
            case .emitPrevToken:
                return !canAutoCommitNonBlank
            case .emitToken:
                return true
            case .always, .autoNonBlank:
                return false
            }
        }
    }

    struct LongformWindowPlan {
        let key: LongformWindowCacheKey
    }

    private struct RewritePrefixDecoderCheckpoint {
        let previousToken: Int
        let state1: MLMultiArray
        let state2: MLMultiArray
    }

    private struct RewritePrefixEncoderWindowCacheEntry {
        let key: LongformWindowCacheKey
        let encoderSteps: Int
        let decoderInputTensor: MLMultiArray
        var tokenIDs: [Int]
        var decoderCheckpoint: RewritePrefixDecoderCheckpoint?
    }

    private struct RewritePrefixEncoderWindowCache {
        var leftContextFrames: Int = -1
        var rightContextFrames: Int = -1
        var allowRightContext: Bool = false
        var encoderFrameCount: Int = -1
        var hopFrames: Int = -1
        var entries: [RewritePrefixEncoderWindowCacheEntry] = []

        mutating func reset() {
            leftContextFrames = -1
            rightContextFrames = -1
            allowRightContext = false
            encoderFrameCount = -1
            hopFrames = -1
            entries.removeAll(keepingCapacity: true)
        }

        mutating func configure(
            leftContextFrames: Int,
            rightContextFrames: Int,
            allowRightContext: Bool,
            encoderFrameCount: Int,
            hopFrames: Int
        ) {
            guard self.leftContextFrames != leftContextFrames ||
                    self.rightContextFrames != rightContextFrames ||
                    self.allowRightContext != allowRightContext ||
                    self.encoderFrameCount != encoderFrameCount ||
                    self.hopFrames != hopFrames else {
                return
            }
            reset()
            self.leftContextFrames = leftContextFrames
            self.rightContextFrames = rightContextFrames
            self.allowRightContext = allowRightContext
            self.encoderFrameCount = encoderFrameCount
            self.hopFrames = hopFrames
        }
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
    private let encoderStreamingSidecar: EncoderStreamingSidecar?

    private let decoderEncoderInputName: String
    private let decoderStateInputNames: [String]
    private let useStatefulDecoderRuntime: Bool
    private var decoderLogitsOutputName: String?
    private var decoderStateOutputByInputName: [String: String]
    private let decoderEncoderInputShape: [Int]
    private let decoderEncoderInputDataType: MLMultiArrayDataType
    private let decoderFrameCount: Int
    private let decoderStateUpdateGateInputName: String?
    private let decoderStepIndexInputName: String?
    private let statefulCommitPolicy: StatefulCommitPolicy
    private let streamingFinalizeDraftEnabled: Bool
    private let streamingEmitDraftEnabled: Bool
    private let streamingDeterministicCursorEnabled: Bool
    private let streamingRollingStatelessEnabled: Bool
    private let streamingRollingDecodeFrames: Int
    private let streamingAnchoredRollingDecodeFrames: Int
    private let streamingAnchoredRollingStepMultiplier: Int
    private let streamingRollingAnchorMinTokenCount: Int
    private let streamingRollingAnchorRequiredStablePasses: Int
    private let streamingMode: StreamingMode
    private let streamingPrefixDecodeStrideSamples: Int
    private let streamingPrefixAdaptiveEnabled: Bool
    private let streamingPrefixAdaptiveTargetUtilization: Double
    private let streamingPrefixAdaptiveMinStrideSamples: Int
    private let streamingPrefixAdaptiveMaxStrideSamples: Int
    private let streamingPrefixAdaptiveEWMAAlpha: Double
    private let streamingPrefixMaxSamples: Int
    private let streamingPrefixOnsetStrideSamples: Int
    private let streamingPrefixOnsetMaxSamples: Int
    private let streamingPrefixOnsetMinWordCount: Int
    private let rewritePrefixEncoderCacheEnabled: Bool
    private let rewritePrefixDecoderResumeOnsetMaxSamples: Int
    private let rewritePrefixDecoderResumeMinWordCount: Int
    private let streamingPrefixLeftContextFrames: Int
    private let streamingPrefixRightContextFrames: Int
    private let streamingPrefixAllowRightContext: Bool

    private var state1: MLMultiArray
    private var state2: MLMultiArray
    private var encoderAudioBuffer: MLMultiArray
    private var encoderLengthBuffer: MLMultiArray?
    private var encoderStreamingStateBuffer: MLMultiArray?
    private var encoderStreamingTimeStateBuffer: MLMultiArray?
    private var encoderStreamingStateLengthBuffer: MLMultiArray?
    private var decoderEncoderBuffer: MLMultiArray
    private var decoderTargetsBuffer: MLMultiArray
    private var decoderTargetLengthBuffer: MLMultiArray
    private var decoderStateUpdateGateBuffer: MLMultiArray?
    private var decoderStepIndexBuffer: MLMultiArray?
    private var decoderState: MLState?
    private var previousToken: Int
    private var committedTokenIDs: [Int]
    private var draftTokenIDs: [Int]
    private var streamingSampleHistory: [Float]
    private let streamingHistorySampleLimit: Int
    private var streamingPrefixSamples: [Float]
    private var streamingPrefixPendingNewSampleCount: Int
    private var streamingPrefixHasTrimmedHistory: Bool
    private var streamingPrefixLastText: String
    private var streamingPrefixAdaptiveDecodeSecEWMA: Double
    private var streamingPrefixAdaptiveRequiredSamples: Int
    private var rewritePrefixEncoderWindowCache: RewritePrefixEncoderWindowCache
    private var streamingDecodedFrameCursor: Int
    private var streamingEncoderStepScheduler: StreamingEncoderStepScheduler
    private var streamingRollingAnchorStablePassCount: Int
    private var streamingRollingAnchorActive: Bool
    private var streamingDiagnosticCallIndex: Int
    private var streamingFeatureCache: StreamingMelFeatureCache
    private var streamingEncoderSeenSampleCount: Int
    private var streamingEncoderProcessedFeatureFrames: Int
    private let decoderTraceWriter: DecoderTraceWriter?
    private let decoderTraceTopK: Int
    private let blankDurationTieMargin: Float
    private var decoderTraceChunkIndex: Int
    private let debugLoggingEnabled: Bool

    private let featureExtractor: MelFeatureExtractor
    private let minNonZeroDuration: Int
    public private(set) var lastStreamingDiagnostic: ParakeetStreamingDiagnostic?

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
        self.decoderTraceTopK = max(0, Int(ProcessInfo.processInfo.environment["PARAKEET_DECODER_TRACE_TOPK"] ?? "") ?? 0)
        self.blankDurationTieMargin = max(0, Float(ProcessInfo.processInfo.environment["PARAKEET_BLANK_DURATION_TIE_MARGIN"] ?? "0.0") ?? 0.0)
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
        self.encoderStreamingSidecar = EncoderStreamingSidecar.load(from: encoderURL)

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
            (ProcessInfo.processInfo.environment["PARAKEET_STREAM_ROLLING_STATELESS"] ?? "1") != "0"
        self.streamingRollingDecodeFrames = max(
            config.streamingMinTailDecodeFrames,
            Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_ROLLING_DECODE_FRAMES"] ?? "") ?? 48
        )
        self.streamingAnchoredRollingDecodeFrames = max(
            config.streamingMinTailDecodeFrames,
            Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_ANCHORED_ROLLING_DECODE_FRAMES"] ?? "") ?? 12
        )
        self.streamingAnchoredRollingStepMultiplier = max(
            1,
            Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_ANCHORED_ROLLING_STEP_MULTIPLIER"] ?? "") ?? 4
        )
        self.streamingRollingAnchorMinTokenCount = max(
            2,
            Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_ROLLING_ANCHOR_MIN_TOKENS"] ?? "") ?? 6
        )
        self.streamingRollingAnchorRequiredStablePasses = max(
            1,
            Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_ROLLING_ANCHOR_STABLE_PASSES"] ?? "") ?? 2
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
        let onsetStrideSec = Double(ProcessInfo.processInfo.environment["PARAKEET_STREAM_PREFIX_ONSET_STRIDE_SEC"] ?? "") ?? 0.0
        let resolvedOnsetStrideSec = onsetStrideSec > 0 ? onsetStrideSec : strideSec
        self.streamingPrefixOnsetStrideSamples = max(
            1,
            Int(round(resolvedOnsetStrideSec * Double(config.expectedSampleRate)))
        )
        let onsetMaxSec = Double(ProcessInfo.processInfo.environment["PARAKEET_STREAM_PREFIX_ONSET_MAX_SEC"] ?? "") ?? 0.0
        self.streamingPrefixOnsetMaxSamples = onsetMaxSec <= 0 ? 0 : max(
            1,
            Int(round(onsetMaxSec * Double(config.expectedSampleRate)))
        )
        self.streamingPrefixOnsetMinWordCount = max(
            0,
            Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_PREFIX_ONSET_MIN_WORDS"] ?? "") ?? 0
        )
        self.rewritePrefixEncoderCacheEnabled =
            (ProcessInfo.processInfo.environment["PARAKEET_STREAM_PREFIX_ENCODER_CACHE"] ?? "1") != "0"
        let resumeOnsetMaxSec = Double(ProcessInfo.processInfo.environment["PARAKEET_STREAM_PREFIX_RESUME_ONSET_MAX_SEC"] ?? "") ?? 0.0
        self.rewritePrefixDecoderResumeOnsetMaxSamples = resumeOnsetMaxSec <= 0 ? 0 : max(
            1,
            Int(round(resumeOnsetMaxSec * Double(config.expectedSampleRate)))
        )
        self.rewritePrefixDecoderResumeMinWordCount = max(
            0,
            Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_PREFIX_RESUME_MIN_WORDS"] ?? "") ?? 0
        )
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
        let statefulCommitPolicyRaw =
            (ProcessInfo.processInfo.environment["PARAKEET_STATEFUL_COMMIT_POLICY"] ?? "emit_prevtoken").lowercased()
        if let gateInput = decoderInputs["state_update_gate"] ??
            decoderInputs.values.first(where: { $0.name.lowercased().contains("state_update_gate") }) {
            self.decoderStateUpdateGateInputName = gateInput.name
        } else {
            self.decoderStateUpdateGateInputName = nil
        }
        if let stepInput = decoderInputs["step_index"] ??
            decoderInputs.values.first(where: { $0.name.lowercased().contains("step_index") }) {
            self.decoderStepIndexInputName = stepInput.name
        } else {
            self.decoderStepIndexInputName = nil
        }
        let requestedCommitPolicy = StatefulCommitPolicy(raw: statefulCommitPolicyRaw)
        if useStatefulDecoderRuntime,
           (requestedCommitPolicy == .emitPrevToken || requestedCommitPolicy == .emitToken || requestedCommitPolicy == .autoNonBlank),
           decoderStateUpdateGateInputName == nil {
            if debugLoggingEnabled {
                fputs(
                    "[ParakeetSwift] warning: stateful commit policy requires state_update_gate; falling back to always\n",
                    stderr
                )
            }
            self.statefulCommitPolicy = .always
        } else if useStatefulDecoderRuntime,
                  requestedCommitPolicy == .autoNonBlank,
                  decoderStepIndexInputName == nil {
            if debugLoggingEnabled {
                fputs(
                    "[ParakeetSwift] warning: auto_nonblank requires step_index input; falling back to always\n",
                    stderr
                )
            }
            self.statefulCommitPolicy = .always
        } else {
            self.statefulCommitPolicy = requestedCommitPolicy
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
        if let sidecar = encoderStreamingSidecar,
           let cacheSpec = encoderInputs[sidecar.stateInputName],
           let cacheLengthSpec = encoderInputs[sidecar.stateLengthInputName] {
            self.encoderStreamingStateBuffer = try Self.makeArray(
                shape: cacheSpec.shape,
                dataType: cacheSpec.dataType
            )
            self.encoderStreamingStateLengthBuffer = try Self.makeArray(
                shape: cacheLengthSpec.shape,
                dataType: cacheLengthSpec.dataType
            )
            try Self.fill(array: self.encoderStreamingStateBuffer!, with: 0)
            try Self.setScalarInt(self.encoderStreamingStateLengthBuffer!, value: 0)
            if let timeStateInputName = sidecar.timeStateInputName,
               let timeStateSpec = encoderInputs[timeStateInputName] {
                self.encoderStreamingTimeStateBuffer = try Self.makeArray(
                    shape: timeStateSpec.shape,
                    dataType: timeStateSpec.dataType
                )
                try Self.fill(array: self.encoderStreamingTimeStateBuffer!, with: 0)
            } else {
                self.encoderStreamingTimeStateBuffer = nil
            }
        } else {
            self.encoderStreamingStateBuffer = nil
            self.encoderStreamingTimeStateBuffer = nil
            self.encoderStreamingStateLengthBuffer = nil
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
        if let stepName = decoderStepIndexInputName, let stepSpec = decoderInputs[stepName] {
            self.decoderStepIndexBuffer = try Self.makeArray(
                shape: stepSpec.shape,
                dataType: stepSpec.dataType
            )
            try Self.setScalarInt(self.decoderStepIndexBuffer!, value: 0)
        } else {
            self.decoderStepIndexBuffer = nil
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
        self.streamingPrefixPendingNewSampleCount = 0
        self.streamingPrefixHasTrimmedHistory = false
        self.streamingPrefixLastText = ""
        self.streamingPrefixAdaptiveDecodeSecEWMA = 0
        self.streamingPrefixAdaptiveRequiredSamples = self.streamingPrefixDecodeStrideSamples
        self.rewritePrefixEncoderWindowCache = RewritePrefixEncoderWindowCache()
        self.streamingDecodedFrameCursor = 0
        self.streamingEncoderStepScheduler = StreamingEncoderStepScheduler()
        self.streamingRollingAnchorStablePassCount = 0
        self.streamingRollingAnchorActive = false
        self.streamingDiagnosticCallIndex = 0
        self.lastStreamingDiagnostic = nil
        self.streamingEncoderSeenSampleCount = 0
        self.streamingEncoderProcessedFeatureFrames = 0

        self.featureExtractor = MelFeatureExtractor(
            sampleRate: config.expectedSampleRate,
            nFFT: config.nFFT,
            windowLength: config.windowLength,
            hopLength: config.hopLength,
            melBins: config.melBins
        )
        self.streamingFeatureCache = StreamingMelFeatureCache(extractor: self.featureExtractor)
        self.minNonZeroDuration = config.durations.filter { $0 > 0 }.min() ?? 1
        if debugLoggingEnabled {
            let mode = useStatefulDecoderRuntime ? "stateful" : "stateless"
            fputs("[ParakeetSwift] decoder runtime mode=\(mode)\n", stderr)
            let commitPolicy: String
            switch statefulCommitPolicy {
            case .always: commitPolicy = "always"
            case .emitPrevToken: commitPolicy = "emit_prevtoken"
            case .emitToken: commitPolicy = "emit_token"
            case .autoNonBlank: commitPolicy = "auto_nonblank"
            }
            fputs("[ParakeetSwift] stateful commit policy=\(commitPolicy)\n", stderr)
            let streamMode = streamingMode == .rewritePrefix ? "rewrite-prefix" : "incremental"
            fputs("[ParakeetSwift] streaming mode=\(streamMode)\n", stderr)
            if let sidecar = encoderStreamingSidecar {
                fputs(
                    "[ParakeetSwift] encoder streaming sidecar kind=\(sidecar.kind) input_frames=\(sidecar.inputFeatureFrames) shift_frames=\(sidecar.shiftFeatureFrames)\n",
                    stderr
                )
            }
        }
        progress?("Model ready.")
    }

    public convenience init(
        modelDirectory: URL,
        modelSuffix: String = "odmbp-approx",
        encoderModelSuffix: String? = nil,
        decoderModelSuffix: String? = nil,
        vocabFileName: String = "vocab.txt",
        config: ParakeetCoreMLTDTConfig = .init(),
        progress: ((String) -> Void)? = nil
    ) throws {
        let resolvedEncoderSuffix = encoderModelSuffix ?? modelSuffix
        let resolvedDecoderSuffix = decoderModelSuffix ?? modelSuffix
        try self.init(
            modelDirectory: modelDirectory,
            encoderModelName: "encoder-model-\(resolvedEncoderSuffix).mlpackage",
            decoderModelName: "decoder_joint-model-\(resolvedDecoderSuffix).mlpackage",
            vocabFileName: vocabFileName,
            config: config,
            progress: progress
        )
    }

    public func transcribeChunk(_ samples: [Float], sampleRate: Int) throws -> String {
        guard sampleRate == config.expectedSampleRate else {
            throw ParakeetCoreMLTDTError.sampleRateMismatch(expected: config.expectedSampleRate, actual: sampleRate)
        }
        if encoderStreamingSidecar != nil {
            return try autoreleasepool {
                try transcribeChunkWithStreamingEncoderCache(samples, sampleRate: sampleRate)
            }
        }
        if streamingMode == .rewritePrefix {
            return try autoreleasepool {
                try transcribeChunkRewritePrefix(samples, sampleRate: sampleRate)
            }
        }
        if samples.isEmpty {
            return Self.decodePieces(from: committedTokenIDs + draftTokenIDs, vocab: vocab)
        }
        let diagnosticCallIndex = streamingDiagnosticCallIndex
        streamingDiagnosticCallIndex += 1
        let chunkStartTime = CFAbsoluteTimeGetCurrent()

        // Streaming path: maintain a rolling causal context so tiny realtime chunks
        // are decoded with sufficient left context.
        streamingSampleHistory.append(contentsOf: samples)
        let droppedSamples = max(0, streamingSampleHistory.count - streamingHistorySampleLimit)
        if droppedSamples > 0 {
            streamingSampleHistory.removeFirst(droppedSamples)
        }

        let featureExtractStartTime = CFAbsoluteTimeGetCurrent()
        let (featureMatrix, frameCount) = streamingFeatureCache.extract(
            samples: streamingSampleHistory,
            droppedSampleCount: droppedSamples
        )
        let featureExtractMs = (CFAbsoluteTimeGetCurrent() - featureExtractStartTime) * 1000.0
        let historyFrames = min(frameCount, encoderFrameCount)
        let windowFrames = min(historyFrames, config.streamingHistoryFrames)
        let clampedFrames = max(1, windowFrames)
        let startFrame = max(0, frameCount - clampedFrames)
        let approxNewFrames = max(1, Int(round(Double(samples.count) / Double(config.hopLength))))
        let committedTokenCountBefore = committedTokenIDs.count
        let decodedCursorBefore = streamingDecodedFrameCursor
        var decodeStrategy = "none"
        var decodeSourceStartStep = 0
        var decodeRequestedFrames = 0
        var decodeCopiedFrames = 0
        var mergeDiagnostic: ParakeetStreamingDiagnostic.MergeDiagnostic?
        var encoderPrepareMs = 0.0
        var encoderPredictMs = 0.0
        var decodeMs = 0.0
        var mergeMs = 0.0
        if debugLoggingEnabled {
            let decodeWindowEstimate = max(config.streamingMinTailDecodeFrames * 4, approxNewFrames * 6)
            let preview = featureMatrix.prefix(10).map { String(format: "%.6f", $0) }.joined(separator: ", ")
            fputs(
                "[ParakeetSwift] feature frames=\(frameCount) window=\(clampedFrames) start=\(startFrame) decodeWindowEst=\(decodeWindowEstimate) first10=[\(preview)]\n",
                stderr
            )
        }

        let encoderPrepareStartTime = CFAbsoluteTimeGetCurrent()
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
        encoderPrepareMs = (CFAbsoluteTimeGetCurrent() - encoderPrepareStartTime) * 1000.0
        let encoderPredictStartTime = CFAbsoluteTimeGetCurrent()
        let encoderOutput = try encoder.prediction(from: encoderProvider)
        encoderPredictMs = (CFAbsoluteTimeGetCurrent() - encoderPredictStartTime) * 1000.0
        let (encoderTensor, encoderLength) = try Self.pickEncoderTensorAndLength(
            from: encoderOutput,
            debug: debugLoggingEnabled
        )
        let steps = max(0, min(encoderLength ?? clampedFrames, decoderFrameCount))
        if steps == 0 {
            updateStreamingDiagnostic(
                callIndex: diagnosticCallIndex,
                inputSamples: samples.count,
                historySamples: streamingSampleHistory.count,
                droppedSamples: droppedSamples,
                frameCount: frameCount,
                windowFrames: clampedFrames,
                windowStartFrame: startFrame,
                encoderSteps: steps,
                approxNewFrames: approxNewFrames,
                approxNewSteps: 0,
                decodeStrategy: "no-encoder-steps",
                decodeSourceStartStep: 0,
                decodeRequestedFrames: 0,
                decodeCopiedFrames: 0,
                decodedCursorBefore: decodedCursorBefore,
                decodedCursorAfter: streamingDecodedFrameCursor,
                featureExtractMs: featureExtractMs,
                encoderPrepareMs: encoderPrepareMs,
                encoderPredictMs: encoderPredictMs,
                decodeMs: decodeMs,
                mergeMs: mergeMs,
                totalModelMs: (CFAbsoluteTimeGetCurrent() - chunkStartTime) * 1000.0,
                committedTokenCountBefore: committedTokenCountBefore,
                committedTokenCountAfter: committedTokenIDs.count,
                draftTokenCountAfter: draftTokenIDs.count
            )
            return Self.decodePieces(from: committedTokenIDs + draftTokenIDs, vocab: vocab)
        }

        let minNewEncoderSteps = max(
            1,
            Int(ProcessInfo.processInfo.environment["PARAKEET_STREAM_MIN_NEW_ENCODER_STEPS"] ?? "") ?? 1
        )
        let useEncoderStepAdvance = (ProcessInfo.processInfo.environment["PARAKEET_STREAM_USE_ENCODER_STEP_ADVANCE"] ?? "1") != "0"
        let approxNewSteps: Int
        if useEncoderStepAdvance {
            approxNewSteps = streamingEncoderStepScheduler.consumeStepBudget(
                newFeatureFrames: approxNewFrames,
                encoderSteps: steps,
                windowFrames: clampedFrames,
                minNewSteps: minNewEncoderSteps
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
            let baseTokenIDs = committedTokenIDs
            let shouldUseAnchoredTail = streamingRollingAnchorActive || droppedSamples > 0

            func decodeRollingWindow(startStep: Int, requestedFrames: Int) throws -> ([Int], Int) {
                guard requestedFrames > 0 else { return ([], 0) }
                try Self.fill(array: decoderEncoderBuffer, with: 0)
                let copiedFrames = try Self.copyTensorWindowToDecoderInput(
                    source: encoderTensor,
                    sourceStartFrame: startStep,
                    sourceFrameCount: requestedFrames,
                    destination: decoderEncoderBuffer
                )
                guard copiedFrames > 0 else { return ([], 0) }
                try resetDecoderContextForStreamingWindow()
                let decodeStartTime = CFAbsoluteTimeGetCurrent()
                let tokenIDs = try decodeChunk(
                    decoderEncoderInput: decoderEncoderBuffer,
                    encoderSteps: copiedFrames,
                    commitState: false
                )
                decodeMs += (CFAbsoluteTimeGetCurrent() - decodeStartTime) * 1000.0
                return (tokenIDs, copiedFrames)
            }

            if shouldUseAnchoredTail {
                let anchoredFrames = Self.anchoredRollingDecodeFrameCount(
                    encoderSteps: steps,
                    approxNewSteps: approxNewSteps,
                    baseDecodeFrames: streamingAnchoredRollingDecodeFrames,
                    stepMultiplier: streamingAnchoredRollingStepMultiplier,
                    minTailFrames: config.streamingMinTailDecodeFrames
                )
                decodeStrategy = droppedSamples > 0 ? "rolling-stateless-trimmed-tail" : "rolling-stateless-anchored-tail"
                decodeSourceStartStep = max(0, steps - anchoredFrames)
                decodeRequestedFrames = anchoredFrames

                let (chunkTokenIDs, copiedFrames) = try decodeRollingWindow(
                    startStep: decodeSourceStartStep,
                    requestedFrames: decodeRequestedFrames
                )
                decodeCopiedFrames = copiedFrames

                if copiedFrames > 0 {
                    let mergeStartTime = CFAbsoluteTimeGetCurrent()
                    let mergeResult = Self.refineRollingWindowMergeResult(
                        Self._mergeWindowHypothesisWithResult(base: baseTokenIDs, next: chunkTokenIDs),
                        base: baseTokenIDs,
                        next: chunkTokenIDs,
                        vocab: vocab
                    )
                    mergeDiagnostic = makeStreamingMergeDiagnostic(
                        result: mergeResult,
                        base: baseTokenIDs,
                        next: chunkTokenIDs,
                        merged: mergeResult.merged
                    )
                    mergeMs += (CFAbsoluteTimeGetCurrent() - mergeStartTime) * 1000.0

                    if Self.shouldAcceptRollingTailMerge(
                        result: mergeResult,
                        baseCount: baseTokenIDs.count,
                        nextCount: chunkTokenIDs.count
                    ) {
                        committedTokenIDs = mergeResult.merged
                    } else {
                        // Tail decode drifted too far. Re-run the full available prefix
                        // immediately and reset anchor state instead of carrying a bad merge.
                        decodeStrategy = "rolling-stateless-full-fallback"
                        streamingRollingAnchorActive = false
                        streamingRollingAnchorStablePassCount = 0
                        decodeSourceStartStep = 0
                        decodeRequestedFrames = steps
                        let (fallbackTokenIDs, fallbackCopiedFrames) = try decodeRollingWindow(
                            startStep: 0,
                            requestedFrames: steps
                        )
                        decodeCopiedFrames = fallbackCopiedFrames
                        if fallbackCopiedFrames > 0 {
                            committedTokenIDs = fallbackTokenIDs
                        }
                    }
                }
            } else {
                decodeStrategy = "rolling-stateless-full-prefix"
                decodeSourceStartStep = 0
                decodeRequestedFrames = steps
                let (chunkTokenIDs, copiedFrames) = try decodeRollingWindow(
                    startStep: 0,
                    requestedFrames: steps
                )
                decodeCopiedFrames = copiedFrames
                if copiedFrames > 0 {
                    committedTokenIDs = chunkTokenIDs
                    if Self.shouldPromoteRollingAnchor(
                        previous: baseTokenIDs,
                        current: chunkTokenIDs,
                        minTokenCount: streamingRollingAnchorMinTokenCount
                    ) {
                        streamingRollingAnchorStablePassCount += 1
                        if streamingRollingAnchorStablePassCount >= streamingRollingAnchorRequiredStablePasses {
                            streamingRollingAnchorActive = true
                            decodeStrategy = "rolling-stateless-anchor-established"
                        }
                    } else {
                        streamingRollingAnchorStablePassCount = 0
                    }
                }
            }
            draftTokenIDs.removeAll(keepingCapacity: true)
            updateStreamingDiagnostic(
                callIndex: diagnosticCallIndex,
                inputSamples: samples.count,
                historySamples: streamingSampleHistory.count,
                droppedSamples: droppedSamples,
                frameCount: frameCount,
                windowFrames: clampedFrames,
                windowStartFrame: startFrame,
                encoderSteps: steps,
                approxNewFrames: approxNewFrames,
                approxNewSteps: approxNewSteps,
                decodeStrategy: decodeStrategy,
                decodeSourceStartStep: decodeSourceStartStep,
                decodeRequestedFrames: decodeRequestedFrames,
                decodeCopiedFrames: decodeCopiedFrames,
                decodedCursorBefore: decodedCursorBefore,
                decodedCursorAfter: streamingDecodedFrameCursor,
                featureExtractMs: featureExtractMs,
                encoderPrepareMs: encoderPrepareMs,
                encoderPredictMs: encoderPredictMs,
                decodeMs: decodeMs,
                mergeMs: mergeMs,
                totalModelMs: (CFAbsoluteTimeGetCurrent() - chunkStartTime) * 1000.0,
                committedTokenCountBefore: committedTokenCountBefore,
                committedTokenCountAfter: committedTokenIDs.count,
                draftTokenCountAfter: draftTokenIDs.count,
                mergeDiagnostic: mergeDiagnostic
            )
            return Self.decodePieces(from: committedTokenIDs, vocab: vocab)
        }

        if !streamingFinalizeDraftEnabled {
            if streamingDeterministicCursorEnabled {
                decodeStrategy = "deterministic-cursor"
                // Decode each encoder frame once using a global cursor.
                let windowGlobalEnd = max(steps, streamingDecodedFrameCursor + approxNewSteps)
                let windowGlobalStart = max(0, windowGlobalEnd - steps)
                var decodeGlobalStart = max(streamingDecodedFrameCursor, windowGlobalStart)
                decodeGlobalStart = min(decodeGlobalStart, windowGlobalEnd)
                let decodeFrames = max(0, windowGlobalEnd - decodeGlobalStart)
                decodeSourceStartStep = max(0, decodeGlobalStart - windowGlobalStart)
                decodeRequestedFrames = decodeFrames

                if decodeFrames > 0 {
                    try Self.fill(array: decoderEncoderBuffer, with: 0)
                    let copiedFrames = try Self.copyTensorWindowToDecoderInput(
                        source: encoderTensor,
                        sourceStartFrame: decodeSourceStartStep,
                        sourceFrameCount: decodeFrames,
                        destination: decoderEncoderBuffer
                    )
                    decodeCopiedFrames = copiedFrames
                    if copiedFrames > 0 {
                        let decodeStartTime = CFAbsoluteTimeGetCurrent()
                        let chunkTokenIDs = try decodeChunk(
                            decoderEncoderInput: decoderEncoderBuffer,
                            encoderSteps: copiedFrames,
                            commitState: true
                        )
                        decodeMs += (CFAbsoluteTimeGetCurrent() - decodeStartTime) * 1000.0
                        committedTokenIDs.append(contentsOf: chunkTokenIDs)
                        streamingDecodedFrameCursor = decodeGlobalStart + copiedFrames
                    } else {
                        streamingDecodedFrameCursor = windowGlobalEnd
                    }
                } else {
                    streamingDecodedFrameCursor = windowGlobalEnd
                }
            } else {
                decodeStrategy = "tail-overlap"
                // Legacy mode: decode recent tail and overlap-merge token chunks.
                let tailDecodeFrames = min(steps, max(config.streamingMinTailDecodeFrames, approxNewSteps))
                decodeSourceStartStep = max(0, steps - tailDecodeFrames)
                decodeRequestedFrames = tailDecodeFrames
                try Self.fill(array: decoderEncoderBuffer, with: 0)
                let copiedFrames = try Self.copyTensorWindowToDecoderInput(
                    source: encoderTensor,
                    sourceStartFrame: decodeSourceStartStep,
                    sourceFrameCount: tailDecodeFrames,
                    destination: decoderEncoderBuffer
                )
                decodeCopiedFrames = copiedFrames
                if copiedFrames > 0 {
                    let decodeStartTime = CFAbsoluteTimeGetCurrent()
                    let chunkTokenIDs = try decodeChunk(
                        decoderEncoderInput: decoderEncoderBuffer,
                        encoderSteps: copiedFrames,
                        commitState: true
                    )
                    decodeMs += (CFAbsoluteTimeGetCurrent() - decodeStartTime) * 1000.0
                    let mergeStartTime = CFAbsoluteTimeGetCurrent()
                    committedTokenIDs = Self.appendWithOverlap(base: committedTokenIDs, next: chunkTokenIDs)
                    mergeMs += (CFAbsoluteTimeGetCurrent() - mergeStartTime) * 1000.0
                }
            }

            draftTokenIDs.removeAll(keepingCapacity: true)
            updateStreamingDiagnostic(
                callIndex: diagnosticCallIndex,
                inputSamples: samples.count,
                historySamples: streamingSampleHistory.count,
                droppedSamples: droppedSamples,
                frameCount: frameCount,
                windowFrames: clampedFrames,
                windowStartFrame: startFrame,
                encoderSteps: steps,
                approxNewFrames: approxNewFrames,
                approxNewSteps: approxNewSteps,
                decodeStrategy: decodeStrategy,
                decodeSourceStartStep: decodeSourceStartStep,
                decodeRequestedFrames: decodeRequestedFrames,
                decodeCopiedFrames: decodeCopiedFrames,
                decodedCursorBefore: decodedCursorBefore,
                decodedCursorAfter: streamingDecodedFrameCursor,
                featureExtractMs: featureExtractMs,
                encoderPrepareMs: encoderPrepareMs,
                encoderPredictMs: encoderPredictMs,
                decodeMs: decodeMs,
                mergeMs: mergeMs,
                totalModelMs: (CFAbsoluteTimeGetCurrent() - chunkStartTime) * 1000.0,
                committedTokenCountBefore: committedTokenCountBefore,
                committedTokenCountAfter: committedTokenIDs.count,
                draftTokenCountAfter: draftTokenIDs.count
            )
            return Self.decodePieces(from: committedTokenIDs, vocab: vocab)
        }

        decodeStrategy = "finalize-tail"
        // Strict-causal finalized path: commit only newly advanced tail frames.
        // The previous overlap-based finalized window caused repeated commits in
        // realtime because old context frames were being re-committed.
        let finalizedSteps = min(steps, max(config.streamingMinTailDecodeFrames, approxNewSteps))
        let finalizedStart = max(0, steps - finalizedSteps)
        decodeSourceStartStep = finalizedStart
        decodeRequestedFrames = finalizedSteps

        if finalizedSteps > 0 {
            try Self.fill(array: decoderEncoderBuffer, with: 0)
            let finalCopied = try Self.copyTensorWindowToDecoderInput(
                source: encoderTensor,
                sourceStartFrame: finalizedStart,
                sourceFrameCount: finalizedSteps,
                destination: decoderEncoderBuffer
            )
            decodeCopiedFrames = finalCopied
            if finalCopied > 0 {
                let decodeStartTime = CFAbsoluteTimeGetCurrent()
                let finalizedTokenIDs = try decodeChunk(
                    decoderEncoderInput: decoderEncoderBuffer,
                    encoderSteps: finalCopied,
                    commitState: true
                )
                decodeMs += (CFAbsoluteTimeGetCurrent() - decodeStartTime) * 1000.0
                let mergeStartTime = CFAbsoluteTimeGetCurrent()
                committedTokenIDs = Self.appendWithOverlap(base: committedTokenIDs, next: finalizedTokenIDs)
                mergeMs += (CFAbsoluteTimeGetCurrent() - mergeStartTime) * 1000.0
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
                    let decodeStartTime = CFAbsoluteTimeGetCurrent()
                    draftTokenIDs = try decodeChunk(
                        decoderEncoderInput: decoderEncoderBuffer,
                        encoderSteps: draftCopied,
                        commitState: false
                    )
                    decodeMs += (CFAbsoluteTimeGetCurrent() - decodeStartTime) * 1000.0
                }
            }
        }

        if streamingEmitDraftEnabled {
            return Self.decodePieces(from: committedTokenIDs + draftTokenIDs, vocab: vocab)
        }
        updateStreamingDiagnostic(
            callIndex: diagnosticCallIndex,
            inputSamples: samples.count,
            historySamples: streamingSampleHistory.count,
            droppedSamples: droppedSamples,
            frameCount: frameCount,
            windowFrames: clampedFrames,
            windowStartFrame: startFrame,
            encoderSteps: steps,
            approxNewFrames: approxNewFrames,
            approxNewSteps: approxNewSteps,
            decodeStrategy: decodeStrategy,
            decodeSourceStartStep: decodeSourceStartStep,
            decodeRequestedFrames: decodeRequestedFrames,
            decodeCopiedFrames: decodeCopiedFrames,
            decodedCursorBefore: decodedCursorBefore,
            decodedCursorAfter: streamingDecodedFrameCursor,
            featureExtractMs: featureExtractMs,
            encoderPrepareMs: encoderPrepareMs,
            encoderPredictMs: encoderPredictMs,
            decodeMs: decodeMs,
            mergeMs: mergeMs,
            totalModelMs: (CFAbsoluteTimeGetCurrent() - chunkStartTime) * 1000.0,
            committedTokenCountBefore: committedTokenCountBefore,
            committedTokenCountAfter: committedTokenIDs.count,
            draftTokenCountAfter: draftTokenIDs.count
        )
        return Self.decodePieces(from: committedTokenIDs, vocab: vocab)
    }

    private func transcribeChunkWithStreamingEncoderCache(_ samples: [Float], sampleRate: Int) throws -> String {
        guard sampleRate == config.expectedSampleRate else {
            throw ParakeetCoreMLTDTError.sampleRateMismatch(expected: config.expectedSampleRate, actual: sampleRate)
        }
        guard let sidecar = encoderStreamingSidecar,
              let initialStateBuffer = encoderStreamingStateBuffer,
              let initialStateLengthBuffer = encoderStreamingStateLengthBuffer,
              encoderInputSpecs[sidecar.stateInputName] != nil,
              encoderInputSpecs[sidecar.stateLengthInputName] != nil else {
            throw ParakeetCoreMLTDTError.invalidModelIO("Encoder streaming sidecar is present but encoder cache IO is unavailable.")
        }
        var stateBuffer = initialStateBuffer
        var stateLengthBuffer = initialStateLengthBuffer
        var timeStateBuffer = encoderStreamingTimeStateBuffer
        if let timeStateInputName = sidecar.timeStateInputName {
            guard let existingTimeStateBuffer = timeStateBuffer,
                  encoderInputSpecs[timeStateInputName] != nil else {
                throw ParakeetCoreMLTDTError.invalidModelIO("Encoder streaming sidecar requires time-cache IO that is unavailable.")
            }
            timeStateBuffer = existingTimeStateBuffer
        }

        let diagnosticCallIndex = streamingDiagnosticCallIndex
        streamingDiagnosticCallIndex += 1
        let chunkStartTime = CFAbsoluteTimeGetCurrent()
        if !samples.isEmpty {
            streamingSampleHistory.append(contentsOf: samples)
            streamingEncoderSeenSampleCount += samples.count
        }
        if streamingSampleHistory.isEmpty {
            return Self.decodePieces(from: committedTokenIDs + draftTokenIDs, vocab: vocab)
        }

        let historyFrameLimit = max(sidecar.inputFeatureFrames, sidecar.inputFeatureFrames + sidecar.shiftFeatureFrames * 4)
        let historySampleLimit = historyFrameLimit * config.hopLength + config.windowLength
        let droppedSamples = max(0, streamingSampleHistory.count - historySampleLimit)
        if droppedSamples > 0 {
            streamingSampleHistory.removeFirst(droppedSamples)
        }

        let featureExtractStartTime = CFAbsoluteTimeGetCurrent()
        let (featureMatrix, frameCount) = streamingFeatureCache.extract(
            samples: streamingSampleHistory,
            droppedSampleCount: droppedSamples
        )
        let featureExtractMs = (CFAbsoluteTimeGetCurrent() - featureExtractStartTime) * 1000.0
        let totalFeatureFramesSeen = featureExtractor.frameCount(forSampleCount: streamingEncoderSeenSampleCount)
        let historyGlobalStartFrame = max(0, totalFeatureFramesSeen - frameCount)

        let committedTokenCountBefore = committedTokenIDs.count
        var encoderPrepareMs = 0.0
        var encoderPredictMs = 0.0
        var decodeMs = 0.0
        var processedAnyStep = false

        if historyGlobalStartFrame > streamingEncoderProcessedFeatureFrames {
            try Self.fill(array: stateBuffer, with: 0)
            try Self.setScalarInt(stateLengthBuffer, value: 0)
            if let timeStateBuffer {
                try Self.fill(array: timeStateBuffer, with: 0)
            }
            try resetDecoderContextForStreamingWindow()
            streamingEncoderProcessedFeatureFrames = historyGlobalStartFrame
        }

        while totalFeatureFramesSeen - streamingEncoderProcessedFeatureFrames >= sidecar.shiftFeatureFrames {
            let targetEndGlobal = streamingEncoderProcessedFeatureFrames + sidecar.shiftFeatureFrames
            let targetEndRelative = targetEndGlobal - historyGlobalStartFrame
            guard targetEndRelative > 0, targetEndRelative <= frameCount else {
                break
            }

            let copyFrames = min(sidecar.inputFeatureFrames, targetEndRelative)
            let sourceStart = max(0, targetEndRelative - copyFrames)
            let destinationStart = sidecar.inputFeatureFrames - copyFrames

            let encoderPrepareStartTime = CFAbsoluteTimeGetCurrent()
            try Self.fill(array: encoderAudioBuffer, with: 0)
            try Self.copyFeatureWindowToEncoderInputRightAligned(
                featureMatrix: featureMatrix,
                melBins: config.melBins,
                totalFrames: frameCount,
                startFrame: sourceStart,
                copyFrames: copyFrames,
                destination: encoderAudioBuffer,
                destinationStartFrame: destinationStart
            )

            var encoderFeed: [String: Any] = [encoderAudioInputName: encoderAudioBuffer]
            if let lengthName = encoderLengthInputName {
                if let lengthArray = encoderLengthBuffer {
                    try Self.setScalarInt(lengthArray, value: sidecar.inputFeatureFrames)
                    encoderFeed[lengthName] = lengthArray
                } else if let lengthType = encoderLengthDataType {
                    let lengthArray = try Self.makeArray(shape: [1], dataType: lengthType)
                    try Self.setScalarInt(lengthArray, value: sidecar.inputFeatureFrames)
                    encoderFeed[lengthName] = lengthArray
                }
            }
            encoderFeed[sidecar.stateInputName] = stateBuffer
            encoderFeed[sidecar.stateLengthInputName] = stateLengthBuffer
            if let timeStateInputName = sidecar.timeStateInputName, let timeStateBuffer {
                encoderFeed[timeStateInputName] = timeStateBuffer
            }

            let encoderProvider = try MLDictionaryFeatureProvider(dictionary: encoderFeed)
            encoderPrepareMs += (CFAbsoluteTimeGetCurrent() - encoderPrepareStartTime) * 1000.0

            let encoderPredictStartTime = CFAbsoluteTimeGetCurrent()
            let encoderOutput = try encoder.prediction(from: encoderProvider)
            encoderPredictMs += (CFAbsoluteTimeGetCurrent() - encoderPredictStartTime) * 1000.0

            let pickedOutputs = try Self.pickStreamingEncoderOutputs(
                from: encoderOutput,
                expectedStateShape: stateBuffer.intShape,
                expectedTimeStateShape: timeStateBuffer?.intShape,
                debug: debugLoggingEnabled
            )
            let encoderTensor = pickedOutputs.tensor
            let availableSteps = encoderTensor.intShape.count >= 3 ? encoderTensor.intShape[2] : 0
            let steps = max(0, min(sidecar.validOutputSteps, pickedOutputs.length ?? availableSteps, availableSteps, decoderFrameCount))

            if let nextState = encoderOutput.featureValue(for: sidecar.stateOutputName)?.multiArrayValue ?? pickedOutputs.state {
                if nextState.count == stateBuffer.count, nextState.dataType == stateBuffer.dataType {
                    try Self.copyArrayContents(source: nextState, destination: stateBuffer)
                } else {
                    stateBuffer = try Self.cloneArray(nextState)
                    encoderStreamingStateBuffer = stateBuffer
                }
            }
            if let nextStateLength = encoderOutput.featureValue(for: sidecar.stateLengthOutputName)?.multiArrayValue {
                if nextStateLength.count == stateLengthBuffer.count, nextStateLength.dataType == stateLengthBuffer.dataType {
                    try Self.copyArrayContents(source: nextStateLength, destination: stateLengthBuffer)
                } else {
                    stateLengthBuffer = try Self.cloneArray(nextStateLength)
                    encoderStreamingStateLengthBuffer = stateLengthBuffer
                }
            } else if let stateLength = pickedOutputs.stateLength {
                try Self.setScalarInt(stateLengthBuffer, value: min(max(0, stateLength), stateBuffer.intShape[2]))
            }
            if let currentTimeStateBuffer = timeStateBuffer {
                if let nextTimeState = (sidecar.timeStateOutputName.flatMap { encoderOutput.featureValue(for: $0)?.multiArrayValue })
                    ?? pickedOutputs.timeState
                {
                    if nextTimeState.count == currentTimeStateBuffer.count, nextTimeState.dataType == currentTimeStateBuffer.dataType {
                        try Self.copyArrayContents(source: nextTimeState, destination: currentTimeStateBuffer)
                    } else {
                        let clonedTimeState = try Self.cloneArray(nextTimeState)
                        timeStateBuffer = clonedTimeState
                        encoderStreamingTimeStateBuffer = clonedTimeState
                    }
                } else {
                    throw ParakeetCoreMLTDTError.invalidModelIO("Streaming encoder did not return the expected time-cache output.")
                }
            }

            if steps > 0 {
                try Self.fill(array: decoderEncoderBuffer, with: 0)
                let copiedFrames = try Self.copyTensorWindowToDecoderInput(
                    source: encoderTensor,
                    sourceStartFrame: 0,
                    sourceFrameCount: steps,
                    destination: decoderEncoderBuffer
                )
                if copiedFrames > 0 {
                    let decodeStartTime = CFAbsoluteTimeGetCurrent()
                    let chunkTokenIDs = try decodeChunk(
                        decoderEncoderInput: decoderEncoderBuffer,
                        encoderSteps: copiedFrames,
                        commitState: true
                    )
                    decodeMs += (CFAbsoluteTimeGetCurrent() - decodeStartTime) * 1000.0
                    committedTokenIDs.append(contentsOf: chunkTokenIDs)
                }
            }

            streamingEncoderProcessedFeatureFrames = targetEndGlobal
            processedAnyStep = true
        }

        draftTokenIDs.removeAll(keepingCapacity: true)
        updateStreamingDiagnostic(
            callIndex: diagnosticCallIndex,
            inputSamples: samples.count,
            historySamples: streamingSampleHistory.count,
            droppedSamples: droppedSamples,
            frameCount: frameCount,
            windowFrames: min(frameCount, sidecar.inputFeatureFrames),
            windowStartFrame: max(0, frameCount - min(frameCount, sidecar.inputFeatureFrames)),
            encoderSteps: processedAnyStep ? sidecar.validOutputSteps : 0,
            approxNewFrames: max(1, Int(round(Double(samples.count) / Double(config.hopLength)))),
            approxNewSteps: processedAnyStep ? sidecar.validOutputSteps : 0,
            decodeStrategy: sidecar.kind,
            decodeSourceStartStep: 0,
            decodeRequestedFrames: processedAnyStep ? sidecar.validOutputSteps : 0,
            decodeCopiedFrames: processedAnyStep ? sidecar.validOutputSteps : 0,
            decodedCursorBefore: 0,
            decodedCursorAfter: streamingEncoderProcessedFeatureFrames,
            featureExtractMs: featureExtractMs,
            encoderPrepareMs: encoderPrepareMs,
            encoderPredictMs: encoderPredictMs,
            decodeMs: decodeMs,
            mergeMs: 0,
            totalModelMs: (CFAbsoluteTimeGetCurrent() - chunkStartTime) * 1000.0,
            committedTokenCountBefore: committedTokenCountBefore,
            committedTokenCountAfter: committedTokenIDs.count,
            draftTokenCountAfter: draftTokenIDs.count
        )
        return Self.decodePieces(from: committedTokenIDs, vocab: vocab)
    }

    private func transcribeChunkRewritePrefix(_ samples: [Float], sampleRate: Int) throws -> String {
        if !samples.isEmpty {
            streamingPrefixSamples.append(contentsOf: samples)
            streamingPrefixPendingNewSampleCount += samples.count
            if streamingPrefixMaxSamples > 0, streamingPrefixSamples.count > streamingPrefixMaxSamples {
                let drop = streamingPrefixSamples.count - streamingPrefixMaxSamples
                streamingPrefixSamples.removeFirst(drop)
                streamingPrefixHasTrimmedHistory = true
                rewritePrefixEncoderWindowCache.reset()
            }
        }
        if streamingPrefixSamples.isEmpty {
            return streamingPrefixLastText
        }

        let requiredSamples = Self.rewritePrefixRequiredSamples(
            stableRequiredSamples: max(streamingPrefixDecodeStrideSamples, streamingPrefixAdaptiveRequiredSamples),
            onsetStrideSamples: streamingPrefixOnsetStrideSamples,
            onsetMaxSamples: streamingPrefixOnsetMaxSamples,
            onsetMinWordCount: streamingPrefixOnsetMinWordCount,
            currentText: streamingPrefixLastText,
            bufferedSamples: streamingPrefixSamples.count
        )
        let shouldDecode = Self.shouldDecodeRewritePrefix(
            hasPreviousText: !streamingPrefixLastText.isEmpty,
            pendingNewSamples: streamingPrefixPendingNewSampleCount,
            requiredSamples: requiredSamples
        )
        if !shouldDecode {
            return streamingPrefixLastText
        }

        // `transcribeLongform` performs a model-state reset internally.
        // Preserve rewrite-prefix bookkeeping around that internal reset.
        let prefixCopy = streamingPrefixSamples
        let priorMergedText = streamingPrefixLastText
        let hadTrimmedHistory = streamingPrefixHasTrimmedHistory
        let adaptiveDecodeSecEWMA = streamingPrefixAdaptiveDecodeSecEWMA
        let adaptiveRequiredSamples = streamingPrefixAdaptiveRequiredSamples
        let started = CFAbsoluteTimeGetCurrent()
        let allowDecoderResume = Self.shouldAllowRewritePrefixDecoderResume(
            currentText: streamingPrefixLastText,
            bufferedSamples: prefixCopy.count,
            onsetMaxSamples: rewritePrefixDecoderResumeOnsetMaxSamples,
            minWordCount: rewritePrefixDecoderResumeMinWordCount
        )
        let text: String
        if rewritePrefixEncoderCacheEnabled {
            text = try transcribeLongformRewritePrefixCached(
                prefixCopy,
                sampleRate: sampleRate,
                leftContextFrames: streamingPrefixLeftContextFrames,
                rightContextFrames: streamingPrefixRightContextFrames,
                allowRightContext: streamingPrefixAllowRightContext,
                allowDecoderResume: allowDecoderResume
            )
        } else {
            text = try transcribeLongform(
                prefixCopy,
                sampleRate: sampleRate,
                leftContextFrames: streamingPrefixLeftContextFrames,
                rightContextFrames: streamingPrefixRightContextFrames,
                allowRightContext: streamingPrefixAllowRightContext
            )
        }
        let decodeSec = max(0, CFAbsoluteTimeGetCurrent() - started)
        streamingPrefixSamples = prefixCopy
        streamingPrefixHasTrimmedHistory = hadTrimmedHistory
        streamingPrefixAdaptiveDecodeSecEWMA = adaptiveDecodeSecEWMA
        streamingPrefixAdaptiveRequiredSamples = adaptiveRequiredSamples
        streamingPrefixPendingNewSampleCount = 0
        streamingPrefixLastText = Self.mergeRewritePrefixTranscript(
            previous: priorMergedText,
            currentWindow: text,
            hasTrimmedHistory: hadTrimmedHistory
        )
        updatePrefixAdaptiveStride(decodeSec: decodeSec)
        return streamingPrefixLastText
    }

    private func transcribeLongformRewritePrefixCached(
        _ samples: [Float],
        sampleRate: Int,
        leftContextFrames: Int,
        rightContextFrames: Int,
        allowRightContext: Bool = true,
        allowDecoderResume: Bool
    ) throws -> String {
        guard sampleRate == config.expectedSampleRate else {
            throw ParakeetCoreMLTDTError.sampleRateMismatch(expected: config.expectedSampleRate, actual: sampleRate)
        }
        if samples.isEmpty { return "" }

        resetDecodeSequenceState()
        let (featureMatrix, totalFrames) = featureExtractor.extract(samples: samples)
        if totalFrames <= 0 { return "" }

        let maxContextTotal = max(0, encoderFrameCount - 1)
        let leftCtx = min(max(0, leftContextFrames), maxContextTotal)
        let configuredRightCtx = allowRightContext ? max(0, rightContextFrames) : 0
        let rightCtx = min(configuredRightCtx, max(0, maxContextTotal - leftCtx))
        let hopFrames = max(1, encoderFrameCount - leftCtx - rightCtx)
        let plans = Self.makeLongformWindowPlans(
            totalFrames: totalFrames,
            encoderFrameCount: encoderFrameCount,
            leftContextFrames: leftCtx,
            rightContextFrames: rightCtx,
            allowRightContext: allowRightContext
        )

        rewritePrefixEncoderWindowCache.configure(
            leftContextFrames: leftCtx,
            rightContextFrames: rightCtx,
            allowRightContext: allowRightContext,
            encoderFrameCount: encoderFrameCount,
            hopFrames: hopFrames
        )

        let reuseCount = Self.matchingLongformWindowPrefixCount(
            previous: rewritePrefixEncoderWindowCache.entries.map(\.key),
            current: plans.map(\.key)
        )
        var entries = Array(rewritePrefixEncoderWindowCache.entries.prefix(reuseCount))
        if entries.count < plans.count {
            entries.reserveCapacity(plans.count)
        }

        for plan in plans.dropFirst(reuseCount) {
            let entry = try buildRewritePrefixEncoderCacheEntry(
                plan: plan,
                featureMatrix: featureMatrix,
                totalFrames: totalFrames,
                hopFrames: hopFrames
            )
            entries.append(entry)
        }
        rewritePrefixEncoderWindowCache.entries = entries

        var allTokenIDs: [Int] = []
        var decodeStartIndex = 0
        if allowDecoderResume, !useStatefulDecoderRuntime {
            let resumableCount = Self.resumableLongformWindowPrefixCount(
                hasCheckpoints: entries.map { $0.decoderCheckpoint != nil },
                reuseCount: reuseCount
            )
            if resumableCount > 0,
               let checkpoint = entries[resumableCount - 1].decoderCheckpoint {
                try restoreRewritePrefixDecoderCheckpoint(checkpoint)
                let reusedTokenCapacity = entries[..<resumableCount].reduce(0) { $0 + $1.tokenIDs.count }
                allTokenIDs.reserveCapacity(reusedTokenCapacity + max(64, (entries.count - resumableCount) * 16))
                for idx in 0..<resumableCount {
                    allTokenIDs.append(contentsOf: entries[idx].tokenIDs)
                }
                decodeStartIndex = resumableCount
            }
        }

        for idx in decodeStartIndex..<entries.count {
            let entry = entries[idx]
            guard entry.encoderSteps > 0 else { continue }
            let chunkTokenIDs = try decodeChunk(
                decoderEncoderInput: entry.decoderInputTensor,
                encoderSteps: entry.encoderSteps,
                commitState: true
            )
            entries[idx].tokenIDs = chunkTokenIDs
            if allowDecoderResume, !useStatefulDecoderRuntime {
                entries[idx].decoderCheckpoint = try makeRewritePrefixDecoderCheckpoint()
            } else {
                entries[idx].decoderCheckpoint = nil
            }
            allTokenIDs.append(contentsOf: chunkTokenIDs)
        }
        rewritePrefixEncoderWindowCache.entries = entries

        committedTokenIDs = allTokenIDs
        draftTokenIDs.removeAll(keepingCapacity: true)
        return Self.decodePieces(from: allTokenIDs, vocab: vocab)
    }

    private func buildRewritePrefixEncoderCacheEntry(
        plan: LongformWindowPlan,
        featureMatrix: [Float],
        totalFrames: Int,
        hopFrames: Int
    ) throws -> RewritePrefixEncoderWindowCacheEntry {
        try Self.fill(array: encoderAudioBuffer, with: 0)
        try Self.copyFeatureWindowToEncoderInput(
            featureMatrix: featureMatrix,
            melBins: config.melBins,
            totalFrames: totalFrames,
            startFrame: plan.key.inputStart,
            sourceFrames: plan.key.actualFrames,
            destination: encoderAudioBuffer,
            copyFrames: plan.key.actualFrames
        )

        var encoderFeed: [String: Any] = [encoderAudioInputName: encoderAudioBuffer]
        if let lengthName = encoderLengthInputName {
            if let lengthArray = encoderLengthBuffer {
                try Self.setScalarInt(lengthArray, value: plan.key.actualFrames)
                encoderFeed[lengthName] = lengthArray
            } else if let lengthType = encoderLengthDataType {
                let lengthArray = try Self.makeArray(shape: [1], dataType: lengthType)
                try Self.setScalarInt(lengthArray, value: plan.key.actualFrames)
                encoderFeed[lengthName] = lengthArray
            }
        }

        return try autoreleasepool {
            let encoderProvider = try MLDictionaryFeatureProvider(dictionary: encoderFeed)
            let encoderOutput = try encoder.prediction(from: encoderProvider)
            let (rawEncoderTensor, rawEncoderLength) = try Self.pickEncoderTensorAndLength(from: encoderOutput, debug: false)
            let rawSteps = max(0, min(rawEncoderLength ?? plan.key.actualFrames, decoderFrameCount))
            if rawSteps <= 0 {
                return .init(
                    key: plan.key,
                    encoderSteps: 0,
                    decoderInputTensor: try Self.makeArray(shape: decoderEncoderInputShape, dataType: decoderEncoderInputDataType),
                    tokenIDs: [],
                    decoderCheckpoint: nil
                )
            }

            let (sourceStartOut, sourceCopyFrames) = Self.projectLongformWindowToEncoderSteps(
                rawSteps: rawSteps,
                inputStart: plan.key.inputStart,
                centerStart: plan.key.centerStart,
                actualFrames: plan.key.actualFrames,
                centerFrames: plan.key.centerFrames,
                hopFrames: hopFrames
            )
            let encoderSteps = max(0, min(rawSteps - sourceStartOut, sourceCopyFrames))
            let decoderInputTensor = try Self.makeArray(shape: decoderEncoderInputShape, dataType: decoderEncoderInputDataType)
            try Self.fill(array: decoderInputTensor, with: 0)
            if encoderSteps > 0 {
                _ = try Self.copyTensorWindowToDecoderInput(
                    source: rawEncoderTensor,
                    sourceStartFrame: sourceStartOut,
                    sourceFrameCount: encoderSteps,
                    destination: decoderInputTensor
                )
            }
            return .init(
                key: plan.key,
                encoderSteps: encoderSteps,
                decoderInputTensor: decoderInputTensor,
                tokenIDs: [],
                decoderCheckpoint: nil
            )
        }
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

    private func resetDecodeSequenceState() {
        if #available(macOS 15.0, iOS 18.0, *), useStatefulDecoderRuntime {
            decoderState = decoder.makeState()
        } else {
            try? Self.fill(array: state1, with: 0)
            try? Self.fill(array: state2, with: 0)
        }
        if let stateBuffer = encoderStreamingStateBuffer {
            try? Self.fill(array: stateBuffer, with: 0)
        }
        if let timeStateBuffer = encoderStreamingTimeStateBuffer {
            try? Self.fill(array: timeStateBuffer, with: 0)
        }
        if let stateLengthBuffer = encoderStreamingStateLengthBuffer {
            try? Self.setScalarInt(stateLengthBuffer, value: 0)
        }
        previousToken = blankID
        committedTokenIDs.removeAll(keepingCapacity: true)
        draftTokenIDs.removeAll(keepingCapacity: true)
        streamingSampleHistory.removeAll(keepingCapacity: true)
        streamingDecodedFrameCursor = 0
        streamingEncoderStepScheduler.reset()
        streamingRollingAnchorStablePassCount = 0
        streamingRollingAnchorActive = false
        streamingFeatureCache.reset()
        streamingEncoderSeenSampleCount = 0
        streamingEncoderProcessedFeatureFrames = 0
        lastStreamingDiagnostic = nil
        decoderTraceChunkIndex = 0
    }

    private func makeRewritePrefixDecoderCheckpoint() throws -> RewritePrefixDecoderCheckpoint {
        .init(
            previousToken: previousToken,
            state1: try Self.cloneArray(state1),
            state2: try Self.cloneArray(state2)
        )
    }

    private func restoreRewritePrefixDecoderCheckpoint(_ checkpoint: RewritePrefixDecoderCheckpoint) throws {
        if checkpoint.state1.intShape == state1.intShape, checkpoint.state1.dataType == state1.dataType {
            try Self.copyArrayContents(source: checkpoint.state1, destination: state1)
        } else {
            state1 = try Self.cloneArray(checkpoint.state1)
        }
        if checkpoint.state2.intShape == state2.intShape, checkpoint.state2.dataType == state2.dataType {
            try Self.copyArrayContents(source: checkpoint.state2, destination: state2)
        } else {
            state2 = try Self.cloneArray(checkpoint.state2)
        }
        previousToken = checkpoint.previousToken
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

        resetDecodeSequenceState()
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
                let centerIn = min(hopFrames, totalFrames - centerStart)
                let projected = Self.projectLongformWindowToEncoderSteps(
                    rawSteps: rawSteps,
                    inputStart: inputStart,
                    centerStart: centerStart,
                    actualFrames: actualFrames,
                    centerFrames: centerIn,
                    hopFrames: hopFrames
                )
                sourceStartOut = projected.start
                sourceCopyFrames = projected.count
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
        resetDecodeSequenceState()
        streamingPrefixSamples.removeAll(keepingCapacity: true)
        streamingPrefixPendingNewSampleCount = 0
        streamingPrefixHasTrimmedHistory = false
        streamingPrefixLastText = ""
        streamingPrefixAdaptiveDecodeSecEWMA = 0
        streamingPrefixAdaptiveRequiredSamples = streamingPrefixDecodeStrideSamples
        rewritePrefixEncoderWindowCache.reset()
        lastStreamingDiagnostic = nil
    }

    private func updateStreamingDiagnostic(
        callIndex: Int,
        inputSamples: Int,
        historySamples: Int,
        droppedSamples: Int,
        frameCount: Int,
        windowFrames: Int,
        windowStartFrame: Int,
        encoderSteps: Int,
        approxNewFrames: Int,
        approxNewSteps: Int,
        decodeStrategy: String,
        decodeSourceStartStep: Int,
        decodeRequestedFrames: Int,
        decodeCopiedFrames: Int,
        decodedCursorBefore: Int,
        decodedCursorAfter: Int,
        featureExtractMs: Double,
        encoderPrepareMs: Double,
        encoderPredictMs: Double,
        decodeMs: Double,
        mergeMs: Double,
        totalModelMs: Double,
        committedTokenCountBefore: Int,
        committedTokenCountAfter: Int,
        draftTokenCountAfter: Int,
        mergeDiagnostic: ParakeetStreamingDiagnostic.MergeDiagnostic? = nil
    ) {
        lastStreamingDiagnostic = ParakeetStreamingDiagnostic(
            callIndex: callIndex,
            inputSamples: inputSamples,
            historySamples: historySamples,
            droppedSamples: droppedSamples,
            frameCount: frameCount,
            windowFrames: windowFrames,
            windowStartFrame: windowStartFrame,
            encoderSteps: encoderSteps,
            approxNewFrames: approxNewFrames,
            approxNewSteps: approxNewSteps,
            decodeStrategy: decodeStrategy,
            decodeSourceStartStep: decodeSourceStartStep,
            decodeRequestedFrames: decodeRequestedFrames,
            decodeCopiedFrames: decodeCopiedFrames,
            decodedCursorBefore: decodedCursorBefore,
            decodedCursorAfter: decodedCursorAfter,
            featureExtractMs: featureExtractMs,
            encoderPrepareMs: encoderPrepareMs,
            encoderPredictMs: encoderPredictMs,
            decodeMs: decodeMs,
            mergeMs: mergeMs,
            totalModelMs: totalModelMs,
            committedTokenCountBefore: committedTokenCountBefore,
            committedTokenCountAfter: committedTokenCountAfter,
            draftTokenCountAfter: draftTokenCountAfter,
            schedulerPendingStepBudget: streamingEncoderStepScheduler.pendingBudget,
            useStatefulDecoderRuntime: useStatefulDecoderRuntime,
            streamingDeterministicCursorEnabled: streamingDeterministicCursorEnabled,
            streamingRollingStatelessEnabled: streamingRollingStatelessEnabled,
            streamingFinalizeDraftEnabled: streamingFinalizeDraftEnabled,
            mergeDiagnostic: mergeDiagnostic
        )
    }

    private func makeStreamingMergeDiagnostic(
        result: WindowMergeResult,
        base: [Int],
        next: [Int],
        merged: [Int]
    ) -> ParakeetStreamingDiagnostic.MergeDiagnostic {
        let baseTail = Array(base.suffix(64))
        let nextTokens = Array(next.prefix(96))
        let mergedTail = Array(merged.suffix(96))
        return ParakeetStreamingDiagnostic.MergeDiagnostic(
            strategy: result.strategy,
            alignmentStart: result.alignmentStart,
            matchLength: result.matchLength,
            baseTokenCount: base.count,
            nextTokenCount: next.count,
            mergedTokenCount: merged.count,
            baseTailTokenIDs: baseTail,
            nextTokenIDs: nextTokens,
            mergedTailTokenIDs: mergedTail,
            baseTailText: Self.decodePieces(from: baseTail, vocab: vocab),
            nextText: Self.decodePieces(from: nextTokens, vocab: vocab),
            mergedTailText: Self.decodePieces(from: mergedTail, vocab: vocab)
        )
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
        let canAutoCommitNonBlank =
            useStatefulDecoderRuntime &&
            decoderStateUpdateGateInputName != nil &&
            decoderStepIndexInputName != nil

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
                if let stepName = decoderStepIndexInputName,
                   let stepArray = decoderStepIndexBuffer {
                    try Self.setScalarInt(stepArray, value: t)
                    feed[stepName] = stepArray
                }
                if useStatefulDecoderRuntime,
                   let gateName = decoderStateUpdateGateInputName,
                   let gateArray = decoderStateUpdateGateBuffer {
                    let firstPassGateValue = statefulCommitPolicy.firstPassGateValue(
                        commitState: commitState,
                        canAutoCommitNonBlank: canAutoCommitNonBlank
                    )
                    try Self.setScalarGate(gateArray, value: firstPassGateValue)
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
                let durationIdx = Self.selectDurationIndex(
                    scores: durationScores,
                    tokenID: tokenID,
                    blankID: blankID,
                    durations: config.durations,
                    blankTieMargin: blankDurationTieMargin
                )
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
                           let state = decoderState,
                           statefulCommitPolicy.requiresExplicitAdvanceAfterEmit(
                               canAutoCommitNonBlank: canAutoCommitNonBlank
                           ) {
                                let advanceToken: Int
                                switch statefulCommitPolicy {
                                case .emitPrevToken:
                                    advanceToken = prevTokenBefore
                                case .emitToken:
                                    advanceToken = tokenID
                                case .always, .autoNonBlank:
                                    advanceToken = prevTokenBefore
                                }
                                try Self.setScalarInt(targets, value: advanceToken)
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
                var traceEvent: [String: Any] = [
                    "source": "swift",
                    "kind": "step",
                    "chunk_index": chunkIndex,
                    "step_index": traceStepIndex,
                    "encoder_steps": encoderSteps,
                    "t": tBefore,
                    "token_id": tokenID,
                    "duration_idx": durationIdx,
                    "duration_value": skip,
                    "skip": skip,
                    "prev_token": prevTokenBefore,
                    "token_piece": Self.traceTokenPiece(tokenID: tokenID, vocab: vocab, blankID: blankID),
                    "emitted": tokenID != blankID,
                    "commit_state": commitState ? 1 : 0
                ]
                if decoderTraceTopK > 0 {
                    traceEvent["token_topk"] = Self.traceTokenTopKEntries(
                        scores: tokenScores,
                        k: decoderTraceTopK,
                        vocab: vocab,
                        blankID: blankID
                    )
                    traceEvent["duration_topk"] = Self.traceDurationTopKEntries(
                        scores: durationScores,
                        k: decoderTraceTopK,
                        durations: config.durations
                    )
                }
                decoderTraceWriter?.write(event: traceEvent)
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

extension ParakeetCoreMLTDTTranscriptionModel {
    struct WindowMergeResult {
        let merged: [Int]
        let strategy: String
        let alignmentStart: Int?
        let matchLength: Int
    }

    static func mergeWindowHypothesis(base: [Int], next: [Int], maxSearchWindow: Int = 4096) -> [Int] {
        _mergeWindowHypothesisWithResult(base: base, next: next, maxSearchWindow: maxSearchWindow).merged
    }

    static func selectDurationIndexForTesting(
        scores: [Float],
        tokenID: Int,
        blankID: Int,
        durations: [Int],
        blankTieMargin: Float
    ) -> Int {
        selectDurationIndex(
            scores: scores,
            tokenID: tokenID,
            blankID: blankID,
            durations: durations,
            blankTieMargin: blankTieMargin
        )
    }

    static func rollingAnchorPrefixMatchLength(previous: [Int], current: [Int]) -> Int {
        let limit = min(previous.count, current.count)
        var idx = 0
        while idx < limit && previous[idx] == current[idx] {
            idx += 1
        }
        return idx
    }

    static func shouldPromoteRollingAnchor(previous: [Int], current: [Int], minTokenCount: Int) -> Bool {
        guard previous.count >= minTokenCount, current.count >= minTokenCount else {
            return false
        }
        let prefixMatch = rollingAnchorPrefixMatchLength(previous: previous, current: current)
        let shorter = min(previous.count, current.count)
        return prefixMatch >= minTokenCount && prefixMatch * 4 >= shorter * 3
    }

    static func shouldDecodeRewritePrefix(
        hasPreviousText: Bool,
        pendingNewSamples: Int,
        requiredSamples: Int
    ) -> Bool {
        !hasPreviousText || pendingNewSamples >= max(1, requiredSamples)
    }

    static func rewritePrefixRequiredSamples(
        stableRequiredSamples: Int,
        onsetStrideSamples: Int,
        onsetMaxSamples: Int,
        onsetMinWordCount: Int,
        currentText: String,
        bufferedSamples: Int
    ) -> Int {
        let stable = max(1, stableRequiredSamples)
        guard onsetMaxSamples > 0 || onsetMinWordCount > 0 else {
            return stable
        }

        let wordCount = tokenizeStreamingText(currentText).count
        let withinOnsetWindow = onsetMaxSamples > 0 && bufferedSamples <= onsetMaxSamples
        let belowOnsetWordFloor = onsetMinWordCount > 0 && wordCount < onsetMinWordCount
        guard withinOnsetWindow || belowOnsetWordFloor else {
            return stable
        }
        return min(stable, max(1, onsetStrideSamples))
    }

    static func shouldAllowRewritePrefixDecoderResume(
        currentText: String,
        bufferedSamples: Int,
        onsetMaxSamples: Int,
        minWordCount: Int
    ) -> Bool {
        guard onsetMaxSamples > 0 || minWordCount > 0 else {
            return true
        }
        let wordCount = tokenizeStreamingText(currentText).count
        if minWordCount > 0 && wordCount >= minWordCount {
            return true
        }
        if onsetMaxSamples > 0 && bufferedSamples > onsetMaxSamples {
            return true
        }
        return false
    }

    static func makeLongformWindowPlans(
        totalFrames: Int,
        encoderFrameCount: Int,
        leftContextFrames: Int,
        rightContextFrames: Int,
        allowRightContext: Bool
    ) -> [LongformWindowPlan] {
        guard totalFrames > 0 else { return [] }
        let maxContextTotal = max(0, encoderFrameCount - 1)
        let leftCtx = min(max(0, leftContextFrames), maxContextTotal)
        let configuredRightCtx = allowRightContext ? max(0, rightContextFrames) : 0
        let rightCtx = min(configuredRightCtx, max(0, maxContextTotal - leftCtx))
        let hopFrames = max(1, encoderFrameCount - leftCtx - rightCtx)

        var plans: [LongformWindowPlan] = []
        plans.reserveCapacity(max(1, totalFrames / hopFrames + 1))
        for centerStart in stride(from: 0, to: totalFrames, by: hopFrames) {
            let inputStart = max(0, centerStart - leftCtx)
            let actualFrames = max(0, min(encoderFrameCount, totalFrames - inputStart))
            guard actualFrames > 0 else { continue }
            let centerFrames = min(hopFrames, totalFrames - centerStart)
            plans.append(
                LongformWindowPlan(
                    key: .init(
                        centerStart: centerStart,
                        inputStart: inputStart,
                        actualFrames: actualFrames,
                        centerFrames: centerFrames
                    )
                )
            )
        }
        return plans
    }

    static func matchingLongformWindowPrefixCount(
        previous: [LongformWindowCacheKey],
        current: [LongformWindowCacheKey]
    ) -> Int {
        let limit = min(previous.count, current.count)
        var idx = 0
        while idx < limit && previous[idx] == current[idx] {
            idx += 1
        }
        return idx
    }

    static func resumableLongformWindowPrefixCount(hasCheckpoints: [Bool], reuseCount: Int) -> Int {
        guard reuseCount > 0 else { return 0 }
        let limit = min(reuseCount, hasCheckpoints.count)
        var idx = 0
        while idx < limit && hasCheckpoints[idx] {
            idx += 1
        }
        return idx
    }

    static func projectLongformWindowToEncoderSteps(
        rawSteps: Int,
        inputStart: Int,
        centerStart: Int,
        actualFrames: Int,
        centerFrames: Int,
        hopFrames: Int
    ) -> (start: Int, count: Int) {
        guard rawSteps > 0, actualFrames > 0 else { return (0, 0) }
        let leftIn = centerStart - inputStart
        let scale = Double(rawSteps) / Double(actualFrames)
        var leftOut = Int((Double(leftIn) * scale).rounded())
        var centerOut = Int((Double(centerFrames) * scale).rounded())

        leftOut = max(0, min(leftOut, rawSteps))
        centerOut = max(1, centerOut)
        let endOut: Int
        if centerFrames < hopFrames {
            endOut = rawSteps
        } else {
            endOut = max(leftOut, min(rawSteps, leftOut + centerOut))
        }
        return (leftOut, max(0, endOut - leftOut))
    }

    static func mergeRewritePrefixTranscript(
        previous: String,
        currentWindow: String,
        hasTrimmedHistory: Bool
    ) -> String {
        guard hasTrimmedHistory else { return currentWindow }
        let previousWords = tokenizeStreamingText(previous)
        let currentWords = tokenizeStreamingText(currentWindow)
        guard !previousWords.isEmpty else { return currentWindow }
        guard !currentWords.isEmpty else { return previous }
        let merged = RealtimeTranscriptBuffer.appendWithOverlap(base: previousWords, segment: currentWords)
        return merged.joined(separator: " ")
    }

    static func anchoredRollingDecodeFrameCount(
        encoderSteps: Int,
        approxNewSteps: Int,
        baseDecodeFrames: Int,
        stepMultiplier: Int,
        minTailFrames: Int
    ) -> Int {
        min(
            encoderSteps,
            max(minTailFrames, max(baseDecodeFrames, approxNewSteps * stepMultiplier))
        )
    }

    static func shouldAcceptRollingTailMerge(
        result: WindowMergeResult,
        baseCount: Int,
        nextCount: Int
    ) -> Bool {
        switch result.strategy {
        case "seed_next":
            return baseCount == 0 && nextCount > 0
        case "keep_base",
             "keep_base_empty_append",
             "keep_base_word_replay",
             "defer_short_append",
             "defer_word_overlap_append",
             "defer_high_overlap_append",
             "aligned_replace":
            return true
        default:
            return false
        }
    }

    static func refineRollingWindowMergeResult(
        _ result: WindowMergeResult,
        base: [Int],
        next: [Int],
        vocab: [String]
    ) -> WindowMergeResult {
        guard result.strategy == "append_with_overlap" else {
            return result
        }

        let baseWords = normalizedStreamingMergeWords(from: Array(base.suffix(128)), vocab: vocab)
        let nextWords = normalizedStreamingMergeWords(from: Array(next.prefix(128)), vocab: vocab)
        guard !nextWords.isEmpty else {
            return WindowMergeResult(
                merged: base,
                strategy: "keep_base_empty_append",
                alignmentStart: nil,
                matchLength: 0
            )
        }

        let shortWordOverlap = suffixPrefixWordOverlap(baseWords, nextWords, maxOverlap: 4)
        let shortSharedWordCount = sharedNormalizedWordCount(Array(baseWords.suffix(8)), nextWords)

        // Single-word fallback appends are usually unstable. For two-word
        // tails, only defer if they are already overlapping/replaying recent
        // suffix words; otherwise let novel short extensions through.
        if nextWords.count == 1 || (nextWords.count == 2 && (shortWordOverlap > 0 || shortSharedWordCount > 0)) {
            return WindowMergeResult(
                merged: base,
                strategy: "defer_short_append",
                alignmentStart: nil,
                matchLength: nextWords.count
            )
        }

        if containsSubsequence(haystack: baseWords, needle: nextWords) {
            return WindowMergeResult(
                merged: base,
                strategy: "keep_base_word_replay",
                alignmentStart: nil,
                matchLength: nextWords.count
            )
        }

        let wordOverlap = suffixPrefixWordOverlap(baseWords, nextWords, maxOverlap: 8)
        if wordOverlap >= min(3, max(0, nextWords.count - 1)) {
            return WindowMergeResult(
                merged: base,
                strategy: "defer_word_overlap_append",
                alignmentStart: nil,
                matchLength: wordOverlap
            )
        }

        let commonWordCount = sharedNormalizedWordCount(Array(baseWords.suffix(16)), nextWords)
        if nextWords.count <= 8 && commonWordCount * 2 >= nextWords.count {
            return WindowMergeResult(
                merged: base,
                strategy: "defer_high_overlap_append",
                alignmentStart: nil,
                matchLength: commonWordCount
            )
        }

        return result
    }

    static func tokenizeStreamingText(_ text: String) -> [String] {
        text
            .split(whereSeparator: \.isWhitespace)
            .map(String.init)
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

    static func cloneArray(_ source: MLMultiArray) throws -> MLMultiArray {
        let out = try makeArray(shape: source.intShape, dataType: source.dataType)
        try copyArrayContents(source: source, destination: out)
        return out
    }

    static func copyArrayContents(source: MLMultiArray, destination: MLMultiArray) throws {
        guard source.dataType == destination.dataType, source.count == destination.count else {
            throw ParakeetCoreMLTDTError.invalidModelIO("MultiArray copy shape/dtype mismatch.")
        }
        let bytesPerElement: Int
        switch source.dataType {
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
            memcpy(destination.dataPointer, source.dataPointer, source.count * bytesPerElement)
            return
        }
        for idx in 0..<source.count {
            destination.setFloatValue(source.floatValue(flatIndex: idx), flatIndex: idx)
        }
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

    static func copyFeatureWindowToEncoderInputRightAligned(
        featureMatrix: [Float],
        melBins: Int,
        totalFrames: Int,
        startFrame: Int,
        copyFrames: Int,
        destination: MLMultiArray,
        destinationStartFrame: Int
    ) throws {
        guard destination.intShape.count == 3 else {
            throw ParakeetCoreMLTDTError.unsupportedArrayRank(destination.intShape.count)
        }
        let shape = destination.intShape
        guard shape[0] >= 1,
              shape[1] >= melBins,
              destinationStartFrame >= 0,
              destinationStartFrame + copyFrames <= shape[2] else {
            throw ParakeetCoreMLTDTError.invalidModelIO("Encoder right-aligned input shape mismatch.")
        }
        guard totalFrames > 0, copyFrames > 0 else { return }

        let strides = destination.intStrides
        let dstMelStride = strides[1]
        let dstFrameStride = strides[2]

        if destination.dataType == .float32 {
            let dst = destination.dataPointer.bindMemory(to: Float.self, capacity: destination.count)
            for m in 0..<melBins {
                let srcBase = m * totalFrames + startFrame
                let dstBase = m * dstMelStride + destinationStartFrame * dstFrameStride
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
                destination.setFloatValue(
                    featureMatrix[base + startFrame + t],
                    indices: [0, m, destinationStartFrame + t]
                )
            }
        }
    }

    static func pickEncoderTensorAndLength(from provider: MLFeatureProvider, debug: Bool = false) throws -> (MLMultiArray, Int?) {
        var length: Int?
        var bestName: String?
        var bestScore: (Int, Int, Int, Int)?

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
            // Prefer the actual encoder sequence tensor (rank 3) over higher-rank
            // cache tensors that can appear in streaming encoder wrappers.
            let score = (shape.count == 3 ? 1 : 0, shape.count, shape.last ?? 0, arr.count)
            if let b = bestScore {
                if score.0 > b.0
                    || (score.0 == b.0 && score.1 > b.1)
                    || (score.0 == b.0 && score.1 == b.1 && score.2 > b.2)
                    || (score.0 == b.0 && score.1 == b.1 && score.2 == b.2 && score.3 > b.3)
                {
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

    struct PickedStreamingEncoderOutputs {
        let tensor: MLMultiArray
        let length: Int?
        let state: MLMultiArray?
        let timeState: MLMultiArray?
        let stateLength: Int?
    }

    static func pickStreamingEncoderOutputs(
        from provider: MLFeatureProvider,
        expectedStateShape: [Int],
        expectedTimeStateShape: [Int]? = nil,
        debug: Bool = false
    ) throws -> PickedStreamingEncoderOutputs {
        var scalarValues: [(name: String, value: Int)] = []
        var floatArrays: [(name: String, array: MLMultiArray)] = []

        let orderedNames = Array(provider.featureNames).sorted()
        for name in orderedNames {
            guard let arr = provider.featureValue(for: name)?.multiArrayValue else { continue }
            let shape = arr.intShape
            if debug {
                fputs("[ParakeetSwift] encoder output \(name): dtype=\(arr.dataType.rawValue) shape=\(shape)\n", stderr)
            }
            if arr.isIntegerLike, shape.count == 1, arr.count == 1 {
                scalarValues.append((name: name, value: Int(arr.floatValue(flatIndex: 0))))
                continue
            }
            guard arr.isFloatLike, shape.count >= 2 else { continue }
            floatArrays.append((name: name, array: arr))
        }

        let stateCount = expectedStateShape.reduce(1, *)
        var pickedState: MLMultiArray?
        var pickedStateName: String?
        for candidate in floatArrays {
            let shape = candidate.array.intShape
            if shape == expectedStateShape || candidate.array.count == stateCount {
                pickedState = candidate.array
                pickedStateName = candidate.name
                break
            }
        }

        var pickedTimeState: MLMultiArray?
        var pickedTimeStateName: String?
        if let expectedTimeStateShape {
            let timeStateCount = expectedTimeStateShape.reduce(1, *)
            for candidate in floatArrays where candidate.name != pickedStateName {
                let shape = candidate.array.intShape
                if shape == expectedTimeStateShape || candidate.array.count == timeStateCount {
                    pickedTimeState = candidate.array
                    pickedTimeStateName = candidate.name
                    break
                }
            }
        }

        let tensorCandidates = floatArrays.filter { $0.name != pickedStateName && $0.name != pickedTimeStateName }
        guard !tensorCandidates.isEmpty else {
            throw ParakeetCoreMLTDTError.missingOutput("Could not infer encoder tensor output.")
        }

        let rank3Candidates = tensorCandidates.filter { $0.array.intShape.count == 3 }
        let preferredTensorPool = rank3Candidates.isEmpty ? tensorCandidates : rank3Candidates

        var pickedTensor: MLMultiArray?
        var pickedTensorName: String?
        var bestScore: (Int, Int, Int)?
        for candidate in preferredTensorPool {
            let shape = candidate.array.intShape
            let score = (shape.last ?? 0, shape.count, candidate.array.count)
            if let currentBest = bestScore {
                if score.0 > currentBest.0
                    || (score.0 == currentBest.0 && score.1 > currentBest.1)
                    || (score.0 == currentBest.0 && score.1 == currentBest.1 && score.2 > currentBest.2)
                {
                    pickedTensor = candidate.array
                    pickedTensorName = candidate.name
                    bestScore = score
                }
            } else {
                pickedTensor = candidate.array
                pickedTensorName = candidate.name
                bestScore = score
            }
        }

        guard let tensor = pickedTensor else {
            throw ParakeetCoreMLTDTError.missingOutput("Could not infer encoder tensor output.")
        }

        let tensorTimeDim = tensor.intShape.last ?? 0
        let length = scalarValues
            .filter { $0.value >= 0 && (tensorTimeDim == 0 || $0.value <= tensorTimeDim) }
            .map(\.value)
            .max() ?? scalarValues.map(\.value).max()

        let maxStateLength = expectedStateShape.count >= 3 ? expectedStateShape[2] : 0
        let stateLength = scalarValues
            .filter { $0.value >= 0 && (maxStateLength == 0 || $0.value <= maxStateLength) }
            .map(\.value)
            .max() ?? length

        if debug {
            fputs(
                "[ParakeetSwift] selected encoder tensor \(pickedTensorName ?? "<unknown>") shape=\(tensor.intShape) length=\(length.map(String.init) ?? "nil")\n",
                stderr
            )
            if let state = pickedState {
                fputs(
                    "[ParakeetSwift] selected encoder cache \(pickedStateName ?? "<unknown>") shape=\(state.intShape) state_length=\(stateLength.map(String.init) ?? "nil")\n",
                    stderr
                )
            }
            if let timeState = pickedTimeState {
                fputs(
                    "[ParakeetSwift] selected encoder time cache \(pickedTimeStateName ?? "<unknown>") shape=\(timeState.intShape)\n",
                    stderr
                )
            }
            if tensor.intShape.count == 3, tensor.intShape[0] > 0, tensor.intShape[1] > 0 {
                let limit = min(10, tensor.intShape[2])
                let preview = (0..<limit).map { String(format: "%.6f", tensor.floatValue(indices: [0, 0, $0])) }.joined(separator: ", ")
                fputs("[ParakeetSwift] encoder[0,0,0:\(limit)] = [\(preview)]\n", stderr)
            }
        }

        return PickedStreamingEncoderOutputs(
            tensor: tensor,
            length: length,
            state: pickedState,
            timeState: pickedTimeState,
            stateLength: stateLength
        )
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

    static func topKIndices(_ values: [Float], k: Int) -> [Int] {
        guard k > 0, !values.isEmpty else { return [] }
        return values.enumerated()
            .sorted { lhs, rhs in
                if lhs.element == rhs.element {
                    return lhs.offset < rhs.offset
                }
                return lhs.element > rhs.element
            }
            .prefix(k)
            .map(\.offset)
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

    static func selectDurationIndex(
        scores: [Float],
        tokenID: Int,
        blankID: Int,
        durations: [Int],
        blankTieMargin: Float
    ) -> Int {
        let bestIndex = argmax(scores)
        guard tokenID == blankID, blankTieMargin > 0, !scores.isEmpty else {
            return bestIndex
        }

        let bestScore = scores[bestIndex]
        var preferredIndex = bestIndex
        var preferredDuration = durations.indices.contains(bestIndex) ? durations[bestIndex] : Int.max

        for (index, score) in scores.enumerated() {
            guard durations.indices.contains(index) else { continue }
            let durationValue = durations[index]
            guard durationValue > 0 else { continue }
            guard bestScore - score <= blankTieMargin else { continue }
            if preferredDuration <= 0 || durationValue < preferredDuration {
                preferredIndex = index
                preferredDuration = durationValue
            }
        }

        return preferredIndex
    }

    static func traceTokenPiece(tokenID: Int, vocab: [String], blankID: Int) -> String {
        if tokenID == blankID {
            return "<blank>"
        }
        guard tokenID >= 0 && tokenID < vocab.count else {
            return "<invalid>"
        }
        return vocab[tokenID]
    }

    static func traceTokenTopKEntries(
        scores: [Float],
        k: Int,
        vocab: [String],
        blankID: Int
    ) -> [[String: Any]] {
        topKIndices(scores, k: k).map { index in
            [
                "token_id": index,
                "token_piece": traceTokenPiece(tokenID: index, vocab: vocab, blankID: blankID),
                "logp": Double(scores[index]),
            ]
        }
    }

    static func traceDurationTopKEntries(
        scores: [Float],
        k: Int,
        durations: [Int]
    ) -> [[String: Any]] {
        topKIndices(scores, k: k).map { index in
            [
                "duration_idx": index,
                "duration_value": durations.indices.contains(index) ? durations[index] : 1,
                "logp": Double(scores[index]),
            ]
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

    static func _mergeWindowHypothesisWithResult(
        base: [Int],
        next: [Int],
        maxSearchWindow: Int = 4096
    ) -> WindowMergeResult {
        guard !base.isEmpty else {
            return WindowMergeResult(merged: next, strategy: "seed_next", alignmentStart: nil, matchLength: 0)
        }
        guard !next.isEmpty else {
            return WindowMergeResult(merged: base, strategy: "keep_base", alignmentStart: nil, matchLength: 0)
        }

        let searchStart = max(0, base.count - maxSearchWindow)
        let minRequiredMatch: Int
        if next.count >= 16 {
            minRequiredMatch = 4
        } else if next.count >= 8 {
            minRequiredMatch = 3
        } else {
            minRequiredMatch = 2
        }

        var bestStart: Int?
        var bestMatchLength = 0
        for start in searchStart..<base.count {
            let limit = min(base.count - start, next.count)
            guard limit > 0 else { continue }
            var matchLength = 0
            while matchLength < limit && base[start + matchLength] == next[matchLength] {
                matchLength += 1
            }
            guard matchLength > 0 else { continue }
            if matchLength > bestMatchLength ||
                (matchLength == bestMatchLength && start < (bestStart ?? Int.max)) {
                bestStart = start
                bestMatchLength = matchLength
            }
        }

        if let bestStart, bestMatchLength >= minRequiredMatch {
            return WindowMergeResult(
                merged: Array(base.prefix(bestStart)) + next,
                strategy: "aligned_replace",
                alignmentStart: bestStart,
                matchLength: bestMatchLength
            )
        }

        if next.count >= base.count {
            return WindowMergeResult(
                merged: next,
                strategy: "replace_all",
                alignmentStart: nil,
                matchLength: 0
            )
        }
        return WindowMergeResult(
            merged: appendWithOverlap(base: base, next: next),
            strategy: "append_with_overlap",
            alignmentStart: nil,
            matchLength: 0
        )
    }

    static func normalizedStreamingMergeWords(from tokenIDs: [Int], vocab: [String]) -> [String] {
        decodePieces(from: tokenIDs, vocab: vocab)
            .split(whereSeparator: \.isWhitespace)
            .map { normalizeStreamingMergeWord(String($0)) }
            .filter { !$0.isEmpty }
    }

    static func normalizeStreamingMergeWord(_ word: String) -> String {
        let scalars = word.lowercased().unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) }
        return String(String.UnicodeScalarView(scalars))
    }

    static func suffixPrefixWordOverlap(_ lhs: [String], _ rhs: [String], maxOverlap: Int) -> Int {
        let limit = min(maxOverlap, min(lhs.count, rhs.count))
        guard limit > 0 else { return 0 }
        for candidate in stride(from: limit, through: 1, by: -1) {
            if Array(lhs.suffix(candidate)) == Array(rhs.prefix(candidate)) {
                return candidate
            }
        }
        return 0
    }

    static func sharedNormalizedWordCount(_ lhs: [String], _ rhs: [String]) -> Int {
        guard !lhs.isEmpty, !rhs.isEmpty else { return 0 }
        let lhsSet = Set(lhs)
        let rhsSet = Set(rhs)
        return lhsSet.intersection(rhsSet).count
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

    static func containsSubsequence(haystack: [String], needle: [String]) -> Bool {
        guard !needle.isEmpty else { return true }
        guard needle.count <= haystack.count else { return false }
        let limit = haystack.count - needle.count
        for i in 0...limit {
            if Array(haystack[i..<(i + needle.count)]) == needle {
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

struct StreamingEncoderStepScheduler {
    private var pendingStepBudget: Double = 0

    var pendingBudget: Double {
        pendingStepBudget
    }

    mutating func reset() {
        pendingStepBudget = 0
    }

    mutating func consumeStepBudget(
        newFeatureFrames: Int,
        encoderSteps: Int,
        windowFrames: Int,
        minNewSteps: Int
    ) -> Int {
        guard encoderSteps > 0, windowFrames > 0 else {
            pendingStepBudget = 0
            return 0
        }

        let normalizedFeatureFrames = max(0, newFeatureFrames)
        let stepRatio = Double(encoderSteps) / Double(max(1, windowFrames))
        pendingStepBudget += Double(normalizedFeatureFrames) * stepRatio

        let wholeSteps = Int(floor(pendingStepBudget + 1e-9))
        guard wholeSteps > 0 else {
            return 0
        }

        let scheduled = max(minNewSteps, wholeSteps)
        pendingStepBudget = max(0, pendingStepBudget - Double(scheduled))
        return scheduled
    }
}

struct StreamingMelFeatureCache {
    private let extractor: MelFeatureExtractor
    private var rawFrames: [[Float]]
    private var cachedSampleCount: Int

    init(extractor: MelFeatureExtractor) {
        self.extractor = extractor
        self.rawFrames = []
        self.cachedSampleCount = -1
    }

    mutating func reset() {
        rawFrames.removeAll(keepingCapacity: true)
        cachedSampleCount = -1
    }

    mutating func extract(samples: [Float], droppedSampleCount: Int) -> ([Float], Int) {
        let targetFrameCount = extractor.frameCount(forSampleCount: samples.count)
        let previousSampleCount = cachedSampleCount
        let expectedPreviousFrameCount = previousSampleCount >= 0 ? extractor.frameCount(forSampleCount: previousSampleCount) : 0
        let canSlideWindow = droppedSampleCount == 0 || droppedSampleCount % extractor.hopLength == 0
        let droppedFrames = canSlideWindow ? droppedSampleCount / extractor.hopLength : 0

        let needsFullRebuild =
            rawFrames.isEmpty ||
            previousSampleCount < 0 ||
            rawFrames.count != expectedPreviousFrameCount ||
            !canSlideWindow ||
            droppedFrames > rawFrames.count

        if needsFullRebuild {
            rawFrames = extractor.extractRawFrames(samples: samples)
            cachedSampleCount = samples.count
            return extractor.normalize(rawFrames: rawFrames)
        }

        if droppedFrames > 0 {
            rawFrames.removeFirst(droppedFrames)
        }

        let preservedFrameCount = rawFrames.count
        if rawFrames.count > targetFrameCount {
            rawFrames.removeLast(rawFrames.count - targetFrameCount)
        } else if rawFrames.count < targetFrameCount {
            rawFrames.append(
                contentsOf: Array(
                    repeating: Array(repeating: 0, count: extractor.melBins),
                    count: targetFrameCount - rawFrames.count
                )
            )
        }

        if droppedFrames > 0 {
            recomputeFrames(
                samples: samples,
                range: 0..<min(extractor.boundaryRecomputeFrames, targetFrameCount)
            )
        }
        let tailStart = max(0, preservedFrameCount - extractor.boundaryRecomputeFrames)
        recomputeFrames(samples: samples, range: tailStart..<targetFrameCount)

        cachedSampleCount = samples.count
        return extractor.normalize(rawFrames: rawFrames)
    }

    private mutating func recomputeFrames(samples: [Float], range: Range<Int>) {
        guard !range.isEmpty else { return }
        var fftBuffer = Array(repeating: Float(0), count: extractor.nFFT)
        var powerSpectrum = Array(repeating: Float(0), count: extractor.nFFT / 2 + 1)
        for frameIndex in range {
            rawFrames[frameIndex] = extractor.rawLogMelFrame(
                samples: samples,
                frameIndex: frameIndex,
                fftBuffer: &fftBuffer,
                powerSpectrum: &powerSpectrum
            )
        }
    }
}

struct MelFeatureExtractor {
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
        let rawFrames = extractRawFrames(samples: samples)
        return normalize(rawFrames: rawFrames)
    }

    var boundaryRecomputeFrames: Int {
        max(2, ((nFFT / 2) + hopLength - 1) / hopLength + 2)
    }

    func frameCount(forSampleCount sampleCount: Int) -> Int {
        max(1, 1 + max(0, sampleCount) / hopLength)
    }

    func extractRawFrames(samples: [Float]) -> [[Float]] {
        let frameCount = frameCount(forSampleCount: samples.count)
        var frames = Array(repeating: Array(repeating: Float(0), count: melBins), count: frameCount)
        var fftBuffer = Array(repeating: Float(0), count: nFFT)
        var powerSpectrum = Array(repeating: Float(0), count: spectrumBins)
        for frameIdx in 0..<frameCount {
            frames[frameIdx] = rawLogMelFrame(
                samples: samples,
                frameIndex: frameIdx,
                fftBuffer: &fftBuffer,
                powerSpectrum: &powerSpectrum
            )
        }
        return frames
    }

    func normalize(rawFrames: [[Float]]) -> ([Float], Int) {
        let frameCount = max(1, rawFrames.count)
        var logMel = Array(repeating: Float(0), count: melBins * frameCount)

        for m in 0..<melBins {
            var mean: Float = 0
            for t in 0..<frameCount {
                mean += rawFrames[t][m]
            }
            mean /= Float(frameCount)

            var variance: Float = 0
            for t in 0..<frameCount {
                let d = rawFrames[t][m] - mean
                variance += d * d
            }
            variance /= Float(frameCount)
            let std = sqrt(variance) + 1e-5

            for t in 0..<frameCount {
                logMel[m * frameCount + t] = (rawFrames[t][m] - mean) / std
            }
        }

        return (logMel, frameCount)
    }

    func rawLogMelFrame(
        samples: [Float],
        frameIndex: Int,
        fftBuffer: inout [Float],
        powerSpectrum: inout [Float]
    ) -> [Float] {
        let start = frameIndex * hopLength
        let centerPad = nFFT / 2
        let windowOffset = max(0, (nFFT - windowLength) / 2)
        for i in 0..<nFFT {
            fftBuffer[i] = 0
        }
        for i in 0..<windowLength {
            let sampleIndex = start + i - centerPad
            let sample = sampleIndex >= 0 && sampleIndex < samples.count ? samples[sampleIndex] : 0
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

        var frame = Array(repeating: Float(0), count: melBins)
        for m in 0..<melBins {
            var energy: Float = 0
            let weights = melFilters[m]
            for k in 0..<spectrumBins {
                energy += weights[k] * powerSpectrum[k]
            }
            frame[m] = log(max(energy, 1e-6))
        }
        return frame
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
