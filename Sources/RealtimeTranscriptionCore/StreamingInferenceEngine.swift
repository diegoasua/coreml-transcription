import Foundation

public struct StreamingInferenceEvent: Equatable {
    public let transcript: TranscriptState
    public let revision: Int
    public let isSpeech: Bool
    public let didFlushSegment: Bool
    public let energyDBFS: Float
    public let rawPartialText: String?

    public init(
        transcript: TranscriptState,
        revision: Int = 0,
        isSpeech: Bool,
        didFlushSegment: Bool,
        energyDBFS: Float,
        rawPartialText: String? = nil
    ) {
        self.transcript = transcript
        self.revision = revision
        self.isSpeech = isSpeech
        self.didFlushSegment = didFlushSegment
        self.energyDBFS = energyDBFS
        self.rawPartialText = rawPartialText
    }
}

public struct StreamingInferenceEngine<Model: TranscriptionModel, VAD: VoiceActivityDetecting> {
    private var model: Model
    private var vad: VAD
    private var textController: StreamingTextController
    private var ringBuffer: AudioRingBuffer
    private let ringBufferCapacity: Int
    private(set) public var policy: StreamingPolicy
    private let decodeOnlyWhenSpeech: Bool
    private let flushOnSpeechEnd: Bool
    private let maxSpeechChunkRunBeforeReset: Int?
    private let maxStagnantSpeechChunks: Int?
    private var previouslyInSpeech: Bool
    private var speechChunkRunCount: Int
    private var stagnantSpeechChunkCount: Int
    private var lastSpeechFingerprint: String

    public init(
        model: Model,
        vad: VAD,
        policy: StreamingPolicy = .init(),
        requiredAgreementCount: Int = 2,
        draftAgreementCount: Int = 1,
        decodeOnlyWhenSpeech: Bool = true,
        flushOnSpeechEnd: Bool = true,
        maxSpeechChunkRunBeforeReset: Int? = nil,
        maxStagnantSpeechChunks: Int? = nil,
        ringBufferCapacity: Int? = nil
    ) {
        self.model = model
        self.vad = vad
        self.policy = policy
        self.textController = StreamingTextController(
            requiredAgreementCount: requiredAgreementCount,
            draftAgreementCount: draftAgreementCount
        )
        self.decodeOnlyWhenSpeech = decodeOnlyWhenSpeech
        self.flushOnSpeechEnd = flushOnSpeechEnd
        self.maxSpeechChunkRunBeforeReset = maxSpeechChunkRunBeforeReset
        self.maxStagnantSpeechChunks = maxStagnantSpeechChunks
        let capacity = ringBufferCapacity ?? max(policy.chunkSamples * 4, policy.chunkSamples + policy.hopSamples)
        self.ringBufferCapacity = capacity
        self.ringBuffer = AudioRingBuffer(capacity: capacity)
        self.previouslyInSpeech = false
        self.speechChunkRunCount = 0
        self.stagnantSpeechChunkCount = 0
        self.lastSpeechFingerprint = ""
    }

    public mutating func process(samples: [Float]) throws -> [StreamingInferenceEvent] {
        guard !samples.isEmpty else { return [] }
        ringBuffer.append(samples)

        var events: [StreamingInferenceEvent] = []
        while ringBuffer.availableToRead >= policy.chunkSamples {
            let chunk = ringBuffer.peek(count: policy.chunkSamples)
            let decision = vad.process(samples: chunk, sampleRate: policy.sampleRate)
            var decodedState: TranscriptState?
            let shouldDecode = decision.isSpeech || !decodeOnlyWhenSpeech
            if shouldDecode {
                // Decode only newly advanced audio (hop window). Using the full
                // overlapping chunk here causes repeated audio exposure when
                // chunkMs > hopMs, which inflates latency and can duplicate text.
                let decodeSliceCount = min(policy.hopSamples, ringBuffer.availableToRead)
                let decodeSamples = ringBuffer.peek(count: decodeSliceCount)
                let partial = try model.transcribeChunk(decodeSamples, sampleRate: policy.sampleRate)
                let state = textController.update(partialText: partial)
                decodedState = state
                events.append(
                    StreamingInferenceEvent(
                        transcript: state,
                        revision: textController.revision,
                        isSpeech: decision.isSpeech,
                        didFlushSegment: false,
                        energyDBFS: decision.energyDBFS,
                        rawPartialText: partial
                    )
                )
            }

            var shouldForceFlush = false
            if decision.isSpeech {
                speechChunkRunCount += 1
                if let maxRun = maxSpeechChunkRunBeforeReset, maxRun > 0, speechChunkRunCount >= maxRun {
                    shouldForceFlush = true
                }
            } else {
                speechChunkRunCount = 0
            }

            // Decoder-stagnation guard: if decoded transcript stops evolving for too many
            // chunks, force a segment flush + state reset to recover forward progress.
            if decision.isSpeech, let state = decodedState {
                let fingerprint = state.confirmed + "|" + state.hypothesis
                if fingerprint == lastSpeechFingerprint {
                    stagnantSpeechChunkCount += 1
                } else {
                    lastSpeechFingerprint = fingerprint
                    stagnantSpeechChunkCount = 0
                }
                if let maxStagnant = maxStagnantSpeechChunks,
                   maxStagnant > 0,
                   stagnantSpeechChunkCount >= maxStagnant {
                    shouldForceFlush = true
                }
            } else if !shouldDecode {
                // If no decode happened due to gating, clear stagnation state.
                stagnantSpeechChunkCount = 0
                lastSpeechFingerprint = ""
            } else {
                // Do not accumulate stagnation while VAD is inactive.
                stagnantSpeechChunkCount = 0
                lastSpeechFingerprint = ""
            }

            if shouldForceFlush {
                let state = textController.endSegment()
                model.resetState()
                speechChunkRunCount = 0
                stagnantSpeechChunkCount = 0
                lastSpeechFingerprint = ""
                events.append(
                    StreamingInferenceEvent(
                        transcript: state,
                        revision: textController.revision,
                        isSpeech: decision.isSpeech,
                        didFlushSegment: true,
                        energyDBFS: decision.energyDBFS
                    )
                )
            }

            if flushOnSpeechEnd, !decision.isSpeech, previouslyInSpeech {
                let state = textController.endSegment()
                model.resetState()
                speechChunkRunCount = 0
                stagnantSpeechChunkCount = 0
                lastSpeechFingerprint = ""
                events.append(
                    StreamingInferenceEvent(
                        transcript: state,
                        revision: textController.revision,
                        isSpeech: false,
                        didFlushSegment: true,
                        energyDBFS: decision.energyDBFS
                    )
                )
            }

            previouslyInSpeech = decision.isSpeech
            _ = ringBuffer.pop(count: policy.hopSamples)
        }
        return events
    }

    public mutating func finishStream() -> TranscriptState {
        let final = textController.endSegment()
        resetRuntimeState()
        return final
    }

    public mutating func discardStream(preserveHypothesis: Bool = false) -> TranscriptState {
        let snapshot = textController.discardCurrentSegment(preserveHypothesis: preserveHypothesis)
        resetRuntimeState()
        return snapshot
    }

    private mutating func resetRuntimeState() {
        model.resetState()
        vad.reset()
        ringBuffer = AudioRingBuffer(capacity: ringBufferCapacity)
        previouslyInSpeech = false
        speechChunkRunCount = 0
        stagnantSpeechChunkCount = 0
        lastSpeechFingerprint = ""
    }
}
