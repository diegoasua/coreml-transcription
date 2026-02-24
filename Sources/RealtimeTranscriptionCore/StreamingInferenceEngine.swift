import Foundation

public struct StreamingInferenceEvent: Equatable {
    public let transcript: TranscriptState
    public let isSpeech: Bool
    public let didFlushSegment: Bool
    public let energyDBFS: Float

    public init(transcript: TranscriptState, isSpeech: Bool, didFlushSegment: Bool, energyDBFS: Float) {
        self.transcript = transcript
        self.isSpeech = isSpeech
        self.didFlushSegment = didFlushSegment
        self.energyDBFS = energyDBFS
    }
}

public struct StreamingInferenceEngine<Model: TranscriptionModel, VAD: VoiceActivityDetecting> {
    private var model: Model
    private var vad: VAD
    private var textController: StreamingTextController
    private var ringBuffer: AudioRingBuffer
    private let ringBufferCapacity: Int
    private(set) public var policy: StreamingPolicy
    private var previouslyInSpeech: Bool

    public init(
        model: Model,
        vad: VAD,
        policy: StreamingPolicy = .init(),
        requiredAgreementCount: Int = 2,
        ringBufferCapacity: Int? = nil
    ) {
        self.model = model
        self.vad = vad
        self.policy = policy
        self.textController = StreamingTextController(requiredAgreementCount: requiredAgreementCount)
        let capacity = ringBufferCapacity ?? max(policy.chunkSamples * 4, policy.chunkSamples + policy.hopSamples)
        self.ringBufferCapacity = capacity
        self.ringBuffer = AudioRingBuffer(capacity: capacity)
        self.previouslyInSpeech = false
    }

    public mutating func process(samples: [Float]) throws -> [StreamingInferenceEvent] {
        guard !samples.isEmpty else { return [] }
        ringBuffer.append(samples)

        var events: [StreamingInferenceEvent] = []
        while ringBuffer.availableToRead >= policy.chunkSamples {
            let chunk = ringBuffer.peek(count: policy.chunkSamples)
            let decision = vad.process(samples: chunk, sampleRate: policy.sampleRate)
            if decision.isSpeech {
                let partial = try model.transcribeChunk(chunk, sampleRate: policy.sampleRate)
                let state = textController.update(partialText: partial)
                events.append(
                    StreamingInferenceEvent(
                        transcript: state,
                        isSpeech: true,
                        didFlushSegment: false,
                        energyDBFS: decision.energyDBFS
                    )
                )
            } else if previouslyInSpeech {
                let state = textController.endSegment()
                events.append(
                    StreamingInferenceEvent(
                        transcript: state,
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
        model.resetState()
        vad.reset()
        ringBuffer = AudioRingBuffer(capacity: ringBufferCapacity)
        previouslyInSpeech = false
        return final
    }
}
