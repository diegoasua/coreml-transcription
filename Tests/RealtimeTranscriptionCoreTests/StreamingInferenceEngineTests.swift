import XCTest
@testable import RealtimeTranscriptionCore

final class StreamingInferenceEngineTests: XCTestCase {
    func testSpeechUpdatesThenFlushOnSilence() throws {
        let model = MockModel(outputs: ["hello", "hello world"])
        let vad = MockVAD(decisions: [true, true, false], energyDBFS: -20)
        var engine = StreamingInferenceEngine(
            model: model,
            vad: vad,
            policy: .init(sampleRate: 1000, chunkMs: 100, hopMs: 100),
            requiredAgreementCount: 2,
            ringBufferCapacity: 400
        )

        let samples = Array(repeating: Float(0.1), count: 300)
        let events = try engine.process(samples: samples)

        XCTAssertEqual(events.count, 3)

        XCTAssertTrue(events[0].isSpeech)
        XCTAssertEqual(events[0].transcript.confirmed, "")
        XCTAssertEqual(events[0].transcript.hypothesis, "hello")

        XCTAssertTrue(events[1].isSpeech)
        XCTAssertEqual(events[1].transcript.confirmed, "hello")
        XCTAssertEqual(events[1].transcript.hypothesis, "world")

        XCTAssertFalse(events[2].isSpeech)
        XCTAssertTrue(events[2].didFlushSegment)
        XCTAssertEqual(events[2].transcript.confirmed, "hello world")
        XCTAssertEqual(events[2].transcript.hypothesis, "")
    }

    func testDecodeEvenWhenVADSaysNonSpeechWhenConfigured() throws {
        let model = MockModel(outputs: ["alpha", "alpha beta", "alpha beta gamma"])
        let vad = MockVAD(decisions: [false, false, false], energyDBFS: -60)
        var engine = StreamingInferenceEngine(
            model: model,
            vad: vad,
            policy: .init(sampleRate: 1000, chunkMs: 100, hopMs: 100),
            requiredAgreementCount: 2,
            decodeOnlyWhenSpeech: false,
            ringBufferCapacity: 400
        )

        let samples = Array(repeating: Float(0.1), count: 300)
        let events = try engine.process(samples: samples)

        XCTAssertEqual(events.count, 3)
        XCTAssertFalse(events[0].isSpeech)
        XCTAssertFalse(events[1].isSpeech)
        XCTAssertFalse(events[2].isSpeech)
        XCTAssertFalse(events[0].didFlushSegment)
        XCTAssertFalse(events[1].didFlushSegment)
        XCTAssertFalse(events[2].didFlushSegment)
        XCTAssertEqual(events[0].transcript.confirmed, "")
        XCTAssertEqual(events[0].transcript.hypothesis, "alpha")
        XCTAssertEqual(events[1].transcript.confirmed, "alpha")
        XCTAssertEqual(events[1].transcript.hypothesis, "beta")
    }

    func testForceFlushOnStagnantSpeech() throws {
        let model = MockModel(outputs: ["hello", "hello", "hello", "hello", "hello"])
        let vad = MockVAD(decisions: [true, true, true, true, true], energyDBFS: -20)
        var engine = StreamingInferenceEngine(
            model: model,
            vad: vad,
            policy: .init(sampleRate: 1000, chunkMs: 100, hopMs: 100),
            requiredAgreementCount: 2,
            decodeOnlyWhenSpeech: true,
            maxSpeechChunkRunBeforeReset: nil,
            maxStagnantSpeechChunks: 1,
            ringBufferCapacity: 600
        )

        let samples = Array(repeating: Float(0.1), count: 500)
        let events = try engine.process(samples: samples)

        XCTAssertTrue(events.contains(where: { $0.didFlushSegment }))
    }

    func testNoSilenceDrivenFlushWhenAlwaysDecodeMode() throws {
        let model = MockModel(outputs: ["mm", "mm", "mm", "mm"])
        let vad = MockVAD(decisions: [true, false, false, false], energyDBFS: -55)
        var engine = StreamingInferenceEngine(
            model: model,
            vad: vad,
            policy: .init(sampleRate: 1000, chunkMs: 100, hopMs: 100),
            requiredAgreementCount: 1,
            decodeOnlyWhenSpeech: false,
            flushOnSpeechEnd: false,
            maxSpeechChunkRunBeforeReset: nil,
            maxStagnantSpeechChunks: 1,
            ringBufferCapacity: 500
        )

        let samples = Array(repeating: Float(0.1), count: 400)
        let events = try engine.process(samples: samples)

        XCTAssertEqual(events.filter(\.didFlushSegment).count, 0)
        XCTAssertEqual(events.count, 4)
    }

    func testForceFlushOnDecodedChunkRunEvenWithoutSpeechVAD() throws {
        let model = MockModel(outputs: ["alpha", "alpha beta", "gamma"])
        let vad = MockVAD(decisions: [false, false, false], energyDBFS: -60)
        var engine = StreamingInferenceEngine(
            model: model,
            vad: vad,
            policy: .init(sampleRate: 1000, chunkMs: 100, hopMs: 100),
            requiredAgreementCount: 1,
            decodeOnlyWhenSpeech: false,
            flushOnSpeechEnd: false,
            maxSpeechChunkRunBeforeReset: 2,
            maxStagnantSpeechChunks: nil,
            ringBufferCapacity: 400
        )

        let samples = Array(repeating: Float(0.1), count: 300)
        let events = try engine.process(samples: samples)

        XCTAssertEqual(events.filter(\.didFlushSegment).count, 1)
        XCTAssertTrue(events.contains(where: { $0.didFlushSegment && $0.transcript.confirmed == "alpha beta" }))
    }

    func testDecodesHopSliceWhenChunkOverlaps() throws {
        let model = RecordingModel()
        let vad = MockVAD(decisions: Array(repeating: true, count: 16), energyDBFS: -20)
        var engine = StreamingInferenceEngine(
            model: model,
            vad: vad,
            policy: .init(sampleRate: 1000, chunkMs: 100, hopMs: 50),
            requiredAgreementCount: 1,
            decodeOnlyWhenSpeech: true,
            ringBufferCapacity: 800
        )

        let samples = Array(repeating: Float(0.1), count: 400)
        _ = try engine.process(samples: samples)

        XCTAssertFalse(model.seenCounts.isEmpty)
        XCTAssertTrue(model.seenCounts.allSatisfy { $0 == 50 })
    }

    func testPreservesCommittedTranscriptAcrossSegments() throws {
        let model = NonResettingMockModel(outputs: ["hello", "hello world", "how", "how are"])
        let vad = MockVAD(decisions: [true, true, false, true, true, false], energyDBFS: -20)
        var engine = StreamingInferenceEngine(
            model: model,
            vad: vad,
            policy: .init(sampleRate: 1000, chunkMs: 100, hopMs: 100),
            requiredAgreementCount: 2,
            decodeOnlyWhenSpeech: true,
            flushOnSpeechEnd: true,
            ringBufferCapacity: 800
        )

        let samples = Array(repeating: Float(0.1), count: 600)
        let events = try engine.process(samples: samples)
        XCTAssertEqual(events.count, 6)

        // First segment flush
        XCTAssertTrue(events[2].didFlushSegment)
        XCTAssertEqual(events[2].transcript.confirmed, "hello world")

        // Second segment starts from committed prefix (not from empty transcript).
        XCTAssertFalse(events[3].didFlushSegment)
        XCTAssertEqual(events[3].transcript.confirmed, "hello world")
        XCTAssertEqual(events[3].transcript.hypothesis, "how")

        // Final flush includes both segments.
        XCTAssertTrue(events[5].didFlushSegment)
        XCTAssertEqual(events[5].transcript.confirmed, "hello world how are")
        XCTAssertEqual(events[5].transcript.hypothesis, "")
    }

    func testDiscardStreamDropsInFlightSegmentButKeepsCommittedTranscript() throws {
        let model = NonResettingMockModel(outputs: ["hello", "hello world", "how"])
        let vad = MockVAD(decisions: [true, true, false, true], energyDBFS: -20)
        var engine = StreamingInferenceEngine(
            model: model,
            vad: vad,
            policy: .init(sampleRate: 1000, chunkMs: 100, hopMs: 100),
            requiredAgreementCount: 2,
            decodeOnlyWhenSpeech: true,
            flushOnSpeechEnd: true,
            ringBufferCapacity: 800
        )

        let firstSegmentEvents = try engine.process(samples: Array(repeating: Float(0.1), count: 300))
        XCTAssertEqual(firstSegmentEvents.last?.transcript.confirmed, "hello world")

        let secondSegmentEvents = try engine.process(samples: Array(repeating: Float(0.1), count: 100))
        XCTAssertEqual(secondSegmentEvents.count, 1)
        XCTAssertEqual(secondSegmentEvents[0].transcript.confirmed, "hello world")
        XCTAssertEqual(secondSegmentEvents[0].transcript.hypothesis, "how")

        let snapshot = engine.discardStream()
        XCTAssertEqual(snapshot.confirmed, "hello world")
        XCTAssertEqual(snapshot.hypothesis, "")
    }

    func testDiscardStreamPreserveCarriesMatchingHypothesisAcrossReset() throws {
        let model = NonResettingMockModel(outputs: ["so i have", "so i have created", "so i have created"])
        let vad = MockVAD(decisions: [true, true, true], energyDBFS: -20)
        var engine = StreamingInferenceEngine(
            model: model,
            vad: vad,
            policy: .init(sampleRate: 1000, chunkMs: 100, hopMs: 100),
            requiredAgreementCount: 2,
            decodeOnlyWhenSpeech: true,
            flushOnSpeechEnd: false,
            ringBufferCapacity: 500
        )

        let firstEvents = try engine.process(samples: Array(repeating: Float(0.1), count: 100))
        XCTAssertEqual(firstEvents.last?.transcript.hypothesis, "so i have")

        let snapshot = engine.discardStream(preserveHypothesis: true)
        XCTAssertEqual(snapshot.confirmed, "")
        XCTAssertEqual(snapshot.hypothesis, "so i have")

        let resumedEvents = try engine.process(samples: Array(repeating: Float(0.1), count: 200))
        XCTAssertEqual(resumedEvents.first?.transcript.confirmed, "so i have")
        XCTAssertEqual(resumedEvents.first?.transcript.hypothesis, "created")
    }
}

private struct MockModel: TranscriptionModel {
    private var outputs: [String]
    private var index: Int = 0

    init(outputs: [String]) {
        self.outputs = outputs
    }

    mutating func transcribeChunk(_ samples: [Float], sampleRate: Int) throws -> String {
        let safeIndex = min(index, max(outputs.count - 1, 0))
        let value = outputs[safeIndex]
        index += 1
        return value
    }

    mutating func resetState() {
        index = 0
    }
}

private struct MockVAD: VoiceActivityDetecting {
    private let decisions: [Bool]
    private let energy: Float
    private var index: Int = 0

    init(decisions: [Bool], energyDBFS: Float) {
        self.decisions = decisions
        self.energy = energyDBFS
    }

    mutating func process(samples: [Float], sampleRate: Int) -> VADDecision {
        let i = min(index, max(decisions.count - 1, 0))
        let decision = decisions[i]
        index += 1
        return VADDecision(isSpeech: decision, energyDBFS: energy)
    }

    mutating func reset() {
        index = 0
    }
}

private final class RecordingModel: TranscriptionModel {
    private(set) var seenCounts: [Int] = []

    func transcribeChunk(_ samples: [Float], sampleRate: Int) throws -> String {
        seenCounts.append(samples.count)
        return "x"
    }

    func resetState() {}
}

private final class NonResettingMockModel: TranscriptionModel {
    private let outputs: [String]
    private var index: Int = 0

    init(outputs: [String]) {
        self.outputs = outputs
    }

    func transcribeChunk(_ samples: [Float], sampleRate: Int) throws -> String {
        let safeIndex = min(index, max(outputs.count - 1, 0))
        let value = outputs[safeIndex]
        index += 1
        return value
    }

    func resetState() {
        // Intentionally no-op to simulate a model keeping streaming state
        // across VAD segment boundaries.
    }
}
