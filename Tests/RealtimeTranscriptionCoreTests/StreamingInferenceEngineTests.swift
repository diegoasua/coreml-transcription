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
