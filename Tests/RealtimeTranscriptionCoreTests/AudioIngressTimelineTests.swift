import XCTest
@testable import RealtimeTranscriptionCore

final class AudioIngressTimelineTests: XCTestCase {
    func testReturnsIngressTimeForSampleCursor() {
        var timeline = AudioIngressTimeline(sampleRate: 16_000)
        _ = timeline.recordIngress(sampleCount: 4000, receivedAtSec: 1.0)
        _ = timeline.recordIngress(sampleCount: 4000, receivedAtSec: 1.25)

        XCTAssertEqual(timeline.ingressTime(forSampleCursor: 1) ?? -1, 1.0, accuracy: 1e-9)
        XCTAssertEqual(timeline.ingressTime(forSampleCursor: 4000) ?? -1, 1.0, accuracy: 1e-9)
        XCTAssertEqual(timeline.ingressTime(forSampleCursor: 4001) ?? -1, 1.25, accuracy: 1e-9)
        XCTAssertEqual(timeline.ingressTime(forAudioCursorSec: 0.5) ?? -1, 1.25, accuracy: 1e-9)
    }

    func testResetClearsRecordedIngress() {
        var timeline = AudioIngressTimeline(sampleRate: 16_000)
        _ = timeline.recordIngress(sampleCount: 4000, receivedAtSec: 1.0)

        timeline.reset()

        XCTAssertNil(timeline.ingressTime(forSampleCursor: 1))
        XCTAssertEqual(timeline.totalRecordedSamples, 0)
    }
}
