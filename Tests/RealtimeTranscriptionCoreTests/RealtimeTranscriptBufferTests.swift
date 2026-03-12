import XCTest
@testable import RealtimeTranscriptionCore

final class RealtimeTranscriptBufferTests: XCTestCase {
    func testUpdateTracksCommittedAndMutable() {
        var buffer = RealtimeTranscriptBuffer()
        let first = buffer.update(
            confirmedSegmentWords: ["hello"],
            hypothesisWords: ["world"]
        )

        XCTAssertEqual(first.confirmedText, "hello")
        XCTAssertEqual(first.hypothesisText, "world")
        XCTAssertEqual(first.mergedText, "hello world")
        XCTAssertEqual(first.revision, 1)
    }

    func testRevisionOnlyAdvancesOnTextChange() {
        var buffer = RealtimeTranscriptBuffer()
        _ = buffer.update(confirmedSegmentWords: ["alpha"], hypothesisWords: ["beta"])
        let revAfterFirst = buffer.revision

        _ = buffer.update(confirmedSegmentWords: ["alpha"], hypothesisWords: ["beta"])
        XCTAssertEqual(buffer.revision, revAfterFirst)

        _ = buffer.update(confirmedSegmentWords: ["alpha", "beta"], hypothesisWords: [])
        XCTAssertEqual(buffer.revision, revAfterFirst)
    }

    func testFinalizeSegmentCommitsTail() {
        var buffer = RealtimeTranscriptBuffer()
        _ = buffer.update(confirmedSegmentWords: ["how"], hypothesisWords: ["are"])

        let final = buffer.finalizeSegment(finalSegmentWords: ["how", "are", "you"])
        XCTAssertEqual(final.confirmedText, "how are you")
        XCTAssertEqual(final.hypothesisText, "")
    }

    func testOverlapMergeDoesNotDuplicateAcrossSegments() {
        var buffer = RealtimeTranscriptBuffer()
        _ = buffer.finalizeSegment(finalSegmentWords: ["so", "i", "have", "created"])
        let second = buffer.finalizeSegment(finalSegmentWords: ["so", "i", "have", "created", "here"])
        XCTAssertEqual(second.confirmedText, "so i have created here")
    }

    func testDiscardCurrentSegmentKeepsStableConfirmedPrefix() {
        var buffer = RealtimeTranscriptBuffer()
        _ = buffer.update(confirmedSegmentWords: ["so", "i", "have"], hypothesisWords: ["created"])

        let snapshot = buffer.discardCurrentSegment()
        XCTAssertEqual(snapshot.confirmedText, "so i have")
        XCTAssertEqual(snapshot.hypothesisText, "")
    }

    func testDiscardCurrentSegmentPreserveKeepsStableConfirmedPrefixAndTail() {
        var buffer = RealtimeTranscriptBuffer()
        _ = buffer.update(confirmedSegmentWords: ["so", "i", "have"], hypothesisWords: ["created", "here"])

        let snapshot = buffer.discardCurrentSegment(
            preservingMergedWords: ["so", "i", "have", "created", "here"]
        )
        XCTAssertEqual(snapshot.confirmedText, "so i have")
        XCTAssertEqual(snapshot.hypothesisText, "created here")
    }
}
