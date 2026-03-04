import XCTest
@testable import RealtimeTranscriptionCore

final class StreamingTextControllerTests: XCTestCase {
    func testSplitDraftAndConfirmedChannels() {
        var controller = StreamingTextController(requiredAgreementCount: 2, draftAgreementCount: 1)

        let first = controller.update(partialText: "hello")
        XCTAssertEqual(first.confirmed, "")
        XCTAssertEqual(first.hypothesis, "hello")

        let second = controller.update(partialText: "hello world")
        XCTAssertEqual(second.confirmed, "hello")
        XCTAssertEqual(second.hypothesis, "world")
    }

    func testEndSegmentMergeDoesNotDuplicateOverlappingPrefix() {
        var controller = StreamingTextController(requiredAgreementCount: 1)

        _ = controller.update(partialText: "so i have created")
        let firstFlush = controller.endSegment()
        XCTAssertEqual(firstFlush.confirmed, "so i have created")

        _ = controller.update(partialText: "so i have created here")
        let secondFlush = controller.endSegment()
        XCTAssertEqual(secondFlush.confirmed, "so i have created here")
    }

    func testUpdatePreviewAvoidsCommittedPrefixDuplication() {
        var controller = StreamingTextController(requiredAgreementCount: 1)

        _ = controller.update(partialText: "alpha beta gamma")
        _ = controller.endSegment()

        let preview = controller.update(partialText: "alpha beta gamma delta")
        XCTAssertEqual(preview.confirmed, "alpha beta gamma delta")
    }
}
