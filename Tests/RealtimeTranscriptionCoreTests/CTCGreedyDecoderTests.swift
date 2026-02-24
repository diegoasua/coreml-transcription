import XCTest
@testable import RealtimeTranscriptionCore

final class CTCGreedyDecoderTests: XCTestCase {
    func testCollapseCTCRemovesBlanksAndRepeats() {
        let decoder = CTCGreedyDecoder(vocabulary: ["<blk>", "h", "e", "l", "o"], blankTokenID: 0)
        let raw = [0, 1, 1, 0, 2, 3, 3, 4, 0]
        XCTAssertEqual(decoder.collapseCTC(raw), [1, 2, 3, 4])
    }

    func testSentencePieceStyleDecoding() throws {
        let decoder = CTCGreedyDecoder(
            vocabulary: ["<blk>", "▁hello", "▁world", "!"],
            blankTokenID: 0
        )
        let text = try decoder.decode(tokenIDs: [1, 2, 3])
        XCTAssertEqual(text, "hello world!")
    }
}
