import XCTest
@testable import RealtimeTranscriptionCore

final class StreamingSchedulerSupportTests: XCTestCase {
    func testProtectsLatestFirstDuringUnconfirmedSpeechOnset() {
        XCTAssertTrue(
            StreamingSchedulerSupport.shouldProtectLatestFirstOnset(
                elapsedSec: 0.6,
                currentConfirmed: "",
                baselineConfirmed: "",
                minConfirmedWords: 2,
                maxOnsetSec: 1.6
            )
        )
    }

    func testStopsProtectingOnceConfirmedPrefixAdvancesEnough() {
        XCTAssertFalse(
            StreamingSchedulerSupport.shouldProtectLatestFirstOnset(
                elapsedSec: 0.8,
                currentConfirmed: "so i have",
                baselineConfirmed: "",
                minConfirmedWords: 2,
                maxOnsetSec: 1.6
            )
        )
    }

    func testStopsProtectingAfterOnsetWindowExpires() {
        XCTAssertFalse(
            StreamingSchedulerSupport.shouldProtectLatestFirstOnset(
                elapsedSec: 2.0,
                currentConfirmed: "",
                baselineConfirmed: "",
                minConfirmedWords: 2,
                maxOnsetSec: 1.6
            )
        )
    }

    func testDoesNotProtectAfterTranscriptAlreadyHasConfirmedPrefix() {
        XCTAssertFalse(
            StreamingSchedulerSupport.shouldProtectLatestFirstOnset(
                elapsedSec: 0.4,
                currentConfirmed: "so i have created",
                baselineConfirmed: "so i have",
                minConfirmedWords: 2,
                maxOnsetSec: 1.6
            )
        )
    }
}
