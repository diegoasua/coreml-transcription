import XCTest
@testable import RealtimeTranscriptionCore

final class StreamingSchedulerSupportTests: XCTestCase {
    func testCatchUpBatchCountUsesConfiguredTarget() {
        XCTAssertEqual(
            StreamingSchedulerSupport.catchUpBatchCount(
                queuedUnitCount: 7,
                baseBatchCount: 2,
                catchUpTargetUnitCount: 6,
                minContextUnitCount: 2
            ),
            6
        )
    }

    func testLatestFirstCatchUpTailCountKeepsTargetSizedTail() {
        XCTAssertEqual(
            StreamingSchedulerSupport.latestFirstCatchUpTailCount(
                queuedUnitCount: 7,
                baseBatchCount: 2,
                catchUpTargetUnitCount: 6,
                minContextUnitCount: 2
            ),
            6
        )
    }

    func testLatestFirstCatchUpTailCountNeverDropsBelowBaseBatch() {
        XCTAssertEqual(
            StreamingSchedulerSupport.latestFirstCatchUpTailCount(
                queuedUnitCount: 2,
                baseBatchCount: 2,
                catchUpTargetUnitCount: 6,
                minContextUnitCount: 2
            ),
            2
        )
    }

    func testUsesLatestFirstCatchUpOnlyAfterSoftBacklogOverflow() {
        XCTAssertFalse(
            StreamingSchedulerSupport.shouldUseLatestFirstCatchUp(
                latestFirstEnabled: true,
                protectOnset: false,
                queuedSampleCount: 3_000,
                catchUpTriggerSamples: 4_000
            )
        )
        XCTAssertTrue(
            StreamingSchedulerSupport.shouldUseLatestFirstCatchUp(
                latestFirstEnabled: true,
                protectOnset: false,
                queuedSampleCount: 4_001,
                catchUpTriggerSamples: 4_000
            )
        )
    }

    func testDoesNotUseLatestFirstCatchUpWhenDisabledOrOnsetProtected() {
        XCTAssertFalse(
            StreamingSchedulerSupport.shouldUseLatestFirstCatchUp(
                latestFirstEnabled: false,
                protectOnset: false,
                queuedSampleCount: 8_000,
                catchUpTriggerSamples: 4_000
            )
        )
        XCTAssertFalse(
            StreamingSchedulerSupport.shouldUseLatestFirstCatchUp(
                latestFirstEnabled: true,
                protectOnset: true,
                queuedSampleCount: 8_000,
                catchUpTriggerSamples: 4_000
            )
        )
    }

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
