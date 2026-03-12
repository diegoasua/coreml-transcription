#if canImport(CoreML)
import XCTest
@testable import RealtimeTranscriptionCore

final class ParakeetDecoderSchedulingTests: XCTestCase {
    func testEmitPrevTokenUsesAutoCommitGateWhenStepIndexIsAvailable() {
        let policy = ParakeetCoreMLTDTTranscriptionModel.StatefulCommitPolicy.emitPrevToken
        XCTAssertEqual(policy.firstPassGateValue(commitState: true, canAutoCommitNonBlank: true), 2)
        XCTAssertFalse(policy.requiresExplicitAdvanceAfterEmit(canAutoCommitNonBlank: true))
    }

    func testEmitPrevTokenFallsBackToExplicitAdvanceWithoutAutoCommitSupport() {
        let policy = ParakeetCoreMLTDTTranscriptionModel.StatefulCommitPolicy.emitPrevToken
        XCTAssertEqual(policy.firstPassGateValue(commitState: true, canAutoCommitNonBlank: false), 0)
        XCTAssertTrue(policy.requiresExplicitAdvanceAfterEmit(canAutoCommitNonBlank: false))
    }

    func testEmitTokenStillRequiresExplicitAdvance() {
        let policy = ParakeetCoreMLTDTTranscriptionModel.StatefulCommitPolicy.emitToken
        XCTAssertEqual(policy.firstPassGateValue(commitState: true, canAutoCommitNonBlank: true), 0)
        XCTAssertTrue(policy.requiresExplicitAdvanceAfterEmit(canAutoCommitNonBlank: true))
    }

    func testStreamingEncoderStepSchedulerCarriesFractionalBudget() {
        var scheduler = StreamingEncoderStepScheduler()

        XCTAssertEqual(
            scheduler.consumeStepBudget(newFeatureFrames: 3, encoderSteps: 16, windowFrames: 64, minNewSteps: 1),
            0
        )
        XCTAssertEqual(
            scheduler.consumeStepBudget(newFeatureFrames: 3, encoderSteps: 16, windowFrames: 64, minNewSteps: 1),
            1
        )
        XCTAssertEqual(
            scheduler.consumeStepBudget(newFeatureFrames: 3, encoderSteps: 16, windowFrames: 64, minNewSteps: 1),
            1
        )
    }

    func testStreamingEncoderStepSchedulerTracksWindowRatio() {
        var scheduler = StreamingEncoderStepScheduler()

        let scheduled = (0..<4).map { _ in
            scheduler.consumeStepBudget(newFeatureFrames: 8, encoderSteps: 24, windowFrames: 96, minNewSteps: 1)
        }

        XCTAssertEqual(scheduled, [2, 2, 2, 2])
    }

    func testRollingAnchorPromotesOnlyAfterStrongSharedPrefix() {
        XCTAssertTrue(
            ParakeetCoreMLTDTTranscriptionModel.shouldPromoteRollingAnchor(
                previous: [1, 2, 3, 4, 5, 6, 7, 8],
                current: [1, 2, 3, 4, 5, 6, 9, 10],
                minTokenCount: 6
            )
        )
        XCTAssertFalse(
            ParakeetCoreMLTDTTranscriptionModel.shouldPromoteRollingAnchor(
                previous: [1, 2, 3, 4, 5, 6],
                current: [1, 2, 9, 10, 11, 12],
                minTokenCount: 6
            )
        )
    }

    func testRewritePrefixDecodesImmediatelyWithoutPreviousText() {
        XCTAssertTrue(
            ParakeetCoreMLTDTTranscriptionModel.shouldDecodeRewritePrefix(
                hasPreviousText: false,
                pendingNewSamples: 0,
                requiredSamples: 4000
            )
        )
    }

    func testRewritePrefixDecodeGateUsesPendingSamplesInsteadOfWindowLength() {
        XCTAssertTrue(
            ParakeetCoreMLTDTTranscriptionModel.shouldDecodeRewritePrefix(
                hasPreviousText: true,
                pendingNewSamples: 7200,
                requiredSamples: 5600
            )
        )
        XCTAssertFalse(
            ParakeetCoreMLTDTTranscriptionModel.shouldDecodeRewritePrefix(
                hasPreviousText: true,
                pendingNewSamples: 3200,
                requiredSamples: 5600
            )
        )
    }

    func testRewritePrefixRequiredSamplesUsesFasterOnsetStrideBeforeTranscriptIsAnchored() {
        XCTAssertEqual(
            ParakeetCoreMLTDTTranscriptionModel.rewritePrefixRequiredSamples(
                stableRequiredSamples: 9600,
                onsetStrideSamples: 4000,
                onsetMaxSamples: 32000,
                onsetMinWordCount: 4,
                currentText: "So I",
                bufferedSamples: 12000
            ),
            4000
        )
    }

    func testRewritePrefixRequiredSamplesFallsBackToStableStrideAfterOnset() {
        XCTAssertEqual(
            ParakeetCoreMLTDTTranscriptionModel.rewritePrefixRequiredSamples(
                stableRequiredSamples: 9600,
                onsetStrideSamples: 4000,
                onsetMaxSamples: 32000,
                onsetMinWordCount: 4,
                currentText: "So I have created here a MVVM kind",
                bufferedSamples: 64000
            ),
            9600
        )
    }

    func testRewritePrefixDecoderResumeStaysOffDuringOnset() {
        XCTAssertFalse(
            ParakeetCoreMLTDTTranscriptionModel.shouldAllowRewritePrefixDecoderResume(
                currentText: "So",
                bufferedSamples: 9_600,
                onsetMaxSamples: 12_800,
                minWordCount: 2
            )
        )
    }

    func testRewritePrefixDecoderResumeTurnsOnAfterAnchorWords() {
        XCTAssertTrue(
            ParakeetCoreMLTDTTranscriptionModel.shouldAllowRewritePrefixDecoderResume(
                currentText: "So I",
                bufferedSamples: 9_600,
                onsetMaxSamples: 12_800,
                minWordCount: 2
            )
        )
    }

    func testRewritePrefixDecoderResumeTurnsOnAfterOnsetWindowExpires() {
        XCTAssertTrue(
            ParakeetCoreMLTDTTranscriptionModel.shouldAllowRewritePrefixDecoderResume(
                currentText: "",
                bufferedSamples: 16_000,
                onsetMaxSamples: 12_800,
                minWordCount: 2
            )
        )
    }

    func testLongformWindowPrefixReuseKeepsUnchangedLeadingWindows() {
        let previous = ParakeetCoreMLTDTTranscriptionModel.makeLongformWindowPlans(
            totalFrames: 2200,
            encoderFrameCount: 1200,
            leftContextFrames: 160,
            rightContextFrames: 0,
            allowRightContext: false
        ).map(\.key)
        let current = ParakeetCoreMLTDTTranscriptionModel.makeLongformWindowPlans(
            totalFrames: 2600,
            encoderFrameCount: 1200,
            leftContextFrames: 160,
            rightContextFrames: 0,
            allowRightContext: false
        ).map(\.key)

        XCTAssertEqual(
            ParakeetCoreMLTDTTranscriptionModel.matchingLongformWindowPrefixCount(
                previous: previous,
                current: current
            ),
            2
        )
    }

    func testResumableLongformWindowPrefixStopsAtFirstMissingCheckpoint() {
        XCTAssertEqual(
            ParakeetCoreMLTDTTranscriptionModel.resumableLongformWindowPrefixCount(
                hasCheckpoints: [true, true, false, true],
                reuseCount: 4
            ),
            2
        )
        XCTAssertEqual(
            ParakeetCoreMLTDTTranscriptionModel.resumableLongformWindowPrefixCount(
                hasCheckpoints: [true, true, true],
                reuseCount: 2
            ),
            2
        )
    }

    func testProjectLongformWindowUsesFullTailForFinalPartialWindow() {
        let projected = ParakeetCoreMLTDTTranscriptionModel.projectLongformWindowToEncoderSteps(
            rawSteps: 300,
            inputStart: 2080,
            centerStart: 2080,
            actualFrames: 920,
            centerFrames: 520,
            hopFrames: 1040
        )

        XCTAssertEqual(projected.start, 0)
        XCTAssertEqual(projected.count, 300)
    }

    func testMergeRewritePrefixTranscriptKeepsFullPrefixBeforeTrim() {
        XCTAssertEqual(
            ParakeetCoreMLTDTTranscriptionModel.mergeRewritePrefixTranscript(
                previous: "So I have",
                currentWindow: "So I have created here",
                hasTrimmedHistory: false
            ),
            "So I have created here"
        )
    }

    func testMergeRewritePrefixTranscriptAppendsSlidingWindowTailAfterTrim() {
        XCTAssertEqual(
            ParakeetCoreMLTDTTranscriptionModel.mergeRewritePrefixTranscript(
                previous: "So I have created here a M V V M kind of style for a chatbot",
                currentWindow: "kind of style for a chatbot that works locally within the computer",
                hasTrimmedHistory: true
            ),
            "So I have created here a M V V M kind of style for a chatbot that works locally within the computer"
        )
    }

    func testAnchoredRollingDecodeFrameCountTracksNewStepsWithFloor() {
        XCTAssertEqual(
            ParakeetCoreMLTDTTranscriptionModel.anchoredRollingDecodeFrameCount(
                encoderSteps: 24,
                approxNewSteps: 1,
                baseDecodeFrames: 12,
                stepMultiplier: 4,
                minTailFrames: 8
            ),
            12
        )
        XCTAssertEqual(
            ParakeetCoreMLTDTTranscriptionModel.anchoredRollingDecodeFrameCount(
                encoderSteps: 24,
                approxNewSteps: 5,
                baseDecodeFrames: 12,
                stepMultiplier: 4,
                minTailFrames: 8
            ),
            20
        )
    }

    func testRollingTailMergeAcceptsAlignedReplaceButRejectsReplaceAll() {
        XCTAssertTrue(
            ParakeetCoreMLTDTTranscriptionModel.shouldAcceptRollingTailMerge(
                result: .init(merged: [1, 2, 3, 4], strategy: "aligned_replace", alignmentStart: 2, matchLength: 3),
                baseCount: 8,
                nextCount: 4
            )
        )
        XCTAssertFalse(
            ParakeetCoreMLTDTTranscriptionModel.shouldAcceptRollingTailMerge(
                result: .init(merged: [4, 5, 6], strategy: "replace_all", alignmentStart: nil, matchLength: 0),
                baseCount: 8,
                nextCount: 3
            )
        )
    }

    func testBlankDurationTieBreakerPrefersShorterNearTie() {
        let scores: [Float] = [-14.1, -1.17, -1.00, -1.60, -2.11]

        let selected = ParakeetCoreMLTDTTranscriptionModel.selectDurationIndexForTesting(
            scores: scores,
            tokenID: 1024,
            blankID: 1024,
            durations: [0, 1, 2, 3, 4],
            blankTieMargin: 0.25
        )

        XCTAssertEqual(selected, 1)
    }

    func testBlankDurationTieBreakerKeepsBestWhenGapIsLarge() {
        let scores: [Float] = [-10.0, -1.50, -0.90, -2.50, -3.00]

        let selected = ParakeetCoreMLTDTTranscriptionModel.selectDurationIndexForTesting(
            scores: scores,
            tokenID: 1024,
            blankID: 1024,
            durations: [0, 1, 2, 3, 4],
            blankTieMargin: 0.25
        )

        XCTAssertEqual(selected, 2)
    }

    func testMergeWindowHypothesisReplacesUnstableTailFromBestAlignment() {
        let base = [99, 10, 11, 77, 10, 11, 12, 13]
        let next = [10, 11, 12, 13, 14, 15]

        let merged = ParakeetCoreMLTDTTranscriptionModel.mergeWindowHypothesis(base: base, next: next)

        XCTAssertEqual(merged, [99, 10, 11, 77, 10, 11, 12, 13, 14, 15])
    }

    func testMergeWindowHypothesisReplacesWholeHypothesisWhenIncomingWindowIsLonger() {
        let base = [1, 2, 3]
        let next = [4, 5, 6, 7, 8]

        let merged = ParakeetCoreMLTDTTranscriptionModel.mergeWindowHypothesis(base: base, next: next)

        XCTAssertEqual(merged, next)
    }

    func testRefineRollingWindowMergeDefersShortAppendFallback() {
        let vocab = ["<blk>", "▁So", "▁I", "▁have", "▁Um"]
        let base = [1, 2, 3]
        let next = [4]
        let raw = ParakeetCoreMLTDTTranscriptionModel.WindowMergeResult(
            merged: base + next,
            strategy: "append_with_overlap",
            alignmentStart: nil,
            matchLength: 0
        )

        let refined = ParakeetCoreMLTDTTranscriptionModel.refineRollingWindowMergeResult(
            raw,
            base: base,
            next: next,
            vocab: vocab
        )

        XCTAssertEqual(refined.strategy, "defer_short_append")
        XCTAssertEqual(refined.merged, base)
    }

    func testRefineRollingWindowMergeDefersWordOverlapAppendFallback() {
        let vocab = ["<blk>", "▁So", "▁I", "▁have", "▁creat", "▁here", "▁upgrade", "▁path", "▁path", "▁again"]
        let base = [1, 2, 3, 4, 5, 6, 7, 8]
        let next = [6, 7, 8, 9]
        let raw = ParakeetCoreMLTDTTranscriptionModel.WindowMergeResult(
            merged: base + next,
            strategy: "append_with_overlap",
            alignmentStart: nil,
            matchLength: 0
        )

        let refined = ParakeetCoreMLTDTTranscriptionModel.refineRollingWindowMergeResult(
            raw,
            base: base,
            next: next,
            vocab: vocab
        )

        XCTAssertEqual(refined.strategy, "defer_word_overlap_append")
        XCTAssertEqual(refined.merged, base)
    }

    func testRefineRollingWindowMergeDefersHighOverlapReplayFallback() {
        let vocab = ["<blk>", "▁alpha", "▁beta", "▁gamma", "▁delta", "▁epsilon", "▁zeta"]
        let base = [1, 2, 3, 4, 5]
        let next = [6, 2, 3, 4]
        let raw = ParakeetCoreMLTDTTranscriptionModel.WindowMergeResult(
            merged: base + next,
            strategy: "append_with_overlap",
            alignmentStart: nil,
            matchLength: 0
        )

        let refined = ParakeetCoreMLTDTTranscriptionModel.refineRollingWindowMergeResult(
            raw,
            base: base,
            next: next,
            vocab: vocab
        )

        XCTAssertEqual(refined.strategy, "defer_high_overlap_append")
        XCTAssertEqual(refined.merged, base)
    }

}
#endif
