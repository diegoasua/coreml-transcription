import Foundation

public enum StreamingSchedulerSupport {
    public static func wordCount(in text: String) -> Int {
        text.split(whereSeparator: \.isWhitespace).count
    }

    public static func catchUpBatchCount(
        queuedUnitCount: Int,
        baseBatchCount: Int,
        catchUpTargetUnitCount: Int,
        minContextUnitCount: Int
    ) -> Int {
        guard queuedUnitCount > 0 else {
            return 0
        }
        let target = max(baseBatchCount, catchUpTargetUnitCount, minContextUnitCount)
        return min(queuedUnitCount, target)
    }

    public static func latestFirstCatchUpTailCount(
        queuedUnitCount: Int,
        baseBatchCount: Int,
        catchUpTargetUnitCount: Int,
        minContextUnitCount: Int
    ) -> Int {
        catchUpBatchCount(
            queuedUnitCount: queuedUnitCount,
            baseBatchCount: baseBatchCount,
            catchUpTargetUnitCount: catchUpTargetUnitCount,
            minContextUnitCount: minContextUnitCount
        )
    }

    public static func shouldUseLatestFirstCatchUp(
        latestFirstEnabled: Bool,
        protectOnset: Bool,
        queuedSampleCount: Int,
        catchUpTriggerSamples: Int
    ) -> Bool {
        guard latestFirstEnabled,
              !protectOnset,
              catchUpTriggerSamples > 0 else {
            return false
        }
        return queuedSampleCount > catchUpTriggerSamples
    }

    public static func shouldProtectLatestFirstOnset(
        elapsedSec: Double?,
        currentConfirmed: String,
        baselineConfirmed: String,
        minConfirmedWords: Int,
        maxOnsetSec: Double
    ) -> Bool {
        guard let elapsedSec,
              elapsedSec >= 0,
              maxOnsetSec > 0,
              elapsedSec <= maxOnsetSec else {
            return false
        }

        guard wordCount(in: baselineConfirmed) == 0 else {
            return false
        }

        let requiredWords = max(1, minConfirmedWords)
        let confirmedAdvance = max(
            0,
            wordCount(in: currentConfirmed) - wordCount(in: baselineConfirmed)
        )
        return confirmedAdvance < requiredWords
    }
}
