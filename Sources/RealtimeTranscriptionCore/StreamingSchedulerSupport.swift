import Foundation

public enum StreamingSchedulerSupport {
    public static func wordCount(in text: String) -> Int {
        text.split(whereSeparator: \.isWhitespace).count
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
