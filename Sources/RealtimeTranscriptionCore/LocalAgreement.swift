import Foundation

public struct LocalAgreementConfig {
    public let requiredAgreementCount: Int

    public init(requiredAgreementCount: Int = 2) {
        self.requiredAgreementCount = max(1, requiredAgreementCount)
    }
}

public struct LocalAgreementStabilizer {
    private let config: LocalAgreementConfig
    private var confirmedWords: [String]
    private var recentUnconfirmedHypotheses: [[String]]

    public init(config: LocalAgreementConfig = .init()) {
        self.config = config
        self.confirmedWords = []
        self.recentUnconfirmedHypotheses = []
    }

    public mutating func push(partial: String) -> TranscriptState {
        let words = Self.tokenize(partial)
        let tailWords = Self.unconfirmedTail(words: words, confirmedWords: confirmedWords)
        recentUnconfirmedHypotheses.append(tailWords)
        if recentUnconfirmedHypotheses.count > config.requiredAgreementCount {
            recentUnconfirmedHypotheses.removeFirst(recentUnconfirmedHypotheses.count - config.requiredAgreementCount)
        }

        // LocalAgreement-n over unconfirmed tails:
        // promote only the common prefix shared by the last n tails.
        if recentUnconfirmedHypotheses.count == config.requiredAgreementCount {
            let stableTailPrefix = Self.commonPrefix(in: recentUnconfirmedHypotheses)
            if !stableTailPrefix.isEmpty {
                confirmedWords.append(contentsOf: stableTailPrefix)
                recentUnconfirmedHypotheses = recentUnconfirmedHypotheses.map { hypothesis in
                    Array(hypothesis.dropFirst(min(stableTailPrefix.count, hypothesis.count)))
                }
            }
        }

        let hypothesisWords = recentUnconfirmedHypotheses.last ?? []
        return TranscriptState(
            confirmed: confirmedWords.joined(separator: " "),
            hypothesis: hypothesisWords.joined(separator: " ")
        )
    }

    public mutating func flushSegment() -> TranscriptState {
        guard let latest = recentUnconfirmedHypotheses.last else {
            return TranscriptState(confirmed: confirmedWords.joined(separator: " "), hypothesis: "")
        }
        if !latest.isEmpty {
            confirmedWords.append(contentsOf: latest)
        }
        recentUnconfirmedHypotheses.removeAll(keepingCapacity: true)
        return TranscriptState(confirmed: confirmedWords.joined(separator: " "), hypothesis: "")
    }

    public mutating func resetAll() {
        confirmedWords.removeAll(keepingCapacity: true)
        recentUnconfirmedHypotheses.removeAll(keepingCapacity: true)
    }

    private static func commonPrefix(in sequences: [[String]]) -> [String] {
        guard var prefix = sequences.first else { return [] }
        for seq in sequences.dropFirst() {
            let count = min(prefix.count, seq.count)
            var idx = 0
            while idx < count && prefix[idx] == seq[idx] {
                idx += 1
            }
            prefix = Array(prefix.prefix(idx))
            if prefix.isEmpty {
                return []
            }
        }
        return prefix
    }

    private static func tokenize(_ text: String) -> [String] {
        text
            .split(whereSeparator: \.isWhitespace)
            .map(String.init)
    }

    private static func unconfirmedTail(words: [String], confirmedWords: [String]) -> [String] {
        guard !words.isEmpty else { return [] }
        guard !confirmedWords.isEmpty else { return words }

        // Confirmed words are immutable once committed. Even if the model
        // revises them retroactively, keep them frozen and expose only the tail.
        let overlap = suffixPrefixOverlap(confirmedWords, words, maxOverlap: 64)
        let drop = min(words.count, max(confirmedWords.count, overlap))
        return Array(words.dropFirst(drop))
    }

    private static func suffixPrefixOverlap(_ lhs: [String], _ rhs: [String], maxOverlap: Int) -> Int {
        let limit = min(maxOverlap, min(lhs.count, rhs.count))
        guard limit > 0 else { return 0 }
        for candidate in stride(from: limit, through: 1, by: -1) {
            if Array(lhs.suffix(candidate)) == Array(rhs.prefix(candidate)) {
                return candidate
            }
        }
        return 0
    }
}
