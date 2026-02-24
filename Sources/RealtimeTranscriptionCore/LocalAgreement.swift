import Foundation

public struct LocalAgreementConfig {
    public let requiredAgreementCount: Int

    public init(requiredAgreementCount: Int = 2) {
        self.requiredAgreementCount = max(2, requiredAgreementCount)
    }
}

public struct LocalAgreementStabilizer {
    private let config: LocalAgreementConfig
    private var confirmedWords: [String]
    private var recentHypotheses: [[String]]

    public init(config: LocalAgreementConfig = .init()) {
        self.config = config
        self.confirmedWords = []
        self.recentHypotheses = []
    }

    public mutating func push(partial: String) -> TranscriptState {
        let words = Self.tokenize(partial)
        recentHypotheses.append(words)
        if recentHypotheses.count > config.requiredAgreementCount {
            recentHypotheses.removeFirst(recentHypotheses.count - config.requiredAgreementCount)
        }

        if recentHypotheses.count == config.requiredAgreementCount {
            let stablePrefix = Self.commonPrefix(in: recentHypotheses)
            if stablePrefix.count > confirmedWords.count {
                confirmedWords = Array(stablePrefix)
            }
        }

        let hypothesisWords = words.dropFirst(min(words.count, confirmedWords.count))
        return TranscriptState(
            confirmed: confirmedWords.joined(separator: " "),
            hypothesis: hypothesisWords.joined(separator: " ")
        )
    }

    public mutating func flushSegment() -> TranscriptState {
        guard let latest = recentHypotheses.last else {
            return TranscriptState(confirmed: confirmedWords.joined(separator: " "), hypothesis: "")
        }
        if latest.count > confirmedWords.count {
            confirmedWords = latest
        }
        recentHypotheses.removeAll(keepingCapacity: true)
        return TranscriptState(confirmed: confirmedWords.joined(separator: " "), hypothesis: "")
    }

    public mutating func resetAll() {
        confirmedWords.removeAll(keepingCapacity: true)
        recentHypotheses.removeAll(keepingCapacity: true)
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
}
