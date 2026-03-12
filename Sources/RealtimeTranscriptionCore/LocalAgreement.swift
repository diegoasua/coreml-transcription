import Foundation

public struct LocalAgreementConfig {
    public let requiredAgreementCount: Int

    public init(requiredAgreementCount: Int = 2) {
        self.requiredAgreementCount = max(1, requiredAgreementCount)
    }
}

public struct LocalAgreementStabilizer {
    private let config: LocalAgreementConfig
    private var recentPartialHypotheses: [[String]]

    public init(config: LocalAgreementConfig = .init()) {
        self.config = config
        self.recentPartialHypotheses = []
    }

    public mutating func push(partial: String) -> TranscriptState {
        let words = Self.tokenize(partial)
        recentPartialHypotheses.append(words)
        if recentPartialHypotheses.count > config.requiredAgreementCount {
            recentPartialHypotheses.removeFirst(recentPartialHypotheses.count - config.requiredAgreementCount)
        }

        let confirmedWords: [String]
        if recentPartialHypotheses.count == config.requiredAgreementCount {
            confirmedWords = Self.commonPrefix(in: recentPartialHypotheses)
        } else {
            confirmedWords = []
        }

        let latestWords = recentPartialHypotheses.last ?? []
        let hypothesisWords = Array(latestWords.dropFirst(min(confirmedWords.count, latestWords.count)))
        return TranscriptState(
            confirmed: confirmedWords.joined(separator: " "),
            hypothesis: hypothesisWords.joined(separator: " ")
        )
    }

    public mutating func seed(unconfirmedHypothesisWords words: [String]) {
        guard !words.isEmpty else { return }
        recentPartialHypotheses = [words]
    }

    public mutating func flushSegment() -> TranscriptState {
        guard let latest = recentPartialHypotheses.last else {
            return TranscriptState(confirmed: "", hypothesis: "")
        }
        recentPartialHypotheses.removeAll(keepingCapacity: true)
        return TranscriptState(confirmed: latest.joined(separator: " "), hypothesis: "")
    }

    public mutating func resetAll() {
        recentPartialHypotheses.removeAll(keepingCapacity: true)
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
