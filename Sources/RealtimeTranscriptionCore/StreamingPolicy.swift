import Foundation

public struct StreamingPolicy {
    public let sampleRate: Int
    public let chunkMs: Int
    public let hopMs: Int

    public init(sampleRate: Int = 16_000, chunkMs: Int = 960, hopMs: Int = 320) {
        self.sampleRate = sampleRate
        self.chunkMs = chunkMs
        self.hopMs = hopMs
    }

    public var chunkSamples: Int {
        (sampleRate * chunkMs) / 1000
    }

    public var hopSamples: Int {
        (sampleRate * hopMs) / 1000
    }
}

public struct StreamingTextController {
    private let config: LocalAgreementConfig
    private let draftAgreementCount: Int
    private var stabilizer: LocalAgreementStabilizer
    private var draftHypotheses: [[String]]
    private var transcriptBuffer: RealtimeTranscriptBuffer

    public init(requiredAgreementCount: Int = 2, draftAgreementCount: Int = 1) {
        let cfg = LocalAgreementConfig(requiredAgreementCount: requiredAgreementCount)
        self.config = cfg
        self.draftAgreementCount = max(1, draftAgreementCount)
        self.stabilizer = LocalAgreementStabilizer(config: cfg)
        self.draftHypotheses = []
        self.transcriptBuffer = RealtimeTranscriptBuffer()
    }

    public mutating func update(partialText: String) -> TranscriptState {
        let segment = stabilizer.push(partial: partialText)
        let partialWords = Self.tokenize(partialText)
        draftHypotheses.append(partialWords)
        if draftHypotheses.count > draftAgreementCount {
            draftHypotheses.removeFirst(draftHypotheses.count - draftAgreementCount)
        }

        let confirmedWords = Self.tokenize(segment.confirmed)
        let draftWords = Self.commonPrefix(in: draftHypotheses)
        let hypothesisWords = Self.draftTail(confirmedWords: confirmedWords, draftWords: draftWords)
        let snapshot = transcriptBuffer.update(
            confirmedSegmentWords: confirmedWords,
            hypothesisWords: hypothesisWords
        )
        return TranscriptState(confirmed: snapshot.confirmedText, hypothesis: snapshot.hypothesisText)
    }

    public mutating func endSegment() -> TranscriptState {
        let segment = stabilizer.flushSegment()
        _ = transcriptBuffer.finalizeSegment(finalSegmentWords: Self.tokenize(segment.confirmed))
        stabilizer = LocalAgreementStabilizer(config: config)
        draftHypotheses.removeAll(keepingCapacity: true)
        let snapshot = transcriptBuffer.snapshot()
        return TranscriptState(confirmed: snapshot.confirmedText, hypothesis: snapshot.hypothesisText)
    }

    public mutating func reset() {
        stabilizer = LocalAgreementStabilizer(config: config)
        draftHypotheses.removeAll(keepingCapacity: true)
        transcriptBuffer.reset()
    }

    public func snapshot() -> TranscriptBufferSnapshot {
        transcriptBuffer.snapshot()
    }

    public var revision: Int {
        transcriptBuffer.revision
    }

    private static func tokenize(_ text: String) -> [String] {
        text
            .split(whereSeparator: \.isWhitespace)
            .map(String.init)
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

    private static func draftTail(confirmedWords: [String], draftWords: [String]) -> [String] {
        guard !draftWords.isEmpty else { return [] }
        guard !confirmedWords.isEmpty else { return draftWords }

        let prefixCount = longestCommonPrefixLength(confirmedWords, draftWords)
        let overlapCount = suffixPrefixOverlap(confirmedWords, draftWords, maxOverlap: 64)
        let dropCount = max(prefixCount, overlapCount)
        return Array(draftWords.dropFirst(min(dropCount, draftWords.count)))
    }

    private static func longestCommonPrefixLength(_ lhs: [String], _ rhs: [String]) -> Int {
        let count = min(lhs.count, rhs.count)
        var idx = 0
        while idx < count && lhs[idx] == rhs[idx] {
            idx += 1
        }
        return idx
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
