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
    private var carryoverHypothesisWords: [String]

    public init(requiredAgreementCount: Int = 2, draftAgreementCount: Int = 1) {
        let cfg = LocalAgreementConfig(requiredAgreementCount: requiredAgreementCount)
        self.config = cfg
        self.draftAgreementCount = max(1, draftAgreementCount)
        self.stabilizer = LocalAgreementStabilizer(config: cfg)
        self.draftHypotheses = []
        self.transcriptBuffer = RealtimeTranscriptBuffer()
        self.carryoverHypothesisWords = []
    }

    public mutating func update(partialText: String) -> TranscriptState {
        let partialWords = Self.collapseStreamingSelfOverlap(words: Self.tokenize(partialText))
        if !carryoverHypothesisWords.isEmpty, !partialWords.isEmpty, draftHypotheses.isEmpty {
            stabilizer.seed(unconfirmedHypothesisWords: carryoverHypothesisWords)
            draftHypotheses = [carryoverHypothesisWords]
            carryoverHypothesisWords.removeAll(keepingCapacity: true)
        }

        let segment = stabilizer.push(partial: partialWords.joined(separator: " "))
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
        carryoverHypothesisWords.removeAll(keepingCapacity: true)
        let snapshot = transcriptBuffer.snapshot()
        return TranscriptState(confirmed: snapshot.confirmedText, hypothesis: snapshot.hypothesisText)
    }

    public mutating func discardCurrentSegment(preserveHypothesis: Bool = false) -> TranscriptState {
        let mergedWords = preserveHypothesis ? Self.tokenize(transcriptBuffer.snapshot().mergedText) : []
        let snapshot: TranscriptBufferSnapshot
        if preserveHypothesis, !mergedWords.isEmpty {
            snapshot = transcriptBuffer.discardCurrentSegment(preservingMergedWords: mergedWords)
            carryoverHypothesisWords = Self.trailingWords(
                Self.tokenize(snapshot.hypothesisText),
                maxCount: 32
            )
        } else {
            carryoverHypothesisWords.removeAll(keepingCapacity: true)
            snapshot = transcriptBuffer.discardCurrentSegment()
        }
        stabilizer = LocalAgreementStabilizer(config: config)
        draftHypotheses.removeAll(keepingCapacity: true)
        return TranscriptState(confirmed: snapshot.confirmedText, hypothesis: snapshot.hypothesisText)
    }

    public mutating func reset() {
        stabilizer = LocalAgreementStabilizer(config: config)
        draftHypotheses.removeAll(keepingCapacity: true)
        transcriptBuffer.reset()
        carryoverHypothesisWords.removeAll(keepingCapacity: true)
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

    static func collapseStreamingSelfOverlap(words: [String]) -> [String] {
        var current = words
        while let collapsed = collapseStreamingSelfOverlapOnce(words: current), collapsed.count < current.count {
            current = collapsed
        }
        return current
    }

    private static func collapseStreamingSelfOverlapOnce(words: [String]) -> [String]? {
        guard words.count >= 4 else { return nil }

        var best = words
        var improved = false
        for split in 1..<words.count {
            let prefix = Array(words[..<split])
            let suffixAll = Array(words[split...])
            let maxSkip = min(2, max(0, suffixAll.count - 1))
            for skip in 0...maxSkip {
                let suffix = Array(suffixAll.dropFirst(skip))
                let overlap = suffixPrefixOverlapNormalized(
                    prefix,
                    suffix,
                    maxOverlap: min(prefix.count, suffix.count)
                )
                let fullyRepeatedTail = overlap == suffix.count && suffix.count >= 2
                guard overlap >= 3 || fullyRepeatedTail else { continue }
                let merged = prefix + suffix.dropFirst(overlap)
                if merged.count < best.count {
                    best = merged
                    improved = true
                }
            }
        }
        return improved ? best : nil
    }

    private static func suffixPrefixOverlapNormalized(_ lhs: [String], _ rhs: [String], maxOverlap: Int) -> Int {
        let limit = min(maxOverlap, min(lhs.count, rhs.count))
        guard limit > 0 else { return 0 }
        let normalizedLHS = lhs.map(normalizedStreamingWord)
        let normalizedRHS = rhs.map(normalizedStreamingWord)
        for candidate in stride(from: limit, through: 1, by: -1) {
            if Array(normalizedLHS.suffix(candidate)) == Array(normalizedRHS.prefix(candidate)) {
                return candidate
            }
        }
        return 0
    }

    private static func normalizedStreamingWord(_ word: String) -> String {
        word.trimmingCharacters(in: .punctuationCharacters).lowercased()
    }

    private static func trailingWords(_ words: [String], maxCount: Int) -> [String] {
        guard words.count > maxCount else { return words }
        return Array(words.suffix(maxCount))
    }
}
