import Foundation

public struct TimedToken: Equatable {
    public let id: Int
    public let text: String
    public let startStep: Int?
    public let endStep: Int?
    public let logProb: Float?
    public let revision: Int

    public init(
        id: Int,
        text: String,
        startStep: Int? = nil,
        endStep: Int? = nil,
        logProb: Float? = nil,
        revision: Int
    ) {
        self.id = id
        self.text = text
        self.startStep = startStep
        self.endStep = endStep
        self.logProb = logProb
        self.revision = revision
    }
}

public struct TranscriptBufferSnapshot: Equatable {
    public let committed: [TimedToken]
    public let mutable: [TimedToken]
    public let revision: Int

    public init(committed: [TimedToken], mutable: [TimedToken], revision: Int) {
        self.committed = committed
        self.mutable = mutable
        self.revision = revision
    }

    public var confirmedText: String {
        committed.map(\.text).joined(separator: " ")
    }

    public var hypothesisText: String {
        mutable.map(\.text).joined(separator: " ")
    }

    public var mergedText: String {
        (committed + mutable).map(\.text).joined(separator: " ")
    }
}

public struct RealtimeTranscriptBuffer {
    private(set) public var committed: [TimedToken]
    private(set) public var mutable: [TimedToken]
    private(set) public var revision: Int

    private var persistedCommittedWords: [String]
    private var nextTokenID: Int

    public init() {
        self.committed = []
        self.mutable = []
        self.revision = 0
        self.persistedCommittedWords = []
        self.nextTokenID = 1
    }

    public mutating func update(confirmedSegmentWords: [String], hypothesisWords: [String]) -> TranscriptBufferSnapshot {
        // Preview-only update: compute transient confirmed text from persisted
        // committed prefix + stable words of current segment.
        let previewCommittedWords = Self.appendWithOverlap(
            base: persistedCommittedWords,
            segment: confirmedSegmentWords
        )
        apply(committedWords: previewCommittedWords, mutableWords: hypothesisWords)
        return snapshot()
    }

    public mutating func finalizeSegment(finalSegmentWords: [String]) -> TranscriptBufferSnapshot {
        let mergedCommittedWords = Self.appendWithOverlap(
            base: persistedCommittedWords,
            segment: finalSegmentWords
        )
        persistedCommittedWords = mergedCommittedWords
        apply(committedWords: mergedCommittedWords, mutableWords: [])
        return snapshot()
    }

    public mutating func reset() {
        committed.removeAll(keepingCapacity: true)
        mutable.removeAll(keepingCapacity: true)
        revision = 0
        persistedCommittedWords.removeAll(keepingCapacity: true)
        nextTokenID = 1
    }

    public func snapshot() -> TranscriptBufferSnapshot {
        TranscriptBufferSnapshot(committed: committed, mutable: mutable, revision: revision)
    }

    private mutating func apply(committedWords: [String], mutableWords: [String]) {
        let oldMerged = (committed + mutable).map(\.text)
        let newMerged = committedWords + mutableWords
        if oldMerged != newMerged {
            revision += 1
        }
        committed = retokenizePreservingPrefix(existing: committed, newWords: committedWords)
        mutable = retokenizePreservingPrefix(existing: mutable, newWords: mutableWords)
    }

    private mutating func retokenizePreservingPrefix(existing: [TimedToken], newWords: [String]) -> [TimedToken] {
        var out: [TimedToken] = []
        out.reserveCapacity(newWords.count)

        var i = 0
        while i < existing.count && i < newWords.count && existing[i].text == newWords[i] {
            out.append(existing[i])
            i += 1
        }

        if i < newWords.count {
            for word in newWords[i...] {
                out.append(
                    TimedToken(
                        id: nextTokenID,
                        text: word,
                        revision: revision
                    )
                )
                nextTokenID += 1
            }
        }
        return out
    }

    static func appendWithOverlap(base: [String], segment: [String], maxOverlap: Int = 256) -> [String] {
        guard !base.isEmpty else { return segment }
        guard !segment.isEmpty else { return base }

        let searchWindow = min(base.count, 2048)
        if segment.count >= 4 {
            let start = base.count - searchWindow
            if containsSubsequence(haystack: Array(base[start...]), needle: segment) {
                return base
            }
        }

        let overlapLimit = min(maxOverlap, min(base.count, segment.count))
        var overlap = 0
        if overlapLimit > 0 {
            for candidate in stride(from: overlapLimit, through: 1, by: -1) {
                let lhs = Array(base.suffix(candidate))
                let rhs = Array(segment.prefix(candidate))
                if lhs == rhs {
                    overlap = candidate
                    break
                }
            }
        }

        if overlap > 0 {
            return base + segment.dropFirst(overlap)
        }

        if segment.count >= 4 {
            let start = max(0, base.count - 2048)
            let window = Array(base[start...])
            let maxPrefix = min(segment.count, 512)
            for prefixLen in stride(from: maxPrefix, through: 4, by: -1) {
                let prefix = Array(segment.prefix(prefixLen))
                if containsSubsequence(haystack: window, needle: prefix) {
                    return base + segment.dropFirst(prefixLen)
                }
            }
        }

        return base + segment
    }

    private static func containsSubsequence(haystack: [String], needle: [String]) -> Bool {
        guard !needle.isEmpty else { return true }
        guard needle.count <= haystack.count else { return false }
        let limit = haystack.count - needle.count
        for i in 0...limit {
            var matched = true
            for j in 0..<needle.count where haystack[i + j] != needle[j] {
                matched = false
                break
            }
            if matched {
                return true
            }
        }
        return false
    }
}
