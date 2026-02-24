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
    private var stabilizer: LocalAgreementStabilizer

    public init(requiredAgreementCount: Int = 2) {
        self.stabilizer = LocalAgreementStabilizer(
            config: .init(requiredAgreementCount: requiredAgreementCount)
        )
    }

    public mutating func update(partialText: String) -> TranscriptState {
        stabilizer.push(partial: partialText)
    }

    public mutating func endSegment() -> TranscriptState {
        stabilizer.flushSegment()
    }

    public mutating func reset() {
        stabilizer.resetAll()
    }
}
