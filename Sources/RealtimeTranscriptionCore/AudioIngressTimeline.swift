import Foundation

public struct AudioIngressTimeline: Sendable {
    private struct Mark: Sendable {
        let endSampleCursor: Int
        let receivedAtSec: Double
    }

    public let sampleRate: Int
    private var marks: [Mark] = []
    private(set) public var totalRecordedSamples: Int = 0

    public init(sampleRate: Int) {
        self.sampleRate = max(1, sampleRate)
    }

    @discardableResult
    public mutating func recordIngress(sampleCount: Int, receivedAtSec: Double) -> Range<Int> {
        let clampedCount = max(0, sampleCount)
        let start = totalRecordedSamples
        totalRecordedSamples += clampedCount
        if clampedCount > 0 {
            marks.append(Mark(endSampleCursor: totalRecordedSamples, receivedAtSec: receivedAtSec))
        }
        return start ..< totalRecordedSamples
    }

    public func ingressTime(forSampleCursor sampleCursor: Int) -> Double? {
        guard !marks.isEmpty else { return nil }
        let target = max(1, sampleCursor)
        var low = 0
        var high = marks.count - 1
        while low < high {
            let mid = (low + high) / 2
            if marks[mid].endSampleCursor >= target {
                high = mid
            } else {
                low = mid + 1
            }
        }
        guard marks[low].endSampleCursor >= target else { return nil }
        return marks[low].receivedAtSec
    }

    public func ingressTime(forAudioCursorSec audioCursorSec: Double) -> Double? {
        let sampleCursor = Int(ceil(max(0, audioCursorSec) * Double(sampleRate)))
        return ingressTime(forSampleCursor: sampleCursor)
    }

    public mutating func reset() {
        marks.removeAll(keepingCapacity: false)
        totalRecordedSamples = 0
    }
}
