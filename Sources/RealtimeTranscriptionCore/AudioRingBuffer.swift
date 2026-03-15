import Foundation

public struct AudioRingBuffer {
    private var storage: [Float]
    private var readIndex: Int
    private var writeIndex: Int
    private(set) public var count: Int

    public init(capacity: Int) {
        precondition(capacity > 1, "capacity must be > 1")
        self.storage = Array(repeating: 0.0, count: capacity)
        self.readIndex = 0
        self.writeIndex = 0
        self.count = 0
    }

    public var capacity: Int { storage.count }
    public var availableToRead: Int { count }
    public var availableToWrite: Int { capacity - count }

    public mutating func append(_ samples: [Float]) {
        for sample in samples {
            if count == capacity {
                _ = pop(count: 1)
            }
            storage[writeIndex] = sample
            writeIndex = (writeIndex + 1) % capacity
            count += 1
        }
    }

    @discardableResult
    public mutating func pop(count requested: Int) -> [Float] {
        let n = min(requested, count)
        guard n > 0 else { return [] }

        var out: [Float] = []
        out.reserveCapacity(n)
        for _ in 0..<n {
            out.append(storage[readIndex])
            readIndex = (readIndex + 1) % capacity
            count -= 1
        }
        return out
    }

    public func peek(count requested: Int) -> [Float] {
        let n = min(requested, count)
        guard n > 0 else { return [] }

        var out: [Float] = []
        out.reserveCapacity(n)
        var idx = readIndex
        for _ in 0..<n {
            out.append(storage[idx])
            idx = (idx + 1) % capacity
        }
        return out
    }

    public func peekLast(count requested: Int) -> [Float] {
        let n = min(requested, count)
        guard n > 0 else { return [] }

        var out = Array(repeating: Float(0), count: n)
        var idx = (writeIndex - n + capacity) % capacity
        for outIdx in 0..<n {
            out[outIdx] = storage[idx]
            idx = (idx + 1) % capacity
        }
        return out
    }
}
