#if canImport(CoreML)
import XCTest
@testable import RealtimeTranscriptionCore

final class ParakeetStreamingMelFeatureCacheTests: XCTestCase {
    func testRollingCacheMatchesFullRecomputeAcrossAlignedHops() {
        let extractor = MelFeatureExtractor(
            sampleRate: 16_000,
            nFFT: 64,
            windowLength: 48,
            hopLength: 16,
            melBins: 8
        )
        var cache = StreamingMelFeatureCache(extractor: extractor)
        let historySampleLimit = 12 * extractor.hopLength + extractor.windowLength
        let hopSamples = 64
        let audio = (0..<(hopSamples * 8)).map { idx -> Float in
            let t = Float(idx) / 16_000.0
            return 0.6 * sin(2.0 * .pi * 440.0 * t) + 0.2 * cos(2.0 * .pi * 137.0 * t)
        }

        var history: [Float] = []
        var cursor = 0
        while cursor < audio.count {
            let next = min(hopSamples, audio.count - cursor)
            history.append(contentsOf: audio[cursor..<(cursor + next)])
            let droppedSamples = max(0, history.count - historySampleLimit)
            if droppedSamples > 0 {
                history.removeFirst(droppedSamples)
            }

            let expected = extractor.extract(samples: history)
            let actual = cache.extract(samples: history, droppedSampleCount: droppedSamples)
            assertFeatureMatrix(actual, matches: expected)
            cursor += next
        }
    }

    func testRollingCacheFallsBackForMisalignedHistoryTrim() {
        let extractor = MelFeatureExtractor(
            sampleRate: 16_000,
            nFFT: 64,
            windowLength: 48,
            hopLength: 16,
            melBins: 8
        )
        var cache = StreamingMelFeatureCache(extractor: extractor)
        let historySampleLimit = 150
        let chunkSizes = [19, 23, 17, 29, 31, 13, 27]
        let audio = (0..<chunkSizes.reduce(0, +)).map { idx -> Float in
            let t = Float(idx) / 16_000.0
            return 0.5 * sin(2.0 * .pi * 320.0 * t) + 0.1 * sin(2.0 * .pi * 40.0 * t)
        }

        var history: [Float] = []
        var cursor = 0
        for chunkSize in chunkSizes {
            history.append(contentsOf: audio[cursor..<(cursor + chunkSize)])
            let droppedSamples = max(0, history.count - historySampleLimit)
            if droppedSamples > 0 {
                history.removeFirst(droppedSamples)
            }

            let expected = extractor.extract(samples: history)
            let actual = cache.extract(samples: history, droppedSampleCount: droppedSamples)
            assertFeatureMatrix(actual, matches: expected)
            cursor += chunkSize
        }
    }

    private func assertFeatureMatrix(
        _ actual: ([Float], Int),
        matches expected: ([Float], Int),
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.1, expected.1, file: file, line: line)
        XCTAssertEqual(actual.0.count, expected.0.count, file: file, line: line)
        for (lhs, rhs) in zip(actual.0, expected.0) {
            XCTAssertEqual(lhs, rhs, accuracy: 1e-5, file: file, line: line)
        }
    }
}
#endif
