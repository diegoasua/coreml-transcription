import XCTest
@testable import RealtimeTranscriptionCore

final class VoiceActivityDetectorTests: XCTestCase {
    func testSilenceStaysInactive() {
        var vad = EnergyVAD(
            config: .init(
                startThresholdDBFS: -35,
                endThresholdDBFS: -45,
                minSpeechMs: 100,
                minSilenceMs: 100
            )
        )

        let frame = Array(repeating: Float(0), count: 1600) // 100 ms @ 16k
        let d1 = vad.process(samples: frame, sampleRate: 16_000)
        XCTAssertFalse(d1.isSpeech)
    }

    func testSpeechActivationAndRelease() {
        var vad = EnergyVAD(
            config: .init(
                startThresholdDBFS: -35,
                endThresholdDBFS: -45,
                minSpeechMs: 100,
                minSilenceMs: 100
            )
        )

        let speechFrame = Array(repeating: Float(0.25), count: 1600) // 100 ms @ 16k
        let silenceFrame = Array(repeating: Float(0.0), count: 1600)

        let d1 = vad.process(samples: speechFrame, sampleRate: 16_000)
        XCTAssertTrue(d1.isSpeech)

        let d2 = vad.process(samples: silenceFrame, sampleRate: 16_000)
        XCTAssertFalse(d2.isSpeech)
    }
}
