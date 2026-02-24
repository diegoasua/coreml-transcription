import Foundation

public struct VADDecision: Equatable {
    public let isSpeech: Bool
    public let energyDBFS: Float

    public init(isSpeech: Bool, energyDBFS: Float) {
        self.isSpeech = isSpeech
        self.energyDBFS = energyDBFS
    }
}

public protocol VoiceActivityDetecting {
    mutating func process(samples: [Float], sampleRate: Int) -> VADDecision
    mutating func reset()
}

public struct EnergyVADConfig {
    public let startThresholdDBFS: Float
    public let endThresholdDBFS: Float
    public let minSpeechMs: Int
    public let minSilenceMs: Int

    public init(
        startThresholdDBFS: Float = -35.0,
        endThresholdDBFS: Float = -45.0,
        minSpeechMs: Int = 120,
        minSilenceMs: Int = 250
    ) {
        self.startThresholdDBFS = startThresholdDBFS
        self.endThresholdDBFS = endThresholdDBFS
        self.minSpeechMs = minSpeechMs
        self.minSilenceMs = minSilenceMs
    }
}

public struct EnergyVAD: VoiceActivityDetecting {
    private let config: EnergyVADConfig
    private var activeSpeechMs: Float
    private var activeSilenceMs: Float
    private var speechActive: Bool

    public init(config: EnergyVADConfig = .init()) {
        self.config = config
        self.activeSpeechMs = 0
        self.activeSilenceMs = 0
        self.speechActive = false
    }

    public mutating func process(samples: [Float], sampleRate: Int) -> VADDecision {
        guard sampleRate > 0, !samples.isEmpty else {
            return VADDecision(isSpeech: speechActive, energyDBFS: -120.0)
        }

        let frameMs = (Float(samples.count) / Float(sampleRate)) * 1000.0
        let energy = rms(samples)
        let dbfs = linearToDBFS(energy)

        if !speechActive {
            if dbfs >= config.startThresholdDBFS {
                activeSpeechMs += frameMs
            } else {
                activeSpeechMs = 0
            }

            if activeSpeechMs >= Float(config.minSpeechMs) {
                speechActive = true
                activeSilenceMs = 0
            }
        } else {
            if dbfs <= config.endThresholdDBFS {
                activeSilenceMs += frameMs
            } else {
                activeSilenceMs = 0
            }

            if activeSilenceMs >= Float(config.minSilenceMs) {
                speechActive = false
                activeSpeechMs = 0
            }
        }

        return VADDecision(isSpeech: speechActive, energyDBFS: dbfs)
    }

    public mutating func reset() {
        activeSpeechMs = 0
        activeSilenceMs = 0
        speechActive = false
    }

    private func rms(_ values: [Float]) -> Float {
        let sum = values.reduce(0.0) { partial, value in
            partial + value * value
        }
        return sqrt(sum / Float(values.count))
    }

    private func linearToDBFS(_ value: Float) -> Float {
        let floor: Float = 1e-7
        return 20.0 * log10(max(value, floor))
    }
}
