import Foundation

#if canImport(CoreML)
import CoreML

public enum CoreMLTranscriptionError: Error {
    case sampleRateMismatch(expected: Int, actual: Int)
    case missingOutput(String)
    case unsupportedOutputLayout
}

public struct CoreMLCTCConfig {
    public let audioInputName: String
    public let lengthInputName: String?
    public let logitsOutputName: String
    public let expectedSampleRate: Int
    public let computeUnits: MLComputeUnits

    public init(
        audioInputName: String,
        lengthInputName: String? = nil,
        logitsOutputName: String,
        expectedSampleRate: Int = 16_000,
        computeUnits: MLComputeUnits = .all
    ) {
        self.audioInputName = audioInputName
        self.lengthInputName = lengthInputName
        self.logitsOutputName = logitsOutputName
        self.expectedSampleRate = expectedSampleRate
        self.computeUnits = computeUnits
    }
}

public final class CoreMLCTCTranscriptionModel: TranscriptionModel {
    private let model: MLModel
    private let config: CoreMLCTCConfig
    private let decoder: CTCGreedyDecoder

    public init(modelURL: URL, config: CoreMLCTCConfig, decoder: CTCGreedyDecoder) throws {
        self.config = config
        self.decoder = decoder

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = config.computeUnits
        self.model = try MLModel(contentsOf: modelURL, configuration: mlConfig)
    }

    public func transcribeChunk(_ samples: [Float], sampleRate: Int) throws -> String {
        guard sampleRate == config.expectedSampleRate else {
            throw CoreMLTranscriptionError.sampleRateMismatch(expected: config.expectedSampleRate, actual: sampleRate)
        }

        let audioArray = try MLMultiArray(shape: [1, NSNumber(value: samples.count)], dataType: .float32)
        let audioPointer = audioArray.dataPointer.bindMemory(to: Float.self, capacity: samples.count)
        for idx in 0..<samples.count {
            audioPointer[idx] = samples[idx]
        }

        var dictionary: [String: Any] = [
            config.audioInputName: audioArray
        ]
        if let lengthName = config.lengthInputName {
            let length = try MLMultiArray(shape: [1], dataType: .int32)
            length.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = Int32(samples.count)
            dictionary[lengthName] = length
        }

        let inputProvider = try MLDictionaryFeatureProvider(dictionary: dictionary)
        let result = try model.prediction(from: inputProvider)

        guard let logits = result.featureValue(for: config.logitsOutputName)?.multiArrayValue else {
            let available = result.featureNames.sorted().joined(separator: ", ")
            throw CoreMLTranscriptionError.missingOutput(
                "\(config.logitsOutputName). Available outputs: [\(available)]"
            )
        }

        let rawTokenIDs = try logits.greedyArgmaxTokenIDs()
        let collapsed = decoder.collapseCTC(rawTokenIDs)
        return try decoder.decode(tokenIDs: collapsed)
    }

    public func resetState() {
        // Stateless baseline path. For stateful models (KV cache / RNNT state),
        // this is where recurrent state would be cleared.
    }
}

private extension MLMultiArray {
    func greedyArgmaxTokenIDs() throws -> [Int] {
        let rank = shape.count
        switch rank {
        case 2:
            return argmax2D(timeDim: 0, vocabDim: 1)
        case 3:
            // Assumes [batch, time, vocab], uses batch index 0.
            return argmax3D(batchIndex: 0, timeDim: 1, vocabDim: 2)
        default:
            throw CTCDecoderError.unsupportedLogitsRank(rank)
        }
    }

    private func argmax2D(timeDim: Int, vocabDim: Int) -> [Int] {
        let timeSteps = shape[timeDim].intValue
        let vocabSize = shape[vocabDim].intValue

        let strideT = strides[timeDim].intValue
        let strideV = strides[vocabDim].intValue

        var tokens: [Int] = []
        tokens.reserveCapacity(timeSteps)

        for t in 0..<timeSteps {
            var maxValue = -Float.infinity
            var maxIndex = 0
            for v in 0..<vocabSize {
                let idx = t * strideT + v * strideV
                let value = valueAt(flatElementIndex: idx)
                if value > maxValue {
                    maxValue = value
                    maxIndex = v
                }
            }
            tokens.append(maxIndex)
        }
        return tokens
    }

    private func argmax3D(batchIndex: Int, timeDim: Int, vocabDim: Int) -> [Int] {
        let batchStride = strides[0].intValue
        let base = batchIndex * batchStride
        let timeSteps = shape[timeDim].intValue
        let vocabSize = shape[vocabDim].intValue

        let strideT = strides[timeDim].intValue
        let strideV = strides[vocabDim].intValue

        var tokens: [Int] = []
        tokens.reserveCapacity(timeSteps)

        for t in 0..<timeSteps {
            var maxValue = -Float.infinity
            var maxIndex = 0
            for v in 0..<vocabSize {
                let idx = base + t * strideT + v * strideV
                let value = valueAt(flatElementIndex: idx)
                if value > maxValue {
                    maxValue = value
                    maxIndex = v
                }
            }
            tokens.append(maxIndex)
        }
        return tokens
    }

    private func valueAt(flatElementIndex idx: Int) -> Float {
        switch dataType {
        case .float32:
            let ptr = dataPointer.bindMemory(to: Float.self, capacity: count)
            return ptr[idx]
        case .double:
            let ptr = dataPointer.bindMemory(to: Double.self, capacity: count)
            return Float(ptr[idx])
        case .int32:
            let ptr = dataPointer.bindMemory(to: Int32.self, capacity: count)
            return Float(ptr[idx])
        case .int8:
            let ptr = dataPointer.bindMemory(to: Int8.self, capacity: count)
            return Float(ptr[idx])
        case .float16:
            let ptr = dataPointer.bindMemory(to: UInt16.self, capacity: count)
            return Self.floatFromFloat16Bits(ptr[idx])
        @unknown default:
            return 0
        }
    }

    private static func floatFromFloat16Bits(_ bits: UInt16) -> Float {
        let sign = UInt32(bits & 0x8000) << 16
        var exponent = Int((bits & 0x7C00) >> 10)
        var mantissa = UInt32(bits & 0x03FF)

        if exponent == 0 {
            if mantissa == 0 {
                return Float(bitPattern: sign)
            }
            exponent = 1
            while (mantissa & 0x0400) == 0 {
                mantissa <<= 1
                exponent -= 1
            }
            mantissa &= 0x03FF
        } else if exponent == 0x1F {
            let infNaN = sign | 0x7F80_0000 | (mantissa << 13)
            return Float(bitPattern: infNaN)
        }

        let exp32 = UInt32(exponent + (127 - 15)) << 23
        let mant32 = mantissa << 13
        return Float(bitPattern: sign | exp32 | mant32)
    }
}

#endif
