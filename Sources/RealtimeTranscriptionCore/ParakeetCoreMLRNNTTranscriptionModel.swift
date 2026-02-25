import Foundation

#if canImport(CoreML)
import CoreML

public enum ParakeetCoreMLRNNTError: Error {
    case sampleRateMismatch(expected: Int, actual: Int)
    case missingRequiredInput(String)
    case invalidModelIO(String)
    case missingOutput(String)
    case unsupportedArrayRank(Int)
}

public struct ParakeetCoreMLRNNTConfig {
    public let expectedSampleRate: Int
    public let durations: [Int]
    public let maxSymbolsPerStep: Int
    public let maxTokensPerChunk: Int
    public let computeUnits: MLComputeUnits
    public let melBins: Int
    public let nFFT: Int
    public let windowLength: Int
    public let hopLength: Int

    public init(
        expectedSampleRate: Int = 16_000,
        durations: [Int] = [0, 1, 2, 3, 4],
        maxSymbolsPerStep: Int = 4,
        maxTokensPerChunk: Int = 192,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        melBins: Int = 128,
        nFFT: Int = 512,
        windowLength: Int = 400,
        hopLength: Int = 160
    ) {
        self.expectedSampleRate = expectedSampleRate
        self.durations = durations
        self.maxSymbolsPerStep = max(1, maxSymbolsPerStep)
        self.maxTokensPerChunk = max(0, maxTokensPerChunk)
        self.computeUnits = computeUnits
        self.melBins = melBins
        self.nFFT = nFFT
        self.windowLength = windowLength
        self.hopLength = hopLength
    }
}

public final class ParakeetCoreMLRNNTTranscriptionModel: TranscriptionModel {
    private let encoder: MLModel
    private let decoder: MLModel
    private let config: ParakeetCoreMLRNNTConfig
    private let vocab: [String]
    private let blankID: Int

    private let encoderAudioInputName: String
    private let encoderLengthInputName: String?
    private let encoderFrameCount: Int

    private let decoderEncoderInputName: String
    private let decoderStateInputNames: [String]
    private var decoderLogitsOutputName: String?
    private var decoderStateOutputByInputName: [String: String]
    private let decoderFrameCount: Int

    private var state1: MLMultiArray
    private var state2: MLMultiArray
    private var previousToken: Int
    private var emittedTokenIDs: [Int]
    private let debugLoggingEnabled: Bool

    private let featureExtractor: MelFeatureExtractor
    private let minNonZeroDuration: Int

    public init(
        modelDirectory: URL,
        encoderModelName: String,
        decoderModelName: String,
        vocabFileName: String = "vocab.txt",
        config: ParakeetCoreMLRNNTConfig = .init()
    ) throws {
        self.config = config
        self.decoderStateOutputByInputName = [:]
        self.emittedTokenIDs = []
        self.debugLoggingEnabled = ProcessInfo.processInfo.environment["PARAKEET_SWIFT_DEBUG"] == "1"

        let encoderURL = modelDirectory.appendingPathComponent(encoderModelName)
        let decoderURL = modelDirectory.appendingPathComponent(decoderModelName)
        let vocabURL = modelDirectory.appendingPathComponent(vocabFileName)

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = config.computeUnits
        self.encoder = try Self.loadModel(at: encoderURL, configuration: mlConfig)
        self.decoder = try Self.loadModel(at: decoderURL, configuration: mlConfig)

        self.vocab = try String(contentsOf: vocabURL, encoding: .utf8).split(separator: "\n").map(String.init)
        self.blankID = vocab.count

        let encoderInputs = Self.inputSpecs(model: encoder)
        guard let audioInput = encoderInputs["audio_signal"] ?? encoderInputs.values.first else {
            throw ParakeetCoreMLRNNTError.invalidModelIO("Could not find encoder input.")
        }
        self.encoderAudioInputName = audioInput.name
        self.encoderLengthInputName = encoderInputs["length"]?.name
        self.encoderFrameCount = max(1, audioInput.shape.last ?? 1200)

        let decoderInputs = Self.inputSpecs(model: decoder)
        if let encoderInput = decoderInputs["encoder_outputs"] {
            self.decoderEncoderInputName = encoderInput.name
        } else if let inferred = decoderInputs.values.first(where: { $0.name.lowercased().contains("encoder") }) {
            self.decoderEncoderInputName = inferred.name
        } else {
            throw ParakeetCoreMLRNNTError.invalidModelIO("Could not find decoder encoder input.")
        }

        self.decoderStateInputNames = decoderInputs.values
            .filter { $0.name.hasPrefix("input_states") }
            .map(\.name)
            .sorted()
        guard decoderStateInputNames.count >= 2 else {
            throw ParakeetCoreMLRNNTError.invalidModelIO("Decoder is missing input_states_* tensors.")
        }
        self.decoderFrameCount = max(1, decoderInputs[decoderEncoderInputName]?.shape.last ?? 300)

        self.state1 = try Self.makeArray(
            shape: decoderInputs[decoderStateInputNames[0]]?.shape ?? [2, 1, 640],
            dataType: decoderInputs[decoderStateInputNames[0]]?.dataType ?? .float32
        )
        self.state2 = try Self.makeArray(
            shape: decoderInputs[decoderStateInputNames[1]]?.shape ?? [2, 1, 640],
            dataType: decoderInputs[decoderStateInputNames[1]]?.dataType ?? .float32
        )
        self.previousToken = blankID

        self.featureExtractor = MelFeatureExtractor(
            sampleRate: config.expectedSampleRate,
            nFFT: config.nFFT,
            windowLength: config.windowLength,
            hopLength: config.hopLength,
            melBins: config.melBins
        )
        self.minNonZeroDuration = config.durations.filter { $0 > 0 }.min() ?? 1
    }

    public convenience init(
        modelDirectory: URL,
        modelSuffix: String = "odmbp-approx",
        vocabFileName: String = "vocab.txt",
        config: ParakeetCoreMLRNNTConfig = .init()
    ) throws {
        try self.init(
            modelDirectory: modelDirectory,
            encoderModelName: "encoder-model-\(modelSuffix).mlpackage",
            decoderModelName: "decoder_joint-model-\(modelSuffix).mlpackage",
            vocabFileName: vocabFileName,
            config: config
        )
    }

    public func transcribeChunk(_ samples: [Float], sampleRate: Int) throws -> String {
        guard sampleRate == config.expectedSampleRate else {
            throw ParakeetCoreMLRNNTError.sampleRateMismatch(expected: config.expectedSampleRate, actual: sampleRate)
        }
        if samples.isEmpty {
            return Self.decodePieces(from: emittedTokenIDs, vocab: vocab)
        }

        let (featureMatrix, frameCount) = featureExtractor.extract(samples: samples)
        let clampedFrames = min(frameCount, encoderFrameCount)
        if debugLoggingEnabled {
            let preview = featureMatrix.prefix(10).map { String(format: "%.6f", $0) }.joined(separator: ", ")
            fputs("[ParakeetSwift] feature frames=\(frameCount) clamped=\(clampedFrames) first10=[\(preview)]\n", stderr)
        }

        let encoderAudio = try Self.makeArray(shape: [1, config.melBins, encoderFrameCount], dataType: .float32)
        try Self.fill(array: encoderAudio, with: 0)
        try Self.copyFeatureMatrixToEncoderInput(
            featureMatrix: featureMatrix,
            melBins: config.melBins,
            sourceFrames: frameCount,
            destination: encoderAudio,
            destinationFrames: encoderFrameCount,
            copyFrames: clampedFrames
        )

        var encoderFeed: [String: Any] = [encoderAudioInputName: encoderAudio]
        if let lengthName = encoderLengthInputName {
            let lengthType = Self.inputSpecs(model: encoder)[lengthName]?.dataType ?? .int32
            let lengthArray = try Self.makeArray(shape: [1], dataType: lengthType)
            try Self.setScalarInt(lengthArray, value: clampedFrames)
            encoderFeed[lengthName] = lengthArray
        }

        let encoderProvider = try MLDictionaryFeatureProvider(dictionary: encoderFeed)
        let encoderOutput = try encoder.prediction(from: encoderProvider)
        let (encoderTensor, encoderLength) = try Self.pickEncoderTensorAndLength(
            from: encoderOutput,
            debug: debugLoggingEnabled
        )
        let steps = max(0, min(encoderLength ?? clampedFrames, decoderFrameCount))
        if steps == 0 {
            return Self.decodePieces(from: emittedTokenIDs, vocab: vocab)
        }

        let decoderEncoderInput = try Self.makeArray(
            shape: Self.inputSpecs(model: decoder)[decoderEncoderInputName]?.shape ?? [1, 1024, decoderFrameCount],
            dataType: .float32
        )
        try Self.fill(array: decoderEncoderInput, with: 0)
        try Self.copyOverlapping(source: encoderTensor, destination: decoderEncoderInput)

        let chunkTokenIDs = try decodeChunk(
            decoderEncoderInput: decoderEncoderInput,
            encoderSteps: steps
        )
        emittedTokenIDs.append(contentsOf: chunkTokenIDs)
        return Self.decodePieces(from: emittedTokenIDs, vocab: vocab)
    }

    public func transcribeLongform(
        _ samples: [Float],
        sampleRate: Int,
        leftContextFrames: Int,
        rightContextFrames: Int,
        allowRightContext: Bool = true
    ) throws -> String {
        guard sampleRate == config.expectedSampleRate else {
            throw ParakeetCoreMLRNNTError.sampleRateMismatch(expected: config.expectedSampleRate, actual: sampleRate)
        }
        if samples.isEmpty { return "" }

        resetState()
        let (featureMatrix, totalFrames) = featureExtractor.extract(samples: samples)
        if totalFrames <= 0 { return "" }

        let maxContextTotal = max(0, encoderFrameCount - 1)
        let leftCtx = min(max(0, leftContextFrames), maxContextTotal)
        let configuredRightCtx = allowRightContext ? max(0, rightContextFrames) : 0
        let rightCtx = min(configuredRightCtx, max(0, maxContextTotal - leftCtx))
        let hopFrames = max(1, encoderFrameCount - leftCtx - rightCtx)

        var allTokenIDs: [Int] = []
        for centerStart in stride(from: 0, to: totalFrames, by: hopFrames) {
            let inputStart = max(0, centerStart - leftCtx)
            let actualFrames = max(0, min(encoderFrameCount, totalFrames - inputStart))
            if actualFrames <= 0 { continue }

            let encoderAudio = try Self.makeArray(shape: [1, config.melBins, encoderFrameCount], dataType: .float32)
            try Self.fill(array: encoderAudio, with: 0)
            try Self.copyFeatureWindowToEncoderInput(
                featureMatrix: featureMatrix,
                melBins: config.melBins,
                totalFrames: totalFrames,
                startFrame: inputStart,
                sourceFrames: actualFrames,
                destination: encoderAudio,
                copyFrames: actualFrames
            )

            var encoderFeed: [String: Any] = [encoderAudioInputName: encoderAudio]
            if let lengthName = encoderLengthInputName {
                let lengthType = Self.inputSpecs(model: encoder)[lengthName]?.dataType ?? .int32
                let lengthArray = try Self.makeArray(shape: [1], dataType: lengthType)
                try Self.setScalarInt(lengthArray, value: actualFrames)
                encoderFeed[lengthName] = lengthArray
            }

            let encoderProvider = try MLDictionaryFeatureProvider(dictionary: encoderFeed)
            let encoderOutput = try encoder.prediction(from: encoderProvider)
            let (rawEncoderTensor, rawEncoderLength) = try Self.pickEncoderTensorAndLength(from: encoderOutput, debug: false)
            var encoderTensor = rawEncoderTensor
            var encoderSteps = max(0, min(rawEncoderLength ?? actualFrames, decoderFrameCount))

            if (leftCtx > 0 || rightCtx > 0), encoderSteps > 0, actualFrames > 0 {
                let leftIn = centerStart - inputStart
                let centerIn = min(hopFrames, totalFrames - centerStart)
                let scale = Double(encoderSteps) / Double(actualFrames)
                var leftOut = Int((Double(leftIn) * scale).rounded())
                var centerOut = Int((Double(centerIn) * scale).rounded())

                leftOut = max(0, min(leftOut, encoderSteps))
                centerOut = max(1, centerOut)

                let endOut: Int
                if centerStart + centerIn >= totalFrames {
                    endOut = encoderSteps
                } else {
                    endOut = max(leftOut, min(encoderSteps, leftOut + centerOut))
                }

                encoderTensor = try Self.slice3DTensor(rawEncoderTensor, startFrame: leftOut, endFrame: endOut)
                encoderSteps = max(0, endOut - leftOut)
            }

            if encoderSteps <= 0 { continue }

            let decoderEncoderInput = try Self.makeArray(
                shape: Self.inputSpecs(model: decoder)[decoderEncoderInputName]?.shape ?? [1, 1024, decoderFrameCount],
                dataType: .float32
            )
            try Self.fill(array: decoderEncoderInput, with: 0)
            try Self.copyOverlapping(source: encoderTensor, destination: decoderEncoderInput)

            let chunkTokenIDs = try decodeChunk(
                decoderEncoderInput: decoderEncoderInput,
                encoderSteps: encoderSteps
            )
            allTokenIDs.append(contentsOf: chunkTokenIDs)
        }

        emittedTokenIDs = allTokenIDs
        return Self.decodePieces(from: allTokenIDs, vocab: vocab)
    }

    public func resetState() {
        try? Self.fill(array: state1, with: 0)
        try? Self.fill(array: state2, with: 0)
        previousToken = blankID
        emittedTokenIDs.removeAll(keepingCapacity: true)
    }

    private func decodeChunk(decoderEncoderInput: MLMultiArray, encoderSteps: Int) throws -> [Int] {
        let targetInputName = "targets"
        let targetLengthName = "target_length"
        guard Self.inputSpecs(model: decoder)[targetInputName] != nil else {
            throw ParakeetCoreMLRNNTError.missingRequiredInput(targetInputName)
        }
        guard Self.inputSpecs(model: decoder)[targetLengthName] != nil else {
            throw ParakeetCoreMLRNNTError.missingRequiredInput(targetLengthName)
        }

        let maxTokens = maxTokensForChunk(encoderSteps: encoderSteps)
        var tokenIDs: [Int] = []
        tokenIDs.reserveCapacity(min(maxTokens, encoderSteps * 4))
        let targets = try Self.makeArray(shape: [1, 1], dataType: .int32)
        let targetLength = try Self.makeArray(shape: [1], dataType: .int32)
        try Self.setScalarInt(targetLength, value: 1)

        var t = 0
        while t < encoderSteps && tokenIDs.count < maxTokens {
            var symbolsAdded = 0
            var continueLoop = true
            var skip = 1

            while continueLoop && symbolsAdded < config.maxSymbolsPerStep && tokenIDs.count < maxTokens {
                try Self.setScalarInt(targets, value: previousToken)

                let feed: [String: Any] = [
                    decoderEncoderInputName: decoderEncoderInput,
                    targetInputName: targets,
                    targetLengthName: targetLength,
                    decoderStateInputNames[0]: state1,
                    decoderStateInputNames[1]: state2,
                ]
                let provider = try MLDictionaryFeatureProvider(dictionary: feed)
                let output = try decoder.prediction(from: provider)
                try inferDecoderOutputRolesIfNeeded(output: output, feed: feed)

                guard let logitsName = decoderLogitsOutputName,
                      let logits = output.featureValue(for: logitsName)?.multiArrayValue else {
                    throw ParakeetCoreMLRNNTError.missingOutput("Decoder logits output not found.")
                }
                if debugLoggingEnabled && t == 0 && tokenIDs.isEmpty {
                    Self.debugLogitsCandidates(logits: logits, blankID: blankID)
                }
                let stepLogits = try Self.extractStepLogits(logits: logits, tIndex: t)
                let tokenVocabSize = blankID + 1
                if stepLogits.count < tokenVocabSize {
                    throw ParakeetCoreMLRNNTError.invalidModelIO("Decoder logits smaller than vocab+blank.")
                }

                let tokenID = Self.argmax(stepLogits[0..<tokenVocabSize])
                let durationPart = stepLogits[tokenVocabSize..<stepLogits.count]
                let durationIdx = durationPart.isEmpty ? 1 : Self.argmax(durationPart)
                skip = config.durations.indices.contains(durationIdx) ? config.durations[durationIdx] : 1
                if tokenID == blankID && skip == 0 {
                    skip = minNonZeroDuration
                }
                if skip == 0 {
                    skip = 1
                }

                if tokenID != blankID {
                    tokenIDs.append(tokenID)
                    if let stateOutput1 = decoderStateOutputByInputName[decoderStateInputNames[0]],
                       let stateOutput2 = decoderStateOutputByInputName[decoderStateInputNames[1]],
                       let next1 = output.featureValue(for: stateOutput1)?.multiArrayValue,
                       let next2 = output.featureValue(for: stateOutput2)?.multiArrayValue {
                        state1 = next1
                        state2 = next2
                    } else {
                        throw ParakeetCoreMLRNNTError.missingOutput("Decoder recurrent state outputs not found.")
                    }
                    previousToken = tokenID
                }

                if debugLoggingEnabled && tokenIDs.count <= 16 {
                    fputs(
                        "[ParakeetSwift] t=\(t) token=\(tokenID) skip=\(skip) blank=\(blankID)\n",
                        stderr
                    )
                }

                symbolsAdded += 1
                t += skip
                continueLoop = (skip == 0)
            }

            if symbolsAdded >= config.maxSymbolsPerStep {
                t += 1
            }
        }

        return tokenIDs
    }

    private func maxTokensForChunk(encoderSteps: Int) -> Int {
        if config.maxTokensPerChunk > 0 {
            return config.maxTokensPerChunk
        }
        return max(256, encoderSteps * 4)
    }

    private func inferDecoderOutputRolesIfNeeded(output: MLFeatureProvider, feed: [String: Any]) throws {
        if decoderLogitsOutputName == nil {
            var best: (name: String, rank: Int, lastDim: Int, count: Int)?
            for name in output.featureNames {
                guard let arr = output.featureValue(for: name)?.multiArrayValue else { continue }
                let shape = arr.intShape
                guard arr.isFloatLike, shape.count >= 2 else { continue }
                let candidate: (name: String, rank: Int, lastDim: Int, count: Int) = (
                    name: name,
                    rank: shape.count,
                    lastDim: shape.last ?? 0,
                    count: arr.count
                )
                if let b = best {
                    if candidate.rank > b.rank ||
                        (candidate.rank == b.rank && candidate.lastDim > b.lastDim) ||
                        (candidate.rank == b.rank && candidate.lastDim == b.lastDim && candidate.count > b.count) {
                        best = candidate
                    }
                } else {
                    best = candidate
                }
            }
            decoderLogitsOutputName = best?.name
        }

        if decoderStateOutputByInputName.isEmpty {
            let stateInputs = decoderStateInputNames
            var unusedOutputStateNames: [String] = []
            let orderedOutputNames = Array(output.featureNames).sorted()
            for name in orderedOutputNames {
                guard name != decoderLogitsOutputName else { continue }
                guard let out = output.featureValue(for: name)?.multiArrayValue else { continue }
                guard out.isFloatLike, out.intShape.count == 3 else { continue }
                unusedOutputStateNames.append(name)
            }

            for inName in stateInputs {
                // Prefer explicit name-pair mapping when available.
                let preferredOutName = inName.replacingOccurrences(of: "input_", with: "output_")
                if let preferredIdx = unusedOutputStateNames.firstIndex(of: preferredOutName),
                   let inState = feed[inName] as? MLMultiArray,
                   let out = output.featureValue(for: preferredOutName)?.multiArrayValue,
                   out.intShape == inState.intShape {
                    decoderStateOutputByInputName[inName] = preferredOutName
                    unusedOutputStateNames.remove(at: preferredIdx)
                    continue
                }

                guard let inState = feed[inName] as? MLMultiArray else { continue }
                let inShape = inState.intShape
                if let matched = unusedOutputStateNames.first(where: {
                    guard let out = output.featureValue(for: $0)?.multiArrayValue else { return false }
                    return out.intShape == inShape
                }) {
                    decoderStateOutputByInputName[inName] = matched
                    unusedOutputStateNames.removeAll(where: { $0 == matched })
                }
            }

            if debugLoggingEnabled {
                let pairs = decoderStateInputNames.compactMap { inName -> String? in
                    guard let outName = decoderStateOutputByInputName[inName] else { return nil }
                    return "\(inName)->\(outName)"
                }.joined(separator: ", ")
                fputs("[ParakeetSwift] decoder logits output=\(decoderLogitsOutputName ?? "nil")\n", stderr)
                fputs("[ParakeetSwift] decoder state map: \(pairs)\n", stderr)
            }
        }
    }
}

private struct ModelInputSpec {
    let name: String
    let shape: [Int]
    let dataType: MLMultiArrayDataType
}

private extension ParakeetCoreMLRNNTTranscriptionModel {
    static func loadModel(at url: URL, configuration: MLModelConfiguration) throws -> MLModel {
        let ext = url.pathExtension.lowercased()
        if ext == "mlmodelc" {
            return try MLModel(contentsOf: url, configuration: configuration)
        }
        if ext == "mlpackage" || ext == "mlmodel" {
            let compiledURL = try compiledModelURL(for: url)
            return try MLModel(contentsOf: compiledURL, configuration: configuration)
        }
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    static func compiledModelURL(for sourceURL: URL) throws -> URL {
        let cacheDir = sourceURL.deletingLastPathComponent().appendingPathComponent(".mlmodelc-cache", isDirectory: true)
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        let cacheName = sourceURL.lastPathComponent.replacingOccurrences(of: ".", with: "_") + ".mlmodelc"
        let cachedURL = cacheDir.appendingPathComponent(cacheName, isDirectory: true)

        if FileManager.default.fileExists(atPath: cachedURL.path), try isCachedCompiledModelCurrent(cachedURL: cachedURL, sourceURL: sourceURL) {
            return cachedURL
        }

        let compiledTempURL = try MLModel.compileModel(at: sourceURL)
        if FileManager.default.fileExists(atPath: cachedURL.path) {
            try FileManager.default.removeItem(at: cachedURL)
        }
        try FileManager.default.copyItem(at: compiledTempURL, to: cachedURL)
        return cachedURL
    }

    static func isCachedCompiledModelCurrent(cachedURL: URL, sourceURL: URL) throws -> Bool {
        let fm = FileManager.default
        let sourceAttrs = try fm.attributesOfItem(atPath: sourceURL.path)
        let cachedAttrs = try fm.attributesOfItem(atPath: cachedURL.path)
        let sourceDate = sourceAttrs[.modificationDate] as? Date ?? .distantPast
        let cachedDate = cachedAttrs[.modificationDate] as? Date ?? .distantPast
        return cachedDate >= sourceDate
    }

    static func inputSpecs(model: MLModel) -> [String: ModelInputSpec] {
        var out: [String: ModelInputSpec] = [:]
        for (name, desc) in model.modelDescription.inputDescriptionsByName {
            guard let constraint = desc.multiArrayConstraint else { continue }
            out[name] = ModelInputSpec(
                name: name,
                shape: constraint.shape.map(\.intValue),
                dataType: constraint.dataType
            )
        }
        return out
    }

    static func makeArray(shape: [Int], dataType: MLMultiArrayDataType) throws -> MLMultiArray {
        let nsShape = shape.map { NSNumber(value: $0) }
        return try MLMultiArray(shape: nsShape, dataType: dataType)
    }

    static func fill(array: MLMultiArray, with value: Float) throws {
        for idx in 0..<array.count {
            array.setFloatValue(value, flatIndex: idx)
        }
    }

    static func setScalarInt(_ array: MLMultiArray, value: Int) throws {
        guard array.count > 0 else { return }
        switch array.dataType {
        case .int32:
            let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: 1)
            ptr[0] = Int32(value)
        default:
            array.setFloatValue(Float(value), flatIndex: 0)
        }
    }

    static func copyFeatureMatrixToEncoderInput(
        featureMatrix: [Float],
        melBins: Int,
        sourceFrames: Int,
        destination: MLMultiArray,
        destinationFrames: Int,
        copyFrames: Int
    ) throws {
        guard destination.intShape.count == 3 else {
            throw ParakeetCoreMLRNNTError.unsupportedArrayRank(destination.intShape.count)
        }
        let shape = destination.intShape
        let batch = shape[0]
        let features = shape[1]
        let frames = shape[2]
        guard batch >= 1, features >= melBins, frames >= destinationFrames else {
            throw ParakeetCoreMLRNNTError.invalidModelIO("Encoder input shape mismatch.")
        }
        guard sourceFrames > 0 else { return }

        for m in 0..<melBins {
            for t in 0..<copyFrames {
                let srcIdx = m * sourceFrames + t
                let value = srcIdx < featureMatrix.count ? featureMatrix[srcIdx] : 0
                destination.setFloatValue(value, indices: [0, m, t])
            }
        }
    }

    static func copyFeatureWindowToEncoderInput(
        featureMatrix: [Float],
        melBins: Int,
        totalFrames: Int,
        startFrame: Int,
        sourceFrames: Int,
        destination: MLMultiArray,
        copyFrames: Int
    ) throws {
        guard destination.intShape.count == 3 else {
            throw ParakeetCoreMLRNNTError.unsupportedArrayRank(destination.intShape.count)
        }
        let shape = destination.intShape
        guard shape[0] >= 1, shape[1] >= melBins, shape[2] >= copyFrames else {
            throw ParakeetCoreMLRNNTError.invalidModelIO("Encoder input shape mismatch.")
        }
        guard totalFrames > 0 else { return }
        for m in 0..<melBins {
            let base = m * totalFrames
            for t in 0..<copyFrames {
                let srcT = startFrame + t
                let value: Float
                if srcT >= 0, srcT < sourceFrames + startFrame, base + srcT < featureMatrix.count {
                    value = featureMatrix[base + srcT]
                } else {
                    value = 0
                }
                destination.setFloatValue(value, indices: [0, m, t])
            }
        }
    }

    static func pickEncoderTensorAndLength(from provider: MLFeatureProvider, debug: Bool = false) throws -> (MLMultiArray, Int?) {
        var length: Int?
        var bestName: String?
        var bestScore: (Int, Int, Int)?

        let orderedNames = Array(provider.featureNames).sorted()
        for name in orderedNames {
            guard let arr = provider.featureValue(for: name)?.multiArrayValue else { continue }
            let shape = arr.intShape
            if debug {
                fputs("[ParakeetSwift] encoder output \(name): dtype=\(arr.dataType.rawValue) shape=\(shape)\n", stderr)
            }
            if arr.isIntegerLike, shape.count == 1, arr.count == 1 {
                length = Int(arr.floatValue(flatIndex: 0))
                continue
            }
            guard arr.isFloatLike, shape.count >= 2 else { continue }
            let score = (shape.count, shape.last ?? 0, arr.count)
            if let b = bestScore {
                if score.0 > b.0 || (score.0 == b.0 && score.1 > b.1) || (score.0 == b.0 && score.1 == b.1 && score.2 > b.2) {
                    bestName = name
                    bestScore = score
                }
            } else {
                bestName = name
                bestScore = score
            }
        }

        guard let tensorName = bestName,
              let tensor = provider.featureValue(for: tensorName)?.multiArrayValue else {
            throw ParakeetCoreMLRNNTError.missingOutput("Could not infer encoder tensor output.")
        }
        if debug {
            fputs("[ParakeetSwift] selected encoder tensor \(tensorName) shape=\(tensor.intShape) length=\(length.map(String.init) ?? "nil")\n", stderr)
            if tensor.intShape.count == 3, tensor.intShape[0] > 0, tensor.intShape[1] > 0 {
                let limit = min(10, tensor.intShape[2])
                let preview = (0..<limit).map { String(format: "%.6f", tensor.floatValue(indices: [0, 0, $0])) }.joined(separator: ", ")
                fputs("[ParakeetSwift] encoder[0,0,0:\(limit)] = [\(preview)]\n", stderr)
            }
        }
        return (tensor, length)
    }

    static func copyOverlapping(source: MLMultiArray, destination: MLMultiArray) throws {
        let srcShape = source.intShape
        let dstShape = destination.intShape
        guard srcShape.count == dstShape.count else {
            throw ParakeetCoreMLRNNTError.invalidModelIO("Cannot copy tensors with different ranks.")
        }
        let rank = srcShape.count
        var indices = Array(repeating: 0, count: rank)

        func recurse(_ dim: Int) {
            if dim == rank {
                let v = source.floatValue(indices: indices)
                destination.setFloatValue(v, indices: indices)
                return
            }
            let limit = min(srcShape[dim], dstShape[dim])
            if limit <= 0 {
                return
            }
            for i in 0..<limit {
                indices[dim] = i
                recurse(dim + 1)
            }
        }

        recurse(0)
    }

    static func slice3DTensor(_ source: MLMultiArray, startFrame: Int, endFrame: Int) throws -> MLMultiArray {
        let shape = source.intShape
        guard shape.count == 3 else {
            throw ParakeetCoreMLRNNTError.unsupportedArrayRank(shape.count)
        }
        let start = max(0, min(startFrame, shape[2]))
        let end = max(start, min(endFrame, shape[2]))
        let outFrames = max(0, end - start)
        let out = try makeArray(shape: [shape[0], shape[1], outFrames], dataType: source.dataType)
        if outFrames == 0 { return out }
        for b in 0..<shape[0] {
            for c in 0..<shape[1] {
                for t in 0..<outFrames {
                    out.setFloatValue(source.floatValue(indices: [b, c, start + t]), indices: [b, c, t])
                }
            }
        }
        return out
    }

    static func extractStepLogits(logits: MLMultiArray, tIndex: Int) throws -> [Float] {
        let shape = logits.intShape
        switch shape.count {
        case 4:
            guard shape[0] > 0 else { return [] }
            return try extractStepLogitsRank3(
                logits: logits,
                dims: [shape[1], shape[2], shape[3]],
                tIndex: tIndex,
                batchIndex: 0
            )
        case 3:
            return try extractStepLogitsRank3(
                logits: logits,
                dims: shape,
                tIndex: tIndex,
                batchIndex: nil
            )
        case 2:
            let t = min(max(tIndex, 0), max(0, shape[0] - 1))
            return (0..<shape[1]).map { v in logits.floatValue(indices: [t, v]) }
        case 1:
            return (0..<shape[0]).map { v in logits.floatValue(indices: [v]) }
        default:
            throw ParakeetCoreMLRNNTError.unsupportedArrayRank(shape.count)
        }
    }

    static func extractStepLogitsRank3(
        logits: MLMultiArray,
        dims: [Int],
        tIndex: Int,
        batchIndex: Int?
    ) throws -> [Float] {
        guard dims.count == 3 else {
            throw ParakeetCoreMLRNNTError.unsupportedArrayRank(dims.count)
        }

        // Mirror Python reference logic:
        // 1) treat largest dim as vocab axis,
        // 2) move vocab axis to the last position,
        // 3) collapse U dimension via [:, 0, :] (or [0] when A==1),
        // 4) pick time row t.
        let vocabAxis = dims.enumerated().max(by: { $0.element < $1.element })?.offset ?? 2
        let movedAxes: [Int]
        switch vocabAxis {
        case 0:
            movedAxes = [1, 2, 0]
        case 1:
            movedAxes = [0, 2, 1]
        default:
            movedAxes = [0, 1, 2]
        }

        let aAxis = movedAxes[0]
        let bAxis = movedAxes[1]
        let vAxis = movedAxes[2]

        let aSize = dims[aAxis]
        let bSize = dims[bAxis]
        let vocabSize = dims[vAxis]
        if vocabSize <= 0 {
            return []
        }

        let clampedT = max(0, tIndex)
        let aIndex: Int
        let bIndex: Int
        if aSize == 1 {
            aIndex = 0
            bIndex = min(clampedT, max(0, bSize - 1))
        } else {
            aIndex = min(clampedT, max(0, aSize - 1))
            bIndex = 0
        }

        var out: [Float] = []
        out.reserveCapacity(vocabSize)
        for v in 0..<vocabSize {
            var idx3 = [0, 0, 0]
            idx3[aAxis] = aIndex
            idx3[bAxis] = bIndex
            idx3[vAxis] = v

            if let b = batchIndex {
                out.append(logits.floatValue(indices: [b, idx3[0], idx3[1], idx3[2]]))
            } else {
                out.append(logits.floatValue(indices: [idx3[0], idx3[1], idx3[2]]))
            }
        }
        return out
    }

    static func argmax<T: Collection>(_ values: T) -> Int where T.Element == Float {
        var bestIndex = 0
        var bestValue = -Float.infinity
        var i = 0
        for value in values {
            if value > bestValue {
                bestValue = value
                bestIndex = i
            }
            i += 1
        }
        return bestIndex
    }

    static func decodePieces(from tokenIDs: [Int], vocab: [String]) -> String {
        let pieces = tokenIDs.compactMap { token -> String? in
            guard token >= 0 && token < vocab.count else { return nil }
            return vocab[token]
        }
        let merged = pieces.joined()
            .replacingOccurrences(of: "▁", with: " ")
        return merged.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    static func debugLogitsCandidates(logits: MLMultiArray, blankID: Int) {
        let shape = logits.intShape
        fputs("[ParakeetSwift] logits shape=\(shape) dtype=\(logits.dataType.rawValue)\n", stderr)
        guard shape.count == 4, shape[0] > 0 else { return }

        let axes = [1, 2, 3]
        for vocabAxis in axes {
            let remaining = axes.filter { $0 != vocabAxis }
            for tAxis in remaining {
                let uAxis = remaining.first { $0 != tAxis }!
                let vocabSize = shape[vocabAxis]
                let tokenLimit = min(vocabSize, blankID + 1)
                if tokenLimit <= 0 { continue }

                var bestToken = 0
                var bestValue = -Float.infinity
                for v in 0..<tokenLimit {
                    var idx = [0, 0, 0, 0]
                    idx[0] = 0
                    idx[vocabAxis] = v
                    idx[tAxis] = 0
                    idx[uAxis] = 0
                    let val = logits.floatValue(indices: idx)
                    if val > bestValue {
                        bestValue = val
                        bestToken = v
                    }
                }
                fputs(
                    "[ParakeetSwift] candidate vocabAxis=\(vocabAxis) tAxis=\(tAxis) uAxis=\(uAxis) -> token=\(bestToken)\n",
                    stderr
                )
            }
        }
    }
}

private extension MLMultiArray {
    var intShape: [Int] {
        shape.map(\.intValue)
    }

    var intStrides: [Int] {
        strides.map(\.intValue)
    }

    var isFloatLike: Bool {
        switch dataType {
        case .float16, .float32, .double:
            return true
        default:
            return false
        }
    }

    var isIntegerLike: Bool {
        switch dataType {
        case .int32:
            return true
        default:
            return false
        }
    }

    func flatIndex(for indices: [Int]) -> Int {
        zip(indices, intStrides).reduce(0) { $0 + $1.0 * $1.1 }
    }

    func floatValue(indices: [Int]) -> Float {
        if dataType == .float16 {
            return self[indices.map { NSNumber(value: $0) }].floatValue
        }
        return floatValue(flatIndex: flatIndex(for: indices))
    }

    func setFloatValue(_ value: Float, indices: [Int]) {
        if dataType == .float16 {
            self[indices.map { NSNumber(value: $0) }] = NSNumber(value: value)
            return
        }
        setFloatValue(value, flatIndex: flatIndex(for: indices))
    }

    func floatValue(flatIndex idx: Int) -> Float {
        switch dataType {
        case .float32:
            return dataPointer.bindMemory(to: Float.self, capacity: count)[idx]
        case .double:
            return Float(dataPointer.bindMemory(to: Double.self, capacity: count)[idx])
        case .int32:
            return Float(dataPointer.bindMemory(to: Int32.self, capacity: count)[idx])
        case .int8:
            return Float(dataPointer.bindMemory(to: Int8.self, capacity: count)[idx])
        case .float16:
            let bits = dataPointer.bindMemory(to: UInt16.self, capacity: count)[idx]
            return Self.float16ToFloat32(bits)
        @unknown default:
            return 0
        }
    }

    func setFloatValue(_ value: Float, flatIndex idx: Int) {
        switch dataType {
        case .float32:
            dataPointer.bindMemory(to: Float.self, capacity: count)[idx] = value
        case .double:
            dataPointer.bindMemory(to: Double.self, capacity: count)[idx] = Double(value)
        case .int32:
            dataPointer.bindMemory(to: Int32.self, capacity: count)[idx] = Int32(value)
        case .int8:
            dataPointer.bindMemory(to: Int8.self, capacity: count)[idx] = Int8(value)
        case .float16:
            dataPointer.bindMemory(to: UInt16.self, capacity: count)[idx] = Self.float32ToFloat16(value)
        @unknown default:
            break
        }
    }

    static func float16ToFloat32(_ bits: UInt16) -> Float {
        let sign = UInt32(bits & 0x8000) << 16
        var exponent = Int((bits & 0x7C00) >> 10)
        var mantissa = UInt32(bits & 0x03FF)

        if exponent == 0 {
            if mantissa == 0 { return Float(bitPattern: sign) }
            exponent = 1
            while (mantissa & 0x0400) == 0 {
                mantissa <<= 1
                exponent -= 1
            }
            mantissa &= 0x03FF
        } else if exponent == 0x1F {
            return Float(bitPattern: sign | 0x7F80_0000 | (mantissa << 13))
        }

        let exp32 = UInt32(exponent + (127 - 15)) << 23
        let mant32 = mantissa << 13
        return Float(bitPattern: sign | exp32 | mant32)
    }

    static func float32ToFloat16(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let sign = UInt16((bits >> 16) & 0x8000)
        let exponent = Int((bits >> 23) & 0xFF) - 127 + 15
        let mantissa = bits & 0x7F_FFFF

        if exponent <= 0 {
            if exponent < -10 { return sign }
            let mant = mantissa | 0x80_0000
            let shift = UInt32(14 - exponent)
            var halfMant = UInt16(mant >> shift)
            if (mant >> (shift - 1)) & 1 == 1 {
                halfMant &+= 1
            }
            return sign | halfMant
        } else if exponent >= 0x1F {
            return sign | 0x7C00
        } else {
            let halfExp = UInt16(exponent) << 10
            var halfMant = UInt16(mantissa >> 13)
            if (mantissa >> 12) & 1 == 1 {
                halfMant &+= 1
            }
            return sign | halfExp | halfMant
        }
    }
}

private struct MelFeatureExtractor {
    let sampleRate: Int
    let nFFT: Int
    let windowLength: Int
    let hopLength: Int
    let melBins: Int

    private let window: [Float]
    private let melFilters: [[Float]]
    private let cosTable: [[Float]]
    private let sinTable: [[Float]]
    private let spectrumBins: Int

    init(sampleRate: Int, nFFT: Int, windowLength: Int, hopLength: Int, melBins: Int) {
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.windowLength = windowLength
        self.hopLength = hopLength
        self.melBins = melBins
        self.spectrumBins = nFFT / 2 + 1

        var w: [Float] = []
        w.reserveCapacity(windowLength)
        if windowLength <= 1 {
            w = [1.0]
        } else {
            for n in 0..<windowLength {
                // Match FFT-style periodic Hann window (librosa/scipy fftbins=True).
                let value = 0.5 - 0.5 * cos(2.0 * Float.pi * Float(n) / Float(windowLength))
                w.append(value)
            }
        }
        self.window = w
        self.melFilters = Self.buildMelFilters(
            sampleRate: sampleRate,
            nFFT: nFFT,
            melBins: melBins,
            lowHz: 0,
            highHz: Float(sampleRate) / 2
        )

        var cosMatrix: [[Float]] = Array(repeating: Array(repeating: 0, count: nFFT), count: spectrumBins)
        var sinMatrix: [[Float]] = Array(repeating: Array(repeating: 0, count: nFFT), count: spectrumBins)
        for k in 0..<spectrumBins {
            for n in 0..<nFFT {
                let phase = 2 * Float.pi * Float(k * n) / Float(nFFT)
                cosMatrix[k][n] = cos(phase)
                sinMatrix[k][n] = sin(phase)
            }
        }
        self.cosTable = cosMatrix
        self.sinTable = sinMatrix
    }

    func extract(samples: [Float]) -> ([Float], Int) {
        if samples.isEmpty {
            return (Array(repeating: 0, count: melBins), 1)
        }
        let centerPad = nFFT / 2
        var padded = Array(repeating: Float(0), count: centerPad + samples.count + centerPad)
        for i in 0..<samples.count {
            padded[centerPad + i] = samples[i]
        }

        let frameCount: Int
        if padded.count <= nFFT {
            frameCount = 1
        } else {
            frameCount = 1 + ((padded.count - nFFT) / hopLength)
        }

        var logMel = Array(repeating: Float(0), count: melBins * frameCount)
        var fftBuffer = Array(repeating: Float(0), count: nFFT)
        var powerSpectrum = Array(repeating: Float(0), count: spectrumBins)
        let windowOffset = max(0, (nFFT - windowLength) / 2)

        for frameIdx in 0..<frameCount {
            let start = frameIdx * hopLength
            for i in 0..<nFFT {
                fftBuffer[i] = 0
            }
            for i in 0..<windowLength {
                let src = start + i
                let sample = src < padded.count ? padded[src] : 0
                let dst = windowOffset + i
                if dst < nFFT {
                    fftBuffer[dst] = sample * window[i]
                }
            }

            for k in 0..<spectrumBins {
                var real: Float = 0
                var imag: Float = 0
                let cosRow = cosTable[k]
                let sinRow = sinTable[k]
                for n in 0..<nFFT {
                    let x = fftBuffer[n]
                    real += x * cosRow[n]
                    imag -= x * sinRow[n]
                }
                powerSpectrum[k] = real * real + imag * imag
            }

            for m in 0..<melBins {
                var energy: Float = 0
                let weights = melFilters[m]
                for k in 0..<spectrumBins {
                    energy += weights[k] * powerSpectrum[k]
                }
                logMel[m * frameCount + frameIdx] = log(max(energy, 1e-6))
            }
        }

        // Per-feature normalization.
        for m in 0..<melBins {
            let base = m * frameCount
            var mean: Float = 0
            for t in 0..<frameCount {
                mean += logMel[base + t]
            }
            mean /= Float(frameCount)

            var variance: Float = 0
            for t in 0..<frameCount {
                let d = logMel[base + t] - mean
                variance += d * d
            }
            variance /= Float(frameCount)
            let std = sqrt(variance) + 1e-5

            for t in 0..<frameCount {
                logMel[base + t] = (logMel[base + t] - mean) / std
            }
        }

        return (logMel, frameCount)
    }

    private static func hzToMel(_ hz: Float) -> Float {
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1000.0
        let minLogMel: Float = minLogHz / fSp
        let logStep: Float = log(6.4) / 27.0
        if hz >= minLogHz {
            return minLogMel + log(hz / minLogHz) / logStep
        }
        return hz / fSp
    }

    private static func melToHz(_ mel: Float) -> Float {
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1000.0
        let minLogMel: Float = minLogHz / fSp
        let logStep: Float = log(6.4) / 27.0
        if mel >= minLogMel {
            return minLogHz * exp(logStep * (mel - minLogMel))
        }
        return mel * fSp
    }

    private static func buildMelFilters(
        sampleRate: Int,
        nFFT: Int,
        melBins: Int,
        lowHz: Float,
        highHz: Float
    ) -> [[Float]] {
        let spectrumBins = nFFT / 2 + 1
        let lowMel = hzToMel(lowHz)
        let highMel = hzToMel(highHz)
        let melStep = (highMel - lowMel) / Float(melBins + 1)
        let melPoints = (0..<(melBins + 2)).map { lowMel + Float($0) * melStep }
        let hzPoints = melPoints.map(melToHz)
        let fftFreqs = (0..<spectrumBins).map { Float($0) * Float(sampleRate) / Float(nFFT) }

        var filters = Array(repeating: Array(repeating: Float(0), count: spectrumBins), count: melBins)
        for m in 0..<melBins {
            let leftHz = hzPoints[m]
            let centerHz = hzPoints[m + 1]
            let rightHz = hzPoints[m + 2]

            for k in 0..<spectrumBins {
                let f = fftFreqs[k]
                if f >= leftHz, f <= centerHz, centerHz > leftHz {
                    filters[m][k] = (f - leftHz) / (centerHz - leftHz)
                } else if f > centerHz, f <= rightHz, rightHz > centerHz {
                    filters[m][k] = (rightHz - f) / (rightHz - centerHz)
                }
            }
            // Slaney-style area normalization (librosa default).
            let enorm = rightHz > leftHz ? 2.0 / (rightHz - leftHz) : 0
            if enorm > 0 {
                for k in 0..<spectrumBins {
                    filters[m][k] *= enorm
                }
            }
        }
        return filters
    }
}

#endif
