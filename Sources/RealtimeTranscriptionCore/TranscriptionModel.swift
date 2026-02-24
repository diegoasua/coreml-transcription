import Foundation

public protocol TranscriptionModel {
    mutating func transcribeChunk(_ samples: [Float], sampleRate: Int) throws -> String
    mutating func resetState()
}
