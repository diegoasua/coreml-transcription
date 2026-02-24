import Foundation

public struct TranscriptState: Equatable {
    public let confirmed: String
    public let hypothesis: String

    public init(confirmed: String, hypothesis: String) {
        self.confirmed = confirmed
        self.hypothesis = hypothesis
    }
}
