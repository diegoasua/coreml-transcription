import Foundation

public enum CTCDecoderError: Error {
    case invalidVocabulary
    case unsupportedLogitsRank(Int)
}

public struct CTCGreedyDecoder {
    public let vocabulary: [String]
    public let blankTokenID: Int
    public let wordDelimiterToken: String?

    public init(
        vocabulary: [String],
        blankTokenID: Int = 0,
        wordDelimiterToken: String? = nil
    ) {
        self.vocabulary = vocabulary
        self.blankTokenID = blankTokenID
        self.wordDelimiterToken = wordDelimiterToken
    }

    public func decode(tokenIDs: [Int]) throws -> String {
        guard !vocabulary.isEmpty else { throw CTCDecoderError.invalidVocabulary }
        var text = ""
        for tokenID in tokenIDs {
            guard tokenID >= 0, tokenID < vocabulary.count else { continue }
            let token = vocabulary[tokenID]

            if let delimiter = wordDelimiterToken, token == delimiter {
                if !text.isEmpty && !text.hasSuffix(" ") {
                    text.append(" ")
                }
                continue
            }

            if token.hasPrefix("▁") {
                if !text.isEmpty && !text.hasSuffix(" ") {
                    text.append(" ")
                }
                text.append(contentsOf: token.dropFirst())
            } else {
                text.append(token)
            }
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public func collapseCTC(_ rawTokenIDs: [Int]) -> [Int] {
        var out: [Int] = []
        out.reserveCapacity(rawTokenIDs.count)

        var previous: Int? = nil
        for token in rawTokenIDs {
            defer { previous = token }
            if token == blankTokenID {
                continue
            }
            if let prev = previous, prev == token {
                continue
            }
            out.append(token)
        }
        return out
    }
}
