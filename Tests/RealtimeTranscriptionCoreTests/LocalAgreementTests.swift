import XCTest
@testable import RealtimeTranscriptionCore

final class LocalAgreementTests: XCTestCase {
    func testConfirmsOnlyStablePrefix() {
        var stabilizer = LocalAgreementStabilizer(config: .init(requiredAgreementCount: 2))

        let s1 = stabilizer.push(partial: "hello wor")
        XCTAssertEqual(s1.confirmed, "")
        XCTAssertEqual(s1.hypothesis, "hello wor")

        let s2 = stabilizer.push(partial: "hello world")
        XCTAssertEqual(s2.confirmed, "hello")
        XCTAssertEqual(s2.hypothesis, "world")

        let s3 = stabilizer.push(partial: "hello world from")
        XCTAssertEqual(s3.confirmed, "hello world")
        XCTAssertEqual(s3.hypothesis, "from")
    }

    func testFlushCommitsLatestHypothesis() {
        var stabilizer = LocalAgreementStabilizer(config: .init(requiredAgreementCount: 3))

        _ = stabilizer.push(partial: "testing local")
        _ = stabilizer.push(partial: "testing local")
        let flushed = stabilizer.flushSegment()

        XCTAssertEqual(flushed.confirmed, "testing local")
        XCTAssertEqual(flushed.hypothesis, "")
    }

    func testUnflushedPrefixCanReviseWhenRecentHypothesesChange() {
        var stabilizer = LocalAgreementStabilizer(config: .init(requiredAgreementCount: 2))

        _ = stabilizer.push(partial: "alpha beta")
        _ = stabilizer.push(partial: "alpha beta one")
        let afterFirstAgreement = stabilizer.push(partial: "alpha zeta one two")
        XCTAssertEqual(afterFirstAgreement.confirmed, "alpha")
        XCTAssertEqual(afterFirstAgreement.hypothesis, "zeta one two")

        let afterSecondAgreement = stabilizer.push(partial: "alpha zeta one two three")
        XCTAssertEqual(afterSecondAgreement.confirmed, "alpha zeta one two")
        XCTAssertEqual(afterSecondAgreement.hypothesis, "three")
    }
}
