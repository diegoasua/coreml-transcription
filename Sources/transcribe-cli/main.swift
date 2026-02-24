import Foundation
import RealtimeTranscriptionCore

print("LocalAgreement CLI demo")
print("Enter partial hypotheses line by line. Empty line flushes segment. Ctrl-D exits.")

var controller = StreamingTextController(requiredAgreementCount: 2)

while let line = readLine() {
    if line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        let state = controller.endSegment()
        print("CONFIRMED: \(state.confirmed)")
        print("HYPOTHESIS: \(state.hypothesis)")
        continue
    }

    let state = controller.update(partialText: line)
    print("CONFIRMED: \(state.confirmed)")
    print("HYPOTHESIS: \(state.hypothesis)")
}

let final = controller.endSegment()
if !final.confirmed.isEmpty || !final.hypothesis.isEmpty {
    print("FINAL CONFIRMED: \(final.confirmed)")
    print("FINAL HYPOTHESIS: \(final.hypothesis)")
}
