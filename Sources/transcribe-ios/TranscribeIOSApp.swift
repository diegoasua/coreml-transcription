import SwiftUI
import TranscribeAppleAppSupport

@main
struct TranscribeIOSApp: App {
    var body: some Scene {
        WindowGroup {
            NavigationStack {
                TranscribeAppleRootView()
            }
        }
    }
}
