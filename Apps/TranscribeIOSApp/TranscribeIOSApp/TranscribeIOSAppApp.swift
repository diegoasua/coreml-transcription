import SwiftUI
import TranscribeAppleAppSupport

@main
struct TranscribeIOSAppApp: App {
    var body: some Scene {
        WindowGroup {
            NavigationStack {
                TranscribeAppleRootView()
            }
        }
    }
}
