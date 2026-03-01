// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "coreml-transcription",
    platforms: [
        .macOS("26.0"),
        .iOS("26.0"),
    ],
    products: [
        .library(
            name: "RealtimeTranscriptionCore",
            targets: ["RealtimeTranscriptionCore"]
        ),
        .executable(
            name: "transcribe-cli",
            targets: ["transcribe-cli"]
        ),
        .executable(
            name: "transcribe-macos",
            targets: ["transcribe-macos"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "RealtimeTranscriptionCore"
        ),
        .executableTarget(
            name: "transcribe-cli",
            dependencies: ["RealtimeTranscriptionCore"]
        ),
        .executableTarget(
            name: "transcribe-macos",
            dependencies: ["RealtimeTranscriptionCore"]
        ),
        .testTarget(
            name: "RealtimeTranscriptionCoreTests",
            dependencies: ["RealtimeTranscriptionCore"]
        ),
    ],
    swiftLanguageModes: [.v6]
)
