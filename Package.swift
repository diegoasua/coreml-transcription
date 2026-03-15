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
        .library(
            name: "TranscribeAppleAppSupport",
            targets: ["TranscribeAppleAppSupport"]
        ),
        .executable(
            name: "transcribe-cli",
            targets: ["transcribe-cli"]
        ),
        .executable(
            name: "transcribe-macos",
            targets: ["transcribe-macos"]
        ),
        .executable(
            name: "transcribe-ios",
            targets: ["transcribe-ios"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "RealtimeTranscriptionCore"
        ),
        .target(
            name: "TranscribeAppleAppSupport",
            dependencies: ["RealtimeTranscriptionCore"]
        ),
        .executableTarget(
            name: "transcribe-cli",
            dependencies: ["RealtimeTranscriptionCore"]
        ),
        .executableTarget(
            name: "transcribe-macos",
            dependencies: ["TranscribeAppleAppSupport"]
        ),
        .executableTarget(
            name: "transcribe-ios",
            dependencies: ["TranscribeAppleAppSupport"]
        ),
        .testTarget(
            name: "RealtimeTranscriptionCoreTests",
            dependencies: ["RealtimeTranscriptionCore"]
        ),
    ],
    swiftLanguageModes: [.v6]
)
