// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "coreml-transcription",
    platforms: [
        .macOS(.v13),
        .iOS(.v17),
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
    ]
)
