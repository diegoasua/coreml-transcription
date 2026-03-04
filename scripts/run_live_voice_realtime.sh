#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Tuned defaults from the current best split draft/confirmed realtime run.
export PARAKEET_STREAM_MODE="${PARAKEET_STREAM_MODE:-rewrite-prefix}"
export PARAKEET_STREAM_CHUNK_MS="${PARAKEET_STREAM_CHUNK_MS:-500}"
export PARAKEET_STREAM_HOP_MS="${PARAKEET_STREAM_HOP_MS:-250}"
export PARAKEET_STREAM_MAX_BATCH_SEC="${PARAKEET_STREAM_MAX_BATCH_SEC:-0.5}"
export PARAKEET_STREAM_AGREEMENT="${PARAKEET_STREAM_AGREEMENT:-2}"
export PARAKEET_STREAM_DRAFT_AGREEMENT="${PARAKEET_STREAM_DRAFT_AGREEMENT:-1}"
# Live voice UX profile: prefer continuity over aggressive catch-up drops.
export PARAKEET_STREAM_LATEST_FIRST="${PARAKEET_STREAM_LATEST_FIRST:-0}"
export PARAKEET_STREAM_BACKLOG_SOFT_SEC="${PARAKEET_STREAM_BACKLOG_SOFT_SEC:-12.0}"
export PARAKEET_STREAM_BACKLOG_TARGET_SEC="${PARAKEET_STREAM_BACKLOG_TARGET_SEC:-6.0}"
# Force decode on all chunks; do not rely on VAD gate for initial live bring-up.
export PARAKEET_DECODE_ONLY_WHEN_SPEECH=0
export PARAKEET_STREAM_FLUSH_ON_SPEECH_END="${PARAKEET_STREAM_FLUSH_ON_SPEECH_END:-0}"
export PARAKEET_METRICS_LOG="${PARAKEET_METRICS_LOG:-1}"
# Device compatibility default: disable AVAudio voice-processing path unless opted in.
export PARAKEET_AUDIO_VOICE_PROCESSING="${PARAKEET_AUDIO_VOICE_PROCESSING:-0}"

echo "Launching live voice realtime test (split draft/confirmed)..."
echo "  chunk/hop: ${PARAKEET_STREAM_CHUNK_MS}ms/${PARAKEET_STREAM_HOP_MS}ms"
echo "  agreement: confirmed=${PARAKEET_STREAM_AGREEMENT} draft=${PARAKEET_STREAM_DRAFT_AGREEMENT}"
echo "  scheduling: latest_first=${PARAKEET_STREAM_LATEST_FIRST} backlog_soft=${PARAKEET_STREAM_BACKLOG_SOFT_SEC}s backlog_target=${PARAKEET_STREAM_BACKLOG_TARGET_SEC}s"
echo "  decode_only_when_speech: ${PARAKEET_DECODE_ONLY_WHEN_SPEECH}"
echo "  voice_processing: ${PARAKEET_AUDIO_VOICE_PROCESSING}"

bash scripts/run_transcribe_macos_release.sh
