#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

model_dir="${PARAKEET_COREML_MODEL_DIR:-$repo_root/artifacts/parakeet-tdt-0.6b-v2}"
suffix="${PARAKEET_COREML_MODEL_SUFFIX:-odmbp-approx}"
metrics_log="${PARAKEET_METRICS_LOG:-1}"
agreement="${PARAKEET_STREAM_AGREEMENT:-2}"
draft_agreement="${PARAKEET_STREAM_DRAFT_AGREEMENT:-1}"
vad_start_dbfs="${PARAKEET_VAD_START_DBFS:--50}"
vad_end_dbfs="${PARAKEET_VAD_END_DBFS:--58}"
vad_min_speech_ms="${PARAKEET_VAD_MIN_SPEECH_MS:-40}"
vad_min_silence_ms="${PARAKEET_VAD_MIN_SILENCE_MS:-600}"
max_speech_chunks="${PARAKEET_STREAM_MAX_SPEECH_CHUNKS:-240}"
max_stagnant_chunks="${PARAKEET_STREAM_MAX_STAGNANT_CHUNKS:-0}"
decode_only_when_speech="${PARAKEET_DECODE_ONLY_WHEN_SPEECH:-0}"
stream_mode="${PARAKEET_STREAM_MODE:-rewrite-prefix}"
flush_on_speech_end="${PARAKEET_STREAM_FLUSH_ON_SPEECH_END:-0}"
stream_history_frames="${PARAKEET_STREAM_HISTORY_FRAMES:-300}"
stream_min_tail_frames="${PARAKEET_STREAM_MIN_TAIL_FRAMES:-8}"
if [[ "$stream_mode" == "rewrite-prefix" ]]; then
  default_chunk_ms=500
  default_hop_ms=250
  default_max_batch_sec=0.5
  default_latest_first=1
  default_backlog_soft_sec=5.0
  default_backlog_target_sec=1.5
else
  default_chunk_ms=160
  default_hop_ms=80
  default_max_batch_sec=0.25
  default_latest_first=1
  default_backlog_soft_sec=1.5
  default_backlog_target_sec=0.4
fi

chunk_ms="${PARAKEET_STREAM_CHUNK_MS:-$default_chunk_ms}"
hop_ms="${PARAKEET_STREAM_HOP_MS:-$default_hop_ms}"
max_batch_sec="${PARAKEET_STREAM_MAX_BATCH_SEC:-$default_max_batch_sec}"
max_buffer_sec="${PARAKEET_STREAM_MAX_BUFFER_SEC:-4}"
backlog_soft_sec="${PARAKEET_STREAM_BACKLOG_SOFT_SEC:-$default_backlog_soft_sec}"
backlog_target_sec="${PARAKEET_STREAM_BACKLOG_TARGET_SEC:-$default_backlog_target_sec}"
latest_first="${PARAKEET_STREAM_LATEST_FIRST:-$default_latest_first}"
max_symbols_per_step="${PARAKEET_TDT_MAX_SYMBOLS_PER_STEP:-10}"
max_tokens_per_chunk="${PARAKEET_TDT_MAX_TOKENS_PER_CHUNK:-0}"
voice_processing="${PARAKEET_AUDIO_VOICE_PROCESSING:-0}"
voice_processing_agc="${PARAKEET_AUDIO_VOICE_PROCESSING_AGC:-0}"

encoder="$model_dir/encoder-model-$suffix.mlpackage"
decoder="$model_dir/decoder_joint-model-$suffix.mlpackage"
vocab="$model_dir/vocab.txt"

if [[ ! -d "$encoder" || ! -d "$decoder" || ! -f "$vocab" ]]; then
  cat >&2 <<EOF
Missing Parakeet artifacts for suffix '$suffix'.
Expected:
  $encoder
  $decoder
  $vocab
Set PARAKEET_COREML_MODEL_DIR and/or PARAKEET_COREML_MODEL_SUFFIX correctly.
EOF
  exit 1
fi

echo "Launching transcribe-macos (release)"
echo "  model_dir: $model_dir"
echo "  suffix:    $suffix"
echo "  metrics:   PARAKEET_METRICS_LOG=$metrics_log"
echo "  stream:    chunk=${chunk_ms}ms hop=${hop_ms}ms agreement=${agreement}"
echo "  draft:     agreement=${draft_agreement}"
echo "  vad:       start=${vad_start_dbfs}dBFS end=${vad_end_dbfs}dBFS min_speech=${vad_min_speech_ms}ms min_silence=${vad_min_silence_ms}ms"
echo "  decode:    only_when_speech=${decode_only_when_speech} flush_on_end=${flush_on_speech_end}"
echo "  mode:      stream_mode=${stream_mode}"
echo "  context:   history_frames=${stream_history_frames} min_tail_frames=${stream_min_tail_frames}"
echo "  batching:  max_batch_sec=${max_batch_sec}"
echo "  limits:    max_buffer_sec=${max_buffer_sec} backlog_soft_sec=${backlog_soft_sec} backlog_target_sec=${backlog_target_sec} latest_first=${latest_first}"
echo "  tdt:       max_symbols_per_step=${max_symbols_per_step} max_tokens_per_chunk=${max_tokens_per_chunk}"
echo "  guards:    max_speech_chunks=${max_speech_chunks} max_stagnant_chunks=${max_stagnant_chunks}"
echo "  audio:     voice_processing=${voice_processing} agc=${voice_processing_agc}"

PARAKEET_COREML_MODEL_DIR="$model_dir" \
PARAKEET_COREML_MODEL_SUFFIX="$suffix" \
PARAKEET_METRICS_LOG="$metrics_log" \
PARAKEET_STREAM_CHUNK_MS="$chunk_ms" \
PARAKEET_STREAM_HOP_MS="$hop_ms" \
PARAKEET_STREAM_AGREEMENT="$agreement" \
PARAKEET_STREAM_DRAFT_AGREEMENT="$draft_agreement" \
PARAKEET_VAD_START_DBFS="$vad_start_dbfs" \
PARAKEET_VAD_END_DBFS="$vad_end_dbfs" \
PARAKEET_VAD_MIN_SPEECH_MS="$vad_min_speech_ms" \
PARAKEET_VAD_MIN_SILENCE_MS="$vad_min_silence_ms" \
PARAKEET_STREAM_MAX_SPEECH_CHUNKS="$max_speech_chunks" \
PARAKEET_STREAM_MAX_STAGNANT_CHUNKS="$max_stagnant_chunks" \
PARAKEET_DECODE_ONLY_WHEN_SPEECH="$decode_only_when_speech" \
PARAKEET_STREAM_MODE="$stream_mode" \
PARAKEET_STREAM_FLUSH_ON_SPEECH_END="$flush_on_speech_end" \
PARAKEET_STREAM_HISTORY_FRAMES="$stream_history_frames" \
PARAKEET_STREAM_MIN_TAIL_FRAMES="$stream_min_tail_frames" \
PARAKEET_STREAM_MAX_BATCH_SEC="$max_batch_sec" \
PARAKEET_STREAM_MAX_BUFFER_SEC="$max_buffer_sec" \
PARAKEET_STREAM_BACKLOG_SOFT_SEC="$backlog_soft_sec" \
PARAKEET_STREAM_BACKLOG_TARGET_SEC="$backlog_target_sec" \
PARAKEET_STREAM_LATEST_FIRST="$latest_first" \
PARAKEET_TDT_MAX_SYMBOLS_PER_STEP="$max_symbols_per_step" \
PARAKEET_TDT_MAX_TOKENS_PER_CHUNK="$max_tokens_per_chunk" \
PARAKEET_AUDIO_VOICE_PROCESSING="$voice_processing" \
PARAKEET_AUDIO_VOICE_PROCESSING_AGC="$voice_processing_agc" \
swift run -c release transcribe-macos
