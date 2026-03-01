#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_name="${RUN_NAME:-parakeet-realtime-bench}"
audio="${AUDIO_PATH:-}"
model_dir="${PARAKEET_COREML_MODEL_DIR:-$repo_root/artifacts/parakeet-tdt-0.6b-v2}"
suffix="${PARAKEET_COREML_MODEL_SUFFIX:-odmbp-approx}"
out_dir="${OUT_DIR:-$repo_root/artifacts/realtime-bench-runs/$run_name}"
out_json="$out_dir/summary.json"

if [[ -z "$audio" ]]; then
  cat >&2 <<USAGE
Set AUDIO_PATH to a wav/m4a file before running.
Example:
  AUDIO_PATH=/path/to/audio.wav bash scripts/run_realtime_bench_cli.sh
USAGE
  exit 1
fi

mkdir -p "$out_dir"

echo "Running realtime bench"
echo "  audio:    $audio"
echo "  model:    $model_dir (suffix=$suffix)"
echo "  output:   $out_json"

swift run -c release transcribe-cli \
  --audio "$audio" \
  --model-dir "$model_dir" \
  --suffix "$suffix" \
  --realtime-bench \
  --stream-chunk-ms "${PARAKEET_STREAM_CHUNK_MS:-160}" \
  --stream-hop-ms "${PARAKEET_STREAM_HOP_MS:-80}" \
  --stream-agreement "${PARAKEET_STREAM_AGREEMENT:-2}" \
  --max-symbols-per-step "${PARAKEET_TDT_MAX_SYMBOLS_PER_STEP:-4}" \
  --max-tokens-per-chunk "${PARAKEET_TDT_MAX_TOKENS_PER_CHUNK:-64}" \
  --report-every-ms "${PARAKEET_BENCH_REPORT_MS:-200}" \
  --max-batch-ms "${PARAKEET_STREAM_MAX_BATCH_MS:-200}" \
  --queue-pass-sec "${PARAKEET_BENCH_QUEUE_PASS_SEC:-0.5}" \
  --first-token-pass-ms "${PARAKEET_BENCH_FIRST_TOKEN_PASS_MS:-300}" \
  --confirmed-pass-ms "${PARAKEET_BENCH_CONFIRMED_PASS_MS:-1700}" \
  --backlog-soft-sec "${PARAKEET_STREAM_BACKLOG_SOFT_SEC:-1.5}" \
  --backlog-target-sec "${PARAKEET_STREAM_BACKLOG_TARGET_SEC:-0.25}" \
  --metrics-output "$out_json"

echo "Done: $out_json"
