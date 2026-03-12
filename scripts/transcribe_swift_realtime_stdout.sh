#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

audio_path="${1:-}"
if [[ -z "$audio_path" ]]; then
  echo "usage: $0 /path/to/audio.wav" >&2
  exit 2
fi

keep_artifacts="${PARAKEET_REALTIME_EVAL_KEEP_ARTIFACTS:-0}"
artifacts_root="${PARAKEET_REALTIME_EVAL_ARTIFACTS_DIR:-$repo_root/artifacts/realtime-openbench-runs}"
sample_stub="$(basename "$audio_path")"
sample_stub="${sample_stub%.*}"
sample_stub="$(printf '%s' "$sample_stub" | tr -cs 'A-Za-z0-9._-' '_')"
run_tag="${PARAKEET_REALTIME_EVAL_RUN_TAG:-$(date +%Y%m%d-%H%M%S)-$$}"
out_dir="$artifacts_root/$run_tag/$sample_stub"
mkdir -p "$out_dir"
progress_log="$out_dir/progress.log"

if [[ -z "${PARAKEET_STREAM_MAX_BATCH_MS:-}" && -n "${PARAKEET_STREAM_MAX_BATCH_SEC:-}" ]]; then
  PARAKEET_STREAM_MAX_BATCH_MS="$(python3 - <<'PY' "${PARAKEET_STREAM_MAX_BATCH_SEC}"
import sys
try:
    sec = float(sys.argv[1])
except Exception:
    raise SystemExit(1)
print(max(1, int(round(sec * 1000.0))))
PY
)"
  export PARAKEET_STREAM_MAX_BATCH_MS
fi

cleanup() {
  if [[ "$keep_artifacts" != "1" ]]; then
    rm -rf "$out_dir"
  fi
}
trap cleanup EXIT

AUDIO_PATH="$audio_path" \
OUT_DIR="$out_dir" \
bash "$repo_root/scripts/run_realtime_bench_cli.sh" >"$progress_log" 2>&1

confirmed_txt="$out_dir/summary.confirmed.txt"
if [[ ! -f "$confirmed_txt" ]]; then
  echo "error: realtime benchmark did not produce $confirmed_txt" >&2
  exit 3
fi

cat "$confirmed_txt"
