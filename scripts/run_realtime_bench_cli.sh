#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_name="${RUN_NAME:-parakeet-realtime-bench}"
audio="${AUDIO_PATH:-}"
model_dir="${PARAKEET_COREML_MODEL_DIR:-$repo_root/artifacts/parakeet-tdt-0.6b-v2}"
suffix="${PARAKEET_COREML_MODEL_SUFFIX:-odmbp-approx}"
stream_mode="${PARAKEET_STREAM_MODE:-rewrite-prefix}"
out_dir="${OUT_DIR:-$repo_root/artifacts/realtime-bench-runs/$run_name}"
out_json="$out_dir/summary.json"
reference_path="${PARAKEET_BENCH_REFERENCE_PATH:-${REFERENCE_TRANSCRIPT_PATH:-}}"
reference_max_seconds="${PARAKEET_BENCH_REFERENCE_MAX_SECONDS:-60}"
quality_min_seq_ratio="${PARAKEET_BENCH_QUALITY_MIN_SEQ_RATIO:-0.70}"
quality_min_word_recall="${PARAKEET_BENCH_QUALITY_MIN_WORD_RECALL:-0.60}"
quality_max_trigram_dup="${PARAKEET_BENCH_QUALITY_MAX_TRIGRAM_DUP_RATIO:-0.25}"

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

if [[ "$stream_mode" == "rewrite-prefix" ]]; then
  default_chunk_ms=500
  default_hop_ms=250
  default_max_batch_ms=500
  default_latest_first=1
  default_backlog_soft_sec=5.0
  default_backlog_target_sec=1.5
else
  default_chunk_ms=160
  default_hop_ms=80
  default_max_batch_ms=200
  default_latest_first=1
  default_backlog_soft_sec=1.5
  default_backlog_target_sec=0.25
fi

PARAKEET_STREAM_MODE="$stream_mode" \
PARAKEET_STREAM_LATEST_FIRST="${PARAKEET_STREAM_LATEST_FIRST:-$default_latest_first}" \
PARAKEET_DECODE_ONLY_WHEN_SPEECH="${PARAKEET_DECODE_ONLY_WHEN_SPEECH:-0}" \
swift run -c release transcribe-cli \
  --audio "$audio" \
  --model-dir "$model_dir" \
  --suffix "$suffix" \
  --realtime-bench \
  --stream-chunk-ms "${PARAKEET_STREAM_CHUNK_MS:-$default_chunk_ms}" \
  --stream-hop-ms "${PARAKEET_STREAM_HOP_MS:-$default_hop_ms}" \
  --stream-agreement "${PARAKEET_STREAM_AGREEMENT:-2}" \
  --stream-draft-agreement "${PARAKEET_STREAM_DRAFT_AGREEMENT:-1}" \
  --max-symbols-per-step "${PARAKEET_TDT_MAX_SYMBOLS_PER_STEP:-10}" \
  --max-tokens-per-chunk "${PARAKEET_TDT_MAX_TOKENS_PER_CHUNK:-0}" \
  --report-every-ms "${PARAKEET_BENCH_REPORT_MS:-200}" \
  --max-batch-ms "${PARAKEET_STREAM_MAX_BATCH_MS:-$default_max_batch_ms}" \
  --queue-pass-sec "${PARAKEET_BENCH_QUEUE_PASS_SEC:-0.5}" \
  --first-token-pass-ms "${PARAKEET_BENCH_FIRST_TOKEN_PASS_MS:-300}" \
  --confirmed-pass-ms "${PARAKEET_BENCH_CONFIRMED_PASS_MS:-1700}" \
  --backlog-soft-sec "${PARAKEET_STREAM_BACKLOG_SOFT_SEC:-$default_backlog_soft_sec}" \
  --backlog-target-sec "${PARAKEET_STREAM_BACKLOG_TARGET_SEC:-$default_backlog_target_sec}" \
  --metrics-output "$out_json"

if [[ -n "$reference_path" ]]; then
  confirmed_txt="${out_json%.json}.confirmed.txt"
  compare_report="$out_dir/quality.compare.txt"
  if [[ ! -f "$reference_path" ]]; then
    echo "error: reference transcript not found: $reference_path" >&2
    exit 2
  fi
  if [[ ! -f "$confirmed_txt" ]]; then
    echo "error: confirmed transcript missing: $confirmed_txt" >&2
    exit 2
  fi

  echo "Running quality gate"
  echo "  reference: $reference_path"
  echo "  confirmed: $confirmed_txt"
  python3 scripts/compare_realtime_transcript.py \
    --reference "$reference_path" \
    --confirmed "$confirmed_txt" \
    --max-seconds "$reference_max_seconds" | tee "$compare_report"

  python3 - "$out_json" "$compare_report" "$reference_path" "$quality_min_seq_ratio" "$quality_min_word_recall" "$quality_max_trigram_dup" <<'PY'
import json
import re
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
reference_path = sys.argv[3]
min_seq = float(sys.argv[4])
min_recall = float(sys.argv[5])
max_dup = float(sys.argv[6])

report_text = report_path.read_text(encoding="utf-8", errors="replace")
match = re.search(r"\{[\s\S]*?\}", report_text)
if match is None:
    raise SystemExit("quality gate: failed to parse metrics JSON from compare report")
metrics = json.loads(match.group(0))

summary = json.loads(summary_path.read_text(encoding="utf-8"))
quality_pass = (
    float(metrics.get("sequence_match_ratio", 0.0)) >= min_seq
    and float(metrics.get("word_overlap_recall", 0.0)) >= min_recall
    and float(metrics.get("hypothesis_trigram_duplication_ratio", 1.0)) <= max_dup
)

quality_gate = {
    "enabled": True,
    "reference_path": reference_path,
    "report_path": str(report_path),
    "thresholds": {
        "sequence_match_ratio_min": min_seq,
        "word_overlap_recall_min": min_recall,
        "trigram_duplication_ratio_max": max_dup,
    },
    "metrics": metrics,
    "pass": quality_pass,
}
summary["quality_gate"] = quality_gate

pass_breakdown = summary.get("pass_breakdown") or {}
pass_breakdown["quality"] = quality_pass
summary["pass_breakdown"] = pass_breakdown
summary["pass"] = bool(summary.get("pass", False)) and quality_pass

summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

if not quality_pass:
    raise SystemExit("quality gate failed: transcript quality below threshold")
PY
fi

echo "Done: $out_json"
