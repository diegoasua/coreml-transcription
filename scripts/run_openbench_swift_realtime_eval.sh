#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET="${DATASET:-earnings22-3hours}"
RUN_NAME="${RUN_NAME:-parakeet-swift-realtime-${DATASET}}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
COMMAND_CWD="${COMMAND_CWD:-${REPO_ROOT}}"
TRANSCRIBE_CMD="${TRANSCRIBE_CMD:-bash scripts/transcribe_swift_realtime_stdout.sh {audio_path}}"
REALTIME_ARTIFACTS_DIR="${PARAKEET_REALTIME_EVAL_ARTIFACTS_DIR:-${REPO_ROOT}/artifacts/realtime-openbench-runs}"
REALTIME_RUN_TAG="${PARAKEET_REALTIME_EVAL_RUN_TAG:-${RUN_NAME}}"
PROGRESS_SEC="${PARAKEET_REALTIME_EVAL_PROGRESS_SEC:-300}"
TIMEOUT_SEC="${PARAKEET_REALTIME_EVAL_TIMEOUT_SEC:-7200}"
MONITOR_PID=""

progress_monitor() {
  local run_dir="${REALTIME_ARTIFACTS_DIR}/${REALTIME_RUN_TAG}"
  local interval
  interval="$(python3 - <<'PY' "${PROGRESS_SEC}"
import sys
try:
    value = int(float(sys.argv[1]))
except Exception:
    value = 300
print(max(5, value))
PY
)"
  echo "[swift-realtime-eval] progress monitor: every ${interval}s, artifacts under ${run_dir}" >&2
  while true; do
    sleep "${interval}"
    if [[ ! -d "${run_dir}" ]]; then
      echo "[swift-realtime-eval] waiting for realtime artifacts in ${run_dir}" >&2
      continue
    fi

    local completed
    completed="$(find "${run_dir}" -name 'summary.json' -type f 2>/dev/null | wc -l | tr -d ' ')"
    local latest_log
    latest_log="$(find "${run_dir}" -name 'progress.log' -type f -print0 2>/dev/null | xargs -0 ls -1t 2>/dev/null | head -n 1)"
    if [[ -z "${latest_log}" ]]; then
      echo "[swift-realtime-eval] completed_samples=${completed} | no active progress log yet" >&2
      continue
    fi

    local sample_name
    sample_name="$(basename "$(dirname "${latest_log}")")"
    local last_line
    last_line="$(tail -n 1 "${latest_log}" 2>/dev/null || true)"
    if [[ -z "${last_line}" ]]; then
      echo "[swift-realtime-eval] completed_samples=${completed} | active_sample=${sample_name} | progress log exists, waiting for first update" >&2
    else
      echo "[swift-realtime-eval] completed_samples=${completed} | active_sample=${sample_name} | ${last_line}" >&2
    fi
  done
}

cleanup() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

ARGS=(
  --dataset "${DATASET}"
  --transcribe-cmd "${TRANSCRIBE_CMD}"
  --run-name "${RUN_NAME}"
  --command-cwd "${COMMAND_CWD}"
  --timeout-sec "${TIMEOUT_SEC}"
)

if [[ -n "${NUM_SAMPLES}" ]]; then
  ARGS+=(--num-samples "${NUM_SAMPLES}")
fi

progress_monitor &
MONITOR_PID="$!"

PARAKEET_REALTIME_EVAL_ARTIFACTS_DIR="${REALTIME_ARTIFACTS_DIR}" \
PARAKEET_REALTIME_EVAL_RUN_TAG="${REALTIME_RUN_TAG}" \
bash "${REPO_ROOT}/scripts/run_openbench_eval.sh" "${ARGS[@]}" "$@"
