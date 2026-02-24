#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "error: no python found; set PYTHON_BIN"
    exit 1
  fi
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: '${PYTHON_BIN}' not found"
  exit 1
fi

DATASET_PATH="${DATASET_PATH:-hf-audio/esb-datasets-test-only-sorted}"
DATASET="${DATASET:-}"
SPLIT="${SPLIT:-test}"
RUN_NAME="${RUN_NAME:-parakeet-coreml-earnings22-hf}"
NORMALIZER="${NORMALIZER:-open_asr}"
PRINT_EVERY="${PRINT_EVERY:-10}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
PYTHON_TRANSCRIBER="${PYTHON_TRANSCRIBER:-scripts/parakeet_coreml_rnnt_transcriber.py}"
COMMAND_CWD="${COMMAND_CWD:-${REPO_ROOT}}"

ARGS=(
  --dataset-path "${DATASET_PATH}"
  --split "${SPLIT}"
  --python-transcriber "${PYTHON_TRANSCRIBER}"
  --run-name "${RUN_NAME}"
  --normalizer "${NORMALIZER}"
  --print-every "${PRINT_EVERY}"
  --warmup-steps "${WARMUP_STEPS}"
  --command-cwd "${COMMAND_CWD}"
)

if [[ -n "${MAX_EVAL_SAMPLES}" ]]; then
  ARGS+=(--max-eval-samples "${MAX_EVAL_SAMPLES}")
fi

if [[ -n "${DATASET}" ]]; then
  ARGS+=(--dataset "${DATASET}")
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval_hf_open_asr_style.py" "${ARGS[@]}" "$@"
