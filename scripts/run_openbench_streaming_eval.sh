#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-timit}"
RUN_NAME="${RUN_NAME:-parakeet-coreml-streaming}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
PYTHON_TRANSCRIBER="${PYTHON_TRANSCRIBER:-scripts/parakeet_coreml_tdt_transcriber.py}"

ARGS=(
  --dataset "${DATASET}"
  --pipeline-kind streaming
  --python-transcriber "${PYTHON_TRANSCRIBER}"
  --run-name "${RUN_NAME}"
  --metrics wer streaming_latency confirmed_streaming_latency
)

if [[ -n "${NUM_SAMPLES}" ]]; then
  ARGS+=(--num-samples "${NUM_SAMPLES}")
fi

bash scripts/run_openbench_eval.sh "${ARGS[@]}" "$@"
