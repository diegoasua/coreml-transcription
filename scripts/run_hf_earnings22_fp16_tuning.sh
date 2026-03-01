#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ ! -f "configs/parakeet-coreml-fp16-baseline.env" ]]; then
  echo "error: run from repo root"
  exit 1
fi

# Base runtime profile.
source configs/parakeet-coreml-fp16-baseline.env
source configs/parakeet-coreml-decoder-conservative.env

declare -a RUNS=(
  "l240r0_s4|240|0|4"
  "l240r120_s4|240|120|4"
  "l300r120_s4|300|120|4"
  "l240r120_s6|240|120|6"
)

for item in "${RUNS[@]}"; do
  IFS="|" read -r suffix left right maxsym <<< "${item}"

  export PARAKEET_ENCODER_LEFT_CONTEXT_FRAMES="${left}"
  export PARAKEET_ENCODER_RIGHT_CONTEXT_FRAMES="${right}"
  export PARAKEET_TDT_MAX_SYMBOLS_PER_STEP="${maxsym}"

  run_name="parakeet-coreml-earnings22-hf-full-fp16-${suffix}"
  echo "==> ${run_name} (left=${left}, right=${right}, max_symbols=${maxsym})"

  WARMUP_STEPS=0 \
  RUN_NAME="${run_name}" \
  bash scripts/run_hf_open_asr_eval.sh \
    --python-transcriber scripts/parakeet_coreml_tdt_transcriber.py
done

echo
echo "Completed. Summaries:"
for item in "${RUNS[@]}"; do
  IFS="|" read -r suffix _ <<< "${item}"
  run_name="parakeet-coreml-earnings22-hf-full-fp16-${suffix}"
  summary="artifacts/hf-asr-eval/${run_name}/summary.json"
  if [[ -f "${summary}" ]]; then
    echo "- ${summary}"
  fi
done
