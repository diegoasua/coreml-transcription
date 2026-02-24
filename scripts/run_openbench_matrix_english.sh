#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/artifacts/openbench-runs}"

RUN_PREFIX="${RUN_PREFIX:-parakeet-coreml-english}"
DATASETS_CSV="${DATASETS_CSV:-earnings22}"
PROFILES_CSV="${PROFILES_CSV:-fp16,odmbp-approx,odmbp-approx-beam2}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
PYTHON_TRANSCRIBER="${PYTHON_TRANSCRIBER:-scripts/parakeet_coreml_rnnt_transcriber.py}"
ENABLE_LONGFORM_SEGMENTED="${ENABLE_LONGFORM_SEGMENTED:-0}"

mkdir -p "${RUNS_ROOT}"
MANIFEST_PATH="${RUNS_ROOT}/${RUN_PREFIX}-matrix-manifest.tsv"
REPORT_JSON="${RUNS_ROOT}/${RUN_PREFIX}-matrix-summary.json"
REPORT_CSV="${RUNS_ROOT}/${RUN_PREFIX}-matrix-summary.csv"

echo -e "dataset\tprofile\trun_name\tstatus" > "${MANIFEST_PATH}"

load_profile() {
  local profile="$1"
  case "${profile}" in
    fp16)
      # shellcheck source=/dev/null
      source "${REPO_ROOT}/configs/parakeet-coreml-fp16-baseline.env"
      unset PARAKEET_RNNT_BEAM_WIDTH PARAKEET_RNNT_DURATION_BEAM_WIDTH || true
      ;;
    odmbp-approx)
      # shellcheck source=/dev/null
      source "${REPO_ROOT}/configs/parakeet-coreml-v7-odmbp-approx.env"
      unset PARAKEET_RNNT_BEAM_WIDTH PARAKEET_RNNT_DURATION_BEAM_WIDTH || true
      ;;
    odmbp-approx-beam2)
      # shellcheck source=/dev/null
      source "${REPO_ROOT}/configs/parakeet-coreml-v7-odmbp-approx.env"
      # shellcheck source=/dev/null
      source "${REPO_ROOT}/configs/parakeet-coreml-decoder-beam2.env"
      ;;
    *)
      echo "error: unknown profile '${profile}'"
      exit 1
      ;;
  esac
}

trim() {
  local x="$1"
  # Trim leading/trailing whitespace.
  x="${x#"${x%%[![:space:]]*}"}"
  x="${x%"${x##*[![:space:]]}"}"
  printf "%s" "${x}"
}

IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV}"
IFS=',' read -r -a PROFILES <<< "${PROFILES_CSV}"

fail_count=0
mode_tag="noseg"
if [[ "${ENABLE_LONGFORM_SEGMENTED}" == "1" ]]; then
  mode_tag="seg"
fi

for raw_dataset in "${DATASETS[@]}"; do
  dataset="$(trim "${raw_dataset}")"
  if [[ -z "${dataset}" ]]; then
    continue
  fi
  for raw_profile in "${PROFILES[@]}"; do
    profile="$(trim "${raw_profile}")"
    if [[ -z "${profile}" ]]; then
      continue
    fi

    run_name="${RUN_PREFIX}-${dataset}-${profile}-${mode_tag}"
    echo ""
    echo "==> dataset=${dataset} profile=${profile} run=${run_name}"

    status="ok"
    (
      cd "${REPO_ROOT}"
      load_profile "${profile}"
      if [[ "${ENABLE_LONGFORM_SEGMENTED}" == "1" ]]; then
        # shellcheck source=/dev/null
        source "${REPO_ROOT}/configs/parakeet-coreml-longform-segmented.env"
      fi

      cmd=(
        bash scripts/run_openbench_eval.sh
        --dataset "${dataset}"
        --pipeline-kind transcription
        --python-transcriber "${PYTHON_TRANSCRIBER}"
        --metrics wer
        --run-name "${run_name}"
      )
      if [[ -n "${NUM_SAMPLES}" ]]; then
        cmd+=(--num-samples "${NUM_SAMPLES}")
      fi
      "${cmd[@]}"
    ) || status="failed"

    if [[ "${status}" != "ok" ]]; then
      fail_count=$((fail_count + 1))
    fi
    echo -e "${dataset}\t${profile}\t${run_name}\t${status}" >> "${MANIFEST_PATH}"
  done
done

python3 "${REPO_ROOT}/scripts/build_openbench_matrix_report.py" \
  --manifest "${MANIFEST_PATH}" \
  --runs-root "${RUNS_ROOT}" \
  --output-json "${REPORT_JSON}" \
  --output-csv "${REPORT_CSV}"

echo ""
echo "Matrix manifest: ${MANIFEST_PATH}"
echo "Matrix summary:  ${REPORT_JSON}"
echo "Matrix csv:      ${REPORT_CSV}"

if [[ "${fail_count}" -gt 0 ]]; then
  echo "warning: ${fail_count} run(s) failed; see manifest and report for details."
  exit 2
fi
