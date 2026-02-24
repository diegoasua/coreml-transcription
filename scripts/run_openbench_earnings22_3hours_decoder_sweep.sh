#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/artifacts/openbench-runs}"

# Base profile picks model artifacts (fp16, odmbp-approx, etc.).
MODEL_PROFILE="${MODEL_PROFILE:-configs/parakeet-coreml-fp16-baseline.env}"
DATASET="${DATASET:-earnings22-3hours}"
RUN_PREFIX="${RUN_PREFIX:-parakeet-coreml-${DATASET}-decoder-sweep}"
PYTHON_TRANSCRIBER="${PYTHON_TRANSCRIBER:-scripts/parakeet_coreml_rnnt_transcriber.py}"

# Options:
# - baseline: current decoder settings from model profile.
# - conservative: insertion-guard settings.
# - conservative-beam2: conservative + beam search width 2.
DECODER_VARIANTS_CSV="${DECODER_VARIANTS_CSV:-baseline,conservative,conservative-beam2}"

mkdir -p "${RUNS_ROOT}"
MANIFEST="${RUNS_ROOT}/${RUN_PREFIX}-manifest.tsv"
echo -e "variant\trun_name\tstatus" > "${MANIFEST}"

trim() {
  local x="$1"
  x="${x#"${x%%[![:space:]]*}"}"
  x="${x%"${x##*[![:space:]]}"}"
  printf "%s" "${x}"
}

run_variant() {
  local variant="$1"
  local run_name="${RUN_PREFIX}-${variant}"

  echo ""
  echo "==> dataset=${DATASET} variant=${variant} run=${run_name}"

  (
    cd "${REPO_ROOT}"
    # shellcheck source=/dev/null
    source "${MODEL_PROFILE}"

    # Reset decoder knobs between variants.
    unset PARAKEET_RNNT_MAX_SYMBOLS_PER_STEP PARAKEET_RNNT_MAX_TOKENS_PER_CHUNK || true
    unset PARAKEET_RNNT_BEAM_WIDTH PARAKEET_RNNT_DURATION_BEAM_WIDTH || true

    case "${variant}" in
      baseline)
        ;;
      conservative)
        # shellcheck source=/dev/null
        source "${REPO_ROOT}/configs/parakeet-coreml-decoder-conservative.env"
        ;;
      conservative-beam2)
        # shellcheck source=/dev/null
        source "${REPO_ROOT}/configs/parakeet-coreml-decoder-conservative.env"
        # shellcheck source=/dev/null
        source "${REPO_ROOT}/configs/parakeet-coreml-decoder-beam2.env"
        ;;
      *)
        echo "error: unknown decoder variant '${variant}'"
        exit 1
        ;;
    esac

    bash scripts/run_openbench_eval.sh \
      --dataset "${DATASET}" \
      --pipeline-kind transcription \
      --python-transcriber "${PYTHON_TRANSCRIBER}" \
      --metrics wer \
      --run-name "${run_name}"
  )
}

IFS=',' read -r -a VARIANTS <<< "${DECODER_VARIANTS_CSV}"
fail_count=0

for raw in "${VARIANTS[@]}"; do
  variant="$(trim "${raw}")"
  [[ -z "${variant}" ]] && continue
  if run_variant "${variant}"; then
    status="ok"
  else
    status="failed"
    fail_count=$((fail_count + 1))
  fi
  echo -e "${variant}\t${RUN_PREFIX}-${variant}\t${status}" >> "${MANIFEST}"
done

python3 - "${MANIFEST}" "${RUNS_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
runs_root = Path(sys.argv[2])

rows = []
with manifest.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rows.append(row)

print("")
print("Summary:")
print("variant, status, wer, speed_x, run_dir")
for row in rows:
    variant = row["variant"]
    run_name = row["run_name"]
    status = row["status"]
    summary_path = runs_root / run_name / "custom-openbench-summary.json"
    wer = ""
    speed = ""
    run_dir = str((runs_root / run_name).resolve())
    if status == "ok" and summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        wer = payload.get("global_metrics", {}).get("wer", {}).get("global_result", "")
        speed = payload.get("speed_x", "")
    print(f"{variant}, {status}, {wer}, {speed}, {run_dir}")
PY

echo ""
echo "Manifest: ${MANIFEST}"
if [[ "${fail_count}" -gt 0 ]]; then
  echo "warning: ${fail_count} run(s) failed."
  exit 2
fi
