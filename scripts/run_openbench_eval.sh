#!/usr/bin/env bash
set -euo pipefail

OPENBENCH_DIR="${OPENBENCH_DIR:-external/OpenBench}"
OPENBENCH_PYTHON="${OPENBENCH_PYTHON:-}"
OPENBENCH_REBUILD_TEXTERRORS="${OPENBENCH_REBUILD_TEXTERRORS:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.cache/uv}"
PARAKEET_COREML_MODEL_DIR="${PARAKEET_COREML_MODEL_DIR:-${REPO_ROOT}/artifacts/parakeet-tdt-0.6b-v2}"
PARAKEET_TMPDIR="${PARAKEET_TMPDIR:-${PARAKEET_COREML_MODEL_DIR}/.tmp}"

if [[ ! -d "${OPENBENCH_DIR}" ]]; then
  echo "error: OpenBench directory not found at ${OPENBENCH_DIR}"
  echo "clone it first: git clone https://github.com/argmaxinc/OpenBench ${OPENBENCH_DIR}"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required. Install from https://docs.astral.sh/uv/"
  exit 1
fi

mkdir -p "${UV_CACHE_DIR}"
export UV_CACHE_DIR
mkdir -p "${PARAKEET_TMPDIR}"
export PARAKEET_COREML_MODEL_DIR
export TMPDIR="${PARAKEET_TMPDIR}"

# Avoid uv warning/noise when caller has another venv activated.
unset VIRTUAL_ENV || true

if [[ -z "${OPENBENCH_PYTHON}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    OPENBENCH_PYTHON="python3.11"
  elif command -v python3.12 >/dev/null 2>&1; then
    OPENBENCH_PYTHON="python3.12"
  elif command -v python3.10 >/dev/null 2>&1; then
    OPENBENCH_PYTHON="python3.10"
  else
    OPENBENCH_PYTHON="python3"
  fi
fi

if ! command -v "${OPENBENCH_PYTHON}" >/dev/null 2>&1; then
  echo "error: '${OPENBENCH_PYTHON}' not found."
  echo "set OPENBENCH_PYTHON=python3.11 (or another installed interpreter)"
  exit 1
fi

pushd "${OPENBENCH_DIR}" >/dev/null
echo "Using OpenBench Python: ${OPENBENCH_PYTHON}"
if [[ "${OPENBENCH_REBUILD_TEXTERRORS}" == "1" ]]; then
  uv sync --python "${OPENBENCH_PYTHON}" --reinstall-package texterrors --no-binary-package texterrors
else
  uv sync --python "${OPENBENCH_PYTHON}"
fi
uv run --python "${OPENBENCH_PYTHON}" python ../../scripts/run_openbench_custom_transcription.py --openbench-dir . "$@"
popd >/dev/null
