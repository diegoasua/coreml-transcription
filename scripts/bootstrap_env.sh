#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
  elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
  elif command -v python3.13 >/dev/null 2>&1; then
    PYTHON_BIN="python3.13"
  else
    PYTHON_BIN="python3"
  fi
fi
VENV_DIR="${1:-.venv}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: '${PYTHON_BIN}' not found. Install python3 or set PYTHON_BIN."
  exit 1
fi

echo "[1/3] Creating virtual environment at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

echo "[2/3] Upgrading pip"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip

echo "[3/3] Installing requirements-conversion.txt"
"${VENV_DIR}/bin/pip" install -r requirements-conversion.txt

echo "Environment ready."
echo "Activate with: source ${VENV_DIR}/bin/activate"
