#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/venv}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -e "${ROOT_DIR}[dev]" --no-build-isolation

cat <<EOF
Bootstrap complete.

Reliable source-checkout commands:
  ./scripts/kg-pipeline
  ./scripts/kg-query
  ./scripts/kg-query-cypher
  ./scripts/kg-neo4j-load
  ./scripts/kg-neo4j-status
  ./scripts/kg-neo4j-unload
  ./scripts/kg-health-check

Editable-install entry points are also refreshed under ${VENV_DIR}/bin/.
EOF
