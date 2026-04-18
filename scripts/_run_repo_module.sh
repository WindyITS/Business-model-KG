#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <python-module> [args...]" >&2
  exit 64
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${KG_PYTHON:-${ROOT_DIR}/venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Expected a Python interpreter at ${PYTHON_BIN}." >&2
  echo "Run ./scripts/bootstrap_dev.sh first, or set KG_PYTHON to another interpreter." >&2
  exit 1
fi

MODULE="$1"
shift

PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" exec "${PYTHON_BIN}" -m "${MODULE}" "$@"
