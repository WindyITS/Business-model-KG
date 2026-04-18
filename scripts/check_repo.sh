#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${KG_PYTHON:-${ROOT_DIR}/venv/bin/python}"
SMOKE_DIR=""

cleanup() {
  if [[ -n "${SMOKE_DIR}" && -d "${SMOKE_DIR}" ]]; then
    rm -rf "${SMOKE_DIR}"
  fi
}
trap cleanup EXIT

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Expected a Python interpreter at ${PYTHON_BIN}." >&2
  echo "Run ./scripts/bootstrap_dev.sh first, or set KG_PYTHON to another interpreter." >&2
  exit 1
fi

echo "[1/6] Running repo health check"
bash "${ROOT_DIR}/scripts/kg-health-check" --skip-neo4j

echo "[2/6] Running test suite"
if ! "${PYTHON_BIN}" -c "import pytest" >/dev/null 2>&1; then
  echo "pytest is not installed in ${PYTHON_BIN}." >&2
  echo "Run ./scripts/bootstrap_dev.sh, or install the dev extras with pip install -e .[dev]." >&2
  exit 1
fi
"${PYTHON_BIN}" -m pytest -q

echo "[3/6] Checking Python compilation"
"${PYTHON_BIN}" -m compileall -q src tests

echo "[4/6] Exercising source-checkout wrappers"
for command in \
  kg-pipeline \
  kg-query \
  kg-query-cypher \
  kg-neo4j-load \
  kg-neo4j-status \
  kg-neo4j-unload \
  kg-evaluate-graph \
  kg-health-check; do
  bash "${ROOT_DIR}/scripts/${command}" --help >/dev/null
done

echo "[5/6] Running package smoke install"
SMOKE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kg-v0-check.XXXXXX")"
"${PYTHON_BIN}" -m pip install . --no-build-isolation --no-deps --target "${SMOKE_DIR}" >/dev/null

ENTRY_POINTS="$(find "${SMOKE_DIR}" -maxdepth 2 -name entry_points.txt | head -n 1)"
if [[ -z "${ENTRY_POINTS}" ]]; then
  echo "Could not find entry_points.txt in package smoke install." >&2
  exit 1
fi

grep -q '^kg-health-check = ' "${ENTRY_POINTS}"
grep -q '^kg-neo4j-load = ' "${ENTRY_POINTS}"
grep -q '^kg-neo4j-status = ' "${ENTRY_POINTS}"
grep -q '^kg-neo4j-unload = ' "${ENTRY_POINTS}"

test -f "${SMOKE_DIR}/ontology/ontology.json"
test -f "${SMOKE_DIR}/llm_extraction/_bundled_prompts/canonical/system.txt"
test -f "${SMOKE_DIR}/llm_extraction/_bundled_prompts/analyst/system.txt"

echo "[6/6] Import-checking packaged health check"
PYTHONPATH="${SMOKE_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m runtime.health_check --skip-neo4j --project-root "${ROOT_DIR}" >/dev/null

echo "All repo checks passed."
