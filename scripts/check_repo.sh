#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${KG_PYTHON:-${ROOT_DIR}/venv/bin/python}"
SMOKE_DIR=""
FT_SMOKE_DIR=""

cleanup() {
  if [[ -n "${SMOKE_DIR}" && -d "${SMOKE_DIR}" ]]; then
    rm -rf "${SMOKE_DIR}"
  fi
  if [[ -n "${FT_SMOKE_DIR}" && -d "${FT_SMOKE_DIR}" ]]; then
    rm -rf "${FT_SMOKE_DIR}"
  fi
}
trap cleanup EXIT

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Expected a Python interpreter at ${PYTHON_BIN}." >&2
  echo "Run ./scripts/bootstrap_dev.sh first, or set KG_PYTHON to another interpreter." >&2
  exit 1
fi

echo "[1/7] Running repo health check"
bash "${ROOT_DIR}/scripts/kg-health-check" --skip-neo4j

echo "[2/7] Running test suite"
if ! "${PYTHON_BIN}" -c "import pytest" >/dev/null 2>&1; then
  echo "pytest is not installed in ${PYTHON_BIN}." >&2
  echo "Run ./scripts/bootstrap_dev.sh, or install the dev extras with pip install -e .[dev]." >&2
  exit 1
fi
"${PYTHON_BIN}" -m pytest -q
"${PYTHON_BIN}" -m pytest -q finetuning/tests

echo "[3/7] Checking Python compilation"
"${PYTHON_BIN}" -m compileall -q src tests evaluation finetuning/src finetuning/tests

echo "[4/7] Exercising source-checkout wrappers"
for command in \
  kg-pipeline \
  kg-query \
  kg-query-cypher \
  kg-neo4j-load \
  kg-neo4j-status \
  kg-neo4j-unload \
  kg-health-check; do
  bash "${ROOT_DIR}/scripts/${command}" --help >/dev/null
done
"${PYTHON_BIN}" -m runtime.query_cypher --help >/dev/null

echo "[5/7] Running package smoke install"
# Avoid stale setuptools output from reintroducing removed modules into the smoke package.
rm -rf "${ROOT_DIR}/build" "${ROOT_DIR}/dist" "${ROOT_DIR}/packaging" "${ROOT_DIR}/src/business_model_kg.egg-info"
SMOKE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kg-v0-check.XXXXXX")"
"${PYTHON_BIN}" -m pip install . --no-build-isolation --no-deps --target "${SMOKE_DIR}" >/dev/null

ENTRY_POINTS="$(find "${SMOKE_DIR}" -maxdepth 2 -name entry_points.txt | head -n 1)"
if [[ -z "${ENTRY_POINTS}" ]]; then
  echo "Could not find entry_points.txt in package smoke install." >&2
  exit 1
fi
TOP_LEVEL="$(find "${SMOKE_DIR}" -maxdepth 2 -name top_level.txt | head -n 1)"
if [[ -z "${TOP_LEVEL}" ]]; then
  echo "Could not find top_level.txt in package smoke install." >&2
  exit 1
fi

grep -q '^kg-pipeline = runtime.main:main$' "${ENTRY_POINTS}"
grep -q '^kg-query = runtime.query:main_query$' "${ENTRY_POINTS}"
grep -q '^kg-health-check = ' "${ENTRY_POINTS}"
grep -q '^kg-neo4j-load = ' "${ENTRY_POINTS}"
grep -q '^kg-neo4j-status = ' "${ENTRY_POINTS}"
grep -q '^kg-neo4j-unload = ' "${ENTRY_POINTS}"
grep -q '^kg-query-cypher = runtime.query_cypher:main$' "${ENTRY_POINTS}"
if grep -q '^kg-query-jolly = ' "${ENTRY_POINTS}"; then
  echo "Removed entry point still packaged: kg-query-jolly" >&2
  exit 1
fi
if grep -q '^kg-query-cypher-jolly = ' "${ENTRY_POINTS}"; then
  echo "Removed entry point still packaged: kg-query-cypher-jolly" >&2
  exit 1
fi

for package_name in \
  graph \
  llm \
  llm_extraction \
  ontology \
  runtime \
  training; do
  grep -qx "${package_name}" "${TOP_LEVEL}"
done

for legacy_module in \
  entity_resolver \
  llm_extractor \
  main \
  model_provider \
  neo4j_loader \
  ontology_config \
  ontology_validator \
  place_hierarchy \
  query_cypher; do
  if grep -qx "${legacy_module}" "${TOP_LEVEL}"; then
    echo "Legacy top-level module still packaged: ${legacy_module}" >&2
    exit 1
  fi
  if [[ -f "${SMOKE_DIR}/${legacy_module}.py" ]]; then
    echo "Legacy top-level module file still installed: ${legacy_module}.py" >&2
    exit 1
  fi
done

test -f "${SMOKE_DIR}/ontology/ontology.json"
test -f "${SMOKE_DIR}/llm_extraction/_bundled_prompts/analyst/system.txt"
test -f "${SMOKE_DIR}/llm_extraction/_bundled_prompts/memo_graph_only/memo_foundation.txt"
test -f "${SMOKE_DIR}/llm_extraction/_bundled_prompts/zero-shot/extract.txt"

echo "[6/7] Import-checking packaged modules"
PYTHONPATH="${SMOKE_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m runtime.health_check --skip-neo4j --project-root "${ROOT_DIR}" >/dev/null

echo "[7/7] Running fine-tuning package smoke install"
rm -rf "${ROOT_DIR}/finetuning/build" "${ROOT_DIR}/finetuning/dist" "${ROOT_DIR}/finetuning/src/kg_query_planner_ft.egg-info"
FT_SMOKE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kg-v0-ft-check.XXXXXX")"
"${PYTHON_BIN}" -m pip install "${ROOT_DIR}/finetuning" --no-build-isolation --no-deps --target "${FT_SMOKE_DIR}" >/dev/null

FT_ENTRY_POINTS="$(find "${FT_SMOKE_DIR}" -maxdepth 2 -name entry_points.txt | head -n 1)"
if [[ -z "${FT_ENTRY_POINTS}" ]]; then
  echo "Could not find fine-tuning entry_points.txt in package smoke install." >&2
  exit 1
fi

grep -q '^prepare-data = kg_query_planner_ft.prepare_data:main$' "${FT_ENTRY_POINTS}"
grep -q '^train-router = kg_query_planner_ft.router_train:main$' "${FT_ENTRY_POINTS}"
grep -q '^eval-router = kg_query_planner_ft.router_eval:main$' "${FT_ENTRY_POINTS}"
grep -q '^train-planner = kg_query_planner_ft.planner_train:main$' "${FT_ENTRY_POINTS}"
grep -q '^eval-planner = kg_query_planner_ft.planner_eval:main$' "${FT_ENTRY_POINTS}"
grep -q '^publish-query-stack = kg_query_planner_ft.publish_query_stack:main$' "${FT_ENTRY_POINTS}"

test -f "${FT_SMOKE_DIR}/kg_query_planner_ft/package_config/default.json"
KG_QUERY_PLANNER_FT_ROOT="${FT_SMOKE_DIR}/workspace" \
  PYTHONPATH="${FT_SMOKE_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -c "from kg_query_planner_ft.config import default_config_path, load_config; path = default_config_path(); assert 'package_config' in str(path), path; load_config()" >/dev/null

echo "All repo checks passed."
