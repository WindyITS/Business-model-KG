#!/usr/bin/env bash
set -euo pipefail

FINETUNING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${FINETUNING_DIR}/.." && pwd)"
VENV_DIR="${FINETUNING_DIR}/.venv"
HF_CLI="${VENV_DIR}/bin/huggingface-cli"
DATASET_REPO="${KG_QUERY_PLANNER_DATASET_REPO:-WindyITS/business-model-kg-query-planner-data}"
DEST_DIR="${KG_QUERY_PLANNER_DATA_DIR:-${PROJECT_ROOT}/data/query_planner_curated}"

if [[ ! -x "${HF_CLI}" ]]; then
  echo "Expected Hugging Face CLI at ${HF_CLI}." >&2
  echo "Run bash finetuning/scripts/bootstrap_env.sh first." >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"
"${HF_CLI}" download "${DATASET_REPO}" \
  --repo-type dataset \
  --local-dir "${DEST_DIR}"

if [[ ! -f "${DEST_DIR}/v1_final/train.jsonl" ]]; then
  echo "Downloaded dataset, but ${DEST_DIR}/v1_final/train.jsonl was not found." >&2
  echo "Check the dataset layout or set KG_QUERY_PLANNER_DATA_DIR to the expected parent directory." >&2
  exit 1
fi

cat <<EOF
Downloaded query-planner data to:
  ${DEST_DIR}

Default fine-tuning config expects:
  ${DEST_DIR}/v1_final
EOF
