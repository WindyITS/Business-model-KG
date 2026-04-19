#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${HOME}/projects/.ML-environments/kg-query-planner-ft"

python3 -m venv "${ENV_ROOT}"
"${ENV_ROOT}/bin/python" -m pip install --upgrade pip setuptools wheel
"${ENV_ROOT}/bin/python" -m pip install -e "${ROOT_DIR}[dev]"

cat <<EOF
Bootstrapped fine-tuning environment at:
  ${ENV_ROOT}

Activate with:
  source "${ENV_ROOT}/bin/activate"
EOF
