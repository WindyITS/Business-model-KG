#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

find . -path "./venv" -prune -o -path "./outputs" -prune -o -name ".DS_Store" -type f -delete
find src tests scripts -type d -name "__pycache__" -prune -exec rm -rf {} +

rm -rf .pytest_cache
rm -rf build dist packaging
rm -rf src/business_model_kg.egg-info
rm -f latest_run.log
rmdir configs 2>/dev/null || true

echo "Removed local caches and generated artifacts."
