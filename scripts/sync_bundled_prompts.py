from __future__ import annotations

import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROMPTS_DIR = REPO_ROOT / "prompts"
TARGET_PROMPTS_DIR = REPO_ROOT / "src" / "llm_extraction" / "_bundled_prompts"


def main() -> int:
    if not SOURCE_PROMPTS_DIR.is_dir():
        raise FileNotFoundError(f"Source prompts directory not found: {SOURCE_PROMPTS_DIR}")

    if TARGET_PROMPTS_DIR.exists():
        shutil.rmtree(TARGET_PROMPTS_DIR)
    shutil.copytree(SOURCE_PROMPTS_DIR, TARGET_PROMPTS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
