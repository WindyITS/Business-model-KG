#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_hf_text2cypher_dataset import main


if __name__ == "__main__":
    argv = [
        "--dataset-root",
        "datasets/text2cypher/v3",
        "--packaging-root",
        "packaging/huggingface/text2cypher-v3",
        "--output-root",
        "dist/huggingface/text2cypher-v3",
    ]
    argv.extend(sys.argv[1:])
    raise SystemExit(
        main(argv)
    )
