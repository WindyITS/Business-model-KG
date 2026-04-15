#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from text2cypher_dataset_v2.builder import main


if __name__ == "__main__":
    argv = [
        "--output-root",
        "datasets/text2cypher/v3",
        "--spec-module",
        "text2cypher_dataset_v2.spec_core",
        "--spec-module",
        "text2cypher_dataset_v2.spec_rollups",
        "--spec-module",
        "text2cypher_dataset_v2.spec_negative",
        "--spec-module",
        "text2cypher_dataset_v2.spec_hard_train",
        "--spec-module",
        "text2cypher_dataset_v2.spec_heldout_test",
    ]
    argv.extend(sys.argv[1:])
    raise SystemExit(
        main(argv)
    )
