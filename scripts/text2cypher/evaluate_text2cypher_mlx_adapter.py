#!/usr/bin/env python3
from __future__ import annotations

from _bootstrap import ensure_text2cypher_package

ensure_text2cypher_package()

from text2cypher.cli.evaluate_text2cypher_mlx_adapter import main


if __name__ == "__main__":
    raise SystemExit(main())
