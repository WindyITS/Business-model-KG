from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _repo_src() -> Path:
    return Path(__file__).resolve().parents[3] / "src"


def _ensure_repo_src() -> None:
    src = str(_repo_src())
    if src not in sys.path:
        sys.path.insert(0, src)


def load_runtime_contract() -> tuple[Any, Any, Any]:
    _ensure_repo_src()
    from runtime.query_planner import QueryPlanEnvelope, compile_query_plan, validate_compiled_query

    return QueryPlanEnvelope, compile_query_plan, validate_compiled_query
