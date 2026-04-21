from __future__ import annotations

import os
from pathlib import Path


QUERY_LOCAL_STACK_PYTHON_ENV = "KG_QUERY_LOCAL_STACK_PYTHON"
QUERY_LOCAL_STACK_CONFIG_ENV = "KG_QUERY_LOCAL_STACK_CONFIG"
LOCAL_STACK_MODULE = "kg_query_planner_ft.local_stack"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def local_stack_src(root_dir: Path | None = None) -> Path:
    resolved_root = root_dir or repo_root()
    return resolved_root / "finetuning" / "src"


def default_local_stack_python(root_dir: Path | None = None) -> Path:
    resolved_root = root_dir or repo_root()
    return resolved_root / "finetuning" / ".venv" / "bin" / "python"


def default_local_stack_config(root_dir: Path | None = None) -> Path:
    resolved_root = root_dir or repo_root()
    return resolved_root / "finetuning" / "config" / "default.json"


def resolve_local_stack_python(explicit_path: str | None = None, *, root_dir: Path | None = None) -> str:
    if explicit_path:
        return explicit_path

    env_path = os.getenv(QUERY_LOCAL_STACK_PYTHON_ENV)
    if env_path:
        return env_path

    return str(default_local_stack_python(root_dir))


def resolve_local_stack_config(explicit_path: str | None = None, *, root_dir: Path | None = None) -> str | None:
    if explicit_path:
        return explicit_path

    env_path = os.getenv(QUERY_LOCAL_STACK_CONFIG_ENV)
    if env_path:
        return env_path

    default_path = default_local_stack_config(root_dir)
    return str(default_path) if default_path.exists() else None
