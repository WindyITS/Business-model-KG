from __future__ import annotations

from pathlib import Path

from .config import FineTuningConfig, repo_root


def expand_user(path: str) -> Path:
    return Path(path).expanduser().resolve()


def dataset_root(config: FineTuningConfig) -> Path:
    configured = Path(config.dataset_path)
    if configured.is_absolute():
        return configured
    return (repo_root() / configured).resolve()


def artifact_root(config: FineTuningConfig) -> Path:
    return expand_user(config.artifact_root)


def prepared_router_dir(config: FineTuningConfig) -> Path:
    return artifact_root(config) / "prepared" / "router"


def prepared_planner_raw_dir(config: FineTuningConfig) -> Path:
    return artifact_root(config) / "prepared" / "planner" / "raw"


def prepared_planner_balanced_dir(config: FineTuningConfig) -> Path:
    return artifact_root(config) / "prepared" / "planner" / "balanced"


def router_model_dir(config: FineTuningConfig) -> Path:
    return artifact_root(config) / "router" / "model"


def router_eval_dir(config: FineTuningConfig) -> Path:
    return artifact_root(config) / "router" / "eval"


def planner_adapter_dir(config: FineTuningConfig) -> Path:
    return artifact_root(config) / "planner" / "adapter"


def planner_eval_dir(config: FineTuningConfig) -> Path:
    return artifact_root(config) / "planner" / "eval"
