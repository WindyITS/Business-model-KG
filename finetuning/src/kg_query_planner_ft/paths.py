from __future__ import annotations

from pathlib import Path

from .config import FineTuningConfig, repo_root


def _safe_path_component(value: str) -> str:
    normalized = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in value.strip()
    ).strip("._")
    return normalized or "model"


def _resolve_repo_path(path: str) -> Path:
    configured = Path(path).expanduser()
    if configured.is_absolute():
        return configured.resolve()
    return (repo_root() / configured).resolve()


def dataset_root(config: FineTuningConfig) -> Path:
    return _resolve_repo_path(config.dataset_path)


def artifact_root(config: FineTuningConfig) -> Path:
    return _resolve_repo_path(config.artifact_root)


def prepared_router_dir(config: FineTuningConfig) -> Path:
    return artifact_root(config) / "prepared" / "router"


def planner_train_augmentation_file(config: FineTuningConfig) -> Path:
    return dataset_root(config) / "planner_only_open_literal_copying_augmentation.jsonl"


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


def planner_checkpoint_root_dir(config: FineTuningConfig) -> Path:
    return planner_adapter_dir(config) / "checkpoints"


def planner_eval_dir(
    config: FineTuningConfig,
    *,
    base_only: bool = False,
    backend: str = "mlx",
    model_name: str | None = None,
) -> Path:
    root = artifact_root(config) / "planner" / "eval"
    if backend == "mlx":
        if base_only:
            return root / "base_model"
        return root
    if backend == "lmstudio":
        return root / "lmstudio" / _safe_path_component(model_name or "local-model")
    raise ValueError(f"Unsupported planner eval backend: {backend}")
