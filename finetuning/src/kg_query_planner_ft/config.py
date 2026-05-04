from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path

from pydantic import BaseModel, Field


class RouterConfig(BaseModel):
    base_model: str
    max_length: int = 256
    train_batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 6
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    early_stopping_patience: int = 2
    seed: int = 7
    local_precision_min: float = 0.97


class PlannerConfig(BaseModel):
    base_model: str
    data_variant: str = "balanced"
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    num_layers: int = 16
    batch_size: int = 4
    grad_checkpoint: bool = True
    grad_accumulation_steps: int = 4
    epochs: int = 3
    learning_rate: float = 1e-4
    max_seq_length: int = 4096
    max_tokens: int = 256
    seed: int = 7
    mask_prompt: bool = True
    steps_per_report: int = 10
    checkpoint_every: int = 500
    resume_adapter_file: str | None = None
    resume_checkpoint_dir: str | None = None


class FineTuningConfig(BaseModel):
    artifact_root: str
    dataset_path: str
    router: RouterConfig = Field(default_factory=RouterConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)


def repo_root() -> Path:
    override = os.environ.get("KG_QUERY_PLANNER_FT_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    source_root = Path(__file__).resolve().parents[3]
    if (source_root / "finetuning" / "config" / "default.json").is_file():
        return source_root

    return Path.cwd().resolve()


def finetuning_root() -> Path:
    return repo_root() / "finetuning"


def default_config_path() -> Path:
    source_config = finetuning_root() / "config" / "default.json"
    if source_config.is_file():
        return source_config
    return Path(str(resources.files("kg_query_planner_ft").joinpath("package_config/default.json")))


def load_config(path: str | Path | None = None) -> FineTuningConfig:
    config_path = Path(path) if path is not None else default_config_path()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return FineTuningConfig.model_validate(payload)
