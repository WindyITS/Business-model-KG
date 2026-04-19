from __future__ import annotations

import json
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
    refuse_precision_min: float = 0.95


class PlannerConfig(BaseModel):
    base_model: str
    data_variant: str = "balanced"
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    num_layers: int = 16
    batch_size: int = 4
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


class FineTuningConfig(BaseModel):
    artifact_root: str
    dataset_path: str
    router: RouterConfig = Field(default_factory=RouterConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def finetuning_root() -> Path:
    return repo_root() / "finetuning"


def default_config_path() -> Path:
    return finetuning_root() / "config" / "default.json"


def load_config(path: str | Path | None = None) -> FineTuningConfig:
    config_path = Path(path) if path is not None else default_config_path()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return FineTuningConfig.model_validate(payload)
