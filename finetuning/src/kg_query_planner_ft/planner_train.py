from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path

from .config import load_config
from .json_utils import compact_json, read_jsonl, write_json
from .paths import planner_adapter_dir, prepared_planner_balanced_dir, prepared_planner_raw_dir
from .progress import StepProgress, progress_write


_MLX_PROGRESS_PATTERNS = (
    re.compile(r"\b(?:iter|step)\s+(\d+)\s*/\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\b(?:iter|step)\s+(\d+)\b", re.IGNORECASE),
)


def _yaml_dump(config: dict[str, object]) -> str:
    lines: list[str] = []
    for key, value in config.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for inner_key, inner_value in value.items():
                if isinstance(inner_value, str):
                    lines.append(f"  {inner_key}: {inner_value!r}")
                else:
                    lines.append(f"  {inner_key}: {inner_value}")
            continue
        if isinstance(value, str):
            lines.append(f"{key}: {value!r}")
        elif isinstance(value, bool):
            lines.append(f"{key}: {'true' if value else 'false'}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"


def _maybe_extract_iteration(line: str) -> tuple[int | None, int | None]:
    for pattern in _MLX_PROGRESS_PATTERNS:
        match = pattern.search(line)
        if not match:
            continue
        current = int(match.group(1))
        total = int(match.group(2)) if match.lastindex and match.lastindex >= 2 else None
        return current, total
    return None, None


def _run_mlx_training(command: list[str], *, total_iters: int) -> None:
    from .progress import tqdm

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    current_iter = 0
    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(total=total_iters, desc="planner training", unit="iter", dynamic_ncols=True)
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        parsed_current, parsed_total = _maybe_extract_iteration(line)
        if parsed_total is not None and progress_bar is not None and parsed_total > 0 and progress_bar.total != parsed_total:
            progress_bar.total = parsed_total
            progress_bar.refresh()
        if parsed_current is not None:
            target_iter = min(parsed_current, progress_bar.total if progress_bar is not None and progress_bar.total else total_iters)
            if progress_bar is not None and target_iter > current_iter:
                progress_bar.update(target_iter - current_iter)
            current_iter = max(current_iter, target_iter)
        progress_write(line)
    return_code = process.wait()
    if progress_bar is not None:
        remaining = int(progress_bar.total or total_iters) - current_iter
        if remaining > 0:
            progress_bar.update(remaining)
        progress_bar.close()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def train_planner(config_path: str | None = None) -> dict[str, object]:
    with StepProgress(total=5, desc="train-planner") as progress:
        config = load_config(config_path)
        data_dir = (
            prepared_planner_balanced_dir(config)
            if config.planner.data_variant == "balanced"
            else prepared_planner_raw_dir(config)
        )
        train_rows = read_jsonl(data_dir / "train.jsonl")
        adapter_dir = planner_adapter_dir(config)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        progress.advance("loaded planner training split")

        steps_per_epoch = math.ceil(len(train_rows) / config.planner.batch_size)
        total_iters = steps_per_epoch * config.planner.epochs
        steps_per_eval = max(steps_per_epoch, 1)
        save_every = max(steps_per_epoch, 1)
        progress.advance("computed planner schedule")

        mlx_config = {
            "model": config.planner.base_model,
            "train": True,
            "test": False,
            "data": str(data_dir),
            "seed": config.planner.seed,
            "num_layers": config.planner.num_layers,
            "batch_size": config.planner.batch_size,
            "iters": total_iters,
            "val_batches": -1,
            "learning_rate": config.planner.learning_rate,
            "steps_per_report": config.planner.steps_per_report,
            "steps_per_eval": steps_per_eval,
            "adapter_path": str(adapter_dir),
            "save_every": save_every,
            "max_seq_length": config.planner.max_seq_length,
            "grad_accumulation_steps": config.planner.grad_accumulation_steps,
            "mask_prompt": config.planner.mask_prompt,
            "lora_parameters": {
                "rank": config.planner.rank,
                "dropout": config.planner.dropout,
                "scale": config.planner.alpha,
            },
        }
        yaml_path = adapter_dir / "train_config.yaml"
        yaml_path.write_text(_yaml_dump(mlx_config), encoding="utf-8")
        progress.advance("wrote MLX training config")

        command = [sys.executable, "-m", "mlx_lm.lora", "--config", str(yaml_path)]
        progress.advance("starting planner fine-tune")
        _run_mlx_training(command, total_iters=total_iters)

        summary = {
            "data_dir": str(data_dir),
            "adapter_dir": str(adapter_dir),
            "train_examples": len(train_rows),
            "steps_per_epoch": steps_per_epoch,
            "total_iters": total_iters,
            "effective_batch_size": config.planner.batch_size * config.planner.grad_accumulation_steps,
            "config_path": str(yaml_path),
        }
        write_json(adapter_dir / "training_summary.json", summary)
        progress.advance("saved planner training summary")
        return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the local planner with MLX LoRA.")
    parser.add_argument("--config", type=str, default=None, help="Path to the fine-tuning JSON config.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = train_planner(args.config)
    print(compact_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
