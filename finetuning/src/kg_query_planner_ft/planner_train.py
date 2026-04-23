from __future__ import annotations

import argparse
import math
import types
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Any, Iterator

from .cli_output import render_planner_training_summary
from .config import load_config
from .json_utils import compact_json, read_jsonl, write_json
from .paths import (
    planner_adapter_dir,
    planner_checkpoint_root_dir,
    prepared_planner_balanced_dir,
    prepared_planner_raw_dir,
)
from .progress import StepProgress, progress_write, track, tqdm


def _yaml_dump(config: dict[str, object]) -> str:
    lines: list[str] = []

    def _render_scalar(value: object) -> str:
        if value is None:
            return "null"
        if isinstance(value, str):
            return repr(value)
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    for key, value in config.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for inner_key, inner_value in value.items():
                lines.append(f"  {inner_key}: {_render_scalar(inner_value)}")
            continue
        lines.append(f"{key}: {_render_scalar(value)}")
    return "\n".join(lines) + "\n"


def _planner_length_preflight(
    rows: list[dict[str, Any]],
    *,
    model_id: str,
    max_seq_length: int,
) -> dict[str, int]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only in the training env
        raise RuntimeError(
            "Planner length preflight requires transformers in the fine-tuning environment."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    stats = {
        "count": 0,
        "full_max": 0,
        "prompt_max": 0,
        "completion_max": 0,
        "zero_target_rows": 0,
    }
    for row in track(
        rows,
        total=len(rows),
        desc="planner length preflight",
        unit="row",
    ):
        messages = row["messages"]
        full_tokens = tokenizer.apply_chat_template(messages)
        prompt_tokens = tokenizer.apply_chat_template(
            messages[:-1],
            add_generation_prompt=True,
        )
        completion_tokens = tokenizer.encode(
            messages[-1]["content"],
            add_special_tokens=False,
        )
        stats["count"] += 1
        stats["full_max"] = max(stats["full_max"], len(full_tokens))
        stats["prompt_max"] = max(stats["prompt_max"], len(prompt_tokens))
        stats["completion_max"] = max(stats["completion_max"], len(completion_tokens))
        if min(len(full_tokens), max_seq_length) <= len(prompt_tokens):
            stats["zero_target_rows"] += 1
    return stats


def _build_mlx_training_config(
    config: Any,
    *,
    data_dir: Path,
    adapter_dir: Path,
    total_iters: int,
    steps_per_eval: int,
    save_every: int,
) -> dict[str, object]:
    mlx_config: dict[str, object] = {
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
        "grad_checkpoint": config.planner.grad_checkpoint,
        "grad_accumulation_steps": config.planner.grad_accumulation_steps,
        "mask_prompt": config.planner.mask_prompt,
        "lora_parameters": {
            "rank": config.planner.rank,
            "dropout": config.planner.dropout,
            "scale": config.planner.alpha,
        },
    }
    if config.planner.resume_adapter_file is not None:
        mlx_config["resume_adapter_file"] = config.planner.resume_adapter_file
    if config.planner.resume_checkpoint_dir is not None:
        mlx_config["resume_checkpoint_dir"] = config.planner.resume_checkpoint_dir
    return mlx_config


def _training_args_namespace(mlx_config: dict[str, object]) -> types.SimpleNamespace:
    defaults = {
        "fine_tune_type": "lora",
        "optimizer": "adam",
        "optimizer_config": {"adam": {}},
        "lr_schedule": None,
        "report_to": None,
        "project_name": None,
        "resume_checkpoint_dir": None,
        "resume_adapter_file": None,
    }
    payload = {**defaults, **mlx_config}
    return types.SimpleNamespace(**payload)


def _save_array_tree(path: Path, tree: Any) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), dict(tree_flatten(tree)))


def _load_array_tree(path: Path) -> Any:
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    return tree_unflatten(list(mx.load(str(path)).items()))


def _save_resume_checkpoint(
    checkpoint_dir: Path,
    *,
    adapter_weights: dict[str, Any],
    optimizer_state: Any,
    rng_state: Any,
    trainer_state: dict[str, Any],
) -> None:
    import mlx.core as mx

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(checkpoint_dir / "adapters.safetensors"), adapter_weights)
    _save_array_tree(checkpoint_dir / "optimizer_state.safetensors", optimizer_state)
    _save_array_tree(checkpoint_dir / "rng_state.safetensors", {"state": rng_state})
    write_json(checkpoint_dir / "trainer_state.json", trainer_state)


def _load_resume_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if not trainer_state_path.exists():
        raise FileNotFoundError(f"Missing trainer state file: {trainer_state_path}")
    import json

    trainer_state_payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    rng_state_payload = _load_array_tree(checkpoint_dir / "rng_state.safetensors")
    return {
        "adapter_file": checkpoint_dir / "adapters.safetensors",
        "optimizer_state": _load_array_tree(checkpoint_dir / "optimizer_state.safetensors"),
        "rng_state": rng_state_payload["state"],
        "trainer_state": trainer_state_payload,
    }


def _iter_training_batches(
    dataset: Any,
    *,
    batch_size: int,
    max_seq_length: int,
    seed: int,
    start_iter: int,
    comm_group: Any,
) -> Iterator[tuple[Any, Any]]:
    from mlx_lm.tuner.trainer import iterate_batches

    steps_per_epoch = max(1, len(dataset) // batch_size)
    start_epoch = start_iter // steps_per_epoch
    offset_in_epoch = start_iter % steps_per_epoch
    epoch_index = start_epoch
    while True:
        epoch_batches = iterate_batches(
            dataset=dataset,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            loop=False,
            seed=seed + epoch_index,
            comm_group=comm_group,
        )
        if offset_in_epoch:
            epoch_batches = islice(epoch_batches, offset_in_epoch, None)
            offset_in_epoch = 0
        yield from epoch_batches
        epoch_index += 1


def _run_mlx_training_loop(
    mlx_config: dict[str, object],
    *,
    checkpoint_root: Path,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.nn.utils import average_gradients
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset
    from mlx_lm.tuner.trainer import default_loss, evaluate, grad_checkpoint as mlx_grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
    from mlx_lm.utils import load, save_config

    args = _training_args_namespace(mlx_config)
    if args.grad_accumulation_steps < 1:
        raise ValueError("grad_accumulation_steps must be at least 1.")

    if mx.metal.is_available():
        device_info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        mx.set_wired_limit(device_info["max_recommended_working_set_size"])
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        progress_write(f"Node {rank} of {world_size}")

    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})
    train_set, valid_set, _ = load_dataset(args, tokenizer)
    if len(train_set) < args.batch_size:
        raise ValueError(
            f"Planner training needs at least batch_size={args.batch_size} rows, got {len(train_set)}."
        )

    model.freeze()
    linear_to_lora_layers(model, args.num_layers, args.lora_parameters, use_dora=False)

    resume_state: dict[str, Any] | None = None
    if args.resume_checkpoint_dir:
        resume_state = _load_resume_checkpoint(Path(args.resume_checkpoint_dir))
        args.resume_adapter_file = str(resume_state["adapter_file"])
        progress_write(f"Loading full resume checkpoint from {args.resume_checkpoint_dir}")

    if args.resume_adapter_file is not None:
        progress_write(f"Loading fine-tuned weights from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)

    print_trainable_parameters(model)

    adapter_path = Path(args.adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    adapter_file = adapter_path / "adapters.safetensors"
    save_config(vars(args), adapter_path / "adapter_config.json")

    optimizer = optim.Adam(learning_rate=args.learning_rate)
    start_iter = 0
    trained_tokens = 0
    if resume_state is not None:
        optimizer.state = resume_state["optimizer_state"]
        optimizer.init(model.trainable_parameters())
        mx.random.state = resume_state["rng_state"]
        start_iter = int(resume_state["trainer_state"]["iteration"])
        trained_tokens = int(resume_state["trainer_state"].get("trained_tokens", 0))
        if start_iter >= int(args.iters):
            raise ValueError(
                f"Resume checkpoint iteration {start_iter} is already at or beyond target iters={args.iters}."
            )

    if args.grad_checkpoint:
        mlx_grad_checkpoint(model.layers[0])

    loss_value_and_grad = nn.value_and_grad(model, default_loss)
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, prev_grad, do_update):
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)

        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)

        if do_update:
            grad = average_gradients(grad)
            if args.grad_accumulation_steps > 1:
                grad = tree_map(lambda x: x / args.grad_accumulation_steps, grad)
            optimizer.update(model, grad)
            grad = None

        return lvalue, toks, grad

    model.train()
    losses = 0
    n_tokens = 0
    steps = 0
    train_time = 0.0
    grad_accum = None

    progress_bar = tqdm(total=args.iters, desc="planner training", unit="iter", dynamic_ncols=True) if tqdm is not None else None
    if progress_bar is not None and start_iter:
        progress_bar.update(start_iter)

    training_batches = _iter_training_batches(
        CacheDataset(train_set),
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        start_iter=start_iter,
        comm_group=world,
    )
    try:
        for it, batch in zip(range(start_iter + 1, int(args.iters) + 1), training_batches):
            if progress_bar is not None:
                progress_bar.update(1)
            tic = mx.array(0.0)  # placeholder to keep branch structure simple
            del tic

            import time

            wall_start = time.perf_counter()
            if it == 1 or it % int(args.steps_per_eval) == 0 or it == int(args.iters):
                val_start = time.perf_counter()
                val_loss = evaluate(
                    model=model,
                    dataset=CacheDataset(valid_set),
                    loss=default_loss,
                    batch_size=args.batch_size,
                    num_batches=args.val_batches,
                    max_seq_length=args.max_seq_length,
                )
                model.train()
                val_time = time.perf_counter() - val_start
                if rank == 0:
                    progress_write(
                        f"Iter {it}: Val loss {val_loss:.3f}, Val took {val_time:.3f}s"
                    )
                wall_start = time.perf_counter()

            lvalue, toks, grad_accum = step(
                batch,
                grad_accum,
                it % int(args.grad_accumulation_steps) == 0,
            )

            losses += lvalue
            n_tokens += toks
            steps += 1
            mx.eval(state, losses, n_tokens, grad_accum)
            train_time += time.perf_counter() - wall_start

            if it % int(args.steps_per_report) == 0 or it == int(args.iters):
                train_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
                train_loss /= steps * world_size
                report_tokens = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
                learning_rate = optimizer.learning_rate.item()
                it_sec = steps / train_time if train_time > 0 else 0.0
                tokens_sec = float(report_tokens) / train_time if train_time > 0 else 0.0
                trained_tokens += report_tokens
                peak_mem = mx.get_peak_memory() / 1e9
                if rank == 0:
                    progress_write(
                        f"Iter {it}: Train loss {train_loss:.3f}, "
                        f"Learning Rate {learning_rate:.3e}, "
                        f"It/sec {it_sec:.3f}, "
                        f"Tokens/sec {tokens_sec:.3f}, "
                        f"Trained Tokens {trained_tokens}, "
                        f"Peak mem {peak_mem:.3f} GB"
                    )
                losses = 0
                n_tokens = 0
                steps = 0
                train_time = 0.0

            if it % int(args.save_every) == 0 and rank == 0:
                adapter_weights = dict(tree_flatten(model.trainable_parameters()))
                mx.save_safetensors(str(adapter_file), adapter_weights)
                numbered_adapter_file = adapter_path / f"{it:07d}_adapters.safetensors"
                mx.save_safetensors(str(numbered_adapter_file), adapter_weights)
                checkpoint_dir = checkpoint_root / f"{it:07d}"
                _save_resume_checkpoint(
                    checkpoint_dir,
                    adapter_weights=adapter_weights,
                    optimizer_state=optimizer.state,
                    rng_state=mx.random.state,
                    trainer_state={
                        "iteration": it,
                        "trained_tokens": trained_tokens,
                    },
                )
                progress_write(
                    f"Iter {it}: Saved adapter weights to {adapter_file} and {numbered_adapter_file}."
                )
                progress_write(f"Iter {it}: Saved full resume checkpoint to {checkpoint_dir}.")
    finally:
        if progress_bar is not None:
            progress_bar.close()

    if rank == 0:
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(adapter_file), adapter_weights)
        final_checkpoint_dir = checkpoint_root / f"{int(args.iters):07d}"
        _save_resume_checkpoint(
            final_checkpoint_dir,
            adapter_weights=adapter_weights,
            optimizer_state=optimizer.state,
            rng_state=mx.random.state,
            trainer_state={
                "iteration": int(args.iters),
                "trained_tokens": trained_tokens,
            },
        )
        progress_write(f"Saved final weights to {adapter_file}.")
        progress_write(f"Saved final full resume checkpoint to {final_checkpoint_dir}.")

    return {
        "latest_resume_checkpoint_dir": str(checkpoint_root / f"{int(args.iters):07d}"),
    }


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
        checkpoint_root = planner_checkpoint_root_dir(config)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        progress.advance("loaded planner training split")

        preflight = _planner_length_preflight(
            train_rows,
            model_id=config.planner.base_model,
            max_seq_length=config.planner.max_seq_length,
        )
        progress_write(
            compact_json(
                {
                    "planner_length_preflight": preflight,
                    "max_seq_length": config.planner.max_seq_length,
                }
            )
        )
        if preflight["zero_target_rows"] > 0:
            raise ValueError(
                "Planner fine-tuning would truncate away all assistant target tokens for "
                f"{preflight['zero_target_rows']} training rows at max_seq_length="
                f"{config.planner.max_seq_length}. Increase max_seq_length or shorten the prompt."
            )

        steps_per_epoch = math.ceil(len(train_rows) / config.planner.batch_size)
        total_iters = steps_per_epoch * config.planner.epochs
        steps_per_eval = max(steps_per_epoch, 1)
        save_every = max(config.planner.checkpoint_every, 1)
        if save_every % config.planner.grad_accumulation_steps != 0:
            raise ValueError(
                "checkpoint_every must be divisible by grad_accumulation_steps so resume checkpoints "
                "land on optimizer-update boundaries."
            )
        progress.advance("computed planner schedule")

        mlx_config = _build_mlx_training_config(
            config,
            data_dir=data_dir,
            adapter_dir=adapter_dir,
            total_iters=total_iters,
            steps_per_eval=steps_per_eval,
            save_every=save_every,
        )
        yaml_path = adapter_dir / "train_config.yaml"
        yaml_path.write_text(_yaml_dump(mlx_config), encoding="utf-8")
        progress.advance("wrote MLX training config")

        progress.advance("starting planner fine-tune")
        runtime_summary = _run_mlx_training_loop(mlx_config, checkpoint_root=checkpoint_root)

        summary = {
            "data_dir": str(data_dir),
            "adapter_dir": str(adapter_dir),
            "checkpoint_root_dir": str(checkpoint_root),
            "train_examples": len(train_rows),
            "steps_per_epoch": steps_per_epoch,
            "total_iters": total_iters,
            "checkpoint_every": save_every,
            "resume_adapter_file": config.planner.resume_adapter_file,
            "resume_checkpoint_dir": config.planner.resume_checkpoint_dir,
            "grad_checkpoint": config.planner.grad_checkpoint,
            "effective_batch_size": config.planner.batch_size * config.planner.grad_accumulation_steps,
            "config_path": str(yaml_path),
            **runtime_summary,
        }
        write_json(adapter_dir / "training_summary.json", summary)
        progress.advance("saved planner training summary")
        return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the local planner with MLX QLoRA."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to the fine-tuning JSON config.")
    parser.add_argument("--json", action="store_true", help="Print the final summary as compact JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = train_planner(args.config)
    print(compact_json(summary) if args.json else render_planner_training_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
