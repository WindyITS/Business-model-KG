from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from text2cypher.mlx import (
    DEFAULT_ADAPTER_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRAD_ACCUMULATION_STEPS,
    DEFAULT_MODEL_ID,
    DEFAULT_NUM_LAYERS,
    DEFAULT_PREPARED_DATA_ROOT,
    DEFAULT_TEST_MESSAGES_PATH,
    DEFAULT_TRAIN_ITERS,
    DEFAULT_TRAIN_MESSAGES_PATH,
    build_mlx_lora_command,
    build_mlx_test_command,
    prepare_mlx_chat_dataset,
    run_command,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the text2cypher v3 dataset and launch an MLX LoRA fine-tuning run."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID or local MLX model path.",
    )
    parser.add_argument(
        "--train-messages-path",
        type=Path,
        default=DEFAULT_TRAIN_MESSAGES_PATH,
        help="Path to the train_messages.jsonl artifact used for training.",
    )
    parser.add_argument(
        "--test-messages-path",
        type=Path,
        default=DEFAULT_TEST_MESSAGES_PATH,
        help="Path to the held-out test_messages.jsonl artifact used for test-loss evaluation.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_PREPARED_DATA_ROOT,
        help="Directory where MLX-ready train.jsonl and test.jsonl live.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=DEFAULT_ADAPTER_PATH,
        help="Directory where MLX should save the LoRA adapters.",
    )
    parser.add_argument("--iters", type=int, default=DEFAULT_TRAIN_ITERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=DEFAULT_GRAD_ACCUMULATION_STEPS,
    )
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument(
        "--fine-tune-type",
        type=str,
        choices=("lora", "dora", "full"),
        default="lora",
        help="MLX fine-tuning mode.",
    )
    parser.add_argument(
        "--resume-adapter-file",
        type=Path,
        default=None,
        help="Existing adapters.safetensors file to resume from.",
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default=None,
        help="Optional MLX tracker backend, for example wandb.",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=None,
        help="Optional experiment project name passed through to MLX.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python interpreter to use when invoking mlx_lm.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip dataset preparation and assume data_root already contains train.jsonl and test.jsonl.",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Overwrite an existing prepared dataset directory when re-preparing data.",
    )
    parser.add_argument(
        "--no-mask-prompt",
        action="store_true",
        help="Disable MLX prompt masking and train on all tokens.",
    )
    parser.add_argument(
        "--no-grad-checkpoint",
        action="store_true",
        help="Disable MLX gradient checkpointing.",
    )
    parser.add_argument(
        "--run-test-loss",
        action="store_true",
        help="After training, run `mlx_lm.lora --test` on the held-out test split.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved commands and manifests without executing mlx_lm.",
    )
    parser.add_argument(
        "--extra-mlx-arg",
        action="append",
        default=[],
        help="Additional raw arguments appended to the mlx_lm.lora command.",
    )
    return parser.parse_args(argv)


def _write_request(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    adapter_path = args.adapter_path.resolve()

    prepared_manifest = None
    if not args.skip_prepare:
        prepared_manifest = prepare_mlx_chat_dataset(
            train_messages_path=args.train_messages_path.resolve(),
            test_messages_path=args.test_messages_path.resolve(),
            output_root=data_root,
            force=args.force_prepare,
        )

    train_command = build_mlx_lora_command(
        model=args.model,
        data_dir=data_root,
        adapter_path=adapter_path,
        iters=args.iters,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accumulation_steps,
        num_layers=args.num_layers,
        fine_tune_type=args.fine_tune_type,
        mask_prompt=not args.no_mask_prompt,
        grad_checkpoint=not args.no_grad_checkpoint,
        resume_adapter_file=args.resume_adapter_file.resolve() if args.resume_adapter_file else None,
        report_to=args.report_to,
        project_name=args.project_name,
        python_bin=args.python_bin,
        extra_args=args.extra_mlx_arg,
    )

    test_command = None
    if args.run_test_loss:
        test_command = build_mlx_test_command(
            model=args.model,
            data_dir=data_root,
            adapter_path=adapter_path,
            python_bin=args.python_bin,
        )

    request = {
        "model": args.model,
        "data_root": str(data_root),
        "adapter_path": str(adapter_path),
        "train_command": train_command,
        "test_command": test_command,
        "fine_tune_type": args.fine_tune_type,
        "prepared_manifest": prepared_manifest,
    }
    _write_request(adapter_path / "train_request.json", request)

    print(json.dumps(request, indent=2))
    run_command(train_command, dry_run=args.dry_run)
    if test_command is not None:
        run_command(test_command, dry_run=args.dry_run)
    return 0
