from __future__ import annotations

import argparse
import json
from pathlib import Path

from text2cypher.mlx import (
    DEFAULT_PREPARED_DATA_ROOT,
    DEFAULT_TEST_MESSAGES_PATH,
    DEFAULT_TRAIN_MESSAGES_PATH,
    prepare_mlx_chat_dataset,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the text2cypher v3 SFT corpus as chat JSONL files for MLX LoRA fine-tuning."
    )
    parser.add_argument(
        "--train-messages-path",
        type=Path,
        default=DEFAULT_TRAIN_MESSAGES_PATH,
        help="Path to the train_messages.jsonl artifact.",
    )
    parser.add_argument(
        "--test-messages-path",
        type=Path,
        default=DEFAULT_TEST_MESSAGES_PATH,
        help="Path to the held-out test_messages.jsonl artifact.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_PREPARED_DATA_ROOT,
        help="Directory where MLX-ready train.jsonl and test.jsonl will be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing prepared dataset directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = prepare_mlx_chat_dataset(
        train_messages_path=args.train_messages_path.resolve(),
        test_messages_path=args.test_messages_path.resolve(),
        output_root=args.output_root.resolve(),
        force=args.force,
    )
    print(json.dumps(manifest, indent=2))
    return 0
