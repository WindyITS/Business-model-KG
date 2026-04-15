from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
import sys

from text2cypher.mlx import (
    DEFAULT_ADAPTER_PATH,
    DEFAULT_ENABLE_THINKING,
    DEFAULT_EVAL_OUTPUT_ROOT,
    DEFAULT_FIXTURES_PATH,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_ID,
    DEFAULT_TEST_EXAMPLES_PATH,
    DEFAULT_TEST_MESSAGES_PATH,
    evaluate_execution_matches,
    load_jsonl,
    load_mlx_model_and_tokenizer,
    generate_output_with_loaded_model,
    score_prediction_payload,
    summarize_prediction_metrics,
    write_jsonl,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate held-out predictions with an MLX LoRA adapter and score them against text2cypher v3."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--eval-messages-path", type=Path, default=DEFAULT_TEST_MESSAGES_PATH)
    parser.add_argument("--eval-examples-path", type=Path, default=DEFAULT_TEST_EXAMPLES_PATH)
    parser.add_argument("--fixtures-path", type=Path, default=DEFAULT_FIXTURES_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_EVAL_OUTPUT_ROOT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=DEFAULT_ENABLE_THINKING,
        help="Pass enable_thinking=True into the model chat template during generation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing evaluation output directory.",
    )
    parser.add_argument("--neo4j-uri", type=str, default=None)
    parser.add_argument("--neo4j-user", type=str, default="neo4j")
    parser.add_argument("--neo4j-password", type=str, default="password")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Update the visible progress display every N rows.",
    )
    return parser.parse_args(argv)


def _group_summary(row_results: list[dict[str, object]], field: str) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for result in row_results:
        key = result["message_row"]["metadata"][field]
        grouped[key].append(result)

    grouped_rows = list(grouped.values())
    execution_rows = [rows for rows in grouped_rows if all(row.get("execution") is not None for row in rows)]

    def count_all(predicate_name: str) -> int:
        return sum(all(row["metrics"][predicate_name] for row in rows) for rows in grouped_rows)

    def count_any(predicate_name: str) -> int:
        return sum(any(row["metrics"][predicate_name] for row in rows) for rows in grouped_rows)

    summary = {
        "groups": len(grouped_rows),
        "groups_with_all_valid_json": count_all("valid_json"),
        "groups_with_all_structured_match": count_all("structured_match"),
        "groups_with_any_structured_match": count_any("structured_match"),
    }
    if execution_rows:
        summary["groups_with_all_execution_match"] = sum(
            all(bool(row["execution"]["matched"]) for row in rows) for rows in execution_rows
        )
        summary["execution_group_count"] = len(execution_rows)
    else:
        summary["groups_with_all_execution_match"] = 0
        summary["execution_group_count"] = 0
    return summary


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _prediction_export_row(result: dict[str, object]) -> dict[str, object]:
    message_row = result["message_row"]
    training_ids = message_row["training_example_ids"]
    return {
        "sft_example_id": message_row["sft_example_id"],
        "training_example_ids": training_ids,
        "metadata": message_row["metadata"],
        "prompt": message_row["prompt"],
        "gold_completion": message_row["completion"],
        "raw_output": result["raw_output"],
        "parsed_output": result["metrics"]["parsed_payload"],
        "extracted_json": result["metrics"]["extracted_json"],
        "metrics": {
            key: value
            for key, value in result["metrics"].items()
            if key not in {"parsed_payload", "extracted_json"}
        },
        "execution": result.get("execution"),
    }


def _format_seconds(seconds: float) -> str:
    rounded = max(0, int(seconds))
    minutes, secs = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ProgressPrinter:
    def __init__(self, *, total: int, label: str, stream: object = sys.stdout):
        self.total = total
        self.label = label
        self.stream = stream
        self.start_time = time.monotonic()
        self.last_render_length = 0
        self.is_tty = getattr(stream, "isatty", lambda: False)()

    def update(self, current: int, *, valid_json_rows: int, structured_match_rows: int) -> None:
        elapsed = max(0.001, time.monotonic() - self.start_time)
        rate = current / elapsed if current else 0.0
        remaining = max(0, self.total - current)
        eta_seconds = remaining / rate if rate > 0 else 0.0
        filled = int((current / self.total) * 24) if self.total else 24
        bar = "#" * filled + "-" * (24 - filled)
        line = (
            f"{self.label} [{bar}] {current}/{self.total} "
            f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta_seconds)} "
            f"valid_json={valid_json_rows} structured={structured_match_rows}"
        )

        if self.is_tty:
            padded = line.ljust(self.last_render_length)
            print(f"\r{padded}", end="", file=self.stream, flush=True)
            self.last_render_length = len(padded)
        else:
            print(line, file=self.stream, flush=True)

    def finish(self) -> None:
        if self.is_tty:
            print(file=self.stream, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"Evaluation output directory already exists: {output_root}. Re-run with --force to overwrite it."
            )
        for child in output_root.iterdir():
            if child.is_dir():
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_file():
                        nested.unlink()
                    elif nested.is_dir():
                        nested.rmdir()
                child.rmdir()
            else:
                child.unlink()
    output_root.mkdir(parents=True, exist_ok=True)

    message_rows = load_jsonl(args.eval_messages_path.resolve())
    example_rows = load_jsonl(args.eval_examples_path.resolve())
    if args.limit is not None:
        message_rows = message_rows[: args.limit]

    model, tokenizer, generate_fn = load_mlx_model_and_tokenizer(
        model_path=args.model,
        adapter_path=args.adapter_path.resolve(),
    )

    row_results: list[dict[str, object]] = []
    progress = ProgressPrinter(total=len(message_rows), label="Generating")
    valid_json_rows = 0
    structured_match_rows = 0
    for index, message_row in enumerate(message_rows, start=1):
        raw_output = generate_output_with_loaded_model(
            model=model,
            tokenizer=tokenizer,
            generate_fn=generate_fn,
            messages=message_row["messages"],
            max_tokens=args.max_tokens,
            enable_thinking=args.enable_thinking,
        )
        metrics = score_prediction_payload(raw_output, gold_completion=message_row["completion"])
        valid_json_rows += int(metrics["valid_json"])
        structured_match_rows += int(metrics["structured_match"])
        row_results.append(
            {
                "message_row": message_row,
                "gold_payload": json.loads(message_row["completion"]),
                "raw_output": raw_output,
                "metrics": metrics,
                "execution": None,
            }
        )
        if index == 1 or index == len(message_rows) or index % max(1, args.progress_every) == 0:
            progress.update(
                index,
                valid_json_rows=valid_json_rows,
                structured_match_rows=structured_match_rows,
            )
    progress.finish()

    if args.neo4j_uri:
        print("Running execution checks...", flush=True)
        fixture_rows = load_jsonl(args.fixtures_path.resolve())
        row_results = evaluate_execution_matches(
            row_results=row_results,
            example_rows=example_rows,
            fixture_rows=fixture_rows,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
        )

    predictions_path = output_root / "predictions.jsonl"
    write_jsonl(predictions_path, (_prediction_export_row(result) for result in row_results))

    summary = {
        "model": args.model,
        "adapter_path": str(args.adapter_path.resolve()),
        "eval_messages_path": str(args.eval_messages_path.resolve()),
        "eval_examples_path": str(args.eval_examples_path.resolve()),
        "rows": summarize_prediction_metrics(row_results),
        "by_example_id": _group_summary(row_results, "example_id"),
        "by_intent_id": _group_summary(row_results, "intent_id"),
        "enable_thinking": args.enable_thinking,
        "max_tokens": args.max_tokens,
        "predictions_path": str(predictions_path),
    }
    _write_summary(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0
