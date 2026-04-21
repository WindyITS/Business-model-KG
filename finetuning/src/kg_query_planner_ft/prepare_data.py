from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .cli_output import render_prepare_data_summary
from .config import load_config
from .constants import ROUTE_TO_ROUTER_LABEL
from .frozen_prompt import FROZEN_QUERY_SYSTEM_PROMPT
from .json_utils import compact_json, read_jsonl, write_json, write_jsonl
from .paths import (
    dataset_root,
    prepared_planner_balanced_dir,
    prepared_planner_raw_dir,
    prepared_router_dir,
)
from .progress import StepProgress, track

SOURCE_SPLITS = (
    ("train", "train"),
    ("validation", "valid"),
    ("release_eval", "test"),
)


def _load_source_rows(source_root: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        prepared_name: read_jsonl(source_root / f"{source_name}.jsonl")
        for source_name, prepared_name in track(
            SOURCE_SPLITS,
            total=len(SOURCE_SPLITS),
            desc="load dataset splits",
            unit="split",
        )
    }


def _planner_row(row: dict[str, Any], split_name: str) -> dict[str, Any]:
    plan = row["supervision_target"]["plan"]
    target_json = compact_json(plan)
    return {
        "question": row["question"],
        "split": split_name,
        "family": row["family"],
        "target_json": target_json,
        "gold_plan": plan,
        "messages": [
            {"role": "system", "content": FROZEN_QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": target_json},
        ],
    }


def _router_row(row: dict[str, Any], split_name: str) -> dict[str, Any]:
    return {
        "question": row["question"],
        "label": ROUTE_TO_ROUTER_LABEL[row["route_label"]],
        "split": split_name,
        "source_route_label": row["route_label"],
        "family": row["family"],
    }


def _rebalance_planner_train(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_family[row["family"]].append(row)

    max_family_count = max(len(family_rows) for family_rows in rows_by_family.values())
    balanced: list[dict[str, Any]] = []
    for family in sorted(rows_by_family):
        family_rows = rows_by_family[family]
        target_count = min(max_family_count, len(family_rows) * 3)
        repeated: list[dict[str, Any]] = []
        for index in range(target_count):
            base_row = family_rows[index % len(family_rows)]
            materialized = dict(base_row)
            materialized["rebalance_repeat_index"] = index
            repeated.append(materialized)
        balanced.extend(repeated)
    return balanced


def _write_prompt_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FROZEN_QUERY_SYSTEM_PROMPT + "\n", encoding="utf-8")


def prepare_data(config_path: str | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    with StepProgress(total=6, desc="prepare-data") as progress:
        source_root = dataset_root(config)
        rows_by_split = _load_source_rows(source_root)
        progress.advance("loaded source splits")

        router_rows_by_split: dict[str, list[dict[str, Any]]] = {}
        planner_raw_by_split: dict[str, list[dict[str, Any]]] = {}
        for split_name, rows in track(
            rows_by_split.items(),
            total=len(rows_by_split),
            desc="materialize prepared splits",
            unit="split",
        ):
            router_rows_by_split[split_name] = [_router_row(row, split_name) for row in rows]
            planner_raw_by_split[split_name] = [
                _planner_row(row, split_name)
                for row in rows
                if row["route_label"] == "local_safe"
            ]
        progress.advance("built router and planner rows")

        router_dir = prepared_router_dir(config)
        planner_raw_dir = prepared_planner_raw_dir(config)
        planner_balanced_dir = prepared_planner_balanced_dir(config)

        for split_name, rows in track(
            router_rows_by_split.items(),
            total=len(router_rows_by_split),
            desc="write router dataset",
            unit="split",
        ):
            write_jsonl(router_dir / f"{split_name}.jsonl", rows)
        progress.advance("wrote router dataset")

        for split_name, rows in track(
            planner_raw_by_split.items(),
            total=len(planner_raw_by_split),
            desc="write planner raw dataset",
            unit="split",
        ):
            write_jsonl(planner_raw_dir / f"{split_name}.jsonl", rows)

        planner_balanced_train = _rebalance_planner_train(planner_raw_by_split["train"])
        write_jsonl(planner_balanced_dir / "train.jsonl", planner_balanced_train)
        for split_name in track(
            ("valid", "test"),
            total=2,
            desc="write planner balanced dataset",
            unit="split",
        ):
            write_jsonl(planner_balanced_dir / f"{split_name}.jsonl", planner_raw_by_split[split_name])
        progress.advance("wrote planner datasets")

        for prompt_path in track(
            (
                router_dir / "frozen_prompt.txt",
                planner_raw_dir / "frozen_prompt.txt",
                planner_balanced_dir / "frozen_prompt.txt",
            ),
            total=3,
            desc="write frozen prompts",
            unit="file",
        ):
            _write_prompt_file(prompt_path)
        progress.advance("wrote frozen prompts")

        router_stats = {
            split_name: dict(sorted(Counter(row["label"] for row in rows).items()))
            for split_name, rows in router_rows_by_split.items()
        }
        planner_raw_stats = {
            split_name: dict(sorted(Counter(row["family"] for row in rows).items()))
            for split_name, rows in planner_raw_by_split.items()
        }
        planner_balanced_stats = {
            "train": dict(sorted(Counter(row["family"] for row in planner_balanced_train).items())),
            "valid": planner_raw_stats["valid"],
            "test": planner_raw_stats["test"],
        }

        summary = {
            "source_root": str(source_root),
            "router": {
                "output_dir": str(router_dir),
                "counts_by_split": {split_name: len(rows) for split_name, rows in router_rows_by_split.items()},
                "label_counts_by_split": router_stats,
            },
            "planner_raw": {
                "output_dir": str(planner_raw_dir),
                "counts_by_split": {split_name: len(rows) for split_name, rows in planner_raw_by_split.items()},
                "family_counts_by_split": planner_raw_stats,
            },
            "planner_balanced": {
                "output_dir": str(planner_balanced_dir),
                "counts_by_split": {
                    "train": len(planner_balanced_train),
                    "valid": len(planner_raw_by_split["valid"]),
                    "test": len(planner_raw_by_split["test"]),
                },
                "family_counts_by_split": planner_balanced_stats,
            },
        }
        write_json(router_dir / "summary.json", summary["router"])
        write_json(planner_raw_dir / "summary.json", summary["planner_raw"])
        write_json(planner_balanced_dir / "summary.json", summary["planner_balanced"])
        write_json(router_dir.parent / "summary.json", summary)
        progress.advance("wrote summaries")
        return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare isolated fine-tuning datasets.")
    parser.add_argument("--config", type=str, default=None, help="Path to the fine-tuning JSON config.")
    parser.add_argument("--json", action="store_true", help="Print the final summary as compact JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = prepare_data(args.config)
    print(compact_json(summary) if args.json else render_prepare_data_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
