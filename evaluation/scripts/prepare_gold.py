"""Convert raw gold benchmark CSV files into clean JSONL triples."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_FIELDS: tuple[str, ...] = ("subject", "subject_type", "relation", "object", "object_type")
SPLITS: tuple[str, ...] = ("dev", "test")


@dataclass(frozen=True)
class ConversionSummary:
    split: str
    raw_path: Path
    clean_path: Path
    row_count: int

    def to_json(self, *, root: Path) -> dict[str, Any]:
        return {
            "split": self.split,
            "raw_path": str(self.raw_path.relative_to(root)),
            "clean_path": str(self.clean_path.relative_to(root)),
            "row_count": self.row_count,
        }


def _canonical_header_map(fieldnames: list[str] | None, *, csv_path: Path) -> dict[str, str]:
    if not fieldnames:
        raise ValueError(f"{csv_path} has no header row.")

    header_map = {field.strip().casefold(): field for field in fieldnames}
    missing = [field for field in REQUIRED_FIELDS if field not in header_map]
    if missing:
        expected = ", ".join(REQUIRED_FIELDS)
        found = ", ".join(fieldnames)
        raise ValueError(f"{csv_path} is missing required columns {missing}. Expected {expected}. Found {found}.")
    return header_map


def _clean_cell(value: str | None) -> str:
    return " ".join((value or "").strip().split())


def read_gold_csv(csv_path: Path) -> list[dict[str, str]]:
    triples: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header_map = _canonical_header_map(reader.fieldnames, csv_path=csv_path)
        for row_number, row in enumerate(reader, start=2):
            if row is None or not any(_clean_cell(value) for value in row.values()):
                continue

            triple = {field: _clean_cell(row.get(header_map[field])) for field in REQUIRED_FIELDS}
            empty_fields = [field for field, value in triple.items() if not value]
            if empty_fields:
                raise ValueError(f"{csv_path}:{row_number} has empty required fields: {empty_fields}.")
            triples.append(triple)
    return triples


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def convert_csv(csv_path: Path, *, split: str, clean_dir: Path) -> ConversionSummary:
    triples = read_gold_csv(csv_path)
    clean_path = clean_dir / f"{csv_path.stem}.jsonl"
    write_jsonl(clean_path, triples)
    return ConversionSummary(split=split, raw_path=csv_path, clean_path=clean_path, row_count=len(triples))


def convert_split(root: Path, split: str) -> list[ConversionSummary]:
    split_dir = root / "benchmarks" / split
    raw_dir = split_dir / "raw"
    clean_dir = split_dir / "clean"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(path for path in raw_dir.glob("*.csv") if path.is_file())
    summaries = [convert_csv(path, split=split, clean_dir=clean_dir) for path in csv_paths]
    return summaries


def write_manifest(root: Path, split: str, summaries: list[ConversionSummary]) -> Path:
    manifest_path = root / "benchmarks" / split / "clean" / "manifest.json"
    payload = {
        "split": split,
        "file_count": len(summaries),
        "triple_count": sum(summary.row_count for summary in summaries),
        "files": [summary.to_json(root=root) for summary in summaries],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def convert_benchmarks(root: Path, splits: list[str]) -> list[ConversionSummary]:
    all_summaries: list[ConversionSummary] = []
    for split in splits:
        summaries = convert_split(root, split)
        if summaries:
            write_manifest(root, split, summaries)
        all_summaries.extend(summaries)
    return all_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw gold benchmark CSV files into clean JSONL files.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Evaluation folder root. Defaults to the repo's evaluation/ folder.",
    )
    parser.add_argument(
        "--split",
        choices=("dev", "test", "all"),
        default="all",
        help="Benchmark split to convert.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    splits = list(SPLITS) if args.split == "all" else [args.split]
    summaries = convert_benchmarks(root, splits)

    for summary in summaries:
        print(f"{summary.split}: {summary.raw_path.name} -> {summary.clean_path.name} ({summary.row_count} triples)")
    if not summaries:
        print("No raw CSV files found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
