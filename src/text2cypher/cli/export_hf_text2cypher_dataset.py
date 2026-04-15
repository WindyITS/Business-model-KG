from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = ROOT / "datasets" / "text2cypher" / "v3"
DEFAULT_PACKAGING_ROOT = ROOT / "packaging" / "huggingface" / "text2cypher-v3"
DEFAULT_OUTPUT_ROOT = ROOT / "dist" / "huggingface" / "text2cypher-v3"


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _copy_file_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _build_summary(dataset_root: Path) -> dict:
    fixture_path = dataset_root / "source" / "fixture_instances.jsonl"
    source_path = dataset_root / "source" / "bound_seed_examples.jsonl"
    training_path = dataset_root / "training" / "training_examples.jsonl"
    messages_path = dataset_root / "training" / "messages.jsonl"
    heldout_messages_path = dataset_root / "evaluation" / "test_messages.jsonl"
    manifest_path = dataset_root / "reports" / "training_split_manifest.json"
    sft_manifest_path = dataset_root / "reports" / "sft_manifest.json"
    heldout_manifest_path = dataset_root / "reports" / "heldout_test_manifest.json"

    fixtures = _load_jsonl(fixture_path)
    source_rows = _load_jsonl(source_path)
    training_rows = _load_jsonl(training_path)
    message_rows = _load_jsonl(messages_path) if messages_path.exists() else []
    heldout_message_rows = _load_jsonl(heldout_messages_path) if heldout_messages_path.exists() else []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sft_manifest = json.loads(sft_manifest_path.read_text(encoding="utf-8")) if sft_manifest_path.exists() else {}
    heldout_manifest = (
        json.loads(heldout_manifest_path.read_text(encoding="utf-8"))
        if heldout_manifest_path.exists()
        else {}
    )

    summary = {
        "dataset_root": _display_path(dataset_root),
        "fixtures": len(fixtures),
        "source_examples": len(source_rows),
        "training_examples": len(training_rows),
        "message_examples": len(message_rows),
        "intents": len({row["intent_id"] for row in source_rows}),
        "families": len({row["family_id"] for row in source_rows}),
        "answerable_source_examples": Counter(row["answerable"] for row in source_rows),
        "difficulty_training_rows": Counter(row["difficulty"] for row in training_rows),
        "split_counts": manifest["split_counts"],
        "message_split_counts": sft_manifest.get("split_counts", {}),
        "duplicate_prompt_rows_merged": sft_manifest.get("counts", {}).get("duplicate_prompt_rows_merged", 0),
    }
    if heldout_message_rows or heldout_manifest:
        summary["heldout_message_examples"] = len(heldout_message_rows)
        summary["heldout_message_split_counts"] = heldout_manifest.get("split_counts", {})
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the local text2cypher dataset build into an HF-ready directory."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the local text2cypher dataset build root.",
    )
    parser.add_argument(
        "--packaging-root",
        type=Path,
        default=DEFAULT_PACKAGING_ROOT,
        help="Path to the HF packaging templates and docs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination directory for the exported HF dataset repo.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    packaging_root = args.packaging_root.resolve()
    output_root = args.output_root.resolve()

    for required in (
        dataset_root / "source",
        dataset_root / "reports",
        dataset_root / "training",
        dataset_root / "training" / "messages.jsonl",
        packaging_root / "README.md",
    ):
        _ensure_exists(required)

    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"Output directory already exists: {output_root}. Re-run with --force to overwrite it."
            )
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    _copy_tree(dataset_root / "source", output_root / "source")
    _copy_tree(dataset_root / "reports", output_root / "reports")
    _copy_tree(dataset_root / "training", output_root / "training")
    if (dataset_root / "evaluation").exists():
        _copy_tree(dataset_root / "evaluation", output_root / "evaluation")

    _copy_file_if_exists(packaging_root / "README.md", output_root / "README.md")
    _copy_file_if_exists(packaging_root / ".gitattributes", output_root / ".gitattributes")
    _copy_file_if_exists(packaging_root / "UPLOAD.md", output_root / "UPLOAD.md")

    summary = _build_summary(dataset_root)
    (output_root / "release_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=dict),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "dataset_root": str(dataset_root),
                "packaging_root": str(packaging_root),
                "summary_path": str(output_root / "release_summary.json"),
            },
            indent=2,
        )
    )
    return 0
