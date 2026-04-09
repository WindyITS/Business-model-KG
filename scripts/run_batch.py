"""Run a single batch of the Stage 1 → 2 → 3 pipeline on a slice of FinReflectKG.

Usage:
    # Run batch 1 (of 5)
    python scripts/run_batch.py --batch 1

    # Run batch 3 with a custom model
    python scripts/run_batch.py --batch 3 --model gemma-4-27b-it

    # Merge all completed batches into a single training set
    python scripts/run_batch.py --merge

    # Run with custom chunk window (overrides defaults)
    python scripts/run_batch.py --batch 1 --chunks-per-batch 100000
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT_DIR / "venv" / "bin" / "python")

DEFAULT_CHUNKS_PER_BATCH = 80000
DEFAULT_NUM_BATCHES = 5
DEFAULT_MODEL = "gemma-4-27b-it"
DEFAULT_PROMPT_PROFILE = "default"
DEFAULT_EMPTY_RATIO = 0.3


def batch_dir(batch_num: int) -> Path:
    return ROOT_DIR / "outputs" / f"batch_{batch_num}"


def run_cmd(args: list[str], description: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  $ {' '.join(args)}\n")
    result = subprocess.run(args, cwd=str(ROOT_DIR))
    if result.returncode != 0:
        print(f"\nFAILED: {description} (exit code {result.returncode})")
        sys.exit(result.returncode)


def run_stage1(batch_num: int, chunks_per_batch: int) -> Path:
    skip = (batch_num - 1) * chunks_per_batch
    out_dir = batch_dir(batch_num) / "stage1"
    output_jsonl = out_dir / "projected_examples.jsonl"
    report_path = out_dir / "projection_report.json"

    run_cmd(
        [
            PYTHON, str(ROOT_DIR / "scripts" / "project_finreflectkg_stage1.py"),
            "--limit-chunks", str(chunks_per_batch),
            "--skip-chunks", str(skip),
            "--output-jsonl", str(output_jsonl),
            "--report-path", str(report_path),
        ],
        f"Batch {batch_num} — Stage 1: project chunks {skip:,}–{skip + chunks_per_batch:,}",
    )
    return output_jsonl


def run_stage2(batch_num: int, projected_jsonl: Path, chunks_per_batch: int, empty_ratio: float) -> Path:
    out_dir = batch_dir(batch_num) / "stage2"
    output_jsonl = out_dir / "empty_examples.jsonl"
    merged_output = out_dir / "training_examples.jsonl"
    report_path = out_dir / "empty_sampling_report.json"

    run_cmd(
        [
            PYTHON, str(ROOT_DIR / "scripts" / "sample_empty_chunks.py"),
            "--projected-jsonl", str(projected_jsonl),
            "--empty-ratio", str(empty_ratio),
            "--limit-chunks", str(chunks_per_batch),
            "--output-jsonl", str(output_jsonl),
            "--merged-output-jsonl", str(merged_output),
            "--report-path", str(report_path),
        ],
        f"Batch {batch_num} — Stage 2: sample empty chunks",
    )
    return output_jsonl


def run_stage3(
    batch_num: int,
    projected_jsonl: Path,
    empty_jsonl: Path,
    model: str,
    prompt_profile: str,
    empty_ratio: float,
) -> None:
    out_dir = batch_dir(batch_num) / "stage3"

    run_cmd(
        [
            PYTHON, str(ROOT_DIR / "scripts" / "augment_finreflectkg_stage3.py"),
            "--projected-jsonl", str(projected_jsonl),
            "--candidate-empty-jsonl", str(empty_jsonl),
            "--prompt-profile", prompt_profile,
            "--no-schema", "--no-think",
            "--model", model,
            "--empty-ratio", str(empty_ratio),
            "--augmented-positives-output", str(out_dir / "augmented_positive_examples.jsonl"),
            "--augmented-empty-output", str(out_dir / "augmented_empty_candidates.jsonl"),
            "--augmented-trigger-output", str(out_dir / "augmented_relation_trigger_candidates.jsonl"),
            "--final-positive-output", str(out_dir / "final_positive_examples.jsonl"),
            "--final-empty-output", str(out_dir / "final_empty_examples.jsonl"),
            "--training-output", str(out_dir / "training_examples.jsonl"),
            "--teacher-log-output", str(out_dir / "teacher_logs.jsonl"),
            "--report-path", str(out_dir / "stage3_report.json"),
        ],
        f"Batch {batch_num} — Stage 3: teacher augmentation ({model}, {prompt_profile})",
    )


def merge_batches(num_batches: int) -> None:
    merged_dir = ROOT_DIR / "outputs" / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    all_training: list[dict] = []
    all_positive: list[dict] = []
    all_empty: list[dict] = []
    batch_summaries: list[dict] = []
    seen_chunk_keys: set[str] = set()
    duplicates_skipped = 0

    for batch_num in range(1, num_batches + 1):
        training_path = batch_dir(batch_num) / "stage3" / "training_examples.jsonl"
        report_path = batch_dir(batch_num) / "stage3" / "stage3_report.json"

        if not training_path.exists():
            print(f"Batch {batch_num}: not found at {training_path}, skipping")
            continue

        batch_examples: list[dict] = []
        with open(training_path, encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                chunk_key = json.dumps(ex.get("metadata", {}).get("chunk_key", {}), sort_keys=True, separators=(",", ":"))
                if chunk_key in seen_chunk_keys:
                    duplicates_skipped += 1
                    continue
                seen_chunk_keys.add(chunk_key)
                batch_examples.append(ex)

        batch_pos = [ex for ex in batch_examples if ex.get("output", {}).get("triples")]
        batch_emp = [ex for ex in batch_examples if not ex.get("output", {}).get("triples")]
        all_training.extend(batch_examples)
        all_positive.extend(batch_pos)
        all_empty.extend(batch_emp)

        report = {}
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))

        batch_summaries.append({
            "batch": batch_num,
            "examples": len(batch_examples),
            "positive": len(batch_pos),
            "empty": len(batch_emp),
            "relation_counts": report.get("relation_counts", {}),
        })
        print(f"Batch {batch_num}: {len(batch_examples)} examples ({len(batch_pos)} pos, {len(batch_emp)} empty)")

    # Write merged outputs
    _write_jsonl(merged_dir / "training_examples.jsonl", all_training)
    _write_jsonl(merged_dir / "positive_examples.jsonl", all_positive)
    _write_jsonl(merged_dir / "empty_examples.jsonl", all_empty)

    # Aggregate relation counts
    total_relation_counts: dict[str, int] = {}
    for summary in batch_summaries:
        for rel, count in summary["relation_counts"].items():
            total_relation_counts[rel] = total_relation_counts.get(rel, 0) + count

    # Collect unique tickers
    tickers: set[str] = set()
    for ex in all_training:
        ticker = ex.get("metadata", {}).get("chunk_key", {}).get("ticker", "")
        if ticker:
            tickers.add(ticker)

    merge_report = {
        "batches_merged": len(batch_summaries),
        "total_examples": len(all_training),
        "total_positive": len(all_positive),
        "total_empty": len(all_empty),
        "duplicates_skipped": duplicates_skipped,
        "unique_tickers": len(tickers),
        "relation_counts": dict(sorted(total_relation_counts.items())),
        "batch_summaries": batch_summaries,
    }
    (merged_dir / "merge_report.json").write_text(
        json.dumps(merge_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print(f"  Merged {len(batch_summaries)} batches")
    print(f"  Total: {len(all_training)} examples ({len(all_positive)} pos, {len(all_empty)} empty)")
    print(f"  Companies: {len(tickers)}")
    print(f"  Duplicates skipped: {duplicates_skipped}")
    print(f"  Relations: {dict(sorted(total_relation_counts.items()))}")
    print(f"  Wrote to {merged_dir}/")
    print(f"{'='*60}")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a single batch of the Stage 1→2→3 pipeline, or merge all batches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch", type=int, choices=range(1, DEFAULT_NUM_BATCHES + 1), help="Which batch to run (1–5).")
    parser.add_argument("--merge", action="store_true", help="Merge all completed batches into a single training set.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model id (default: {DEFAULT_MODEL}).")
    parser.add_argument("--prompt-profile", type=str, default=DEFAULT_PROMPT_PROFILE, help=f"Prompt profile (default: {DEFAULT_PROMPT_PROFILE}).")
    parser.add_argument("--chunks-per-batch", type=int, default=DEFAULT_CHUNKS_PER_BATCH, help=f"Chunks per batch (default: {DEFAULT_CHUNKS_PER_BATCH:,}).")
    parser.add_argument("--num-batches", type=int, default=DEFAULT_NUM_BATCHES, help=f"Total number of batches (default: {DEFAULT_NUM_BATCHES}).")
    parser.add_argument("--empty-ratio", type=float, default=DEFAULT_EMPTY_RATIO, help=f"Target empty ratio (default: {DEFAULT_EMPTY_RATIO}).")
    parser.add_argument("--skip-stage1", action="store_true", help="Skip Stage 1 (reuse existing projected examples).")
    parser.add_argument("--skip-stage2", action="store_true", help="Skip Stage 2 (reuse existing empty candidates).")
    args = parser.parse_args()

    if not args.batch and not args.merge:
        parser.error("Specify --batch N or --merge.")

    if args.merge:
        merge_batches(args.num_batches)
        return 0

    batch_num = args.batch
    bd = batch_dir(batch_num)
    projected_jsonl = bd / "stage1" / "projected_examples.jsonl"
    empty_jsonl = bd / "stage2" / "empty_examples.jsonl"

    print(f"\n  Batch {batch_num}/{args.num_batches}")
    print(f"  Chunks: {(batch_num-1)*args.chunks_per_batch:,} – {batch_num*args.chunks_per_batch:,}")
    print(f"  Model: {args.model}")
    print(f"  Profile: {args.prompt_profile}")
    print(f"  Output: {bd}/")

    # Stage 1
    if args.skip_stage1:
        if not projected_jsonl.exists():
            print(f"\nERROR: --skip-stage1 but {projected_jsonl} does not exist.")
            return 1
        print(f"\nSkipping Stage 1, reusing {projected_jsonl}")
    else:
        run_stage1(batch_num, args.chunks_per_batch)

    # Stage 2
    if args.skip_stage2:
        if not empty_jsonl.exists():
            print(f"\nERROR: --skip-stage2 but {empty_jsonl} does not exist.")
            return 1
        print(f"\nSkipping Stage 2, reusing {empty_jsonl}")
    else:
        run_stage2(batch_num, projected_jsonl, args.chunks_per_batch, args.empty_ratio)

    # Stage 3
    run_stage3(
        batch_num,
        projected_jsonl,
        empty_jsonl,
        model=args.model,
        prompt_profile=args.prompt_profile,
        empty_ratio=args.empty_ratio,
    )

    # Print summary
    report_path = bd / "stage3" / "stage3_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"\n{'='*60}")
        print(f"  Batch {batch_num} complete")
        print(f"  Training examples: {report.get('final_training_example_count', '?')}")
        print(f"  Positive: {report.get('final_positive_example_count', '?')}")
        print(f"  Empty: {report.get('selected_empty_count', '?')}")
        print(f"  Relations: {report.get('relation_counts', {})}")
        print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
