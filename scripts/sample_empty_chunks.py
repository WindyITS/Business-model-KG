import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finreflectkg_projection import (
    load_finreflectkg_rows,
    load_jsonl_records,
    sample_empty_examples,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample deterministic empty chunks for stage 2.")
    parser.add_argument("--hf-dataset", type=str, default="domyn/FinReflectKG", help="Hugging Face dataset id.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split.")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/external/huggingface",
        help="Local cache dir for downloaded dataset shards.",
    )
    parser.add_argument(
        "--parquet-file",
        action="append",
        default=[],
        help="Optional local parquet file path. Repeat for multiple files.",
    )
    parser.add_argument("--no-streaming", action="store_true", help="Disable streaming mode.")
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional cap on raw rows read.")
    parser.add_argument("--limit-chunks", type=int, default=None, help="Optional cap on processed chunks.")
    parser.add_argument("--skip-chunks", type=int, default=0, help="Number of grouped chunks to skip before sampling.")
    parser.add_argument(
        "--projected-jsonl",
        type=str,
        default="outputs/finreflectkg_stage1/projected_examples.jsonl",
        help="Stage-1 projected examples JSONL path.",
    )
    parser.add_argument("--empty-ratio", type=float, default=0.3, help="Target empty/non-empty ratio.")
    parser.add_argument("--min-empty-words", type=int, default=80, help="Minimum words for an empty chunk candidate.")
    parser.add_argument("--min-empty-chars", type=int, default=400, help="Minimum chars for an empty chunk candidate.")
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="outputs/finreflectkg_stage2/empty_examples.jsonl",
        help="Where to write sampled empty examples.",
    )
    parser.add_argument(
        "--merged-output-jsonl",
        type=str,
        default="outputs/finreflectkg_stage2/training_examples.jsonl",
        help="Where to write stage-1 examples plus sampled empties.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="outputs/finreflectkg_stage2/empty_sampling_report.json",
        help="Where to write the stage-2 report.",
    )
    args = parser.parse_args()

    projected_jsonl = Path(args.projected_jsonl)
    positive_examples = load_jsonl_records(projected_jsonl)
    rows = load_finreflectkg_rows(
        hf_dataset=None if args.parquet_file else args.hf_dataset,
        split=args.split,
        cache_dir=args.cache_dir,
        parquet_files=args.parquet_file or None,
        streaming=not args.no_streaming,
        limit_rows=args.limit_rows,
    )

    empty_examples, report = sample_empty_examples(
        rows,
        positive_example_count=len(positive_examples),
        empty_ratio=args.empty_ratio,
        limit_chunks=args.limit_chunks,
        skip_chunks=args.skip_chunks,
        min_word_count=args.min_empty_words,
        min_char_count=args.min_empty_chars,
    )

    output_jsonl = Path(args.output_jsonl)
    merged_output_jsonl = Path(args.merged_output_jsonl)
    report_path = Path(args.report_path)
    write_jsonl(output_jsonl, empty_examples)
    write_jsonl(merged_output_jsonl, positive_examples + empty_examples)

    report_payload = {
        "source_dataset": "domyn/FinReflectKG",
        "projected_jsonl": str(projected_jsonl),
        "positive_example_count": len(positive_examples),
        **report,
        "merged_example_count": len(positive_examples) + len(empty_examples),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(empty_examples)} empty examples to {output_jsonl}")
    print(f"Wrote merged dataset to {merged_output_jsonl}")
    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
