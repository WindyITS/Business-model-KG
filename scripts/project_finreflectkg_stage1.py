import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finreflectkg_projection import load_finreflectkg_rows, project_dataset_rows, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Project FinReflectKG into the repo ontology (stage 1).")
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
    parser.add_argument("--skip-chunks", type=int, default=0, help="Number of chunks to skip before processing.")
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="outputs/finreflectkg_stage1/projected_examples.jsonl",
        help="Where to write chunk-level projected examples.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="outputs/finreflectkg_stage1/projection_report.json",
        help="Where to write the stage-1 summary report.",
    )
    args = parser.parse_args()

    rows = load_finreflectkg_rows(
        hf_dataset=None if args.parquet_file else args.hf_dataset,
        split=args.split,
        cache_dir=args.cache_dir,
        parquet_files=args.parquet_file or None,
        streaming=not args.no_streaming,
        limit_rows=args.limit_rows,
    )
    examples, report = project_dataset_rows(rows, limit_chunks=args.limit_chunks, skip_chunks=args.skip_chunks)

    output_jsonl = Path(args.output_jsonl)
    report_path = Path(args.report_path)
    write_jsonl(output_jsonl, examples)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(examples)} projected examples to {output_jsonl}")
    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
