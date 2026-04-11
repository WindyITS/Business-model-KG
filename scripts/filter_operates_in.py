import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from operates_in_cleanup import filter_operates_in_jsonl


def default_batch_paths(batch_num: int) -> tuple[Path, Path, Path]:
    batch_dir = ROOT_DIR / "outputs" / f"batch_{batch_num}" / "stage3"
    return (
        batch_dir / "training_examples.jsonl",
        batch_dir / "training_examples_operates_in_filtered.jsonl",
        batch_dir / "operates_in_filter_report.json",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter noisy OPERATES_IN triples from a dataset without modifying the original JSONL."
    )
    parser.add_argument("--batch", type=int, default=None, help="Batch number under outputs/batch_N/stage3/.")
    parser.add_argument("--input-jsonl", type=str, default=None, help="Input training JSONL to filter.")
    parser.add_argument("--output-jsonl", type=str, default=None, help="Filtered output JSONL path.")
    parser.add_argument("--report-path", type=str, default=None, help="Cleanup report JSON path.")
    args = parser.parse_args()

    if args.batch is None and args.input_jsonl is None:
        parser.error("Specify --batch N or --input-jsonl PATH.")

    if args.batch is not None:
        default_input, default_output, default_report = default_batch_paths(args.batch)
        input_jsonl = Path(args.input_jsonl) if args.input_jsonl else default_input
        output_jsonl = Path(args.output_jsonl) if args.output_jsonl else default_output
        report_path = Path(args.report_path) if args.report_path else default_report
    else:
        input_jsonl = Path(args.input_jsonl)
        output_jsonl = Path(args.output_jsonl) if args.output_jsonl else input_jsonl.with_name(
            f"{input_jsonl.stem}_operates_in_filtered{input_jsonl.suffix}"
        )
        report_path = Path(args.report_path) if args.report_path else input_jsonl.with_name(
            f"{input_jsonl.stem}_operates_in_filter_report.json"
        )

    if not input_jsonl.exists():
        print(f"ERROR: input JSONL does not exist: {input_jsonl}", file=sys.stderr)
        return 1

    report = filter_operates_in_jsonl(input_jsonl, output_jsonl, report_path)
    print(f"Wrote filtered dataset to {output_jsonl}")
    print(f"Wrote cleanup report to {report_path}")
    print(
        "Removed "
        f"{report['dropped_operates_in_triple_count']} noisy OPERATES_IN triples "
        f"and dropped {report['dropped_example_count']} example(s) that became empty."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
