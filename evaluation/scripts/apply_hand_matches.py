"""Compute hand-matched second-tier metrics from reviewed unmatched CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def result_folder_has_files(path: Path) -> bool:
    return path.exists() and any(candidate.is_file() for candidate in path.rglob("*"))


def confirm_overwrite(path: Path) -> bool:
    response = input(
        f"There are already files in the hand-matched results folder {path}. "
        "Proceeding is going to overwrite them. Do you want to proceed? [Y/n] "
    ).strip()
    return response.casefold() not in {"n", "no"}


def prepare_result_folder(path: Path, *, assume_yes: bool = False) -> bool:
    if not result_folder_has_files(path):
        return True
    if not assume_yes and not confirm_overwrite(path):
        return False
    shutil.rmtree(path)
    return True


def metric_payload(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def reviewed_match_count(review_csv: Path) -> tuple[int, list[dict[str, Any]]]:
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"gold": 0, "predicted": 0})
    with review_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            match_id = (row.get("match_id") or "").strip()
            source = (row.get("source") or "").strip()
            if not match_id:
                continue
            if source not in {"gold", "predicted"}:
                continue
            groups[match_id][source] += 1

    accepted_groups: list[dict[str, Any]] = []
    recovered = 0
    for match_id, counts in sorted(groups.items()):
        group_recovered = min(counts["gold"], counts["predicted"])
        if group_recovered <= 0:
            accepted_groups.append(
                {
                    "match_id": match_id,
                    "gold_rows": counts["gold"],
                    "predicted_rows": counts["predicted"],
                    "recovered_matches": 0,
                    "warning": "match_id must label at least one gold row and one predicted row to affect metrics",
                }
            )
            continue
        recovered += group_recovered
        accepted_groups.append(
            {
                "match_id": match_id,
                "gold_rows": counts["gold"],
                "predicted_rows": counts["predicted"],
                "recovered_matches": group_recovered,
            }
        )
    return recovered, accepted_groups


def apply_hand_matches_to_company(company_dir: Path, *, output_dir: Path) -> dict[str, Any] | None:
    metrics_path = company_dir / "metrics.json"
    review_csv = company_dir / "unmatched_for_review.csv"
    if not metrics_path.is_file() or not review_csv.is_file():
        return None

    strict = read_json(metrics_path)
    if strict.get("status") != "evaluated":
        return None

    recovered, accepted_groups = reviewed_match_count(review_csv)
    strict_tp = int(strict["true_positives"])
    strict_fp = int(strict["false_positives"])
    strict_fn = int(strict["false_negatives"])
    recovered = min(recovered, strict_fp, strict_fn)

    adjusted = metric_payload(
        tp=strict_tp + recovered,
        fp=strict_fp - recovered,
        fn=strict_fn - recovered,
    )
    payload = {
        "company": strict.get("company"),
        "company_slug": strict.get("company_slug"),
        "pipeline": strict.get("pipeline"),
        "split": strict.get("split"),
        "status": "hand_matched",
        "review_csv": str(review_csv),
        "strict": metric_payload(strict_tp, strict_fp, strict_fn),
        "recovered_matches": recovered,
        "hand_matched": adjusted,
        "match_groups": accepted_groups,
    }
    write_json(output_dir / "metrics.json", payload)
    return payload


def find_company_dirs(results_dir: Path) -> list[Path]:
    if (results_dir / "metrics.json").is_file():
        return [results_dir]
    companies_dir = results_dir / "companies"
    if companies_dir.is_dir():
        return sorted(path for path in companies_dir.iterdir() if path.is_dir())
    return sorted(path.parent for path in results_dir.rglob("metrics.json"))


def aggregate(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    strict_tp = sum(int(payload["strict"]["true_positives"]) for payload in payloads)
    strict_fp = sum(int(payload["strict"]["false_positives"]) for payload in payloads)
    strict_fn = sum(int(payload["strict"]["false_negatives"]) for payload in payloads)
    hand_tp = sum(int(payload["hand_matched"]["true_positives"]) for payload in payloads)
    hand_fp = sum(int(payload["hand_matched"]["false_positives"]) for payload in payloads)
    hand_fn = sum(int(payload["hand_matched"]["false_negatives"]) for payload in payloads)
    return {
        "evaluated_company_count": len(payloads),
        "strict": metric_payload(strict_tp, strict_fp, strict_fn),
        "recovered_matches": sum(int(payload["recovered_matches"]) for payload in payloads),
        "hand_matched": metric_payload(hand_tp, hand_fp, hand_fn),
    }


def apply_hand_matches(results_dir: Path, *, assume_yes: bool = False) -> dict[str, Any] | None:
    output_root = results_dir / "hand_matched"
    if not prepare_result_folder(output_root, assume_yes=assume_yes):
        return None

    company_payloads = [
        payload
        for company_dir in find_company_dirs(results_dir)
        if (
            payload := apply_hand_matches_to_company(
                company_dir,
                output_dir=output_root / "companies" / company_dir.name,
            )
        )
        is not None
    ]
    summary = {
        "results_dir": str(results_dir),
        "output_dir": str(output_root),
        "companies": company_payloads,
        "aggregate": aggregate(company_payloads),
    }
    write_json(output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute second-tier metrics from manually tagged unmatched CSV files.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Result folder produced by evaluation.scripts.evaluate.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Overwrite existing hand-matched result files without prompting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = apply_hand_matches(args.results_dir.resolve(), assume_yes=args.yes)
    if summary is None:
        print("Hand-matching cancelled. Existing hand-matched results were left unchanged.")
        return 0
    aggregate_payload = summary["aggregate"]
    hand = aggregate_payload["hand_matched"]
    print(
        f"Applied hand matches for {aggregate_payload['evaluated_company_count']} companies. "
        f"Recovered={aggregate_payload['recovered_matches']}. "
        f"Hand-matched F1={hand['f1']:.3f}, precision={hand['precision']:.3f}, recall={hand['recall']:.3f}"
    )
    print(f"Summary: {args.results_dir / 'hand_matched' / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
