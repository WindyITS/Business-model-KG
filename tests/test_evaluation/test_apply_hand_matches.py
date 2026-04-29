import csv
import json
from pathlib import Path

from evaluation.scripts.apply_hand_matches import apply_hand_matches, finalize_result_folder, prepare_result_folder, staging_result_folder


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_review_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_id",
                "match_id",
                "source",
                "subject",
                "subject_type",
                "relation",
                "object",
                "object_type",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "row_id": "gold_0001",
                "match_id": "1",
                "source": "gold",
                "subject": "Intelligent Cloud",
                "subject_type": "BusinessSegment",
                "relation": "OFFERS",
                "object": "Azure and other cloud services",
                "object_type": "Offering",
            }
        )
        writer.writerow(
            {
                "row_id": "predicted_0001",
                "match_id": "1",
                "source": "predicted",
                "subject": "Intelligent Cloud",
                "subject_type": "BusinessSegment",
                "relation": "OFFERS",
                "object": "Azure",
                "object_type": "Offering",
            }
        )
        writer.writerow(
            {
                "row_id": "predicted_0002",
                "match_id": "",
                "source": "predicted",
                "subject": "Microsoft",
                "subject_type": "Company",
                "relation": "OPERATES_IN",
                "object": "Worldwide",
                "object_type": "Place",
            }
        )


def write_ambiguous_review_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_id",
                "match_id",
                "source",
                "subject",
                "subject_type",
                "relation",
                "object",
                "object_type",
            ],
        )
        writer.writeheader()
        for row_id, source in [
            ("gold_0001", "gold"),
            ("gold_0002", "gold"),
            ("predicted_0001", "predicted"),
        ]:
            writer.writerow(
                {
                    "row_id": row_id,
                    "match_id": "1",
                    "source": source,
                    "subject": "Intelligent Cloud",
                    "subject_type": "BusinessSegment",
                    "relation": "OFFERS",
                    "object": "Azure",
                    "object_type": "Offering",
                }
            )


def test_apply_hand_matches_adjusts_metrics_from_review_csv(tmp_path: Path):
    result_dir = tmp_path / "results" / "zero-shot" / "dev"
    company_dir = result_dir / "companies" / "microsoft"
    write_json(
        company_dir / "metrics.json",
        {
            "company": "Microsoft",
            "company_slug": "microsoft",
            "pipeline": "zero-shot",
            "split": "dev",
            "status": "evaluated",
            "true_positives": 3,
            "false_positives": 2,
            "false_negatives": 1,
        },
    )
    write_review_csv(company_dir / "unmatched_for_review.csv")

    summary = apply_hand_matches(result_dir)

    assert summary["aggregate"]["strict"]["true_positives"] == 3
    assert summary["aggregate"]["recovered_matches"] == 1
    assert summary["aggregate"]["hand_matched"]["true_positives"] == 4
    assert summary["aggregate"]["hand_matched"]["false_positives"] == 1
    assert summary["aggregate"]["hand_matched"]["false_negatives"] == 0
    assert (result_dir / "hand_matched" / "companies" / "microsoft" / "metrics.json").is_file()
    assert (result_dir / "hand_matched" / "summary.json").is_file()


def test_apply_hand_matches_rejects_reused_match_ids(tmp_path: Path):
    result_dir = tmp_path / "results" / "zero-shot" / "dev"
    company_dir = result_dir / "companies" / "microsoft"
    write_json(
        company_dir / "metrics.json",
        {
            "company": "Microsoft",
            "company_slug": "microsoft",
            "pipeline": "zero-shot",
            "split": "dev",
            "status": "evaluated",
            "true_positives": 3,
            "false_positives": 1,
            "false_negatives": 2,
        },
    )
    write_ambiguous_review_csv(company_dir / "unmatched_for_review.csv")

    summary = apply_hand_matches(result_dir)

    assert summary["aggregate"]["recovered_matches"] == 0
    group = summary["companies"][0]["match_groups"][0]
    assert group["status"] == "rejected"
    assert "exactly one gold row" in group["warning"]


def test_apply_hand_matches_cancel_keeps_existing_hand_matched_results(tmp_path: Path, monkeypatch):
    result_dir = tmp_path / "results" / "zero-shot" / "dev"
    output_root = result_dir / "hand_matched"
    stale_file = output_root / "summary.json"
    write_json(stale_file, {"old": True})
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    assert prepare_result_folder(output_root) is False
    assert stale_file.is_file()


def test_apply_hand_matches_assume_yes_keeps_existing_hand_matched_results_until_finalize(tmp_path: Path):
    result_dir = tmp_path / "results" / "zero-shot" / "dev"
    output_root = result_dir / "hand_matched"
    stale_file = output_root / "summary.json"
    write_json(stale_file, {"old": True})

    assert prepare_result_folder(output_root, assume_yes=True) is True
    assert stale_file.is_file()


def test_apply_hand_matches_finalize_replaces_existing_results_after_success(tmp_path: Path):
    result_dir = tmp_path / "results" / "zero-shot" / "dev"
    output_root = result_dir / "hand_matched"
    stale_file = output_root / "summary.json"
    write_json(stale_file, {"old": True})
    staging_dir = staging_result_folder(output_root)
    write_json(staging_dir / "summary.json", {"new": True})

    finalize_result_folder(staging_dir, output_root)

    assert not staging_dir.exists()
    assert json.loads(stale_file.read_text(encoding="utf-8")) == {"new": True}
