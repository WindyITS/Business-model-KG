import json
from pathlib import Path

from evaluation.scripts.evaluate import (
    annotation_reliability_payload,
    bootstrap_metrics,
    build_cherry_pick_evaluation_path,
    build_split_evaluation_paths,
    evaluate_paths,
    evaluate_triples,
    finalize_result_folder,
    prepare_result_folder,
    staging_result_folder,
)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_evaluate_triples_uses_normalized_edge_matching():
    gold = [
        {
            "subject": "Microsoft",
            "subject_type": "Company",
            "relation": "HAS_SEGMENT",
            "object": "Intelligent Cloud",
            "object_type": "BusinessSegment",
        },
        {
            "subject": "Intelligent Cloud",
            "subject_type": "BusinessSegment",
            "relation": "SERVES",
            "object": "developers",
            "object_type": "CustomerType",
        },
    ]
    predicted = [
        {
            "subject": " microsoft ",
            "subject_type": "Company",
            "relation": "HAS_SEGMENT",
            "object": "Intelligent   Cloud",
            "object_type": "BusinessSegment",
        },
        {
            "subject": "Intelligent Cloud",
            "subject_type": "Offering",
            "relation": "SERVES",
            "object": "developers",
            "object_type": "CustomerType",
        },
    ]

    result = evaluate_triples(gold, predicted)

    assert result["metrics"] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "relaxed_f1": 0.5,
    }


def test_evaluate_triples_is_case_insensitive_for_entity_values():
    gold = [
        {
            "subject": "LinkedIn",
            "subject_type": "Offering",
            "relation": "MONETIZES_VIA",
            "object": "subscription",
            "object_type": "RevenueModel",
        }
    ]
    predicted = [
        {
            "subject": "linkedin",
            "subject_type": "Offering",
            "relation": "MONETIZES_VIA",
            "object": "Subscription",
            "object_type": "RevenueModel",
        }
    ]

    assert evaluate_triples(gold, predicted)["metrics"]["f1"] == 1.0


def test_evaluate_triples_gives_partial_credit_for_hierarchy_alignment():
    gold = [
        {
            "subject": "Digital Media",
            "subject_type": "BusinessSegment",
            "relation": "OFFERS",
            "object": "Photoshop",
            "object_type": "Offering",
        },
        {
            "subject": "Photoshop",
            "subject_type": "Offering",
            "relation": "MONETIZES_VIA",
            "object": "subscription",
            "object_type": "RevenueModel",
        },
    ]
    predicted = [
        {
            "subject": "Digital Media",
            "subject_type": "BusinessSegment",
            "relation": "OFFERS",
            "object": "Creative Cloud",
            "object_type": "Offering",
        },
        {
            "subject": "Creative Cloud",
            "subject_type": "Offering",
            "relation": "OFFERS",
            "object": "Photoshop",
            "object_type": "Offering",
        },
        {
            "subject": "Creative Cloud",
            "subject_type": "Offering",
            "relation": "MONETIZES_VIA",
            "object": "subscription",
            "object_type": "RevenueModel",
        },
    ]

    result = evaluate_triples(gold, predicted)

    assert result["metrics"]["f1"] == 0.0
    assert result["metrics"]["relaxed_f1"] == 0.6


def test_split_evaluation_writes_pipeline_split_results(tmp_path: Path):
    evaluation_root = tmp_path / "evaluation"
    outputs_root = tmp_path / "outputs"
    write_jsonl(
        evaluation_root / "benchmarks" / "dev" / "clean" / "microsoft.jsonl",
        [
            {
                "subject": "Microsoft",
                "subject_type": "Company",
                "relation": "HAS_SEGMENT",
                "object": "Intelligent Cloud",
                "object_type": "BusinessSegment",
            }
        ],
    )
    write_json(
        outputs_root / "microsoft" / "zero-shot" / "latest" / "resolved_triples.json",
        {
            "triples": [
                {
                    "subject": "Microsoft",
                    "subject_type": "Company",
                    "relation": "HAS_SEGMENT",
                    "object": "Intelligent Cloud",
                    "object_type": "BusinessSegment",
                },
                {
                    "subject": "Microsoft",
                    "subject_type": "Company",
                    "relation": "OPERATES_IN",
                    "object": "Worldwide",
                    "object_type": "Place",
                },
            ]
        },
    )

    paths = build_split_evaluation_paths(
        root=evaluation_root,
        outputs_root=outputs_root,
        pipeline="zero-shot",
        split="dev",
    )
    summary = evaluate_paths(paths, output_root=evaluation_root / "results" / "zero-shot" / "dev")

    assert summary["aggregate"] == {
        "evaluated_company_count": 1,
        "missing_prediction_count": 0,
        "precision": 0.5,
        "recall": 1.0,
        "f1": 2 / 3,
        "macro_f1": 2 / 3,
        "relaxed_f1": 2 / 3,
    }
    assert (evaluation_root / "results" / "zero-shot" / "dev" / "summary.json").is_file()
    assert (
        evaluation_root
        / "results"
        / "zero-shot"
        / "dev"
        / "companies"
        / "microsoft"
        / "false_positives.jsonl"
    ).is_file()
    assert (
        evaluation_root
        / "results"
        / "zero-shot"
        / "dev"
        / "companies"
        / "microsoft"
        / "relaxed_matches.jsonl"
    ).is_file()


def test_cherry_pick_evaluation_uses_cherry_picked_result_folder(tmp_path: Path):
    evaluation_root = tmp_path / "evaluation"
    outputs_root = tmp_path / "outputs"
    write_jsonl(
        evaluation_root / "benchmarks" / "test" / "clean" / "microsoft.jsonl",
        [
            {
                "subject": "Microsoft",
                "subject_type": "Company",
                "relation": "HAS_SEGMENT",
                "object": "Intelligent Cloud",
                "object_type": "BusinessSegment",
            }
        ],
    )

    path = build_cherry_pick_evaluation_path(
        root=evaluation_root,
        outputs_root=outputs_root,
        pipeline="analyst",
        company="Microsoft",
    )

    assert path.output_dir == evaluation_root / "results" / "cherry_picked" / "analyst" / "microsoft"
    assert path.split is None


def test_prepare_result_folder_cancel_keeps_existing_files(tmp_path: Path, monkeypatch):
    result_dir = tmp_path / "results" / "zero-shot" / "dev"
    stale_file = result_dir / "summary.json"
    write_json(stale_file, {"old": True})
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    assert prepare_result_folder(result_dir) is False
    assert stale_file.is_file()


def test_prepare_result_folder_assume_yes_keeps_existing_files_until_finalize(tmp_path: Path):
    result_dir = tmp_path / "results" / "zero-shot" / "dev"
    stale_file = result_dir / "summary.json"
    write_json(stale_file, {"old": True})

    assert prepare_result_folder(result_dir, assume_yes=True) is True
    assert stale_file.is_file()


def test_finalize_result_folder_replaces_existing_results_after_success(tmp_path: Path):
    result_dir = tmp_path / "results" / "zero-shot" / "dev"
    stale_file = result_dir / "summary.json"
    write_json(stale_file, {"old": True})
    staging_dir = staging_result_folder(result_dir)
    write_json(staging_dir / "summary.json", {"new": True})

    finalize_result_folder(staging_dir, result_dir)

    assert not staging_dir.exists()
    assert json.loads(stale_file.read_text(encoding="utf-8")) == {"new": True}


def test_bootstrap_metrics_computes_paper_score_set(tmp_path: Path):
    evaluation_root = tmp_path / "evaluation"
    outputs_root = tmp_path / "outputs"
    gold = {
        "subject": "Acme",
        "subject_type": "Company",
        "relation": "HAS_SEGMENT",
        "object": "Cloud",
        "object_type": "BusinessSegment",
    }
    write_jsonl(evaluation_root / "benchmarks" / "test" / "clean" / "alpha.jsonl", [gold])
    write_jsonl(evaluation_root / "benchmarks" / "test" / "clean" / "beta.jsonl", [gold])
    write_json(outputs_root / "alpha" / "zero-shot" / "latest" / "resolved_triples.json", {"triples": [gold]})
    write_json(outputs_root / "beta" / "zero-shot" / "latest" / "resolved_triples.json", {"triples": []})

    payload = bootstrap_metrics(
        root=evaluation_root,
        outputs_root=outputs_root,
        split="test",
        pipelines=["zero-shot"],
        companies=["alpha", "beta"],
        n_bootstrap=10,
        seed=7,
    )

    assert payload["point_estimates"]["zero-shot"] == {
        "precision": 1.0,
        "recall": 0.5,
        "f1": 2 / 3,
        "macro_f1": 0.5,
        "relaxed_f1": 2 / 3,
    }
    assert set(payload["confidence_intervals"]["zero-shot"]) == {
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "relaxed_f1",
    }


def test_annotation_reliability_payload_uses_jsonl_inputs(tmp_path: Path):
    evaluation_root = tmp_path / "evaluation"
    source_dir = evaluation_root / "benchmarks" / "annotation_reliability"
    write_jsonl(
        source_dir / "amazon_inter_annotator_edges.jsonl",
        [
            {"annotator": "official", "subject": "Amazon", "relation": "HAS_SEGMENT", "object": "AWS"},
            {"annotator": "luca", "subject": "Amazon", "relation": "HAS_SEGMENT", "object": "AWS"},
            {"annotator": "zhong", "subject": "Amazon", "relation": "HAS_SEGMENT", "object": "AWS"},
            {"annotator": "official", "subject": "AWS", "relation": "SERVES", "object": "developers"},
            {"annotator": "luca", "subject": "AWS", "relation": "SERVES", "object": "developers"},
            {"annotator": "zhong", "subject": "AWS", "relation": "SERVES", "object": "enterprises"},
        ],
    )
    write_jsonl(
        source_dir / "intra_annotator_counts.jsonl",
        [
            {
                "label": "Combined micro",
                "true_positives": 1,
                "false_positives": 0,
                "false_negatives": 1,
                "precision": 1.0,
                "recall": 0.5,
                "f1": 2 / 3,
                "jaccard": 0.5,
            },
            {
                "label": "Macro average",
                "true_positives": None,
                "false_positives": None,
                "false_negatives": None,
                "precision": 1.0,
                "recall": 0.5,
                "f1": 2 / 3,
                "jaccard": 0.5,
            },
        ],
    )

    payload = annotation_reliability_payload(evaluation_root)

    inter = payload["summary"]["inter_annotator_amazon"]
    assert inter["official_edges"] == 2
    assert inter["candidate_edges"] == 3
    assert inter["unanimous"] == 1
    assert inter["majority_only"] == 1
    assert inter["single_annotator"] == 1
    assert payload["summary"]["intra_annotator"]["combined_micro"]["f1"] == 2 / 3
