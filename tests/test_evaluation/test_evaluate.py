import json
from pathlib import Path

from evaluation.scripts.evaluate import (
    build_cherry_pick_evaluation_path,
    build_split_evaluation_paths,
    evaluate_paths,
    evaluate_triples,
    finalize_result_folder,
    generate_alias_candidates,
    load_aliases,
    prepare_result_folder,
    staging_result_folder,
)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_evaluate_triples_uses_strict_normalized_typed_matching():
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

    assert result["metrics"]["true_positives"] == 1
    assert result["metrics"]["false_positives"] == 1
    assert result["metrics"]["false_negatives"] == 1
    assert result["metrics"]["precision"] == 0.5
    assert result["metrics"]["recall"] == 0.5
    assert result["metrics"]["f1"] == 0.5


def test_evaluate_triples_applies_approved_aliases():
    gold = [
        {
            "subject": "Intelligent Cloud",
            "subject_type": "BusinessSegment",
            "relation": "OFFERS",
            "object": "Azure and other cloud services",
            "object_type": "Offering",
        }
    ]
    predicted = [
        {
            "subject": "Intelligent Cloud",
            "subject_type": "BusinessSegment",
            "relation": "OFFERS",
            "object": "Azure",
            "object_type": "Offering",
        }
    ]

    strict = evaluate_triples(gold, predicted)
    relaxed = evaluate_triples(gold, predicted, aliases={"Offering": {"azure": "Azure and other cloud services"}})

    assert strict["metrics"]["true_positives"] == 0
    assert relaxed["metrics"]["true_positives"] == 1


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

    assert evaluate_triples(gold, predicted)["metrics"]["true_positives"] == 1


def test_generate_alias_candidates_pairs_compatible_unmatched_triples():
    false_positives = [
        {
            "subject": "Intelligent Cloud",
            "subject_type": "BusinessSegment",
            "relation": "OFFERS",
            "object": "Azure",
            "object_type": "Offering",
        }
    ]
    false_negatives = [
        {
            "subject": "Intelligent Cloud",
            "subject_type": "BusinessSegment",
            "relation": "OFFERS",
            "object": "Azure and other cloud services",
            "object_type": "Offering",
        }
    ]

    candidates = generate_alias_candidates(
        false_positives=false_positives,
        false_negatives=false_negatives,
        min_score=0.1,
    )

    assert len(candidates) == 1
    assert candidates[0]["candidate_aliases"][0]["node_type"] == "Offering"
    assert candidates[0]["candidate_aliases"][0]["predicted_value"] == "Azure"
    assert candidates[0]["candidate_aliases"][0]["gold_value"] == "Azure and other cloud services"


def test_load_aliases_normalizes_alias_keys(tmp_path: Path):
    alias_path = tmp_path / "aliases.json"
    alias_path.write_text(
        json.dumps({"Offering": {" Azure ": "Azure and other cloud services"}}),
        encoding="utf-8",
    )

    assert load_aliases(alias_path) == {"Offering": {"azure": "Azure and other cloud services"}}


def test_load_aliases_missing_file_raises(tmp_path: Path):
    missing_alias_path = tmp_path / "missing_aliases.json"

    try:
        load_aliases(missing_alias_path)
    except FileNotFoundError as exc:
        assert str(missing_alias_path) in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing alias file")


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

    assert summary["aggregate"]["true_positives"] == 1
    assert summary["aggregate"]["false_positives"] == 1
    assert summary["aggregate"]["false_negatives"] == 0
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
        / "unmatched_for_review.csv"
    ).is_file()


def test_split_evaluation_writes_alias_outputs_when_aliases_are_provided(tmp_path: Path):
    evaluation_root = tmp_path / "evaluation"
    outputs_root = tmp_path / "outputs"
    write_jsonl(
        evaluation_root / "benchmarks" / "dev" / "clean" / "microsoft.jsonl",
        [
            {
                "subject": "Intelligent Cloud",
                "subject_type": "BusinessSegment",
                "relation": "OFFERS",
                "object": "Azure and other cloud services",
                "object_type": "Offering",
            }
        ],
    )
    write_json(
        outputs_root / "microsoft" / "zero-shot" / "latest" / "resolved_triples.json",
        {
            "triples": [
                {
                    "subject": "Intelligent Cloud",
                    "subject_type": "BusinessSegment",
                    "relation": "OFFERS",
                    "object": "Azure",
                    "object_type": "Offering",
                }
            ]
        },
    )

    paths = build_split_evaluation_paths(
        root=evaluation_root,
        outputs_root=outputs_root,
        pipeline="zero-shot",
        split="dev",
    )
    summary = evaluate_paths(
        paths,
        output_root=evaluation_root / "results" / "zero-shot" / "dev",
        aliases={"Offering": {"azure": "Azure and other cloud services"}},
    )

    assert summary["aggregate"]["true_positives"] == 0
    assert summary["alias_normalized_aggregate"]["true_positives"] == 1
    assert (
        evaluation_root
        / "results"
        / "zero-shot"
        / "dev"
        / "companies"
        / "microsoft"
        / "alias_candidates.jsonl"
    ).is_file()
    assert (
        evaluation_root
        / "results"
        / "zero-shot"
        / "dev"
        / "companies"
        / "microsoft"
        / "alias_normalized_matched.jsonl"
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
