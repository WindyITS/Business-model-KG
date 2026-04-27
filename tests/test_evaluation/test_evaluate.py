import json
from pathlib import Path

from evaluation.scripts.evaluate import (
    build_cherry_pick_evaluation_path,
    build_split_evaluation_paths,
    evaluate_paths,
    evaluate_triples,
    prepare_result_folder,
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


def test_prepare_result_folder_assume_yes_removes_existing_files(tmp_path: Path):
    result_dir = tmp_path / "results" / "zero-shot" / "dev"
    stale_file = result_dir / "summary.json"
    write_json(stale_file, {"old": True})

    assert prepare_result_folder(result_dir, assume_yes=True) is True
    assert not result_dir.exists()
