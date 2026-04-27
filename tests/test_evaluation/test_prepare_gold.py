import json
from pathlib import Path

from evaluation.scripts.prepare_gold import convert_benchmarks


def test_convert_benchmarks_writes_clean_jsonl_and_manifest(tmp_path: Path):
    raw_dir = tmp_path / "benchmarks" / "dev" / "raw"
    raw_dir.mkdir(parents=True)
    (tmp_path / "benchmarks" / "dev" / "clean").mkdir(parents=True)
    (tmp_path / "benchmarks" / "test" / "raw").mkdir(parents=True)
    (tmp_path / "benchmarks" / "test" / "clean").mkdir(parents=True)

    (raw_dir / "microsoft.csv").write_text(
        "\n".join(
            [
                "subject,subject_type,relation,object,object_type",
                "Microsoft,Company,HAS_SEGMENT,Intelligent Cloud,BusinessSegment",
                "Intelligent Cloud,BusinessSegment,SERVES,developers,CustomerType",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summaries = convert_benchmarks(tmp_path, ["dev"])

    assert len(summaries) == 1
    assert summaries[0].row_count == 2

    clean_path = tmp_path / "benchmarks" / "dev" / "clean" / "microsoft.jsonl"
    rows = [json.loads(line) for line in clean_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
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

    manifest = json.loads((tmp_path / "benchmarks" / "dev" / "clean" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"] == "dev"
    assert manifest["file_count"] == 1
    assert manifest["triple_count"] == 2


def test_convert_benchmarks_accepts_case_insensitive_headers(tmp_path: Path):
    raw_dir = tmp_path / "benchmarks" / "test" / "raw"
    raw_dir.mkdir(parents=True)
    (tmp_path / "benchmarks" / "test" / "clean").mkdir(parents=True)

    (raw_dir / "apple.csv").write_text(
        "Subject,Subject_Type,Relation,Object,Object_Type\nApple,Company,OPERATES_IN,Worldwide,Place\n",
        encoding="utf-8",
    )

    convert_benchmarks(tmp_path, ["test"])

    clean_path = tmp_path / "benchmarks" / "test" / "clean" / "apple.jsonl"
    assert json.loads(clean_path.read_text(encoding="utf-8")) == {
        "subject": "Apple",
        "subject_type": "Company",
        "relation": "OPERATES_IN",
        "object": "Worldwide",
        "object_type": "Place",
    }
