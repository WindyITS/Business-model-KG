import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from text2cypher.dataset.v2.builder import DatasetBuildError, build_dataset, load_dataset_specs, write_dataset
from text2cypher.dataset.v2.models import (
    DatasetSpec,
    FixtureEdgeSpec,
    FixtureNodeSpec,
    FixtureSpec,
    ResultColumnSpec,
    SourceExampleSpec,
)


def _fixture(fixture_id: str) -> FixtureSpec:
    return FixtureSpec(
        fixture_id=fixture_id,
        graph_id=fixture_id,
        graph_purpose="Minimal fixture for dataset builder tests.",
        covered_families=["QF01", "QF99"],
        nodes=[
            FixtureNodeSpec(node_id="company", label="Company", name="Acme Systems"),
            FixtureNodeSpec(
                node_id="segment",
                label="BusinessSegment",
                name="Core Platform",
                properties={"company_name": "Acme Systems"},
            ),
        ],
        edges=[FixtureEdgeSpec(source="company", type="HAS_SEGMENT", target="segment")],
        invariants_satisfied=["company-scoped segment fixture"],
        authoring_notes=["kept intentionally tiny for serialization tests"],
    )


def _answerable_example(
    *,
    example_id: str,
    intent_id: str,
    family_id: str,
    fixture_id: str,
    question: str,
    split: str,
) -> SourceExampleSpec:
    return SourceExampleSpec(
        example_id=example_id,
        intent_id=intent_id,
        family_id=family_id,
        fixture_id=fixture_id,
        graph_id=fixture_id,
        binding_id="acme",
        question_canonical=question,
        gold_cypher=(
            "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->"
            "(s:BusinessSegment {company_name: $company}) RETURN COUNT(DISTINCT s) AS segment_count"
        ),
        params={"company": "Acme Systems"},
        answerable=True,
        refusal_reason=None,
        result_shape=[ResultColumnSpec(column="segment_count", type="integer")],
        difficulty="low",
        split=split,
        paraphrases=(),
    )


def _refusal_example(
    *,
    example_id: str,
    intent_id: str,
    family_id: str,
    fixture_id: str,
    question: str,
    split: str,
) -> SourceExampleSpec:
    return SourceExampleSpec(
        example_id=example_id,
        intent_id=intent_id,
        family_id=family_id,
        fixture_id=fixture_id,
        graph_id=fixture_id,
        binding_id="acme",
        question_canonical=question,
        gold_cypher=None,
        params={},
        answerable=False,
        refusal_reason="not_in_graph",
        result_shape=None,
        difficulty="low",
        split=split,
        paraphrases=(),
    )


def _module_map(*pairs: tuple[str, DatasetSpec]) -> dict[str, SimpleNamespace]:
    return {module_name: SimpleNamespace(build_spec=lambda spec=spec: spec) for module_name, spec in pairs}


class Text2CypherDatasetBuilderTests(unittest.TestCase):
    def test_write_dataset_emits_chat_style_messages_rows(self):
        spec = DatasetSpec(
            fixtures=[_fixture("fx_builder_messages")],
            source_examples=[
                _answerable_example(
                    example_id="qf01_answerable",
                    intent_id="qf01_answerable",
                    family_id="QF01",
                    fixture_id="fx_builder_messages",
                    question="How many business segments does Acme Systems have?",
                    split="train",
                ),
                _refusal_example(
                    example_id="qf99_refusal",
                    intent_id="qf99_refusal",
                    family_id="QF99",
                    fixture_id="fx_builder_messages",
                    question="What filings does Acme Systems publish?",
                    split="train",
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "dataset"
            dataset = build_dataset(spec, output_root)
            write_dataset(dataset, output_root)

            messages_path = output_root / "training" / "messages.jsonl"
            self.assertTrue(messages_path.exists())

            rows = [
                json.loads(line)
                for line in messages_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 2)

            answerable_row = next(row for row in rows if row["metadata"]["answerable"])
            refusal_row = next(row for row in rows if not row["metadata"]["answerable"])

            self.assertEqual([message["role"] for message in answerable_row["messages"]], ["system", "user", "assistant"])
            self.assertIn("Cypher", answerable_row["messages"][0]["content"])
            self.assertIn("JSON", answerable_row["messages"][0]["content"])
            self.assertEqual(
                answerable_row["messages"][1]["content"],
                "How many business segments does Acme Systems have?",
            )
            self.assertEqual(
                json.loads(answerable_row["messages"][2]["content"]),
                {
                    "answerable": True,
                    "cypher": (
                        "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->"
                        "(s:BusinessSegment {company_name: $company}) RETURN COUNT(DISTINCT s) AS segment_count"
                    ),
                    "params": {"company": "Acme Systems"},
                },
            )
            self.assertEqual(answerable_row["metadata"]["example_id"], "qf01_answerable")
            self.assertEqual(answerable_row["metadata"]["intent_id"], "qf01_answerable")
            self.assertEqual(answerable_row["metadata"]["answerable"], True)

            self.assertEqual([message["role"] for message in refusal_row["messages"]], ["system", "user", "assistant"])
            self.assertEqual(
                json.loads(refusal_row["messages"][2]["content"]),
                {"answerable": False, "reason": "not_in_graph"},
            )
            self.assertEqual(refusal_row["metadata"]["answerable"], False)
            self.assertEqual(refusal_row["metadata"]["example_id"], "qf99_refusal")

    def test_build_dataset_requires_train_coverage_for_each_family(self):
        spec = DatasetSpec(
            fixtures=[_fixture("fx_builder_coverage")],
            source_examples=[
                _answerable_example(
                    example_id="qf99_test_only",
                    intent_id="qf99_test_only",
                    family_id="QF99",
                    fixture_id="fx_builder_coverage",
                    question="How many business segments does Acme Systems have in the test split?",
                    split="test",
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "dataset"
            modules = _module_map(("module_coverage", spec))
            with patch("text2cypher.dataset.v2.builder.importlib.import_module", side_effect=lambda name: modules[name]):
                with self.assertRaisesRegex(DatasetBuildError, r"train|coverage"):
                    loaded_spec = load_dataset_specs(["module_coverage"])
                    build_dataset(loaded_spec, output_root)

    def test_load_dataset_specs_rejects_duplicate_question_conflicts(self):
        first = DatasetSpec(
            fixtures=[_fixture("fx_builder_duplicate_a")],
            source_examples=[
                _answerable_example(
                    example_id="qf01_duplicate_a",
                    intent_id="qf01_duplicate_a",
                    family_id="QF01",
                    fixture_id="fx_builder_duplicate_a",
                    question="Does Acme Systems have a core platform segment?",
                    split="train",
                )
            ],
        )
        second = DatasetSpec(
            fixtures=[_fixture("fx_builder_duplicate_b")],
            source_examples=[
                _answerable_example(
                    example_id="qf02_duplicate_b",
                    intent_id="qf02_duplicate_b",
                    family_id="QF02",
                    fixture_id="fx_builder_duplicate_b",
                    question="Does Acme Systems have a core platform segment?",
                    split="test",
                )
            ],
        )

        modules = _module_map(("module_a", first), ("module_b", second))

        with patch("text2cypher.dataset.v2.builder.importlib.import_module", side_effect=lambda name: modules[name]):
            with self.assertRaisesRegex(DatasetBuildError, r"duplicate question|conflict"):
                spec = load_dataset_specs(["module_a", "module_b"])
                with tempfile.TemporaryDirectory() as tmp_dir:
                    build_dataset(spec, Path(tmp_dir) / "dataset")

    def test_write_dataset_with_heldout_split_emits_separate_evaluation_files(self):
        spec = DatasetSpec(
            fixtures=[_fixture("fx_builder_train"), _fixture("fx_builder_heldout")],
            source_examples=[
                _answerable_example(
                    example_id="qf01_train_row",
                    intent_id="qf01_train_row",
                    family_id="QF01",
                    fixture_id="fx_builder_train",
                    question="How many business segments does Acme Systems have in the train pool?",
                    split="train",
                ),
                _answerable_example(
                    example_id="qf99_heldout_row",
                    intent_id="qf99_heldout_row",
                    family_id="QF01",
                    fixture_id="fx_builder_heldout",
                    question="How many business segments does Acme Systems have in the held-out set?",
                    split="heldout_test",
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "dataset"
            dataset = build_dataset(spec, output_root)
            write_dataset(dataset, output_root)

            self.assertTrue((output_root / "training" / "train_messages.jsonl").exists())
            self.assertTrue((output_root / "evaluation" / "test_messages.jsonl").exists())
            self.assertTrue((output_root / "reports" / "heldout_test_manifest.json").exists())
            self.assertTrue((output_root / "reports" / "leakage_report.json").exists())

            train_rows = [
                json.loads(line)
                for line in (output_root / "training" / "train_messages.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            heldout_rows = [
                json.loads(line)
                for line in (output_root / "evaluation" / "test_messages.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(train_rows), 1)
            self.assertEqual(len(heldout_rows), 1)
            self.assertEqual(train_rows[0]["split"], "train")
            self.assertEqual(heldout_rows[0]["split"], "heldout_test")

    def test_build_dataset_rejects_question_overlap_between_train_and_heldout(self):
        spec = DatasetSpec(
            fixtures=[_fixture("fx_builder_train_overlap"), _fixture("fx_builder_heldout_overlap")],
            source_examples=[
                _answerable_example(
                    example_id="qf01_train_overlap",
                    intent_id="qf01_train_overlap",
                    family_id="QF01",
                    fixture_id="fx_builder_train_overlap",
                    question="Which segments does Acme Systems have?",
                    split="train",
                ),
                _answerable_example(
                    example_id="qf99_heldout_overlap",
                    intent_id="qf99_heldout_overlap",
                    family_id="QF01",
                    fixture_id="fx_builder_heldout_overlap",
                    question="Which segments does Acme Systems have?",
                    split="heldout_test",
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(DatasetBuildError, r"leaks into the training pool|leakage"):
                build_dataset(spec, Path(tmp_dir) / "dataset")


if __name__ == "__main__":
    unittest.main()
