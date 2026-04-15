import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "export_hf_text2cypher_dataset.py"

spec_root = ROOT / "src"
sys.path.insert(0, str(spec_root))


def _load_export_module():
    spec = importlib.util.spec_from_file_location("export_hf_text2cypher_dataset", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row) for row in rows)
    path.write_text(payload + "\n", encoding="utf-8")


class ExportText2CypherDatasetTests(unittest.TestCase):
    def test_export_preserves_derived_messages_layer(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
            tmp_root = Path(tmp_dir)
            dataset_root = tmp_root / "datasets" / "text2cypher" / "v3"
            packaging_root = tmp_root / "packaging" / "huggingface" / "text2cypher-v3"
            output_root = tmp_root / "dist" / "huggingface" / "text2cypher-v3"

            _write_jsonl(
                dataset_root / "source" / "fixture_instances.jsonl",
                [{"fixture_id": "fx_export", "graph_id": "fx_export"}],
            )
            _write_jsonl(
                dataset_root / "source" / "bound_seed_examples.jsonl",
                [
                    {
                        "example_id": "example_1",
                        "intent_id": "qf01_export",
                        "family_id": "QF01",
                        "answerable": True,
                    }
                ],
            )
            _write_jsonl(
                dataset_root / "training" / "training_examples.jsonl",
                [
                    {
                        "training_example_id": "example_1__v00",
                        "intent_id": "qf01_export",
                        "difficulty": "low",
                    }
                ],
            )
            _write_jsonl(
                dataset_root / "training" / "messages.jsonl",
                [
                    {
                        "messages": [
                            {"role": "system", "content": "Translate to Cypher and return JSON only."},
                            {"role": "user", "content": "How many business segments does Acme Systems have?"},
                            {
                                "role": "assistant",
                                "content": json.dumps(
                                    {
                                        "answerable": True,
                                        "cypher": "MATCH ...",
                                        "params": {"company": "Acme Systems"},
                                    }
                                ),
                            },
                        ],
                        "metadata": {
                            "example_id": "example_1",
                            "intent_id": "qf01_export",
                            "answerable": True,
                        },
                    }
                ],
            )
            _write_jsonl(
                dataset_root / "training" / "train.jsonl",
                [{"training_example_id": "example_1__v00"}],
            )
            _write_jsonl(dataset_root / "training" / "dev.jsonl", [])
            _write_jsonl(dataset_root / "training" / "test.jsonl", [])
            _write_jsonl(
                dataset_root / "evaluation" / "test_messages.jsonl",
                [
                    {
                        "messages": [
                            {"role": "system", "content": "Translate to Cypher and return JSON only."},
                            {"role": "user", "content": "Which companies operate in Iberia and monetize via subscription?"},
                            {
                                "role": "assistant",
                                "content": json.dumps(
                                    {
                                        "answerable": True,
                                        "cypher": "MATCH ...",
                                        "params": {"place": "Iberia", "revenue_model": "subscription"},
                                    }
                                ),
                            },
                        ],
                        "metadata": {
                            "example_id": "heldout_1",
                            "intent_id": "heldout_qf31_export",
                            "answerable": True,
                        },
                    }
                ],
            )
            _write_jsonl(
                dataset_root / "evaluation" / "test_examples.jsonl",
                [{"training_example_id": "heldout_1__v00"}],
            )
            (dataset_root / "reports").mkdir(parents=True, exist_ok=True)
            (dataset_root / "reports" / "training_split_manifest.json").write_text(
                json.dumps(
                    {
                        "dataset_path": str(dataset_root / "training" / "training_examples.jsonl"),
                        "split_counts": {"train": {"rows": 1, "intents": 1, "source_examples": 1}},
                        "family_intent_split_counts": {"QF01": {"train": 1}},
                        "intent_split_map": {"qf01_export": "train"},
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (dataset_root / "reports" / "bound_seed_validation_report.json").write_text(
                json.dumps({"summary": {"passed": 1, "failed": 0}}),
                encoding="utf-8",
            )
            (dataset_root / "reports" / "heldout_test_manifest.json").write_text(
                json.dumps(
                    {
                        "dataset_path": str(dataset_root / "evaluation" / "test_messages.jsonl"),
                        "split_counts": {
                            "heldout_test": {"rows": 1, "answerable_rows": 1, "refusal_rows": 0}
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            packaging_root.mkdir(parents=True, exist_ok=True)
            (packaging_root / "README.md").write_text("# Text2Cypher v3\n", encoding="utf-8")
            (packaging_root / ".gitattributes").write_text("* text=auto\n", encoding="utf-8")
            (packaging_root / "UPLOAD.md").write_text("Upload notes.\n", encoding="utf-8")

            module = _load_export_module()
            args = SimpleNamespace(
                dataset_root=dataset_root,
                packaging_root=packaging_root,
                output_root=output_root,
                force=True,
            )

            with patch.object(module, "parse_args", return_value=args):
                exit_code = module.main()

            self.assertEqual(exit_code, 0)
            exported_messages = output_root / "training" / "messages.jsonl"
            self.assertTrue(exported_messages.exists())
            self.assertTrue((output_root / "evaluation" / "test_messages.jsonl").exists())
            exported_rows = [
                json.loads(line)
                for line in exported_messages.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(exported_rows[0]["metadata"]["intent_id"], "qf01_export")
            self.assertEqual(
                exported_rows[0]["messages"][2]["content"],
                json.dumps(
                    {
                        "answerable": True,
                        "cypher": "MATCH ...",
                        "params": {"company": "Acme Systems"},
                    }
                ),
            )
            summary = json.loads((output_root / "release_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["training_examples"], 1)
            self.assertEqual(summary["heldout_message_examples"], 1)


if __name__ == "__main__":
    unittest.main()
