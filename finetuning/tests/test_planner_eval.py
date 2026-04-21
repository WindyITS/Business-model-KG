import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kg_query_planner_ft.planner_eval import evaluate_planner


class PlannerEvalTests(unittest.TestCase):
    def test_base_only_eval_uses_no_adapter_and_writes_separate_artifacts(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            artifact_root = tmp / "artifacts"
            prepared_dir = artifact_root / "prepared" / "planner" / "raw"
            prepared_dir.mkdir(parents=True, exist_ok=True)

            row = {
                "question": "Which companies partner with Dell?",
                "family": "companies_by_partner",
                "gold_plan": {
                    "answerable": True,
                    "family": "companies_by_partner",
                    "payload": {"partners": ["Dell"]},
                },
            }
            for name in ("valid", "test"):
                (prepared_dir / f"{name}.jsonl").write_text(
                    json.dumps(row) + "\n",
                    encoding="utf-8",
                )

            config_path = tmp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "artifact_root": str(artifact_root),
                        "dataset_path": str(tmp / "dataset"),
                        "router": {"base_model": "microsoft/deberta-v3-small"},
                        "planner": {"base_model": "mlx-community/Qwen3-4B-Instruct-2507-4bit"},
                    }
                ),
                encoding="utf-8",
            )

            metrics = {
                "count": 1,
                "json_parse_rate": 1.0,
                "contract_valid_rate": 1.0,
                "family_accuracy": 1.0,
                "exact_plan_match_rate": 1.0,
                "per_family": {},
            }
            predictions = [{"question": row["question"], "generated_text": "{\"answerable\":true}"}]

            with (
                patch("kg_query_planner_ft.planner_eval.PlannerGenerator") as mock_generator,
                patch("kg_query_planner_ft.planner_eval._evaluate_split", return_value=(metrics, predictions)),
            ):
                summary = evaluate_planner(str(config_path), base_only=True)

            mock_generator.assert_called_once_with(
                model_path="mlx-community/Qwen3-4B-Instruct-2507-4bit",
                adapter_path=None,
            )
            self.assertEqual(summary["mode"], "base_model")
            self.assertIsNone(summary["adapter_path"])
            self.assertTrue((artifact_root / "planner" / "eval" / "base_model" / "summary.json").exists())
            self.assertFalse((artifact_root / "planner" / "eval" / "summary.json").exists())

    def test_lmstudio_eval_uses_served_model_and_writes_model_scoped_artifacts(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            artifact_root = tmp / "artifacts"
            prepared_dir = artifact_root / "prepared" / "planner" / "raw"
            prepared_dir.mkdir(parents=True, exist_ok=True)

            row = {
                "question": "Which companies partner with Dell?",
                "family": "companies_by_partner",
                "gold_plan": {
                    "answerable": True,
                    "family": "companies_by_partner",
                    "payload": {"partners": ["Dell"]},
                },
            }
            for name in ("valid", "test"):
                (prepared_dir / f"{name}.jsonl").write_text(
                    json.dumps(row) + "\n",
                    encoding="utf-8",
                )

            config_path = tmp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "artifact_root": str(artifact_root),
                        "dataset_path": str(tmp / "dataset"),
                        "router": {"base_model": "microsoft/deberta-v3-small"},
                        "planner": {"base_model": "mlx-community/Qwen3-4B-Instruct-2507-4bit"},
                    }
                ),
                encoding="utf-8",
            )

            metrics = {
                "count": 1,
                "json_parse_rate": 1.0,
                "contract_valid_rate": 1.0,
                "family_accuracy": 1.0,
                "exact_plan_match_rate": 1.0,
                "per_family": {},
            }
            predictions = [{"question": row["question"], "generated_text": "{\"answerable\":true}"}]

            with (
                patch("kg_query_planner_ft.planner_eval.LMStudioPlannerGenerator") as mock_generator,
                patch("kg_query_planner_ft.planner_eval._evaluate_split", return_value=(metrics, predictions)),
            ):
                summary = evaluate_planner(
                    str(config_path),
                    backend="lmstudio",
                    lmstudio_model="Qwen3-32B-Instruct",
                    lmstudio_base_url="http://localhost:1234/v1",
                    lmstudio_api_key="lm-studio",
                )

            mock_generator.assert_called_once_with(
                model_name="Qwen3-32B-Instruct",
                base_url="http://localhost:1234/v1",
                api_key="lm-studio",
            )
            self.assertEqual(summary["backend"], "lmstudio")
            self.assertEqual(summary["mode"], "lmstudio_model")
            self.assertIsNone(summary["adapter_path"])
            self.assertEqual(summary["served_model"], "Qwen3-32B-Instruct")
            self.assertTrue(
                (
                    artifact_root
                    / "planner"
                    / "eval"
                    / "lmstudio"
                    / "Qwen3-32B-Instruct"
                    / "summary.json"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
