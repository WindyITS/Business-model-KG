import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kg_query_planner_ft.local_stack import run_local_stack


class LocalStackTests(unittest.TestCase):
    def _write_config(self, tmp: Path) -> Path:
        config_path = tmp / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "env_root": str(tmp / "env"),
                    "artifact_root": str(tmp / "artifacts"),
                    "dataset_path": str(tmp / "dataset"),
                    "router": {"base_model": "microsoft/deberta-v3-small"},
                    "planner": {"base_model": "Qwen/Qwen3-4B-Instruct"},
                }
            ),
            encoding="utf-8",
        )
        thresholds_dir = tmp / "artifacts" / "router" / "eval"
        thresholds_dir.mkdir(parents=True, exist_ok=True)
        (thresholds_dir / "thresholds.json").write_text(
            json.dumps(
                {
                    "local_threshold": {"threshold": 0.97},
                    "refuse_threshold": {"threshold": 0.95},
                    "planner_gate_open": True,
                }
            ),
            encoding="utf-8",
        )
        return config_path

    def test_local_stack_runs_planner_for_local_routes(self):
        with TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            result = run_local_stack(
                "Which companies partner with Dell?",
                str(config_path),
                router_predictor=lambda _: {"local": 0.99, "refuse": 0.01, "api_fallback": 0.0},
                planner_runner=lambda _: {
                    "generated_text": "{\"answerable\":true}",
                    "plan": {"answerable": True, "family": "companies_by_partner", "payload": {"partners": ["Dell"]}},
                    "compiled": {"cypher": "RETURN 1", "params": {}},
                },
            )
            self.assertEqual(result["decision"], "local")
            self.assertEqual(result["plan"]["family"], "companies_by_partner")

    def test_local_stack_downgrades_failed_planner_runs(self):
        with TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            result = run_local_stack(
                "Which companies partner with Dell?",
                str(config_path),
                router_predictor=lambda _: {"local": 0.99, "refuse": 0.01, "api_fallback": 0.0},
                planner_runner=lambda _: (_ for _ in ()).throw(ValueError("invalid json")),
            )
            self.assertEqual(result["decision"], "api_fallback")
            self.assertIn("invalid json", result["planner"]["error"])

    def test_local_stack_skips_planner_for_refuse(self):
        with TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            result = run_local_stack(
                "Delete Aurora.",
                str(config_path),
                router_predictor=lambda _: {"local": 0.10, "refuse": 0.99, "api_fallback": 0.0},
                planner_runner=lambda _: self.fail("planner should not run for refuse"),
            )
            self.assertEqual(result["decision"], "refuse")
            self.assertIsNone(result["planner"])

    def test_local_stack_respects_closed_planner_gate(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_path = self._write_config(tmp)
            thresholds_path = tmp / "artifacts" / "router" / "eval" / "thresholds.json"
            thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
            thresholds["planner_gate_open"] = False
            thresholds_path.write_text(json.dumps(thresholds), encoding="utf-8")

            result = run_local_stack(
                "Which companies partner with Dell?",
                str(config_path),
                router_predictor=lambda _: {"local": 0.99, "refuse": 0.01, "api_fallback": 0.0},
                planner_runner=lambda _: self.fail("planner should not run when the gate is closed"),
            )

            self.assertEqual(result["decision"], "api_fallback")
            self.assertEqual(result["router"]["gate_reason"], "planner_gate_closed")

    def test_local_stack_default_router_predictor_uses_router_eval_helper(self):
        with TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            with patch("kg_query_planner_ft.local_stack.predict_router_probabilities") as mock_predict:
                mock_predict.return_value = {"local": 0.10, "refuse": 0.99, "api_fallback": 0.0}
                result = run_local_stack(
                    "Delete Aurora.",
                    str(config_path),
                    planner_runner=lambda _: self.fail("planner should not run for refuse"),
                )

            mock_predict.assert_called_once()
            self.assertEqual(result["decision"], "refuse")


if __name__ == "__main__":
    unittest.main()
