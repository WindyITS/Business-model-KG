import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from runtime.local_query_stack import run_local_query_stack


class LocalQueryStackTests(unittest.TestCase):
    def _write_bundle(self, root_dir: Path, *, planner_gate_open: bool = True) -> Path:
        bundle_dir = root_dir / "runtime_assets" / "query_stack" / "current"
        (bundle_dir / "router" / "model").mkdir(parents=True, exist_ok=True)
        (bundle_dir / "planner" / "adapter").mkdir(parents=True, exist_ok=True)
        (bundle_dir / "router" / "thresholds.json").write_text(
            json.dumps(
                {
                    "local_threshold": {"threshold": 0.97},
                    "refuse_threshold": {"threshold": 0.95},
                    "planner_gate_open": planner_gate_open,
                    "temperature": 1.0,
                }
            ),
            encoding="utf-8",
        )
        (bundle_dir / "planner" / "system_prompt.txt").write_text("Planner prompt\n", encoding="utf-8")
        (bundle_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "bundle_format_version": 1,
                    "router": {
                        "model_dir": "router/model",
                        "thresholds_path": "router/thresholds.json",
                        "base_model": "microsoft/deberta-v3-small",
                        "max_length": 256,
                    },
                    "planner": {
                        "base_model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
                        "adapter_dir": "planner/adapter",
                        "max_tokens": 256,
                        "system_prompt_path": "planner/system_prompt.txt",
                    },
                }
            ),
            encoding="utf-8",
        )
        return bundle_dir

    def test_run_local_query_stack_returns_compiled_query_for_local_route(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = self._write_bundle(Path(tmp_dir))
            router = Mock()
            router.predict.return_value = {"local": 0.99, "refuse": 0.01, "api_fallback": 0.0}
            planner = Mock()
            planner.generate.return_value = (
                'preface {"answerable": true, "family": "companies_by_partner", "payload": {"partners": ["Dell"]}} suffix'
            )

            with patch("runtime.local_query_stack._router_predictor_for", return_value=router), patch(
                "runtime.local_query_stack._planner_generator_for",
                return_value=planner,
            ):
                result = run_local_query_stack("Which companies partner with Dell?", bundle_dir=bundle_dir)

        self.assertEqual(result["decision"], "local")
        self.assertEqual(result["plan"]["family"], "companies_by_partner")
        self.assertIn("MATCH", result["compiled"]["cypher"])
        self.assertEqual(result["compiled"]["params"], {"partners": ["Dell"]})
        planner.generate.assert_called_once()

    def test_run_local_query_stack_downgrades_when_planner_gate_is_closed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = self._write_bundle(Path(tmp_dir), planner_gate_open=False)
            router = Mock()
            router.predict.return_value = {"local": 0.99, "refuse": 0.01, "api_fallback": 0.0}

            with patch("runtime.local_query_stack._router_predictor_for", return_value=router), patch(
                "runtime.local_query_stack._planner_generator_for"
            ) as mock_planner_factory:
                result = run_local_query_stack("Which companies partner with Dell?", bundle_dir=bundle_dir)

        self.assertEqual(result["decision"], "api_fallback")
        self.assertEqual(result["router"]["gate_reason"], "planner_gate_closed")
        mock_planner_factory.assert_not_called()

    def test_run_local_query_stack_downgrades_failed_planner_runs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = self._write_bundle(Path(tmp_dir))
            router = Mock()
            router.predict.return_value = {"local": 0.99, "refuse": 0.01, "api_fallback": 0.0}
            planner = Mock()
            planner.generate.return_value = "not valid planner output"

            with patch("runtime.local_query_stack._router_predictor_for", return_value=router), patch(
                "runtime.local_query_stack._planner_generator_for",
                return_value=planner,
            ):
                result = run_local_query_stack("Which companies partner with Dell?", bundle_dir=bundle_dir)

        self.assertEqual(result["decision"], "api_fallback")
        self.assertIn("No JSON object found", result["planner"]["error"])

    def test_run_local_query_stack_skips_planner_for_refuse(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = self._write_bundle(Path(tmp_dir))
            router = Mock()
            router.predict.return_value = {"local": 0.05, "refuse": 0.99, "api_fallback": 0.0}

            with patch("runtime.local_query_stack._router_predictor_for", return_value=router), patch(
                "runtime.local_query_stack._planner_generator_for"
            ) as mock_planner_factory:
                result = run_local_query_stack("Delete Aurora.", bundle_dir=bundle_dir)

        self.assertEqual(result["decision"], "refuse")
        self.assertIsNone(result["planner"])
        mock_planner_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
