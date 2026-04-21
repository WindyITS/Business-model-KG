import json
import tempfile
import unittest
from pathlib import Path

from runtime.query_stack import load_query_stack_bundle


class QueryStackBundleTests(unittest.TestCase):
    @staticmethod
    def _write_manifest(bundle_dir: Path, *, bundle_format_version: int = 1) -> None:
        (bundle_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "bundle_format_version": bundle_format_version,
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

    def test_load_query_stack_bundle_rejects_unsupported_manifest_version(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = Path(tmp_dir) / "runtime_assets" / "query_stack" / "current"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            self._write_manifest(bundle_dir, bundle_format_version=2)

            with self.assertRaisesRegex(ValueError, "unsupported bundle format version"):
                load_query_stack_bundle(bundle_dir)

    def test_load_query_stack_bundle_resolves_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = Path(tmp_dir) / "runtime_assets" / "query_stack" / "current"
            (bundle_dir / "router" / "model").mkdir(parents=True, exist_ok=True)
            (bundle_dir / "planner" / "adapter").mkdir(parents=True, exist_ok=True)
            (bundle_dir / "router" / "thresholds.json").write_text("{}", encoding="utf-8")
            (bundle_dir / "planner" / "system_prompt.txt").write_text("prompt\n", encoding="utf-8")
            self._write_manifest(bundle_dir)

            bundle = load_query_stack_bundle(bundle_dir)

        self.assertEqual(bundle.router_model_dir, (bundle_dir / "router" / "model").resolve())
        self.assertEqual(bundle.router_thresholds_path, (bundle_dir / "router" / "thresholds.json").resolve())
        self.assertEqual(bundle.planner_adapter_dir, (bundle_dir / "planner" / "adapter").resolve())
        self.assertEqual(bundle.planner_system_prompt_path, (bundle_dir / "planner" / "system_prompt.txt").resolve())


if __name__ == "__main__":
    unittest.main()
