import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kg_query_planner_ft.publish_query_stack import publish_query_stack


class PublishQueryStackTests(unittest.TestCase):
    def test_publish_query_stack_writes_runtime_bundle_layout(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            artifact_root = tmp / "artifacts"
            router_model_dir = artifact_root / "router" / "model"
            router_eval_dir = artifact_root / "router" / "eval"
            planner_adapter_dir = artifact_root / "planner" / "adapter"
            router_model_dir.mkdir(parents=True, exist_ok=True)
            router_eval_dir.mkdir(parents=True, exist_ok=True)
            planner_adapter_dir.mkdir(parents=True, exist_ok=True)
            (router_model_dir / "config.json").write_text("{}", encoding="utf-8")
            (router_model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            (router_eval_dir / "thresholds.json").write_text(
                json.dumps(
                    {
                        "local_threshold": {"threshold": 0.95},
                        "policy": "local_if_probability_at_least_0.95_else_best_nonlocal",
                        "planner_gate_open": True,
                    }
                ),
                encoding="utf-8",
            )
            (planner_adapter_dir / "adapters.safetensors").write_text("weights", encoding="utf-8")

            config_path = tmp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "artifact_root": str(artifact_root),
                        "dataset_path": str(tmp / "dataset"),
                        "router": {
                            "base_model": "microsoft/deberta-v3-small",
                            "max_length": 384,
                        },
                        "planner": {
                            "base_model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
                            "max_tokens": 192,
                        },
                    }
                ),
                encoding="utf-8",
            )

            destination_dir = tmp / "published" / "query_stack"
            summary = publish_query_stack(str(config_path), destination=str(destination_dir))

            manifest = json.loads((destination_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["destination_dir"], str(destination_dir.resolve()))
            self.assertTrue((destination_dir / "router" / "model" / "config.json").exists())
            self.assertTrue((destination_dir / "router" / "model" / "tokenizer.json").exists())
            self.assertTrue((destination_dir / "router" / "thresholds.json").exists())
            self.assertTrue((destination_dir / "planner" / "adapter" / "adapters.safetensors").exists())
            self.assertTrue((destination_dir / "planner" / "system_prompt.txt").exists())
            self.assertEqual(manifest["bundle_format_version"], 1)
            self.assertEqual(manifest["router"]["base_model"], "microsoft/deberta-v3-small")
            self.assertEqual(manifest["router"]["max_length"], 384)
            self.assertEqual(manifest["router"]["model_dir"], "router/model")
            self.assertEqual(manifest["router"]["thresholds_path"], "router/thresholds.json")
            self.assertEqual(manifest["planner"]["base_model"], "mlx-community/Qwen3-4B-Instruct-2507-4bit")
            self.assertEqual(manifest["planner"]["max_tokens"], 192)
            self.assertEqual(manifest["planner"]["adapter_dir"], "planner/adapter")
            self.assertEqual(manifest["planner"]["system_prompt_path"], "planner/system_prompt.txt")
            self.assertIn("published_at", manifest)
            prompt = (destination_dir / "planner" / "system_prompt.txt").read_text(encoding="utf-8")
            self.assertIn("answerable must always be true", prompt)
            self.assertIn("router, not this planner, owns refusals", prompt)
            self.assertNotIn('"answerable": false', prompt)
            self.assertNotIn("Valid refusal reasons", prompt)


if __name__ == "__main__":
    unittest.main()
