import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kg_query_planner_ft.planner_train import train_planner


class PlannerTrainTests(unittest.TestCase):
    def test_train_planner_writes_checkpoint_and_resume_settings(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            artifact_root = tmp / "artifacts"
            train_dir = artifact_root / "prepared" / "planner" / "balanced"
            train_dir.mkdir(parents=True, exist_ok=True)
            (train_dir / "train.jsonl").write_text(
                json.dumps(
                    {
                        "question": "Which companies partner with Dell?",
                        "messages": [
                            {"role": "system", "content": "sys"},
                            {"role": "user", "content": "Which companies partner with Dell?"},
                            {"role": "assistant", "content": "{\"answerable\":true}"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            resume_file = artifact_root / "planner" / "adapter" / "0000500_adapters.safetensors"
            config_path = tmp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "artifact_root": str(artifact_root),
                        "dataset_path": str(tmp / "dataset"),
                        "router": {"base_model": "microsoft/deberta-v3-small"},
                        "planner": {
                            "base_model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
                            "checkpoint_every": 500,
                            "resume_adapter_file": str(resume_file),
                        },
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch(
                    "kg_query_planner_ft.planner_train._planner_length_preflight",
                    return_value={
                        "count": 1,
                        "full_max": 10,
                        "prompt_max": 5,
                        "completion_max": 5,
                        "zero_target_rows": 0,
                    },
                ),
                patch("kg_query_planner_ft.planner_train._run_mlx_training"),
            ):
                summary = train_planner(str(config_path))

            yaml_path = artifact_root / "planner" / "adapter" / "train_config.yaml"
            yaml_text = yaml_path.read_text(encoding="utf-8")
            self.assertIn("save_every: 500", yaml_text)
            self.assertIn(f"resume_adapter_file: '{resume_file}'", yaml_text)
            self.assertEqual(summary["checkpoint_every"], 500)
            self.assertEqual(summary["resume_adapter_file"], str(resume_file))


if __name__ == "__main__":
    unittest.main()
