import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kg_query_planner_ft.config import load_config, repo_root
from kg_query_planner_ft.paths import artifact_root


class PathsTests(unittest.TestCase):
    def test_relative_artifact_root_resolves_from_repo_root(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_path = tmp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "artifact_root": "finetuning/artifacts/kg-query-planner",
                        "dataset_path": "data/query_planner_curated/v1_final",
                        "router": {"base_model": "microsoft/deberta-v3-small"},
                        "planner": {"base_model": "mlx-community/Qwen3-4B-Instruct-2507-4bit"},
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(
                artifact_root(config),
                (repo_root() / "finetuning" / "artifacts" / "kg-query-planner").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
