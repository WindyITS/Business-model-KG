import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kg_query_planner_ft.config import default_config_path, load_config, repo_root
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

    def test_default_config_is_loadable(self):
        config_path = default_config_path()
        self.assertTrue(config_path.is_file())
        config = load_config()
        self.assertEqual(config.dataset_path, "data/query_planner_curated/v1_final")

    def test_repo_root_can_be_overridden_for_installed_runs(self):
        with TemporaryDirectory() as tmpdir:
            previous = os.environ.get("KG_QUERY_PLANNER_FT_ROOT")
            os.environ["KG_QUERY_PLANNER_FT_ROOT"] = tmpdir
            try:
                self.assertEqual(repo_root(), Path(tmpdir).resolve())
            finally:
                if previous is None:
                    os.environ.pop("KG_QUERY_PLANNER_FT_ROOT", None)
                else:
                    os.environ["KG_QUERY_PLANNER_FT_ROOT"] = previous


if __name__ == "__main__":
    unittest.main()
