import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from runtime import health_check


class HealthCheckTests(unittest.TestCase):
    def test_check_repo_venv_warns_when_default_venv_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)

            result = health_check._check_repo_venv(root_dir)

        self.assertEqual(result.status, "warn")
        self.assertIn("bootstrap_dev.sh", result.hint or "")

    def test_check_outputs_reports_latest_company_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            latest_dir = root_dir / "outputs" / "apple" / "analyst" / "latest"
            latest_dir.mkdir(parents=True, exist_ok=True)
            (latest_dir / "run_summary.json").write_text(
                '{"company_name":"Apple","source_file":"data/apple_10k.txt","run_dir":"latest"}',
                encoding="utf-8",
            )

            result = health_check._check_outputs(root_dir, Path("outputs"), "analyst")

        self.assertEqual(result.status, "ok")
        self.assertIn('1 company with latest "analyst" output', result.detail)

    def test_main_skips_neo4j_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            (root_dir / ".env.example").write_text("NEO4J_AUTH=neo4j/password\n", encoding="utf-8")
            (root_dir / "prompts").mkdir()
            bundled_prompts = root_dir / "src" / "llm_extraction" / "_bundled_prompts"
            bundled_prompts.mkdir(parents=True, exist_ok=True)
            ontology_dir = root_dir / "src" / "ontology"
            ontology_dir.mkdir(parents=True, exist_ok=True)
            (ontology_dir / "ontology.json").write_text("{}", encoding="utf-8")
            venv_python = root_dir / "venv" / "bin"
            venv_python.mkdir(parents=True, exist_ok=True)
            (venv_python / "python").write_text("#!/usr/bin/env python\n", encoding="utf-8")
            outputs_dir = root_dir / "outputs" / "apple" / "analyst" / "latest"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            (outputs_dir / "run_summary.json").write_text(
                '{"company_name":"Apple","source_file":"data/apple_10k.txt","run_dir":"latest"}',
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with patch.object(health_check, "_project_root", return_value=root_dir), patch.object(
                health_check,
                "_check_packaging_tools",
                return_value=health_check.HealthCheckResult("packaging tools", "ok", "pip, setuptools, wheel"),
            ), redirect_stdout(stdout):
                exit_code = health_check.main(["--skip-neo4j"])

        self.assertEqual(exit_code, 0)
        self.assertIn("REPO HEALTH CHECK", stdout.getvalue())
        self.assertNotIn("neo4j:", stdout.getvalue().lower())

    def test_check_neo4j_warns_when_connection_fails(self):
        fake_loader = MagicMock()
        fake_loader.graph_counts.side_effect = RuntimeError("connection refused")

        with patch.object(health_check, "Neo4jLoader", return_value=fake_loader):
            result = health_check._check_neo4j("bolt://localhost:7687", "neo4j", "password", require=False)

        self.assertEqual(result.status, "warn")
        self.assertIn("connection refused", result.detail)

    def test_check_packaging_tools_fails_when_setuptools_is_missing(self):
        real_import = __import__

        def import_side_effect(name, *args, **kwargs):
            if name == "setuptools":
                raise ModuleNotFoundError("No module named 'setuptools'")
            return real_import(name, *args, **kwargs)

        with patch.object(health_check.importlib, "import_module", side_effect=import_side_effect):
            result = health_check._check_packaging_tools()

        self.assertEqual(result.status, "fail")
        self.assertIn("setuptools", result.detail)


if __name__ == "__main__":
    unittest.main()
