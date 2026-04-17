import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from runtime import neo4j_status


class Neo4jStatusTests(unittest.TestCase):
    @staticmethod
    def _write_run(run_dir: Path, *, company_name: str, source_file: str) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "company_name": company_name,
                    "source_file": source_file,
                    "run_dir": str(run_dir),
                }
            ),
            encoding="utf-8",
        )

    def test_main_reports_loaded_and_not_loaded_companies(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            self._write_run(output_dir / "apple" / "analyst" / "latest", company_name="Apple", source_file="data/apple_10k.txt")
            self._write_run(
                output_dir / "google" / "analyst" / "runs" / "20260417T101500Z",
                company_name="Google",
                source_file="data/google_10k.txt",
            )

            fake_loader = MagicMock()
            fake_loader.list_loaded_companies.return_value = ["Apple", "Microsoft"]
            stdout = io.StringIO()

            with patch.object(neo4j_status, "Neo4jLoader", return_value=fake_loader), redirect_stdout(stdout):
                exit_code = neo4j_status.main(["--output-dir", str(output_dir)])

        self.assertEqual(exit_code, 0)
        rendered = stdout.getvalue()
        self.assertIn("Loaded companies: 2", rendered)
        self.assertIn('- Apple: latest "analyst" output available at ', rendered)
        self.assertIn('- Microsoft: no latest "analyst" output available', rendered)
        self.assertIn("Not loaded companies: 1", rendered)
        self.assertIn('- Google: no latest "analyst" output available; 1 archived run(s)', rendered)
        fake_loader.list_loaded_companies.assert_called_once()
        fake_loader.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
