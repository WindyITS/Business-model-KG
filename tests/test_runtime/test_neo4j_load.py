import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from runtime import neo4j_load


class Neo4jLoadTests(unittest.TestCase):
    @staticmethod
    def _write_run(run_dir: Path, *, company_name=None, source_file="data/apple_10k.txt", triple_subject="Apple") -> None:
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
        (run_dir / "resolved_triples.json").write_text(
            json.dumps(
                {
                    "triples": [
                        {
                            "subject": triple_subject,
                            "subject_type": "Company",
                            "relation": "HAS_SEGMENT",
                            "object": "Cloud",
                            "object_type": "BusinessSegment",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    def test_main_refuses_bulk_load_into_nonempty_database_without_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            self._write_run(output_dir / "apple" / "analyst" / "latest")
            fake_loader = MagicMock()
            fake_loader.graph_counts.return_value = {"node_count": 7, "relationship_count": 11}
            stdout = io.StringIO()
            stderr = io.StringIO()

            with patch.object(neo4j_load, "Neo4jLoader", return_value=fake_loader), redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = neo4j_load.main(
                    ["--output-dir", str(output_dir)],
                    input_reader=lambda prompt: "y",
                    is_interactive=lambda: False,
                )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Refusing to bulk-load into a non-empty Neo4j database without confirmation", stderr.getvalue())
        fake_loader.graph_counts.assert_called_once()
        fake_loader.setup_constraints.assert_not_called()
        fake_loader.close.assert_called_once()

    def test_main_bulk_loads_all_latest_analyst_outputs_by_default(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            self._write_run(
                output_dir / "apple" / "analyst" / "latest",
                company_name="Apple",
                source_file="data/apple_10k.txt",
                triple_subject="Apple",
            )
            self._write_run(
                output_dir / "google" / "analyst" / "latest",
                company_name=None,
                source_file="data/google_10k.txt",
                triple_subject="Google",
            )
            self._write_run(
                output_dir / "apple" / "zero-shot" / "latest",
                company_name="Apple",
                source_file="data/apple_10k.txt",
                triple_subject="Apple",
            )

            fake_loader = MagicMock()
            fake_loader.graph_counts.return_value = {"node_count": 0, "relationship_count": 0}
            fake_loader.replace_company_triples.side_effect = lambda triples, company_name: (
                {
                    "company_name": company_name,
                    "scoped_nodes_deleted": 0,
                    "scoped_relationships_deleted": 0,
                    "company_relationships_deleted": 0,
                    "company_node_deleted": 0,
                    "orphan_nodes_deleted": 0,
                },
                len(triples),
            )
            stdout = io.StringIO()

            with patch.object(neo4j_load, "Neo4jLoader", return_value=fake_loader), redirect_stdout(stdout):
                exit_code = neo4j_load.main(["--output-dir", str(output_dir)])

        self.assertEqual(exit_code, 0)
        fake_loader.graph_counts.assert_called_once()
        fake_loader.setup_constraints.assert_called_once()
        self.assertEqual(
            [call.kwargs["company_name"] for call in fake_loader.replace_company_triples.call_args_list],
            ["Apple", "Google"],
        )
        self.assertEqual(
            [call.kwargs["company_name"] for call in fake_loader.replace_company_triples.call_args_list],
            ["Apple", "Google"],
        )
        rendered = stdout.getvalue()
        self.assertIn('Loaded 2 "analyst" output(s) into Neo4j (2 triples total).', rendered)
        fake_loader.close.assert_called_once()

    def test_main_loads_latest_company_output_without_bulk_warning(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            self._write_run(
                output_dir / "apple" / "analyst" / "latest",
                company_name="Apple",
                source_file="data/apple_10k.txt",
                triple_subject="Apple",
            )
            fake_loader = MagicMock()
            fake_loader.company_graph_counts.return_value = {
                "company_node_count": 0,
                "scoped_node_count": 0,
                "relationship_count": 0,
            }
            fake_loader.replace_company_triples.return_value = (
                {
                    "company_name": "Apple",
                    "scoped_nodes_deleted": 0,
                    "scoped_relationships_deleted": 0,
                    "company_relationships_deleted": 0,
                    "company_node_deleted": 0,
                    "orphan_nodes_deleted": 0,
                },
                1,
            )

            with patch.object(neo4j_load, "Neo4jLoader", return_value=fake_loader):
                exit_code = neo4j_load.main(["--output-dir", str(output_dir), "--company", "Apple"])

        self.assertEqual(exit_code, 0)
        fake_loader.graph_counts.assert_not_called()
        fake_loader.company_graph_counts.assert_called_once_with("Apple")
        fake_loader.replace_company_triples.assert_called_once()
        self.assertEqual(fake_loader.replace_company_triples.call_args.kwargs["company_name"], "Apple")
        fake_loader.close.assert_called_once()

    def test_main_loads_exact_company_run_selector(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            latest_dir = output_dir / "apple" / "analyst" / "latest"
            run_dir = output_dir / "apple" / "analyst" / "runs" / "20260417T101500Z"
            self._write_run(latest_dir, company_name="Apple", source_file="data/apple_10k.txt", triple_subject="Apple")
            self._write_run(run_dir, company_name="Apple", source_file="data/apple_10k.txt", triple_subject="Apple")

            fake_loader = MagicMock()
            fake_loader.company_graph_counts.return_value = {
                "company_node_count": 0,
                "scoped_node_count": 0,
                "relationship_count": 0,
            }
            fake_loader.replace_company_triples.return_value = (
                {
                    "company_name": "Apple",
                    "scoped_nodes_deleted": 0,
                    "scoped_relationships_deleted": 0,
                    "company_relationships_deleted": 0,
                    "company_node_deleted": 0,
                    "orphan_nodes_deleted": 0,
                },
                1,
            )
            stdout = io.StringIO()

            with patch.object(neo4j_load, "Neo4jLoader", return_value=fake_loader), redirect_stdout(stdout):
                exit_code = neo4j_load.main(
                    [
                        "--output-dir",
                        str(output_dir),
                        "--company",
                        "Apple",
                        "--run",
                        "20260417T101500Z",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn(str(run_dir), stdout.getvalue())
        fake_loader.replace_company_triples.assert_called_once()
        fake_loader.close.assert_called_once()

    def test_main_prompts_before_replacing_already_loaded_company(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            self._write_run(output_dir / "apple" / "analyst" / "latest", company_name="Apple")
            fake_loader = MagicMock()
            fake_loader.company_graph_counts.return_value = {
                "company_node_count": 1,
                "scoped_node_count": 4,
                "relationship_count": 9,
            }
            stdout = io.StringIO()
            stderr = io.StringIO()

            with patch.object(neo4j_load, "Neo4jLoader", return_value=fake_loader), redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = neo4j_load.main(
                    ["--output-dir", str(output_dir), "--company", "Apple"],
                    input_reader=lambda prompt: "n",
                    is_interactive=lambda: True,
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("Aborted; nothing was loaded.", stderr.getvalue())
        fake_loader.company_graph_counts.assert_called_once_with("Apple")
        fake_loader.replace_company_triples.assert_not_called()
        fake_loader.close.assert_called_once()

    def test_main_bulk_load_reports_company_failures_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            self._write_run(output_dir / "apple" / "analyst" / "latest", company_name="Apple", triple_subject="Apple")
            self._write_run(output_dir / "google" / "analyst" / "latest", company_name="Google", triple_subject="Google")

            fake_loader = MagicMock()
            fake_loader.graph_counts.return_value = {"node_count": 0, "relationship_count": 0}
            def replace_company_triples_side_effect(triples, company_name):
                if company_name == "Google":
                    raise RuntimeError("validation drift")
                return (
                    {
                        "company_name": company_name,
                        "scoped_nodes_deleted": 0,
                        "scoped_relationships_deleted": 0,
                        "company_relationships_deleted": 0,
                        "company_node_deleted": 0,
                        "orphan_nodes_deleted": 0,
                    },
                    len(triples),
                )

            fake_loader.replace_company_triples.side_effect = replace_company_triples_side_effect
            stdout = io.StringIO()
            stderr = io.StringIO()

            with patch.object(neo4j_load, "Neo4jLoader", return_value=fake_loader), redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = neo4j_load.main(["--output-dir", str(output_dir)])

        self.assertEqual(exit_code, 1)
        self.assertIn('Loaded 1 of 2 "analyst" output(s) into Neo4j (1 triples total). 1 company could not be loaded.', stdout.getvalue())
        self.assertIn("Failed to load Google", stderr.getvalue())
        self.assertEqual(fake_loader.replace_company_triples.call_count, 2)
        fake_loader.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
