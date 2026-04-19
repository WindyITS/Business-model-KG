import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from runtime import neo4j_admin


class Neo4jAdminTests(unittest.TestCase):
    def test_main_refuses_noninteractive_unload_without_yes(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.object(neo4j_admin, "Neo4jLoader") as mock_loader, redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = neo4j_admin.main(
                ["--company", "Apple"],
                input_reader=lambda prompt: "y",
                is_interactive=lambda: False,
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Refusing to unload from Neo4j without confirmation", stderr.getvalue())
        self.assertIn("Aborted; nothing was deleted.", stderr.getvalue())
        mock_loader.assert_not_called()

    def test_main_unloads_company_after_interactive_confirmation(self):
        stdout = io.StringIO()
        prompts: list[str] = []
        fake_loader = unittest.mock.MagicMock()
        fake_loader.unload_company.return_value = {
            "company_name": "Apple",
            "scoped_nodes_deleted": 3,
            "scoped_relationships_deleted": 6,
            "company_relationships_deleted": 2,
            "company_node_deleted": 1,
            "orphan_nodes_deleted": 2,
        }

        with patch.object(neo4j_admin, "Neo4jLoader", return_value=fake_loader), redirect_stdout(stdout):
            exit_code = neo4j_admin.main(
                ["--company", "Apple"],
                input_reader=lambda prompt: prompts.append(prompt) or "y",
                is_interactive=lambda: True,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(prompts, ['Unload Neo4j graph footprint for company "Apple"? [y/N] '])
        fake_loader.unload_company.assert_called_once_with("Apple")
        fake_loader.close.assert_called_once()
        rendered = stdout.getvalue()
        self.assertIn("NEO4J COMPANY UNLOAD", rendered)
        self.assertIn("company: Apple", rendered)
        self.assertIn("orphan shared nodes deleted: 2", rendered)

    def test_main_can_skip_confirmation_with_yes(self):
        stdout = io.StringIO()
        fake_loader = unittest.mock.MagicMock()
        fake_loader.unload_company.return_value = {
            "company_name": "Apple",
            "scoped_nodes_deleted": 0,
            "scoped_relationships_deleted": 0,
            "company_relationships_deleted": 0,
            "company_node_deleted": 0,
            "orphan_nodes_deleted": 0,
        }

        def _unexpected_prompt(prompt: str) -> str:
            raise AssertionError(f"prompt should not be shown: {prompt}")

        with patch.object(neo4j_admin, "Neo4jLoader", return_value=fake_loader), redirect_stdout(stdout):
            exit_code = neo4j_admin.main(
                ["--company", "Apple", "--yes"],
                input_reader=_unexpected_prompt,
                is_interactive=lambda: False,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn('No Neo4j graph footprint was found for company "Apple".', stdout.getvalue())
        fake_loader.unload_company.assert_called_once_with("Apple")
        fake_loader.close.assert_called_once()

    def test_main_refuses_full_unload_noninteractive_without_yes(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.object(neo4j_admin, "Neo4jLoader") as mock_loader, redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = neo4j_admin.main(
                [],
                input_reader=lambda prompt: "y",
                is_interactive=lambda: False,
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Refusing to unload from Neo4j without confirmation", stderr.getvalue())
        self.assertIn("Aborted; nothing was deleted.", stderr.getvalue())
        mock_loader.assert_not_called()

    def test_main_full_unload_after_interactive_confirmation(self):
        stdout = io.StringIO()
        prompts: list[str] = []
        fake_loader = unittest.mock.MagicMock()
        fake_loader.graph_counts.return_value = {"node_count": 5, "relationship_count": 9}

        with patch.object(neo4j_admin, "Neo4jLoader", return_value=fake_loader), redirect_stdout(stdout):
            exit_code = neo4j_admin.main(
                [],
                input_reader=lambda prompt: prompts.append(prompt) or "y",
                is_interactive=lambda: True,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(prompts, ["Unload Neo4j graph footprint for the full dataset? [y/N] "])
        fake_loader.graph_counts.assert_called_once()
        fake_loader.clear_graph.assert_called_once()
        fake_loader.unload_company.assert_not_called()
        fake_loader.close.assert_called_once()
        rendered = stdout.getvalue()
        self.assertIn("NEO4J FULL UNLOAD", rendered)
        self.assertIn("nodes deleted: 5", rendered)
        self.assertIn("relationships deleted: 9", rendered)

    def test_main_full_unload_can_skip_confirmation_with_yes(self):
        stdout = io.StringIO()
        fake_loader = unittest.mock.MagicMock()
        fake_loader.graph_counts.return_value = {"node_count": 0, "relationship_count": 0}

        def _unexpected_prompt(prompt: str) -> str:
            raise AssertionError(f"prompt should not be shown: {prompt}")

        with patch.object(neo4j_admin, "Neo4jLoader", return_value=fake_loader), redirect_stdout(stdout):
            exit_code = neo4j_admin.main(
                ["--yes"],
                input_reader=_unexpected_prompt,
                is_interactive=lambda: False,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Neo4j graph is already empty.", stdout.getvalue())
        fake_loader.graph_counts.assert_called_once()
        fake_loader.clear_graph.assert_called_once()
        fake_loader.unload_company.assert_not_called()
        fake_loader.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
