import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from runtime import query as query_module
from runtime.query import QueryResult


class RuntimeQueryTests(unittest.TestCase):
    def _model_settings(self) -> SimpleNamespace:
        return SimpleNamespace(
            provider="local",
            model="local-model",
            base_url="http://localhost:1234/v1",
            api_mode="chat_completions",
            api_key="lm-studio",
            max_output_tokens=None,
        )

    def test_query_cypher_prints_compiled_query_without_execution(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.object(query_module, "resolve_model_settings", return_value=self._model_settings()), patch.object(
            query_module, "LLMExtractor", return_value=object()
        ), patch.object(
            query_module,
            "generate_query",
            return_value=(
                QueryResult(
                    answerable=True,
                    cypher="MATCH (company:Company) RETURN DISTINCT company.name AS company ORDER BY company",
                    params={},
                ),
                None,
                1,
                {},
            ),
        ), patch.object(query_module, "execute_live_query") as mock_execute, redirect_stdout(stdout), redirect_stderr(
            stderr
        ):
            exit_code = query_module.main_query_cypher(["Which companies are in the graph?"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "MATCH (company:Company) RETURN DISTINCT company.name AS company ORDER BY company")
        self.assertIn("Generating query plan...", stderr.getvalue())
        mock_execute.assert_not_called()

    def test_query_executes_generated_cypher_and_prints_rows(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.object(query_module, "resolve_model_settings", return_value=self._model_settings()), patch.object(
            query_module, "LLMExtractor", return_value=object()
        ), patch.object(
            query_module,
            "generate_query",
            return_value=(
                QueryResult(
                    answerable=True,
                    cypher="MATCH (company:Company) RETURN company.name AS company ORDER BY company",
                    params={},
                ),
                None,
                1,
                {},
            ),
        ), patch.object(
            query_module,
            "preflight_live_query",
            return_value="bolt://localhost:7687",
        ), patch.object(
            query_module,
            "execute_live_query",
            return_value=(["company"], [{"company": "Acme"}, {"company": "Globex"}], "bolt://localhost:7687"),
        ) as mock_execute, redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = query_module.main_query(["List the companies."])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "Acme\nGlobex")
        self.assertIn("Generating query plan...", stderr.getvalue())
        self.assertIn("Running query on Neo4j...", stderr.getvalue())
        mock_execute.assert_called_once()

    def test_query_returns_refusal_without_hitting_neo4j(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.object(query_module, "resolve_model_settings", return_value=self._model_settings()), patch.object(
            query_module, "LLMExtractor", return_value=object()
        ), patch.object(
            query_module,
            "generate_query",
            return_value=(
                QueryResult(answerable=False, reason="unsupported_time"),
                None,
                1,
                {},
            ),
        ), patch.object(query_module, "execute_live_query") as mock_execute, redirect_stdout(stdout), redirect_stderr(
            stderr
        ):
            exit_code = query_module.main_query(["What was Microsoft's revenue in 2024?"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Unsupported request: unsupported_time", stderr.getvalue())
        mock_execute.assert_not_called()

    def test_query_cypher_refuses_invalid_compiled_query(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.object(query_module, "resolve_model_settings", return_value=self._model_settings()), patch.object(
            query_module, "LLMExtractor", return_value=object()
        ), patch.object(
            query_module,
            "generate_query",
            return_value=(
                QueryResult(
                    answerable=True,
                    cypher="CREATE (company:Company {name: $company}) RETURN company.name AS company",
                    params={"company": "Acme"},
                ),
                None,
                1,
                {},
            ),
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = query_module.main_query_cypher(["Add Acme to the graph."])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Unsupported request: beyond_local_coverage", stderr.getvalue())

    def test_query_cypher_renders_list_params_as_browser_ready_snippet(self):
        stdout = io.StringIO()

        with patch.object(query_module, "resolve_model_settings", return_value=self._model_settings()), patch.object(
            query_module, "LLMExtractor", return_value=object()
        ), patch.object(
            query_module,
            "generate_query",
            return_value=(
                QueryResult(
                    answerable=True,
                    cypher="MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment) WHERE s.company_name = c.name AND c.name IN $companies RETURN DISTINCT c.name AS company, s.name AS segment ORDER BY company, segment",
                    params={"companies": ["Apple", "Microsoft"]},
                ),
                None,
                1,
                {},
            ),
        ), redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            exit_code = query_module.main_query_cypher(["What are the business segments of Apple and Microsoft?"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            stdout.getvalue().strip(),
            'MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment) WHERE s.company_name = c.name AND c.name IN ["Apple", "Microsoft"] RETURN DISTINCT c.name AS company, s.name AS segment ORDER BY company, segment',
        )

    def test_query_reports_preflight_errors(self):
        stderr = io.StringIO()

        with patch.object(query_module, "resolve_model_settings", return_value=self._model_settings()), patch.object(
            query_module, "LLMExtractor", return_value=object()
        ), patch.object(
            query_module,
            "generate_query",
            return_value=(
                QueryResult(
                    answerable=True,
                    cypher="MATCH (company:Company) RETURN company.name AS company",
                    params={},
                ),
                None,
                1,
                {},
            ),
        ), patch.object(
            query_module,
            "preflight_live_query",
            side_effect=Exception("Query cannot conclude with WITH"),
        ), redirect_stderr(stderr):
            exit_code = query_module.main_query(["List the companies."])

        self.assertEqual(exit_code, 1)
        self.assertIn("Neo4j preflight error: Query cannot conclude with WITH", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
