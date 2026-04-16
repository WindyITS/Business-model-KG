import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from runtime import query as query_module
from runtime.query import Text2CypherQueryResult


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

    def test_query_cypher_prints_generated_query_without_execution(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.object(query_module, "resolve_model_settings", return_value=self._model_settings()), patch.object(
            query_module, "LLMExtractor", return_value=object()
        ), patch.object(
            query_module,
            "generate_text2cypher_query",
            return_value=(
                Text2CypherQueryResult(
                    answerable=True,
                    cypher="MATCH (company:Company) RETURN company.name AS company",
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
        self.assertEqual(stdout.getvalue().strip(), "MATCH (company:Company) RETURN company.name AS company")
        self.assertIn("Generating query...", stderr.getvalue())
        mock_execute.assert_not_called()

    def test_query_executes_generated_cypher_and_prints_rows(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.object(query_module, "resolve_model_settings", return_value=self._model_settings()), patch.object(
            query_module, "LLMExtractor", return_value=object()
        ), patch.object(
            query_module,
            "generate_text2cypher_query",
            return_value=(
                Text2CypherQueryResult(
                    answerable=True,
                    cypher="MATCH (company:Company) RETURN company.name AS company ORDER BY company",
                    params={},
                ),
                None,
                2,
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
        self.assertIn("Generating query...", stderr.getvalue())
        self.assertIn("Running query on Neo4j...", stderr.getvalue())
        mock_execute.assert_called_once()

    def test_query_returns_refusal_without_hitting_neo4j(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.object(query_module, "resolve_model_settings", return_value=self._model_settings()), patch.object(
            query_module, "LLMExtractor", return_value=object()
        ), patch.object(
            query_module,
            "generate_text2cypher_query",
            return_value=(
                Text2CypherQueryResult(answerable=False, reason="unsupported_request"),
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
        self.assertIn("Unsupported request: unsupported_request", stderr.getvalue())
        mock_execute.assert_not_called()

    def test_query_cypher_rejects_generated_write_query(self):
        stderr = io.StringIO()

        with patch.object(query_module, "resolve_model_settings", return_value=self._model_settings()), patch.object(
            query_module, "LLMExtractor", return_value=object()
        ), patch.object(
            query_module,
            "generate_text2cypher_query",
            return_value=(
                Text2CypherQueryResult(
                    answerable=True,
                    cypher="CREATE (company:Company {name: $company}) RETURN company.name AS company",
                    params={"company": "Acme"},
                ),
                None,
                1,
                {},
            ),
        ), redirect_stderr(stderr):
            exit_code = query_module.main_query_cypher(["--repair-attempts", "0", "Add Acme to the graph."])

        self.assertEqual(exit_code, 1)
        rendered_error = stderr.getvalue()
        self.assertIn("Generated query failed validation.", rendered_error)
        self.assertIn("CREATE (company:Company {name: $company}) RETURN company.name AS company", rendered_error)
        self.assertIn("disallowed clause", rendered_error)

    def test_query_cypher_renders_params_as_browser_ready_snippet(self):
        stdout = io.StringIO()

        with patch.object(query_module, "resolve_model_settings", return_value=self._model_settings()), patch.object(
            query_module, "LLMExtractor", return_value=object()
        ), patch.object(
            query_module,
            "generate_text2cypher_query",
            return_value=(
                Text2CypherQueryResult(
                    answerable=True,
                    cypher="MATCH (company:Company {name: $company}) RETURN company.name AS company",
                    params={"company": "Acme"},
                ),
                None,
                1,
                {},
            ),
        ), redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            exit_code = query_module.main_query_cypher(["Show Acme."])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            stdout.getvalue().strip(),
            'MATCH (company:Company {name: "Acme"}) RETURN company.name AS company',
        )

    def test_query_cypher_inlines_multiple_params(self):
        stdout = io.StringIO()

        with patch.object(query_module, "resolve_model_settings", return_value=self._model_settings()), patch.object(
            query_module, "LLMExtractor", return_value=object()
        ), patch.object(
            query_module,
            "generate_text2cypher_query",
            return_value=(
                Text2CypherQueryResult(
                    answerable=True,
                    cypher=(
                        "MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment {company_name: c.name})"
                        "-[:SERVES]->(:CustomerType {name: $customer_type}) "
                        "MATCH (s)-[:SELLS_THROUGH]->(:Channel {name: $channel}) "
                        "RETURN DISTINCT c.name AS company ORDER BY company"
                    ),
                    params={"customer_type": "developers", "channel": "direct sales"},
                ),
                None,
                1,
                {},
            ),
        ), redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            exit_code = query_module.main_query_cypher(["Show companies."])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            stdout.getvalue().strip(),
            'MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment {company_name: c.name})'
            '-[:SERVES]->(:CustomerType {name: "developers"}) '
            'MATCH (s)-[:SELLS_THROUGH]->(:Channel {name: "direct sales"}) '
            'RETURN DISTINCT c.name AS company ORDER BY company',
        )

    def test_query_repairs_after_neo4j_preflight_error(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        bad_result = Text2CypherQueryResult(
            answerable=True,
            cypher=(
                "MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment {company_name: c.name})"
                "-[:SERVES]->(ct:CustomerType) "
                "WITH ct.name AS customer_type, COUNT(DISTINCT c.name) AS company_count "
                "ORDER BY company_count DESC LIMIT 1"
            ),
            params={},
        )
        fixed_result = Text2CypherQueryResult(
            answerable=True,
            cypher=(
                "MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment {company_name: c.name})"
                "-[:SERVES]->(ct:CustomerType) "
                "WITH ct.name AS customer_type, COUNT(DISTINCT c.name) AS company_count "
                "ORDER BY company_count DESC LIMIT 1 "
                "RETURN customer_type"
            ),
            params={},
        )

        with patch.object(query_module, "resolve_model_settings", return_value=self._model_settings()), patch.object(
            query_module, "LLMExtractor", return_value=object()
        ), patch.object(
            query_module,
            "generate_text2cypher_query",
            return_value=(bad_result, None, 1, {}),
        ), patch.object(
            query_module,
            "repair_text2cypher_query",
            return_value=(fixed_result, None, 1, {}),
        ) as mock_repair, patch.object(
            query_module,
            "preflight_live_query",
            side_effect=[
                Exception("Query cannot conclude with WITH"),
                "bolt://localhost:7687",
            ],
        ), patch.object(
            query_module,
            "execute_live_query",
            return_value=(["customer_type"], [{"customer_type": "developers"}], "bolt://localhost:7687"),
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = query_module.main_query(
                ["What is the customer type served by the highest number of companies worldwide?"]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "developers")
        rendered_error = stderr.getvalue()
        self.assertIn("Generating query...", rendered_error)
        self.assertIn("Repairing query after error (attempt 1/2)...", rendered_error)
        self.assertIn("Running query on Neo4j...", rendered_error)
        mock_repair.assert_called_once()

    def test_query_repairs_after_empty_result_for_multi_customer_company_query(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        bad_result = Text2CypherQueryResult(
            answerable=True,
            cypher=(
                "MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment {company_name: c.name})"
                "-[:SERVES]->(:CustomerType {name: $customer_type_1}) "
                "MATCH (s)-[:SERVES]->(:CustomerType {name: $customer_type_2}) "
                "RETURN DISTINCT c.name AS company ORDER BY company"
            ),
            params={
                "customer_type_1": "government agencies",
                "customer_type_2": "healthcare organizations",
            },
        )
        fixed_result = Text2CypherQueryResult(
            answerable=True,
            cypher=(
                "MATCH (c:Company)-[:HAS_SEGMENT]->(s1:BusinessSegment {company_name: c.name})"
                "-[:SERVES]->(:CustomerType {name: $customer_type_1}) "
                "MATCH (c)-[:HAS_SEGMENT]->(s2:BusinessSegment {company_name: c.name})"
                "-[:SERVES]->(:CustomerType {name: $customer_type_2}) "
                "RETURN DISTINCT c.name AS company ORDER BY company"
            ),
            params={
                "customer_type_1": "government agencies",
                "customer_type_2": "healthcare organizations",
            },
        )

        with patch.object(query_module, "resolve_model_settings", return_value=self._model_settings()), patch.object(
            query_module, "LLMExtractor", return_value=object()
        ), patch.object(
            query_module,
            "generate_text2cypher_query",
            return_value=(bad_result, None, 1, {}),
        ), patch.object(
            query_module,
            "repair_text2cypher_query",
            return_value=(fixed_result, None, 1, {}),
        ) as mock_repair, patch.object(
            query_module,
            "preflight_live_query",
            return_value="bolt://localhost:7687",
        ), patch.object(
            query_module,
            "execute_live_query",
            side_effect=[
                (["company"], [], "bolt://localhost:7687"),
                (["company"], [{"company": "Microsoft"}], "bolt://localhost:7687"),
            ],
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = query_module.main_query(
                ["What are companies selling to the government AND to healthcare firms?"]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "Microsoft")
        rendered_error = stderr.getvalue()
        self.assertIn("Generating query...", rendered_error)
        self.assertIn("Repairing query after error (attempt 1/2)...", rendered_error)
        self.assertIn("Running query on Neo4j...", rendered_error)
        mock_repair.assert_called_once()

    def test_query_does_not_repair_empty_result_for_single_channel_geography_query(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        result = Text2CypherQueryResult(
            answerable=True,
            cypher=(
                "MATCH (company:Company)-[:OPERATES_IN]->(place:Place) "
                "WITH company, CASE "
                "WHEN place.name = $place THEN 0 "
                "WHEN $place IN coalesce(place.includes_places, []) THEN 1 "
                "WHEN $place IN coalesce(place.within_places, []) THEN 2 "
                "ELSE NULL END AS match_rank "
                "WHERE match_rank IS NOT NULL "
                "MATCH (company)-[:HAS_SEGMENT]->(s:BusinessSegment {company_name: company.name})"
                "-[:SELLS_THROUGH]->(:Channel {name: $channel}) "
                "RETURN DISTINCT company.name AS company, s.name AS segment ORDER BY company, segment"
            ),
            params={"place": "United States", "channel": "marketplaces"},
        )

        with patch.object(query_module, "resolve_model_settings", return_value=self._model_settings()), patch.object(
            query_module, "LLMExtractor", return_value=object()
        ), patch.object(
            query_module,
            "generate_text2cypher_query",
            return_value=(result, None, 1, {}),
        ), patch.object(
            query_module,
            "preflight_live_query",
            return_value="bolt://localhost:7687",
        ), patch.object(
            query_module,
            "execute_live_query",
            return_value=(["company", "segment"], [], "bolt://localhost:7687"),
        ), patch.object(
            query_module,
            "repair_text2cypher_query",
        ) as mock_repair, redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = query_module.main_query(
                ["Which company segments at companies operating in the United States sell through marketplaces?"]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "")
        rendered_error = stderr.getvalue()
        self.assertIn("Generating query...", rendered_error)
        self.assertIn("Running query on Neo4j...", rendered_error)
        self.assertIn("No rows returned from bolt://localhost:7687.", rendered_error)
        mock_repair.assert_not_called()


if __name__ == "__main__":
    unittest.main()
