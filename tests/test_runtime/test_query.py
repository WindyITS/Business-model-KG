import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from llm_extraction.models import ExtractionError
from runtime import query as query_module
from runtime.query import QueryResult


class RuntimeQueryTests(unittest.TestCase):
    def _model_settings(self) -> SimpleNamespace:
        return SimpleNamespace(
            provider="opencode-go",
            model="kimi-k2.5",
            base_url="https://opencode.ai/zen/go/v1",
            api_mode="chat_completions",
            api_key="secret",
            max_output_tokens=20000,
        )

    def _hosted_result(self, *, cypher: str, params: dict | None = None) -> tuple[QueryResult, str | None, int, dict]:
        params = params or {}
        return (
            QueryResult(answerable=True, cypher=cypher, params=params),
            None,
            1,
            {},
        )

    def test_query_cypher_prints_hosted_query_without_execution(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.object(query_module, "_run_local_stack", return_value=({"decision": "api_fallback"}, None)), patch.object(
            query_module,
            "resolve_model_settings",
            return_value=self._model_settings(),
        ), patch.object(query_module, "LLMExtractor", return_value=object()), patch.object(
            query_module,
            "generate_query",
            return_value=self._hosted_result(
                cypher="MATCH (company:Company) RETURN DISTINCT company.name AS company ORDER BY company"
            ),
        ), patch.object(query_module, "execute_live_query") as mock_execute, redirect_stdout(stdout), redirect_stderr(
            stderr
        ):
            exit_code = query_module.main_query_cypher(["Which companies are in the graph?"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "MATCH (company:Company) RETURN DISTINCT company.name AS company ORDER BY company")
        self.assertIn("Generating hosted fallback query...", stderr.getvalue())
        mock_execute.assert_not_called()

    def test_query_executes_hosted_query_and_prints_rows(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.object(query_module, "_run_local_stack", return_value=({"decision": "api_fallback"}, None)), patch.object(
            query_module, "resolve_model_settings", return_value=self._model_settings()
        ), patch.object(query_module, "LLMExtractor", return_value=object()), patch.object(
            query_module,
            "generate_query",
            return_value=self._hosted_result(cypher="MATCH (company:Company) RETURN company.name AS company ORDER BY company"),
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
        self.assertIn("Generating hosted fallback query...", stderr.getvalue())
        self.assertIn("Running query on Neo4j...", stderr.getvalue())
        mock_execute.assert_called_once()

    def test_query_uses_local_stack_compiled_query_without_hosted_generation(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.object(
            query_module,
            "_run_local_stack",
            return_value=(
                {
                    "decision": "local",
                    "compiled": {
                        "cypher": "MATCH (company:Company) RETURN DISTINCT company.name AS company ORDER BY company",
                        "params": {},
                    },
                },
                None,
            ),
        ), patch.object(query_module, "resolve_model_settings") as mock_settings, redirect_stdout(stdout), redirect_stderr(
            stderr
        ):
            exit_code = query_module.main_query_cypher(["Which companies are in the graph?"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "MATCH (company:Company) RETURN DISTINCT company.name AS company ORDER BY company")
        self.assertIn("Router decision: local", stderr.getvalue())
        mock_settings.assert_not_called()

    def test_query_returns_hosted_refusal_without_retrying(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.object(query_module, "_run_local_stack", return_value=({"decision": "api_fallback"}, None)), patch.object(
            query_module, "resolve_model_settings", return_value=self._model_settings()
        ), patch.object(query_module, "LLMExtractor", return_value=object()), patch.object(
            query_module,
            "generate_query",
            return_value=(QueryResult(answerable=False, reason="unsupported_time"), None, 1, {}),
        ) as mock_generate, patch.object(query_module, "execute_live_query") as mock_execute, redirect_stdout(
            stdout
        ), redirect_stderr(stderr):
            exit_code = query_module.main_query(["What was Microsoft's revenue in 2024?"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Unsupported request: unsupported_time", stderr.getvalue())
        self.assertEqual(mock_generate.call_count, 1)
        mock_execute.assert_not_called()

    def test_query_cypher_retries_hosted_validation_once(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.object(query_module, "_run_local_stack", return_value=({"decision": "api_fallback"}, None)), patch.object(
            query_module, "resolve_model_settings", return_value=self._model_settings()
        ), patch.object(query_module, "LLMExtractor", return_value=object()), patch.object(
            query_module,
            "generate_query",
            side_effect=[
                self._hosted_result(
                    cypher="CREATE (company:Company {name: $company}) RETURN company.name AS company",
                    params={"company": "Acme"},
                ),
                self._hosted_result(
                    cypher="MATCH (company:Company) RETURN DISTINCT company.name AS company ORDER BY company"
                ),
            ],
        ) as mock_generate, redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = query_module.main_query_cypher(["Which companies are in the graph?", "--stack", "fallback"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Hosted query validation error (attempt 1)", stderr.getvalue())
        self.assertIn("Retrying hosted query generation once with error context", stderr.getvalue())
        self.assertEqual(mock_generate.call_count, 2)
        retry_question = mock_generate.call_args_list[1].kwargs["question"]
        self.assertIn("Hosted query retry context", retry_question)
        self.assertIn("Failure stage: validation", retry_question)

    def test_query_retries_preflight_failure_with_error_context(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.object(query_module, "_run_local_stack", return_value=({"decision": "api_fallback"}, None)), patch.object(
            query_module, "resolve_model_settings", return_value=self._model_settings()
        ), patch.object(query_module, "LLMExtractor", return_value=object()), patch.object(
            query_module,
            "generate_query",
            side_effect=[
                self._hosted_result(cypher="MATCH (company:Company) RETURN company.name AS company"),
                self._hosted_result(cypher="MATCH (company:Company) RETURN company.name AS company"),
            ],
        ) as mock_generate, patch.object(
            query_module,
            "preflight_live_query",
            side_effect=[Exception("Query cannot conclude with WITH"), "bolt://localhost:7687"],
        ), patch.object(
            query_module,
            "execute_live_query",
            return_value=(["company"], [{"company": "Acme"}], "bolt://localhost:7687"),
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = query_module.main_query(["List the companies.", "--stack", "fallback"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Neo4j preflight error: Query cannot conclude with WITH", stderr.getvalue())
        self.assertIn("Retrying hosted query generation once with error context", stderr.getvalue())
        self.assertEqual(mock_generate.call_count, 2)
        retry_question = mock_generate.call_args_list[1].kwargs["question"]
        self.assertIn("Failure stage: neo4j_preflight", retry_question)

    def test_query_retries_execution_failure_with_error_context(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.object(query_module, "_run_local_stack", return_value=({"decision": "api_fallback"}, None)), patch.object(
            query_module, "resolve_model_settings", return_value=self._model_settings()
        ), patch.object(query_module, "LLMExtractor", return_value=object()), patch.object(
            query_module,
            "generate_query",
            side_effect=[
                self._hosted_result(cypher="MATCH (company:Company) RETURN company.name AS company"),
                self._hosted_result(cypher="MATCH (company:Company) RETURN company.name AS company"),
            ],
        ) as mock_generate, patch.object(
            query_module,
            "preflight_live_query",
            return_value="bolt://localhost:7687",
        ), patch.object(
            query_module,
            "execute_live_query",
            side_effect=[
                Exception("temporary execution failure"),
                (["company"], [{"company": "Acme"}], "bolt://localhost:7687"),
            ],
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = query_module.main_query(["List the companies.", "--stack", "fallback"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Neo4j execution error: temporary execution failure", stderr.getvalue())
        self.assertEqual(mock_generate.call_count, 2)
        retry_question = mock_generate.call_args_list[1].kwargs["question"]
        self.assertIn("Failure stage: neo4j_execution", retry_question)

    def test_query_retries_hosted_generation_once_with_error_context(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        first_error = ExtractionError("temporary API generator failure")

        with patch.object(query_module, "_run_local_stack", return_value=({"decision": "api_fallback"}, None)), patch.object(
            query_module, "resolve_model_settings", return_value=self._model_settings()
        ), patch.object(query_module, "LLMExtractor", return_value=object()), patch.object(
            query_module,
            "generate_query",
            side_effect=[
                first_error,
                self._hosted_result(
                    cypher="MATCH (company:Company) RETURN DISTINCT company.name AS company ORDER BY company"
                ),
            ],
        ) as mock_generate, redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = query_module.main_query_cypher(["Which companies are in the graph?", "--stack", "fallback"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Hosted query generation error (attempt 1)", stderr.getvalue())
        self.assertIn("Retrying hosted query generation once with error context", stderr.getvalue())
        self.assertEqual(mock_generate.call_count, 2)
        retry_question = mock_generate.call_args_list[1].kwargs["question"]
        self.assertIn("Failure stage: generation", retry_question)

    def test_query_errors_after_two_hosted_generation_failures(self):
        stderr = io.StringIO()
        first_error = ExtractionError("generator call 1 failed")
        second_error = ExtractionError("generator call 2 failed")

        with patch.object(query_module, "_run_local_stack", return_value=({"decision": "api_fallback"}, None)), patch.object(
            query_module, "resolve_model_settings", return_value=self._model_settings()
        ), patch.object(query_module, "LLMExtractor", return_value=object()), patch.object(
            query_module,
            "generate_query",
            side_effect=[first_error, second_error],
        ), redirect_stderr(stderr):
            exit_code = query_module.main_query_cypher(["Which companies are in the graph?", "--stack", "fallback"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Warning: hosted query failed twice in a row.", stderr.getvalue())
        self.assertIn("Error: Hosted query failed twice.", stderr.getvalue())

    def test_query_errors_after_two_hosted_validation_failures(self):
        stderr = io.StringIO()

        with patch.object(query_module, "_run_local_stack", return_value=({"decision": "api_fallback"}, None)), patch.object(
            query_module, "resolve_model_settings", return_value=self._model_settings()
        ), patch.object(query_module, "LLMExtractor", return_value=object()), patch.object(
            query_module,
            "generate_query",
            side_effect=[
                self._hosted_result(
                    cypher="CREATE (company:Company {name: $company}) RETURN company.name AS company",
                    params={"company": "Acme"},
                ),
                self._hosted_result(
                    cypher="MERGE (company:Company {name: $company}) RETURN company.name AS company",
                    params={"company": "Acme"},
                ),
            ],
        ), redirect_stderr(stderr):
            exit_code = query_module.main_query_cypher(["Which companies are in the graph?", "--stack", "fallback"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Warning: hosted query failed twice in a row.", stderr.getvalue())
        self.assertIn("Error: Hosted query failed twice.", stderr.getvalue())

    def test_query_stack_fallback_skips_local_stack(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.object(query_module, "_run_local_stack") as mock_local_stack, patch.object(
            query_module, "resolve_model_settings", return_value=self._model_settings()
        ), patch.object(query_module, "LLMExtractor", return_value=object()), patch.object(
            query_module,
            "generate_query",
            return_value=self._hosted_result(
                cypher="MATCH (company:Company) RETURN DISTINCT company.name AS company ORDER BY company"
            ),
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = query_module.main_query_cypher(["Which companies are in the graph?", "--stack", "fallback"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Using hosted query generation only.", stderr.getvalue())
        mock_local_stack.assert_not_called()

    def test_query_auto_falls_back_when_local_planner_errors(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.object(
            query_module,
            "_run_local_stack",
            return_value=({"decision": "api_fallback", "planner": {"error": "local qwen failure"}}, None),
        ), patch.object(
            query_module, "resolve_model_settings", return_value=self._model_settings()
        ), patch.object(query_module, "LLMExtractor", return_value=object()), patch.object(
            query_module,
            "generate_query",
            return_value=self._hosted_result(
                cypher="MATCH (company:Company) RETURN DISTINCT company.name AS company ORDER BY company"
            ),
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = query_module.main_query_cypher(["Which companies are in the graph?"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Local query stack unavailable: local qwen failure", stderr.getvalue())
        self.assertIn("Generating hosted fallback query...", stderr.getvalue())
        self.assertEqual(
            stdout.getvalue().strip(),
            "MATCH (company:Company) RETURN DISTINCT company.name AS company ORDER BY company",
        )

    def test_query_auto_falls_back_when_local_stack_startup_fails(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.object(
            query_module,
            "_run_local_stack",
            return_value=(None, "local stack python not found at /missing/python"),
        ), patch.object(
            query_module, "resolve_model_settings", return_value=self._model_settings()
        ), patch.object(query_module, "LLMExtractor", return_value=object()), patch.object(
            query_module,
            "generate_query",
            return_value=self._hosted_result(
                cypher="MATCH (company:Company) RETURN DISTINCT company.name AS company ORDER BY company"
            ),
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = query_module.main_query_cypher(["Which companies are in the graph?"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Local query stack unavailable: local stack python not found at /missing/python", stderr.getvalue())
        self.assertIn("Falling back to hosted query generation.", stderr.getvalue())
        self.assertIn("Generating hosted fallback query...", stderr.getvalue())

    def test_query_rejects_removed_jolly_stack_flag(self):
        stderr = io.StringIO()

        with redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
            query_module.main_query_cypher(["Which companies are in the graph?", "--stack", "jolly"])

        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("invalid choice", stderr.getvalue())

    def test_query_rejects_removed_local_planner_flag(self):
        stderr = io.StringIO()

        with redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
            query_module.main_query_cypher(["Which companies are in the graph?", "--local-planner", "lmstudio"])

        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("unrecognized arguments", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
