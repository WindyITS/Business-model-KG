import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llm_extraction.models import (
    AnalystBusinessModelMemo,
    AnalystPipelineResult,
    CanonicalPipelineResult,
    KnowledgeGraphExtraction,
    Triple,
)
from runtime import main as main_module
from runtime.main import (
    PipelineConsole,
    _format_duration,
    _format_token_visual,
    _infer_company_name,
    _mode_name,
)


class RuntimeMainTests(unittest.TestCase):
    def test_mode_name_is_canonical_pipeline(self):
        args = SimpleNamespace(pipeline="canonical")
        self.assertEqual(_mode_name(args), "canonical_pipeline")

    def test_mode_name_is_analyst_pipeline(self):
        args = SimpleNamespace(pipeline="analyst")
        self.assertEqual(_mode_name(args), "analyst_pipeline")

    def test_format_duration_uses_compact_seconds_and_minutes(self):
        self.assertEqual(_format_duration(0.42), "0.4s")
        self.assertEqual(_format_duration(75.2), "1m15s")

    def test_format_token_visual_renders_plain_counts(self):
        self.assertEqual(_format_token_visual(3244), "3,244")
        self.assertEqual(_format_token_visual(3244, 20000), "3,244/20,000")

    def test_pipeline_console_renders_pass_progress(self):
        lines: list[str] = []
        console = PipelineConsole(printer=lines.append)

        console.start_run(
            started_at=datetime(2026, 4, 14, 7, 19, 1, tzinfo=timezone.utc),
            source_file=Path("data/palantir_10k.txt"),
            run_dir=Path("outputs/palantir_run"),
            pipeline="canonical",
            provider="local",
            model="local-model",
            neo4j_enabled=True,
            llm_token_cap=None,
        )
        console.handle_progress(
            "stage_start",
            index=2,
            title="Pass 1 - Structural skeleton",
            extracts="HAS_SEGMENT, OFFERS",
        )
        console.handle_progress("llm_call_complete", attempt=1, max_retries=3, tokens=3244)
        console.handle_progress("stage_complete", details=[("result", "9 triples")])

        self.assertIn("KG PIPELINE RUN", lines)
        self.assertIn("Neo4j:     enabled (notifications disabled)", lines)
        self.assertIn("[02/10] Pass 1 - Structural skeleton", lines)
        self.assertTrue(any("extracts:" in line and "HAS_SEGMENT, OFFERS" in line for line in lines))
        self.assertTrue(any("llm:" in line and "attempt 1/3, tokens=3,244" in line for line in lines))
        self.assertTrue(any("result:" in line and "9 triples" in line for line in lines))

    def test_pipeline_console_renders_stage_start_details(self):
        lines: list[str] = []
        console = PipelineConsole(printer=lines.append)

        console.handle_progress(
            "stage_start",
            index=7,
            title="Reflection 1 - Ontology compliance",
            details=[("triples in", 14)],
        )

        self.assertIn("[07/10] Reflection 1 - Ontology compliance", lines)
        self.assertTrue(any("triples in:" in line and "14" in line for line in lines))

    def test_pipeline_console_renders_live_retry_updates(self):
        lines: list[str] = []
        console = PipelineConsole(printer=lines.append)

        console.handle_progress(
            "stage_start",
            index=2,
            title="Pass 1 - Structural skeleton",
            extracts="HAS_SEGMENT, OFFERS",
        )
        console.handle_progress("llm_call_start", attempt=1, max_retries=3)
        console.handle_progress(
            "llm_call_error",
            attempt=1,
            max_retries=3,
            error="Error code: 500 - Unexpected token '<'",
            will_retry=True,
        )
        console.handle_progress("llm_call_start", attempt=2, max_retries=3)

        self.assertTrue(any("llm:" in line and "starting attempt 1/3" in line for line in lines))
        self.assertTrue(
            any(
                "llm:" in line
                and "attempt 1/3 failed, retrying: Error code: 500 - Unexpected token '<'" in line
                for line in lines
            )
        )
        self.assertTrue(any("llm:" in line and "starting attempt 2/3" in line for line in lines))

    def test_infer_company_name_uses_filename_stem(self):
        text = "ITEM 1. BUSINESS\nMicrosoft is a technology company.\n"

        inferred = _infer_company_name(Path("microsoft_10k.txt"), text)

        self.assertEqual(inferred, "Microsoft")

    def test_infer_company_name_ignores_filing_text(self):
        text = (
            "ITEM 1. BUSINESS\n"
            "We have built four principal software platforms, Palantir Gotham and Palantir Foundry, "
            "which enable institutions to transform data.\n"
        )

        inferred = _infer_company_name(Path("palantir_10k.txt"), text)

        self.assertEqual(inferred, "Palantir")

    def test_infer_company_name_prefers_filename_even_when_text_mentions_another_company(self):
        text = "ITEM 1. BUSINESS\nMicrosoft is a technology company.\n"

        inferred = _infer_company_name(Path("google_10k.txt"), text)

        self.assertEqual(inferred, "Google")

    def test_main_only_pass1_writes_artifacts_and_loads_by_default(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            filing_path = tmp_path / "microsoft_10k.txt"
            filing_path.write_text("ITEM 1. BUSINESS\nMicrosoft does business.\n", encoding="utf-8")
            run_dir = tmp_path / "outputs" / "run"
            resolved_triple = Triple(
                subject="Microsoft",
                subject_type="Company",
                relation="HAS_SEGMENT",
                object="Intelligent Cloud",
                object_type="BusinessSegment",
            )

            fake_result = CanonicalPipelineResult(
                success=True,
                skeleton_extraction=KnowledgeGraphExtraction(
                    extraction_notes="skeleton",
                    triples=[resolved_triple],
                ),
            )
            fake_extractor = SimpleNamespace()
            fake_validation_report = {
                "valid_triples": [resolved_triple.model_dump()],
                "summary": {
                    "invalid_triple_count": 0,
                    "duplicate_triple_count": 0,
                },
            }

            load_call: dict[str, object] = {}

            class FakeNeo4jLoader:
                def __init__(self, uri, user, password):
                    self.uri = uri
                    self.user = user
                    self.password = password

                def clear_graph(self):
                    pass

                def setup_constraints(self):
                    pass

                def load_triples(self, triples, company_name):
                    load_call["company_name"] = company_name
                    load_call["triple_count"] = len(triples)
                    return len(triples)

                def close(self):
                    pass

            with patch.object(main_module, "_build_run_dir", return_value=run_dir), patch.object(
                main_module, "resolve_model_settings"
            ) as mock_resolve, patch.object(main_module, "LLMExtractor", return_value=fake_extractor), patch.object(
                main_module, "run_extraction_pipeline", return_value=fake_result
            ) as mock_run_pipeline, patch.object(
                main_module, "resolve_entities", return_value=[resolved_triple]
            ), patch.object(
                main_module, "validate_triples", return_value=fake_validation_report
            ), patch.object(
                main_module, "Neo4jLoader", FakeNeo4jLoader
            ), patch(
                "sys.argv", ["main.py", str(filing_path), "--output-dir", str(tmp_path / "outputs"), "--only-pass1"]
            ):
                mock_resolve.return_value = SimpleNamespace(
                    provider="local",
                    model="local-model",
                    base_url="http://localhost:1234/v1",
                    api_mode="chat_completions",
                    api_key="lm-studio",
                    max_output_tokens=None,
                )
                exit_code = main_module.main()

            self.assertEqual(exit_code, 0)
            self.assertTrue((run_dir / "skeleton_extraction.json").exists())
            self.assertFalse((run_dir / "pass2_channels_extraction.json").exists())
            self.assertTrue((run_dir / "resolved_triples.json").exists())
            self.assertTrue((run_dir / "validation_report.json").exists())

            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["pass1_only"])
            self.assertEqual(summary["stage_count"], 4)
            self.assertFalse(summary["skip_neo4j"])
            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["loaded_triple_count"], 1)
            self.assertEqual(load_call["company_name"], "Microsoft")
            self.assertEqual(load_call["triple_count"], 1)

            mock_resolve.assert_called_once()
            mock_run_pipeline.assert_called_once()
            self.assertIs(mock_run_pipeline.call_args.kwargs["extractor"], fake_extractor)
            self.assertTrue(mock_run_pipeline.call_args.kwargs["stop_after_pass1"])

    def test_main_analyst_pipeline_writes_memo_and_graph_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            filing_path = tmp_path / "microsoft_10k.txt"
            filing_path.write_text("ITEM 1. BUSINESS\nMicrosoft does business.\n", encoding="utf-8")
            run_dir = tmp_path / "outputs" / "run"
            resolved_triple = Triple(
                subject="Microsoft",
                subject_type="Company",
                relation="HAS_SEGMENT",
                object="Intelligent Cloud",
                object_type="BusinessSegment",
            )

            fake_result = AnalystPipelineResult(
                success=True,
                foundation_memo=AnalystBusinessModelMemo(
                    content="ANALYTICAL FRAME\nSummary:\nCloud and software franchises drive the business.\n",
                ),
                augmented_memo=AnalystBusinessModelMemo(
                    content="ANALYTICAL FRAME\nSummary:\nCloud and software franchises drive the business.\n",
                ),
                compiled_graph_extraction=KnowledgeGraphExtraction(
                    extraction_notes="compiled",
                    triples=[resolved_triple],
                ),
                final_extraction=KnowledgeGraphExtraction(
                    extraction_notes="final",
                    triples=[resolved_triple],
                ),
                critique_audit={"kept_triple_count": 1, "raw_triple_count": 1},
            )
            fake_extractor = SimpleNamespace()
            fake_validation_report = {
                "valid_triples": [resolved_triple.model_dump()],
                "summary": {
                    "invalid_triple_count": 0,
                    "duplicate_triple_count": 0,
                },
            }

            with patch.object(main_module, "_build_run_dir", return_value=run_dir), patch.object(
                main_module, "resolve_model_settings"
            ) as mock_resolve, patch.object(main_module, "LLMExtractor", return_value=fake_extractor), patch.object(
                main_module, "run_extraction_pipeline", return_value=fake_result
            ) as mock_run_pipeline, patch.object(
                main_module, "resolve_entities", return_value=[resolved_triple]
            ), patch.object(
                main_module, "validate_triples", return_value=fake_validation_report
            ), patch(
                "sys.argv", ["main.py", str(filing_path), "--pipeline", "analyst", "--output-dir", str(tmp_path / "outputs"), "--skip-neo4j"]
            ):
                mock_resolve.return_value = SimpleNamespace(
                    provider="local",
                    model="local-model",
                    base_url="http://localhost:1234/v1",
                    api_mode="chat_completions",
                    api_key="lm-studio",
                    max_output_tokens=None,
                )
                exit_code = main_module.main()

            self.assertEqual(exit_code, 0)
            self.assertTrue((run_dir / "analyst_memo_foundation.md").exists())
            self.assertTrue((run_dir / "analyst_memo_augmented.md").exists())
            self.assertTrue((run_dir / "analyst_graph_compilation.json").exists())
            self.assertTrue((run_dir / "analyst_graph_critique.json").exists())
            self.assertTrue((run_dir / "resolved_triples.json").exists())

            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["stage_count"], 7)
            self.assertEqual(summary["status"], "success")
            self.assertGreater(summary["foundation_memo_character_count"], 0)
            self.assertGreater(summary["augmented_memo_character_count"], 0)
            self.assertTrue(summary["skip_neo4j"])

            mock_resolve.assert_called_once()
            mock_run_pipeline.assert_called_once()
            self.assertEqual(mock_run_pipeline.call_args.kwargs["pipeline"], "analyst")
            self.assertFalse(mock_run_pipeline.call_args.kwargs["stop_after_pass1"])


if __name__ == "__main__":
    unittest.main()
