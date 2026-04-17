import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
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
    @staticmethod
    def _local_model_settings() -> SimpleNamespace:
        return SimpleNamespace(
            provider="local",
            model="local-model",
            base_url="http://localhost:1234/v1",
            api_mode="chat_completions",
            api_key="lm-studio",
            max_output_tokens=None,
        )

    @staticmethod
    def _output_layout(
        run_dir: Path,
        *,
        planned_output_dir: Path | None = None,
        root_dir: Path | None = None,
        company_name: str = "Microsoft",
        company_slug: str = "microsoft",
        pipeline: str = "canonical",
        keep_current_output: bool = False,
    ) -> SimpleNamespace:
        root_dir = root_dir or run_dir.parent
        return SimpleNamespace(
            company_name=company_name,
            company_slug=company_slug,
            pipeline=pipeline,
            run_token="run",
            root_dir=root_dir,
            latest_dir=root_dir / "latest",
            runs_dir=root_dir / "runs",
            failed_dir=root_dir / "failed",
            staging_root=root_dir / ".staging",
            staging_dir=run_dir,
            preserved_run_dir=root_dir / "runs" / "run",
            keep_current_output=keep_current_output,
            planned_output_dir=planned_output_dir or run_dir,
        )

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

    def test_pipeline_console_renders_stage_warning_messages(self):
        lines: list[str] = []
        console = PipelineConsole(printer=lines.append)

        console.handle_progress(
            "stage_start",
            index=7,
            title="Reflection 1 - Ontology compliance",
            details=[("triples in", 14)],
        )
        console.handle_progress(
            "stage_warning",
            message="Rule reflection failed after retries.",
        )

        self.assertTrue(any("warning:" in line and "Rule reflection failed after retries." in line for line in lines))

    def test_pipeline_console_confirms_graph_fallback_with_default_yes(self):
        lines: list[str] = []
        prompts: list[str] = []
        console = PipelineConsole(
            printer=lines.append,
            input_reader=lambda prompt: prompts.append(prompt) or "",
            is_interactive=lambda: True,
        )

        accepted = console.confirm_graph_fallback(stage_label="Rule reflection", triple_count=14)

        self.assertTrue(accepted)
        self.assertIn(
            "Rule reflection could not produce a usable graph. Load the last good graph from this run (14 triples)? [Y/n] ",
            prompts,
        )
        self.assertTrue(any("fallback:" in line and "kept the last good graph from this run" in line for line in lines))

    def test_pipeline_console_can_decline_graph_fallback(self):
        lines: list[str] = []
        console = PipelineConsole(
            printer=lines.append,
            input_reader=lambda prompt: "n",
            is_interactive=lambda: True,
        )

        accepted = console.confirm_graph_fallback(stage_label="Filing reflection", triple_count=2)

        self.assertFalse(accepted)
        self.assertTrue(any("fallback:" in line and "declined by user; stopping run" in line for line in lines))

    def test_pipeline_console_auto_keeps_graph_fallback_when_noninteractive(self):
        lines: list[str] = []

        def _unexpected_prompt(prompt: str) -> str:
            raise AssertionError(f"prompt should not be shown: {prompt}")

        console = PipelineConsole(
            printer=lines.append,
            input_reader=_unexpected_prompt,
            is_interactive=lambda: False,
        )

        accepted = console.confirm_graph_fallback(stage_label="Analyst critique", triple_count=1)

        self.assertTrue(accepted)
        self.assertTrue(
            any("fallback:" in line and "non-interactive terminal; kept the last good graph from this run" in line for line in lines)
        )

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

                def unload_company(self, company_name):
                    load_call["unloaded_company"] = company_name
                    return {
                        "company_name": company_name,
                        "scoped_nodes_deleted": 1,
                        "scoped_relationships_deleted": 2,
                        "company_relationships_deleted": 0,
                        "company_node_deleted": 1,
                        "orphan_nodes_deleted": 0,
                    }

                def setup_constraints(self):
                    pass

                def load_triples(self, triples, company_name):
                    load_call["company_name"] = company_name
                    load_call["triple_count"] = len(triples)
                    return len(triples)

                def close(self):
                    pass

            layout = self._output_layout(run_dir)

            with patch.object(main_module, "_prepare_output_layout", return_value=layout), patch.object(
                main_module, "finalize_successful_run", return_value=run_dir
            ), patch.object(
                main_module, "finalize_failed_run", return_value=run_dir
            ), patch.object(
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
                mock_resolve.return_value = self._local_model_settings()
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
            self.assertEqual(load_call["unloaded_company"], "Microsoft")

            mock_resolve.assert_called_once()
            mock_run_pipeline.assert_called_once()
            self.assertIs(mock_run_pipeline.call_args.kwargs["extractor"], fake_extractor)
            self.assertTrue(mock_run_pipeline.call_args.kwargs["stop_after_pass1"])

    def test_main_passes_expected_validation_options(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            filing_text = "ITEM 1. BUSINESS\nMicrosoft does business.\n"
            filing_path = tmp_path / "microsoft_10k.txt"
            filing_path.write_text(filing_text, encoding="utf-8")
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
            captured_validation: dict[str, object] = {}

            def validate_side_effect(triples, **kwargs):
                captured_validation["triples"] = triples
                captured_validation.update(kwargs)
                return {
                    "valid_triples": [resolved_triple.model_dump()],
                    "summary": {"invalid_triple_count": 0, "duplicate_triple_count": 0},
                }

            layout = self._output_layout(run_dir)

            with patch.object(main_module, "_prepare_output_layout", return_value=layout), patch.object(
                main_module, "finalize_successful_run", return_value=run_dir
            ), patch.object(
                main_module, "finalize_failed_run", return_value=run_dir
            ), patch.object(
                main_module, "resolve_model_settings", return_value=self._local_model_settings()
            ), patch.object(main_module, "LLMExtractor", return_value=fake_extractor), patch.object(
                main_module, "run_extraction_pipeline", return_value=fake_result
            ), patch.object(
                main_module, "resolve_entities", return_value=[resolved_triple]
            ), patch.object(
                main_module, "validate_triples", side_effect=validate_side_effect
            ), patch(
                "sys.argv", ["main.py", str(filing_path), "--output-dir", str(tmp_path / "outputs"), "--skip-neo4j"]
            ):
                exit_code = main_module.main()

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured_validation["triples"], [resolved_triple.model_dump()])
            self.assertEqual(captured_validation["source_text"], filing_text)
            self.assertFalse(captured_validation["require_text_grounding"])
            self.assertTrue(captured_validation["dedupe"])
            self.assertEqual(captured_validation["ontology_version"], "canonical")

    def test_main_uses_company_name_override_for_outputs_and_pipeline(self):
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
            layout = self._output_layout(
                run_dir,
                company_name="Microsoft Corporation",
                company_slug="microsoft_corporation",
            )

            with patch.object(main_module, "_prepare_output_layout", return_value=layout) as mock_prepare_layout, patch.object(
                main_module, "finalize_successful_run", return_value=run_dir
            ), patch.object(
                main_module, "finalize_failed_run", return_value=run_dir
            ), patch.object(
                main_module, "resolve_model_settings", return_value=self._local_model_settings()
            ), patch.object(main_module, "LLMExtractor", return_value=fake_extractor), patch.object(
                main_module, "run_extraction_pipeline", return_value=fake_result
            ) as mock_run_pipeline, patch.object(
                main_module, "resolve_entities", return_value=[resolved_triple]
            ), patch.object(
                main_module,
                "validate_triples",
                return_value={
                    "valid_triples": [resolved_triple.model_dump()],
                    "summary": {"invalid_triple_count": 0, "duplicate_triple_count": 0},
                },
            ), patch(
                "sys.argv",
                [
                    "main.py",
                    str(filing_path),
                    "--output-dir",
                    str(tmp_path / "outputs"),
                    "--skip-neo4j",
                    "--company-name",
                    "Microsoft Corporation",
                ],
            ):
                exit_code = main_module.main()

            self.assertEqual(exit_code, 0)
            mock_prepare_layout.assert_called_once()
            self.assertEqual(mock_prepare_layout.call_args.kwargs["company_name"], "Microsoft Corporation")
            self.assertEqual(mock_run_pipeline.call_args.kwargs["company_name"], "Microsoft Corporation")
            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["company_name"], "Microsoft Corporation")
            self.assertEqual(summary["company_slug"], "microsoft_corporation")

    def test_main_fails_when_resolution_produces_zero_triples(self):
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
            stdout = io.StringIO()

            layout = self._output_layout(run_dir)

            with patch.object(main_module, "_prepare_output_layout", return_value=layout), patch.object(
                main_module, "finalize_successful_run", return_value=run_dir
            ), patch.object(
                main_module, "finalize_failed_run", return_value=run_dir
            ), patch.object(
                main_module, "resolve_model_settings", return_value=self._local_model_settings()
            ), patch.object(main_module, "LLMExtractor", return_value=SimpleNamespace()), patch.object(
                main_module, "run_extraction_pipeline", return_value=fake_result
            ), patch.object(
                main_module, "resolve_entities", return_value=[]
            ), patch.object(
                main_module, "validate_triples"
            ) as mock_validate, redirect_stdout(stdout), patch(
                "sys.argv", ["main.py", str(filing_path), "--output-dir", str(tmp_path / "outputs"), "--skip-neo4j"]
            ):
                exit_code = main_module.main()

            self.assertEqual(exit_code, 1)
            self.assertIn("RUN FAILED", stdout.getvalue())
            self.assertIn("zero resolved triples", stdout.getvalue())
            self.assertFalse((run_dir / "resolved_triples.json").exists())
            mock_validate.assert_not_called()
            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "failed")
            self.assertIn("zero resolved triples", summary["error"])

    def test_main_fails_when_validation_rejects_all_resolved_triples(self):
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
            stdout = io.StringIO()

            layout = self._output_layout(run_dir)

            with patch.object(main_module, "_prepare_output_layout", return_value=layout), patch.object(
                main_module, "finalize_successful_run", return_value=run_dir
            ), patch.object(
                main_module, "finalize_failed_run", return_value=run_dir
            ), patch.object(
                main_module, "resolve_model_settings", return_value=self._local_model_settings()
            ), patch.object(main_module, "LLMExtractor", return_value=SimpleNamespace()), patch.object(
                main_module, "run_extraction_pipeline", return_value=fake_result
            ), patch.object(
                main_module, "resolve_entities", return_value=[resolved_triple]
            ), patch.object(
                main_module,
                "validate_triples",
                return_value={
                    "valid_triples": [],
                    "summary": {"invalid_triple_count": 1, "duplicate_triple_count": 0},
                },
            ), patch.object(main_module, "Neo4jLoader") as mock_loader, redirect_stdout(stdout), patch(
                "sys.argv", ["main.py", str(filing_path), "--output-dir", str(tmp_path / "outputs"), "--skip-neo4j"]
            ):
                exit_code = main_module.main()

            self.assertEqual(exit_code, 1)
            self.assertIn("RUN FAILED", stdout.getvalue())
            self.assertIn("ontology validation", stdout.getvalue())
            self.assertTrue((run_dir / "validation_report.json").exists())
            self.assertFalse((run_dir / "resolved_triples.json").exists())
            mock_loader.assert_not_called()
            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "failed")
            self.assertIn("ontology validation", summary["error"])

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

            layout = self._output_layout(run_dir, pipeline="analyst")

            with patch.object(main_module, "_prepare_output_layout", return_value=layout), patch.object(
                main_module, "finalize_successful_run", return_value=run_dir
            ), patch.object(
                main_module, "finalize_failed_run", return_value=run_dir
            ), patch.object(
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
                mock_resolve.return_value = self._local_model_settings()
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

    def test_main_analyst_pipeline_persists_partial_memos_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            filing_path = tmp_path / "microsoft_10k.txt"
            filing_path.write_text("ITEM 1. BUSINESS\nMicrosoft does business.\n", encoding="utf-8")
            run_dir = tmp_path / "outputs" / "run"

            failed_result = AnalystPipelineResult(
                success=False,
                foundation_memo=AnalystBusinessModelMemo(
                    content="ANALYTICAL FRAME\nSummary:\nFoundation memo.\n",
                ),
                augmented_memo=AnalystBusinessModelMemo(
                    content="ANALYTICAL FRAME\nSummary:\nAugmented memo.\n",
                ),
                error="Failed after 3 attempts. Last error: Empty response from model.",
            )
            fake_extractor = SimpleNamespace()

            layout = self._output_layout(run_dir, pipeline="analyst")

            with patch.object(main_module, "_prepare_output_layout", return_value=layout), patch.object(
                main_module, "finalize_successful_run", return_value=run_dir
            ), patch.object(
                main_module, "finalize_failed_run", return_value=run_dir
            ), patch.object(
                main_module, "resolve_model_settings"
            ) as mock_resolve, patch.object(main_module, "LLMExtractor", return_value=fake_extractor), patch.object(
                main_module, "run_extraction_pipeline", return_value=failed_result
            ), patch(
                "sys.argv", ["main.py", str(filing_path), "--pipeline", "analyst", "--output-dir", str(tmp_path / "outputs"), "--skip-neo4j"]
            ):
                mock_resolve.return_value = self._local_model_settings()
                exit_code = main_module.main()

            self.assertEqual(exit_code, 1)
            self.assertTrue((run_dir / "analyst_memo_foundation.md").exists())
            self.assertTrue((run_dir / "analyst_memo_augmented.md").exists())
            self.assertFalse((run_dir / "analyst_graph_compilation.json").exists())

            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "failed")
            self.assertIn("Empty response from model", summary["error"])
            mock_resolve.assert_called_once()

    def test_main_keep_current_output_requires_skip_neo4j(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            filing_path = tmp_path / "microsoft_10k.txt"
            filing_path.write_text("ITEM 1. BUSINESS\nMicrosoft does business.\n", encoding="utf-8")

            with self.assertRaises(SystemExit) as exc:
                with patch(
                    "sys.argv",
                    ["main.py", str(filing_path), "--keep-current-output"],
                ):
                    main_module.main()

        self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
