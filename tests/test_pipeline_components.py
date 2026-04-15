import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from os import environ
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import llm_extractor as llm_extractor_module
from entity_resolver import clean_entity_name as clean_resolved_entity_name, resolve_entities
from evaluate_graph import _load_triples_from_json, evaluate
from openai import InternalServerError
from llm_extractor import (
    LLMExtractor,
    KnowledgeGraphExtraction,
    Triple,
    _canonical_pipeline_system_prompt,
    _canonical_rule_reflection_system_prompt,
    _canonical_reflection_system_prompt,
    aggregate_extraction_audits,
    audit_knowledge_graph_payload,
)
from main import (
    PipelineConsole,
    _format_duration,
    _format_token_visual,
    _infer_company_name,
    _mode_name,
)
from model_provider import resolve_model_settings


class PipelineComponentTests(unittest.TestCase):
    def test_entity_resolver_preserves_best_surface_form(self):
        extractions = [
            KnowledgeGraphExtraction(
                extraction_notes="ok",
                triples=[
                    Triple(
                        subject="OpenAI",
                        subject_type="Company",
                        relation="PARTNERS_WITH",
                        object="NASA",
                        object_type="Company",
                    ),
                    Triple(
                        subject="openai",
                        subject_type="Company",
                        relation="PARTNERS_WITH",
                        object="NASA",
                        object_type="Company",
                    ),
                ],
            )
        ]

        resolved = resolve_entities(extractions)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].subject, "OpenAI")

    def test_entity_resolver_strips_curly_quotes(self):
        self.assertEqual(clean_resolved_entity_name('  “Apollo”  '), "Apollo")

        extractions = [
            KnowledgeGraphExtraction(
                extraction_notes="ok",
                triples=[
                    Triple(
                        subject="Palantir",
                        subject_type="Company",
                        relation="OFFERS",
                        object="“Apollo”",
                        object_type="Offering",
                    ),
                    Triple(
                        subject="Palantir",
                        subject_type="Company",
                        relation="OFFERS",
                        object="Apollo",
                        object_type="Offering",
                    ),
                ],
            )
        ]

        resolved = resolve_entities(extractions)

        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].object, "Apollo")

    def test_evaluator_accepts_resolved_triples_payload(self):
        payload = {
            "resolved_triples": [
                {
                    "subject": "Microsoft",
                    "subject_type": "Company",
                    "relation": "OFFERS",
                    "object": "Azure",
                    "object_type": "Offering",
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "resolved.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            triples = _load_triples_from_json(str(path))

        self.assertEqual(len(triples), 1)

    def test_evaluator_accepts_validation_report_valid_triples_payload(self):
        payload = {
            "valid_triples": [
                {
                    "subject": "Microsoft",
                    "subject_type": "Company",
                    "relation": "OFFERS",
                    "object": "Azure",
                    "object_type": "Offering",
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "validation_report.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            triples = _load_triples_from_json(str(path))

        self.assertEqual(len(triples), 1)

    def test_evaluate_scores_exact_match(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            payload = {
                "triples": [
                    {
                        "subject": "Microsoft",
                        "subject_type": "Company",
                        "relation": "OFFERS",
                        "object": "Azure",
                        "object_type": "Offering",
                    }
                ]
            }
            gold_path = Path(tmp_dir) / "gold.json"
            pred_path = Path(tmp_dir) / "pred.json"
            gold_path.write_text(json.dumps(payload), encoding="utf-8")
            pred_path.write_text(json.dumps(payload), encoding="utf-8")

            gold = _load_triples_from_json(str(gold_path))
            predicted = _load_triples_from_json(str(pred_path))
            report = evaluate(predicted, gold)

        self.assertEqual(report["precision"], 1.0)
        self.assertEqual(report["recall"], 1.0)
        self.assertEqual(report["f1"], 1.0)

    def test_mode_name_is_canonical_pipeline(self):
        args = SimpleNamespace(pipeline="canonical")
        self.assertEqual(_mode_name(args), "canonical_pipeline")

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

    def test_payload_audit_counts_malformed_and_ontology_rejections(self):
        payload = {
            "triples": [
                {"subject": "Microsoft", "subject_type": "Company", "relation": "OFFERS", "object": "Azure", "object_type": "Offering"},
                {"subject": "Microsoft", "subject_type": "Company", "relation": "SERVES", "object": "startups", "object_type": "CustomerType"},
                {"subject": "Microsoft", "subject_type": "Company", "relation": "OFFERS", "object": "Copilot"},
                "not-a-dict",
            ]
        }

        valid_triples, audit = audit_knowledge_graph_payload(payload)

        self.assertEqual(len(valid_triples), 1)
        self.assertEqual(audit["raw_triple_count"], 4)
        self.assertEqual(audit["malformed_triple_count"], 2)
        self.assertEqual(audit["ontology_rejected_triple_count"], 1)

    def test_payload_audit_accepts_top_level_triple_list(self):
        payload = [
            {
                "subject": "Microsoft",
                "subject_type": "Company",
                "relation": "OFFERS",
                "object": "Azure",
                "object_type": "Offering",
            }
        ]

        valid_triples, audit = audit_knowledge_graph_payload(payload)

        self.assertEqual(len(valid_triples), 1)
        self.assertEqual(audit["raw_triple_count"], 1)
        self.assertEqual(audit["kept_triple_count"], 1)

    def test_load_json_payload_accepts_markdown_fenced_json_without_truncation_warning(self):
        payload_text = '```json\n{"extraction_notes":"ok","triples":[]}\n```'

        with patch.object(llm_extractor_module.logger, "warning") as mock_warning:
            payload, recovered, used_fallback = LLMExtractor._load_json_payload(
                payload_text,
                '{"extraction_notes":"fallback","triples":[]}',
            )

        self.assertEqual(payload["extraction_notes"], "ok")
        self.assertEqual(payload["triples"], [])
        self.assertTrue(recovered)
        self.assertFalse(used_fallback)
        mock_warning.assert_not_called()

    def test_load_json_payload_warns_for_likely_truncation(self):
        payload_text = '{"extraction_notes":"ok","triples":[]'

        with patch.object(llm_extractor_module.logger, "warning") as mock_warning:
            payload, recovered, used_fallback = LLMExtractor._load_json_payload(
                payload_text,
                '{"extraction_notes":"fallback","triples":[]}',
            )

        self.assertEqual(payload["extraction_notes"], "fallback")
        self.assertEqual(payload["triples"], [])
        self.assertTrue(recovered)
        self.assertTrue(used_fallback)
        mock_warning.assert_any_call("Model response may be truncated. Attempting to salvage JSON prefix...")

    def test_structured_call_retries_when_json_falls_back_to_placeholder(self):
        extractor = LLMExtractor(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="local-model",
            provider="local",
            api_mode="chat_completions",
        )
        events: list[tuple[str, dict[str, object]]] = []
        extractor.progress_callback = lambda event, **payload: events.append((event, payload))

        responses = iter(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(content="<html>temporary upstream error</html>", refusal=None),
                        )
                    ],
                    usage=SimpleNamespace(completion_tokens=4),
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(content='{"extraction_notes":"ok","triples":[]}', refusal=None),
                        )
                    ],
                    usage=SimpleNamespace(completion_tokens=9),
                ),
            ]
        )
        extractor.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: next(responses)))
        )

        with patch.object(llm_extractor_module.logger, "warning"):
            parsed_model, raw_content, attempts_used, audit = extractor._call_structured_messages(
                messages=[{"role": "user", "content": "return json"}],
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"fallback","triples":[]}',
                max_retries=2,
            )

        self.assertEqual(parsed_model.extraction_notes, "ok")
        self.assertEqual(raw_content, '{"extraction_notes":"ok","triples":[]}')
        self.assertEqual(attempts_used, 2)
        self.assertFalse(audit["payload_parse_recovered"])
        self.assertEqual(
            events,
            [
                ("llm_call_start", {"attempt": 1, "max_retries": 2}),
                (
                    "llm_call_error",
                    {
                        "attempt": 1,
                        "max_retries": 2,
                        "error": "Model response was not recoverable as JSON.",
                        "will_retry": True,
                    },
                ),
                ("llm_call_start", {"attempt": 2, "max_retries": 2}),
                ("llm_call_complete", {"attempt": 2, "max_retries": 2, "tokens": 9}),
            ],
        )

    def test_merge_relation_subset_into_base_replaces_only_allowed_relations(self):
        base = KnowledgeGraphExtraction(
            extraction_notes="base",
            triples=[
                Triple(
                    subject="Microsoft",
                    subject_type="Company",
                    relation="HAS_SEGMENT",
                    object="Intelligent Cloud",
                    object_type="BusinessSegment",
                ),
                Triple(
                    subject="Intelligent Cloud",
                    subject_type="BusinessSegment",
                    relation="SERVES",
                    object="large enterprises",
                    object_type="CustomerType",
                ),
            ],
        )
        subset = KnowledgeGraphExtraction(
            extraction_notes="subset",
            triples=[
                Triple(
                    subject="Intelligent Cloud",
                    subject_type="BusinessSegment",
                    relation="SERVES",
                    object="developers",
                    object_type="CustomerType",
                ),
                Triple(
                    subject="Microsoft",
                    subject_type="Company",
                    relation="OPERATES_IN",
                    object="United States",
                    object_type="Place",
                ),
            ],
        )

        merged = LLMExtractor._merge_relation_subset_into_base(
            base,
            subset,
            allowed_relations={"SERVES"},
        )

        self.assertEqual(
            [(triple.relation, triple.object) for triple in merged.triples],
            [("HAS_SEGMENT", "Intelligent Cloud"), ("SERVES", "developers")],
        )

    def test_aggregate_extraction_audits_sums_counts(self):
        aggregated = aggregate_extraction_audits(
            [
                {"raw_triple_count": 2, "malformed_triple_count": 1, "ontology_rejected_triple_count": 0, "duplicate_triple_count": 0, "kept_triple_count": 1, "invalid_issue_counts": {"empty_subject": 1}, "payload_parse_recovered": True},
                {"raw_triple_count": 3, "malformed_triple_count": 0, "ontology_rejected_triple_count": 2, "duplicate_triple_count": 1, "kept_triple_count": 0, "invalid_issue_counts": {"non_canonical_label": 2}, "payload_parse_recovered": False},
            ]
        )

        self.assertEqual(aggregated["raw_triple_count"], 5)
        self.assertEqual(aggregated["malformed_triple_count"], 1)
        self.assertEqual(aggregated["ontology_rejected_triple_count"], 2)
        self.assertEqual(aggregated["duplicate_triple_count"], 1)
        self.assertEqual(aggregated["payload_parse_recovered_count"], 1)

    def test_canonical_system_prompts_explicitly_forbid_markdown_fences(self):
        self.assertIn("Do not wrap the JSON in markdown code fences.", _canonical_pipeline_system_prompt("x"))
        self.assertIn("Do not wrap the JSON in markdown code fences.", _canonical_reflection_system_prompt("x"))

    def test_canonical_pipeline_system_prompt_omits_pass_specific_sections(self):
        prompt = _canonical_pipeline_system_prompt("full filing text here")

        self.assertIn("<ontology>", prompt)
        self.assertIn("<canonical_graph_policy>", prompt)
        self.assertNotIn("<canonical_labels>", prompt)
        self.assertNotIn("<structure_rules>", prompt)
        self.assertNotIn("<normalization_rules>", prompt)
        self.assertNotIn("<corporate_shell_rules>", prompt)
        self.assertNotIn("<inference_policy>", prompt)
        self.assertNotIn("SELLS_THROUGH should default to BusinessSegment.", prompt)

    def test_canonical_reflection_system_prompt_includes_full_text_and_ontology_rules(self):
        prompt = _canonical_reflection_system_prompt("full filing text here")

        self.assertIn("<source_filing>", prompt)
        self.assertIn("full filing text here", prompt)
        self.assertIn("<canonical_label_definitions>", prompt)
        self.assertIn("<canonical_graph_policy>", prompt)
        self.assertIn("Supervisor and editor of an existing draft graph", prompt)
        self.assertIn("SELLS_THROUGH should default to BusinessSegment.", prompt)

    def test_canonical_rule_reflection_system_prompt_includes_rules_without_filing(self):
        prompt = _canonical_rule_reflection_system_prompt()

        self.assertIn("<canonical_label_definitions>", prompt)
        self.assertIn("<canonical_graph_policy>", prompt)
        self.assertIn("No filing text is provided in this step by design.", prompt)
        self.assertNotIn("<source_filing>", prompt)

    def test_local_provider_preserves_system_messages(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
            {"role": "assistant", "content": "asst"},
        ]

        prepared = LLMExtractor._prepare_messages_for_provider(messages, "local")

        self.assertEqual(prepared, messages)

    def test_opencode_go_rewrites_system_messages_to_user(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
            {"role": "assistant", "content": "asst"},
        ]

        prepared = LLMExtractor._prepare_messages_for_provider(messages, "opencode-go")

        self.assertEqual(
            prepared,
            [
                {"role": "user", "content": "sys"},
                {"role": "user", "content": "usr"},
                {"role": "assistant", "content": "asst"},
            ],
        )

    def test_local_provider_defaults_to_lm_studio_shape(self):
        settings = resolve_model_settings(provider="local")

        self.assertEqual(settings.provider, "local")
        self.assertEqual(settings.model, "local-model")
        self.assertEqual(settings.base_url, "http://localhost:1234/v1")
        self.assertEqual(settings.api_mode, "chat_completions")
        self.assertEqual(settings.api_key, "lm-studio")
        self.assertIsNone(settings.max_output_tokens)

    def test_opencode_go_normalizes_full_endpoint_url(self):
        settings = resolve_model_settings(
            provider="opencode-go",
            model="kimi-k2.5",
            base_url="https://opencode.ai/zen/go/v1/chat/completions",
            api_key="secret",
        )

        self.assertEqual(settings.base_url, "https://opencode.ai/zen/go/v1")
        self.assertEqual(settings.api_mode, "chat_completions")
        self.assertEqual(settings.max_output_tokens, 20000)

    def test_opencode_go_normalizes_full_messages_endpoint_url(self):
        settings = resolve_model_settings(
            provider="opencode-go",
            model="minimax-m2.7",
            base_url="https://opencode.ai/zen/go/v1/messages",
            api_key="secret",
        )

        self.assertEqual(settings.base_url, "https://opencode.ai/zen/go/v1")
        self.assertEqual(settings.api_mode, "messages")
        self.assertEqual(settings.max_output_tokens, 20000)

    def test_opencode_go_defaults_to_kimi(self):
        settings = resolve_model_settings(
            provider="opencode-go",
            api_key="secret",
        )

        self.assertEqual(settings.model, "kimi-k2.5")
        self.assertEqual(settings.api_mode, "chat_completions")

    def test_opencode_go_reads_api_key_from_environment(self):
        with patch.dict(environ, {"OPENCODE_API_KEY": "env-secret"}, clear=True):
            settings = resolve_model_settings(
                provider="opencode-go",
                model="kimi-k2.5",
            )

        self.assertEqual(settings.api_key, "env-secret")
        self.assertEqual(settings.max_output_tokens, 20000)

    def test_opencode_go_honors_explicit_output_cap(self):
        settings = resolve_model_settings(
            provider="opencode-go",
            model="kimi-k2.5",
            api_key="secret",
            max_output_tokens=1024,
        )

        self.assertEqual(settings.max_output_tokens, 1024)

    def test_opencode_go_accepts_mimo_as_explicit_override(self):
        settings = resolve_model_settings(
            provider="opencode-go",
            model="mimo-v2-pro",
            api_key="secret",
        )

        self.assertEqual(settings.model, "mimo-v2-pro")
        self.assertEqual(settings.api_mode, "chat_completions")

    def test_opencode_go_accepts_human_friendly_model_aliases(self):
        settings = resolve_model_settings(
            provider="opencode-go",
            model="MiniMax M2.7",
            api_key="secret",
        )

        self.assertEqual(settings.model, "minimax-m2.7")
        self.assertEqual(settings.api_mode, "messages")

    def test_opencode_go_accepts_prefixed_model_ids(self):
        settings = resolve_model_settings(
            provider="opencode-go",
            model="opencode-go/kimi-k2.5",
            api_key="secret",
        )

        self.assertEqual(settings.model, "kimi-k2.5")
        self.assertEqual(settings.api_mode, "chat_completions")

    def test_messages_request_payload_keeps_conversation_history_shape(self):
        payload = LLMExtractor._messages_request_payload(
            [
                {"role": "user", "content": "sys"},
                {"role": "user", "content": "usr"},
                {"role": "assistant", "content": "asst"},
            ],
            model="minimax-m2.7",
            max_output_tokens=2048,
            temperature=0.0,
        )

        self.assertEqual(payload["model"], "minimax-m2.7")
        self.assertEqual(payload["max_tokens"], 2048)
        self.assertEqual(
            payload["messages"],
            [
                {"role": "user", "content": "sys"},
                {"role": "user", "content": "usr"},
                {"role": "assistant", "content": "asst"},
            ],
        )

    def test_messages_api_parses_text_blocks(self):
        class _FakeHTTPResponse:
            status = 200
            headers = {"content-type": "application/json"}

            def __init__(self, payload):
                self._payload = payload

            def read(self):
                return json.dumps(self._payload).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        extractor = LLMExtractor(
            base_url="https://opencode.ai/zen/go/v1",
            api_key="secret",
            model="minimax-m2.7",
            provider="opencode-go",
            api_mode="messages",
            max_output_tokens=2048,
        )

        response_payload = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": '{"extraction_notes":"ok","triples":[]}'}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        with patch.object(
            llm_extractor_module.urllib.request,
            "urlopen",
            return_value=_FakeHTTPResponse(response_payload),
        ) as mock_urlopen:
            content = extractor._call_messages_api(
                request_messages=[{"role": "user", "content": "Hello"}],
                temperature=0.0,
                call_label="test",
                attempt=1,
                max_retries=2,
            )

        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://opencode.ai/zen/go/v1/messages")
        self.assertEqual(content, '{"extraction_notes":"ok","triples":[]}')

    def test_structured_call_emits_retry_progress_and_reports_last_error(self):
        extractor = LLMExtractor(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="local-model",
            provider="local",
            api_mode="chat_completions",
        )
        events: list[tuple[str, dict[str, object]]] = []
        extractor.progress_callback = lambda event, **payload: events.append((event, payload))

        success_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content='{"extraction_notes":"ok","triples":[]}', refusal=None),
                )
            ],
            usage=SimpleNamespace(completion_tokens=17),
        )

        class _CreateCall:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("provider unavailable")
                return success_response

        extractor.client = SimpleNamespace(chat=SimpleNamespace(completions=_CreateCall()))

        with patch.object(llm_extractor_module.logger, "warning"):
            parsed_model, raw_content, attempts_used, audit = extractor._call_structured_messages(
                messages=[{"role": "user", "content": "return json"}],
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"fallback","triples":[]}',
                max_retries=2,
            )

        self.assertEqual(parsed_model.extraction_notes, "ok")
        self.assertEqual(raw_content, '{"extraction_notes":"ok","triples":[]}')
        self.assertEqual(attempts_used, 2)
        self.assertFalse(audit["payload_parse_recovered"])
        self.assertEqual(
            events,
            [
                ("llm_call_start", {"attempt": 1, "max_retries": 2}),
                (
                    "llm_call_error",
                    {
                        "attempt": 1,
                        "max_retries": 2,
                        "error": "provider unavailable",
                        "will_retry": True,
                    },
                ),
                ("llm_call_start", {"attempt": 2, "max_retries": 2}),
                ("llm_call_complete", {"attempt": 2, "max_retries": 2, "tokens": 17}),
            ],
        )

        failing_extractor = LLMExtractor(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="local-model",
            provider="local",
            api_mode="chat_completions",
        )
        failing_extractor.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("still down")))
            )
        )

        with patch.object(llm_extractor_module.logger, "warning"):
            with self.assertRaises(llm_extractor_module.ExtractionError) as ctx:
                failing_extractor._call_structured_messages(
                    messages=[{"role": "user", "content": "return json"}],
                    schema_name="KnowledgeGraphExtraction",
                    schema_model=KnowledgeGraphExtraction,
                    fallback_payload='{"extraction_notes":"fallback","triples":[]}',
                    max_retries=2,
                )

        self.assertEqual(str(ctx.exception), "Failed after 2 attempts. Last error: still down")

    def test_structured_call_logs_http_diagnostics_from_openai_errors(self):
        extractor = LLMExtractor(
            base_url="https://opencode.ai/zen/go/v1",
            api_key="secret",
            model="mimo-v2-pro",
            provider="opencode-go",
            api_mode="chat_completions",
        )
        response_body = (
            '{"type":"error","message":"Unexpected token \'<\', '
            '\\"<!DOCTYPE html>\\" is not valid JSON"}'
        )
        request = httpx.Request("POST", "https://opencode.ai/zen/go/v1/chat/completions")
        response = httpx.Response(
            500,
            request=request,
            headers={
                "content-type": "application/json",
                "x-request-id": "req_123",
                "server": "edge",
            },
            text=response_body,
        )
        api_error = InternalServerError(
            "Error code: 500",
            response=response,
            body={"type": "error", "message": "Unexpected token '<', \"<!DOCTYPE html>\" is not valid JSON"},
        )
        extractor.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: (_ for _ in ()).throw(api_error))
            )
        )

        with patch.object(llm_extractor_module.logger, "warning") as mock_warning:
            with self.assertRaises(llm_extractor_module.ExtractionError):
                extractor._call_structured_messages(
                    messages=[{"role": "user", "content": "return json"}],
                    schema_name="KnowledgeGraphExtraction",
                    schema_model=KnowledgeGraphExtraction,
                    fallback_payload='{"extraction_notes":"fallback","triples":[]}',
                    max_retries=1,
                )

        http_diagnostic_call = next(
            call for call in mock_warning.call_args_list if call.args[0] == "HTTP diagnostics for %s attempt %s/%s: %s"
        )
        self.assertEqual(http_diagnostic_call.args[1:4], ("return json", 1, 1))
        diagnostics = http_diagnostic_call.args[4]
        self.assertIn("status=500", diagnostics)
        self.assertIn('"content-type":"application/json"', diagnostics)
        self.assertIn('"x-request-id":"req_123"', diagnostics)
        self.assertIn("parsed_error=", diagnostics)

    def test_messages_api_logs_non_json_response_diagnostics(self):
        class _FakeHTTPResponse:
            status = 200
            headers = {
                "content-type": "text/html; charset=utf-8",
                "server": "edge",
            }

            def read(self):
                return b"<!DOCTYPE html><html><body>upstream down</body></html>"

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        extractor = LLMExtractor(
            base_url="https://opencode.ai/zen/go/v1",
            api_key="secret",
            model="minimax-m2.7",
            provider="opencode-go",
            api_mode="messages",
            max_output_tokens=2048,
        )

        with patch.object(
            llm_extractor_module.urllib.request,
            "urlopen",
            return_value=_FakeHTTPResponse(),
        ), patch.object(llm_extractor_module.logger, "warning") as mock_warning:
            with self.assertRaises(llm_extractor_module.ExtractionError) as ctx:
                extractor._call_messages_api(
                    request_messages=[{"role": "user", "content": "Hello"}],
                    temperature=0.0,
                    call_label="messages-test",
                    attempt=1,
                    max_retries=2,
                )

        self.assertIn("Messages API returned non-JSON content", str(ctx.exception))
        diagnostic_call = next(
            call
            for call in mock_warning.call_args_list
            if call.args[0] == "Messages API non-JSON diagnostics for %s attempt %s/%s: %s"
        )
        self.assertEqual(diagnostic_call.args[1:4], ("messages-test", 1, 2))
        diagnostics = diagnostic_call.args[4]
        self.assertIn("status=200", diagnostics)
        self.assertIn('"content-type":"text/html; charset=utf-8"', diagnostics)
        self.assertIn("raw_response=<!DOCTYPE html><html><body>upstream down</body></html>", diagnostics)

    def test_reflect_extraction_requires_explicit_user_prompt(self):
        extractor = LLMExtractor(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="local-model",
            provider="local",
            api_mode="chat_completions",
        )

        with self.assertRaises(ValueError) as ctx:
            extractor.reflect_extraction(
                full_text="filing",
                current_extraction=KnowledgeGraphExtraction(),
                company_name="Microsoft",
                user_prompt=None,
            )

        self.assertIn("requires an explicit user_prompt", str(ctx.exception))

    def test_structured_call_filters_ontology_invalid_triples(self):
        extractor = LLMExtractor(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="local-model",
            provider="local",
            api_mode="chat_completions",
        )
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=(
                            '{"extraction_notes":"ok","triples":['
                            '{"subject":"Microsoft","subject_type":"Company","relation":"OFFERS","object":"Azure","object_type":"Offering"},'
                            '{"subject":"Microsoft","subject_type":"Company","relation":"SERVES","object":"developers","object_type":"CustomerType"}'
                            ']}'
                        ),
                        refusal=None,
                    ),
                )
            ],
            usage=SimpleNamespace(completion_tokens=12),
        )
        extractor.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))

        parsed_model, _, _, audit = extractor._call_structured_messages(
            messages=[{"role": "user", "content": "return json"}],
            schema_name="KnowledgeGraphExtraction",
            schema_model=KnowledgeGraphExtraction,
            fallback_payload='{"extraction_notes":"fallback","triples":[]}',
            max_retries=1,
        )

        self.assertEqual(len(parsed_model.triples), 1)
        self.assertEqual(parsed_model.triples[0].relation, "OFFERS")
        self.assertEqual(audit["ontology_rejected_triple_count"], 1)

    def test_reflection_fallback_reaudits_current_extraction_after_empty_result(self):
        extractor = LLMExtractor(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="local-model",
            provider="local",
            api_mode="chat_completions",
        )
        current = KnowledgeGraphExtraction(
            extraction_notes="pre-reflection",
            triples=[
                Triple(
                    subject="Microsoft",
                    subject_type="Company",
                    relation="OFFERS",
                    object="Azure",
                    object_type="Offering",
                )
            ],
        )

        with patch.object(
            extractor,
            "_call_structured",
            return_value=(KnowledgeGraphExtraction(extraction_notes="empty", triples=[]), "{}", 1, {"raw_triple_count": 0}),
        ):
            final_extraction, _, _, audit = extractor.reflect_extraction(
                full_text="filing",
                current_extraction=current,
                company_name="Microsoft",
                strict=False,
                user_prompt="review",
            )

        self.assertEqual(final_extraction.model_dump(), current.model_dump())
        self.assertEqual(audit["raw_triple_count"], 1)
        self.assertEqual(audit["kept_triple_count"], 1)

    def test_reflection_failure_reaudits_current_extraction_after_exception(self):
        extractor = LLMExtractor(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="local-model",
            provider="local",
            api_mode="chat_completions",
        )
        current = KnowledgeGraphExtraction(
            extraction_notes="pre-reflection",
            triples=[
                Triple(
                    subject="Microsoft",
                    subject_type="Company",
                    relation="OPERATES_IN",
                    object="United States",
                    object_type="Place",
                )
            ],
        )

        with patch.object(extractor, "_call_structured", side_effect=llm_extractor_module.ExtractionError("boom")):
            final_extraction, raw_response, attempts_used, audit = extractor.reflect_extraction(
                full_text="filing",
                current_extraction=current,
                company_name="Microsoft",
                strict=False,
                user_prompt="review",
            )

        self.assertEqual(final_extraction.model_dump(), current.model_dump())
        self.assertIsNone(raw_response)
        self.assertEqual(attempts_used, 2)
        self.assertEqual(audit["raw_triple_count"], 1)
        self.assertEqual(audit["kept_triple_count"], 1)

    def test_extract_canonical_pipeline_offloads_rules_into_fresh_pass_prompts(self):
        extractor = LLMExtractor(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="local-model",
            provider="local",
            api_mode="chat_completions",
        )
        skeleton = KnowledgeGraphExtraction(
            extraction_notes="skeleton",
            triples=[
                Triple(
                    subject="Microsoft",
                    subject_type="Company",
                    relation="HAS_SEGMENT",
                    object="Intelligent Cloud",
                    object_type="BusinessSegment",
                )
            ],
        )
        empty = KnowledgeGraphExtraction(extraction_notes="", triples=[])
        captured_messages: list[list[dict[str, str]]] = []

        def structured_side_effect(**kwargs):
            messages = kwargs["messages"]
            captured_messages.append(messages)
            call_index = len(captured_messages)
            if call_index == 1:
                return skeleton, "raw skeleton", 1, {"kept_triple_count": 1}
            return empty, f"raw pass {call_index}", 1, {"kept_triple_count": 0}

        def reflect_side_effect(**kwargs):
            current = kwargs["current_extraction"]
            return current, "raw reflection", 1, {"kept_triple_count": len(current.triples)}

        with patch.object(extractor, "_call_structured_messages", side_effect=structured_side_effect), patch.object(
            extractor, "reflect_extraction", side_effect=reflect_side_effect
        ):
            result = extractor.extract_canonical_pipeline(
                full_text="filing",
                company_name="Microsoft",
                max_retries=2,
            )

        self.assertTrue(result.success)
        self.assertEqual(len(captured_messages), 5)

        pass1_prompt = captured_messages[0][1]["content"]
        self.assertIn(
            "do not compress explicit offering lists into invented summary labels, but if the filing itself uses a named parent heading for the list, keep that named heading as the parent offering.",
            pass1_prompt,
        )
        self.assertIn("BusinessSegment -> OFFERS -> Offering is the primary segment-offering edge.", pass1_prompt)
        self.assertIn("reason carefully about product families", pass1_prompt)

        pass2_channels_messages = captured_messages[1]
        self.assertEqual(len(pass2_channels_messages), 2)
        self.assertIn("<current_structure>", pass2_channels_messages[1]["content"])
        self.assertIn("SELLS_THROUGH should default to BusinessSegment.", pass2_channels_messages[1]["content"])
        self.assertNotIn("PASS 1 - STRUCTURAL SKELETON", pass2_channels_messages[1]["content"])

        pass2_revenue_messages = captured_messages[2]
        self.assertEqual(len(pass2_revenue_messages), 2)
        self.assertIn("<current_structure>", pass2_revenue_messages[1]["content"])
        self.assertIn("use only the exact canonical RevenueModel labels defined below.", pass2_revenue_messages[1]["content"])
        self.assertNotIn("PASS 2A - CHANNELS", pass2_revenue_messages[1]["content"])

        pass3_prompt = captured_messages[3][1]["content"]
        self.assertIn("precision is more important than recall.", pass3_prompt)
        self.assertIn("do not make weak guesses from vague proximity or broad context.", pass3_prompt)

        pass4_prompt = captured_messages[4][1]["content"]
        self.assertIn("U.S. -> United States", pass4_prompt)
        self.assertIn("do not use PARTNERS_WITH for suppliers, customers, competitors, ecosystem mentions, or channel relationships.", pass4_prompt)

    def test_extract_canonical_pipeline_runs_two_reflection_stages(self):
        extractor = LLMExtractor(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="local-model",
            provider="local",
            api_mode="chat_completions",
        )
        lines: list[str] = []
        console = PipelineConsole(printer=lines.append)
        extractor.progress_callback = console.handle_progress
        skeleton = KnowledgeGraphExtraction(
            extraction_notes="skeleton",
            triples=[
                Triple(
                    subject="Microsoft",
                    subject_type="Company",
                    relation="HAS_SEGMENT",
                    object="Intelligent Cloud",
                    object_type="BusinessSegment",
                )
            ],
        )
        empty = KnowledgeGraphExtraction(extraction_notes="", triples=[])
        rule_reflection = KnowledgeGraphExtraction(
            extraction_notes="rule cleaned",
            triples=[
                *skeleton.triples,
                Triple(
                    subject="Intelligent Cloud",
                    subject_type="BusinessSegment",
                    relation="OFFERS",
                    object="Azure",
                    object_type="Offering",
                ),
            ],
        )
        final_reflection = KnowledgeGraphExtraction(
            extraction_notes="final",
            triples=[
                Triple(
                    subject="Intelligent Cloud",
                    subject_type="BusinessSegment",
                    relation="OFFERS",
                    object="Azure",
                    object_type="Offering",
                )
            ],
        )
        reflection_calls: list[tuple[str, dict[str, object]]] = []
        reflection_prompts: dict[str, str] = {}

        def reflect_side_effect(**kwargs):
            reflection_calls.append((kwargs["stage_label"], kwargs["current_extraction"].model_dump()))
            reflection_prompts[kwargs["stage_label"]] = kwargs["user_prompt"]
            if kwargs["stage_label"] == "Rule reflection":
                return rule_reflection, '{"extraction_notes":"rule cleaned","triples":[]}', 1, {"kept_triple_count": 1}
            return final_reflection, '{"extraction_notes":"final","triples":[]}', 2, {"kept_triple_count": 1}

        with patch.object(
            extractor,
            "_call_structured_messages",
            side_effect=[
                (skeleton, "raw skeleton", 1, {"kept_triple_count": 1}),
                (empty, "raw channels", 1, {"kept_triple_count": 0}),
                (empty, "raw revenue", 1, {"kept_triple_count": 0}),
                (empty, "raw serves", 1, {"kept_triple_count": 0}),
                (empty, "raw corporate", 1, {"kept_triple_count": 0}),
            ],
        ), patch.object(extractor, "reflect_extraction", side_effect=reflect_side_effect):
            result = extractor.extract_canonical_pipeline(
                full_text="filing",
                company_name="Microsoft",
                max_retries=2,
            )

        self.assertTrue(result.success)
        self.assertEqual(len(reflection_calls), 2)
        self.assertEqual(reflection_calls[0][0], "Rule reflection")
        self.assertEqual(reflection_calls[0][1], result.pre_reflection_extraction.model_dump())
        self.assertEqual(reflection_calls[1][0], "Filing reflection")
        self.assertEqual(reflection_calls[1][1], rule_reflection.model_dump())
        self.assertEqual(result.rule_reflection_extraction.model_dump(), rule_reflection.model_dump())
        self.assertEqual(result.final_extraction.model_dump(), final_reflection.model_dump())
        self.assertEqual(result.rule_reflection_attempts_used, 1)
        self.assertEqual(result.final_reflection_attempts_used, 2)
        self.assertIn("supervising and editing an existing draft graph", reflection_prompts["Filing reflection"])
        self.assertIn("Preserve existing triples by default", reflection_prompts["Filing reflection"])
        self.assertIn("[07/10] Reflection 1 - Ontology compliance", lines)
        self.assertIn("[08/10] Reflection 2 - Filing reconciliation", lines)
        self.assertTrue(any("triples in:" in line and "1" in line for line in lines))
        self.assertTrue(any("triples added:" in line and "1" in line for line in lines))
        self.assertTrue(any("triples removed:" in line and "0" in line for line in lines))
        self.assertTrue(any("triples in:" in line and "2" in line for line in lines))
        self.assertTrue(any("triples added:" in line and "0" in line for line in lines))
        self.assertTrue(any("triples removed:" in line and "1" in line for line in lines))

    def test_opencode_go_rejects_unsupported_models(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_model_settings(
                provider="opencode-go",
                model="glm-5.1",
                api_key="secret",
            )

        self.assertIn("Unsupported opencode-go model", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
