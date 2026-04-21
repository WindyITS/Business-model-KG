import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from llm import extractor as llm_extractor_module
from llm.extractor import LLMExtractor
from llm_extraction.audit import aggregate_extraction_audits, audit_knowledge_graph_payload
from llm_extraction.models import AnalystBusinessModelMemo, KnowledgeGraphExtraction, Triple
from llm_extraction.pipelines.analyst.runner import AnalystPipelineRunner
from openai import InternalServerError


class LLMExtractorTests(unittest.TestCase):
    def test_payload_audit_counts_malformed_and_ontology_rejections(self):
        payload = {
            "triples": [
                {
                    "subject": "Microsoft",
                    "subject_type": "Company",
                    "relation": "OFFERS",
                    "object": "Azure",
                    "object_type": "Offering",
                },
                {
                    "subject": "Microsoft",
                    "subject_type": "Company",
                    "relation": "SERVES",
                    "object": "startups",
                    "object_type": "CustomerType",
                },
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

    def test_text_call_accepts_non_json_memo_output(self):
        extractor = LLMExtractor(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="local-model",
            provider="local",
            api_mode="chat_completions",
        )
        events: list[tuple[str, dict[str, object]]] = []
        extractor.progress_callback = lambda event, **payload: events.append((event, payload))
        extractor.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                finish_reason="stop",
                                message=SimpleNamespace(
                                    content="ANALYTICAL FRAME\nSummary:\nA plain-text memo.\n",
                                    refusal=None,
                                ),
                            )
                        ],
                        usage=SimpleNamespace(completion_tokens=12),
                    )
                )
            )
        )

        content, attempts_used, audit = extractor._call_text_messages(
            messages=[{"role": "user", "content": "write a memo"}],
            max_retries=2,
        )

        self.assertEqual(content, "ANALYTICAL FRAME\nSummary:\nA plain-text memo.")
        self.assertEqual(attempts_used, 1)
        self.assertEqual(audit["format"], "text")
        self.assertEqual(audit["line_count"], 3)
        self.assertEqual(
            events,
            [
                ("llm_call_start", {"attempt": 1, "max_retries": 2}),
                ("llm_call_complete", {"attempt": 1, "max_retries": 2, "tokens": 12}),
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
                {
                    "raw_triple_count": 2,
                    "malformed_triple_count": 1,
                    "ontology_rejected_triple_count": 0,
                    "duplicate_triple_count": 0,
                    "kept_triple_count": 1,
                    "invalid_issue_counts": {"empty_subject": 1},
                    "payload_parse_recovered": True,
                },
                {
                    "raw_triple_count": 3,
                    "malformed_triple_count": 0,
                    "ontology_rejected_triple_count": 2,
                    "duplicate_triple_count": 1,
                    "kept_triple_count": 0,
                    "invalid_issue_counts": {"non_canonical_label": 2},
                    "payload_parse_recovered": False,
                },
            ]
        )

        self.assertEqual(aggregated["raw_triple_count"], 5)
        self.assertEqual(aggregated["malformed_triple_count"], 1)
        self.assertEqual(aggregated["ontology_rejected_triple_count"], 2)
        self.assertEqual(aggregated["duplicate_triple_count"], 1)
        self.assertEqual(aggregated["payload_parse_recovered_count"], 1)

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
                system_prompt="system",
                user_prompt=None,
            )

        self.assertIn("requires explicit system_prompt and user_prompt", str(ctx.exception))

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
                system_prompt="system",
                user_prompt="review",
            )

        self.assertEqual(final_extraction.model_dump(), current.model_dump())
        self.assertEqual(audit["raw_triple_count"], 1)
        self.assertEqual(audit["kept_triple_count"], 1)

    def test_reflection_fallback_emits_progress_warning_after_empty_result(self):
        events: list[tuple[str, dict[str, object]]] = []
        extractor = LLMExtractor(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="local-model",
            provider="local",
            api_mode="chat_completions",
            progress_callback=lambda event, **payload: events.append((event, payload)),
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
            extractor.reflect_extraction(
                full_text="filing",
                current_extraction=current,
                company_name="Microsoft",
                strict=False,
                system_prompt="system",
                user_prompt="review",
                stage_label="Rule reflection",
            )

        self.assertIn(
            (
                "stage_warning",
                {"message": "Rule reflection returned no triples."},
            ),
            events,
        )

    def test_reflection_fallback_can_be_declined_after_empty_result(self):
        decision_payloads: list[dict[str, object]] = []
        extractor = LLMExtractor(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="local-model",
            provider="local",
            api_mode="chat_completions",
            fallback_confirmation_callback=lambda **payload: decision_payloads.append(payload) or False,
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
        ), self.assertRaisesRegex(
            llm_extractor_module.ExtractionError,
            "Rule reflection returned no usable graph and the last good graph was declined by the user.",
        ):
            extractor.reflect_extraction(
                full_text="filing",
                current_extraction=current,
                company_name="Microsoft",
                strict=False,
                system_prompt="system",
                user_prompt="review",
                stage_label="Rule reflection",
            )

        self.assertEqual(
            decision_payloads,
            [{"stage_label": "Rule reflection", "triple_count": 1}],
        )

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
                system_prompt="system",
                user_prompt="review",
            )

        self.assertEqual(final_extraction.model_dump(), current.model_dump())
        self.assertIsNone(raw_response)
        self.assertEqual(attempts_used, 2)
        self.assertEqual(audit["raw_triple_count"], 1)
        self.assertEqual(audit["kept_triple_count"], 1)

    def test_reflection_failure_emits_progress_warning_after_exception(self):
        events: list[tuple[str, dict[str, object]]] = []
        extractor = LLMExtractor(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="local-model",
            provider="local",
            api_mode="chat_completions",
            progress_callback=lambda event, **payload: events.append((event, payload)),
        )
        current = KnowledgeGraphExtraction(
            extraction_notes="pre-reflection",
            triples=[
                Triple(
                    subject="Microsoft",
                    subject_type="Company",
                    relation="OPERATES_IN",
                    object="Worldwide",
                    object_type="Place",
                )
            ],
        )

        with patch.object(extractor, "_call_structured", side_effect=llm_extractor_module.ExtractionError("boom")):
            extractor.reflect_extraction(
                full_text="filing",
                current_extraction=current,
                company_name="Microsoft",
                strict=False,
                system_prompt="system",
                user_prompt="review",
                stage_label="Filing reflection",
            )

        self.assertIn(
            (
                "stage_warning",
                {"message": "Filing reflection failed after retries."},
            ),
            events,
        )

    def test_analyst_runner_returns_failed_result_when_fallback_is_declined(self):
        extractor = LLMExtractor(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="local-model",
            provider="local",
            api_mode="chat_completions",
        )
        foundation_memo = AnalystBusinessModelMemo(content="foundation memo")
        augmented_memo = AnalystBusinessModelMemo(content="augmented memo")
        compiled_graph = KnowledgeGraphExtraction(
            extraction_notes="compiled",
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
            AnalystPipelineRunner,
            "_run_text_stage",
            side_effect=[
                (foundation_memo, foundation_memo.content, 1, {"line_count": 1}),
                (augmented_memo, augmented_memo.content, 1, {"line_count": 1}),
            ],
        ), patch.object(
            AnalystPipelineRunner,
            "_run_structured_stage",
            return_value=(compiled_graph, "raw graph", 1, {"kept_triple_count": 1}),
        ), patch.object(
            extractor,
            "reflect_extraction",
            side_effect=llm_extractor_module.ExtractionError(
                "Analyst critique failed after retries and the last good graph was declined by the user."
            ),
        ):
            result = AnalystPipelineRunner(extractor).run(
                full_text="filing",
                company_name="Microsoft",
                max_retries=2,
            )

        self.assertFalse(result.success)
        self.assertEqual(
            result.error,
            "Analyst critique failed after retries and the last good graph was declined by the user.",
        )


if __name__ == "__main__":
    unittest.main()
