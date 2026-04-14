import json
import sys
import tempfile
import unittest
from os import environ
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import llm_extractor as llm_extractor_module
from entity_resolver import resolve_entities
from evaluate_graph import _load_triples_from_json, evaluate
from llm_extractor import (
    LLMExtractor,
    KnowledgeGraphExtraction,
    Triple,
    _canonical_pipeline_system_prompt,
    _canonical_reflection_system_prompt,
    aggregate_extraction_audits,
    audit_knowledge_graph_payload,
)
from main import _effective_use_schema, _infer_company_name, _mode_name
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

    def test_infer_company_name_accepts_short_issuer_line(self):
        text = "ITEM 1. BUSINESS\nMicrosoft is a technology company.\n"

        inferred = _infer_company_name(Path("microsoft_10k.txt"), text)

        self.assertEqual(inferred, "Microsoft")

    def test_infer_company_name_falls_back_when_line_is_descriptive_prose(self):
        text = (
            "ITEM 1. BUSINESS\n"
            "We have built four principal software platforms, Palantir Gotham and Palantir Foundry, "
            "which enable institutions to transform data.\n"
        )

        inferred = _infer_company_name(Path("palantir_10k.txt"), text)

        self.assertEqual(inferred, "Palantir")

    def test_local_provider_disables_schema_by_default(self):
        args = SimpleNamespace(provider="local", no_schema=False)
        self.assertFalse(_effective_use_schema(args))

    def test_local_provider_no_schema_flag_keeps_schema_disabled(self):
        args = SimpleNamespace(provider="local", no_schema=True)
        self.assertFalse(_effective_use_schema(args))

    def test_use_schema_flag_enables_schema(self):
        args = SimpleNamespace(provider="local", no_schema=False, use_schema=True)
        self.assertTrue(_effective_use_schema(args))

    def test_opencode_go_disables_schema_by_default(self):
        args = SimpleNamespace(provider="opencode-go", no_schema=False, use_schema=False)
        self.assertFalse(_effective_use_schema(args))

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
            payload, recovered = LLMExtractor._load_json_payload(
                payload_text,
                '{"extraction_notes":"fallback","triples":[]}',
            )

        self.assertEqual(payload["extraction_notes"], "ok")
        self.assertEqual(payload["triples"], [])
        self.assertTrue(recovered)
        mock_warning.assert_not_called()

    def test_load_json_payload_warns_for_likely_truncation(self):
        payload_text = '{"extraction_notes":"ok","triples":[]'

        with patch.object(llm_extractor_module.logger, "warning") as mock_warning:
            payload, recovered = LLMExtractor._load_json_payload(
                payload_text,
                '{"extraction_notes":"fallback","triples":[]}',
            )

        self.assertEqual(payload["extraction_notes"], "fallback")
        self.assertEqual(payload["triples"], [])
        self.assertTrue(recovered)
        mock_warning.assert_any_call("Model response may be truncated. Attempting to salvage JSON prefix...")

    def test_canonical_schema_def_lists_only_supported_relations(self):
        schema_def = LLMExtractor._schema_def(
            "KnowledgeGraphExtraction",
            KnowledgeGraphExtraction,
        )
        relation_enum = schema_def["json_schema"]["schema"]["$defs"]["Triple"]["properties"]["relation"]["enum"]

        self.assertEqual(
            set(relation_enum),
            {
                "HAS_SEGMENT",
                "OFFERS",
                "SERVES",
                "OPERATES_IN",
                "SELLS_THROUGH",
                "PARTNERS_WITH",
                "MONETIZES_VIA",
            },
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
            model="mimo-v2-pro",
            base_url="https://opencode.ai/zen/go/v1/chat/completions",
            api_key="secret",
        )

        self.assertEqual(settings.base_url, "https://opencode.ai/zen/go/v1")
        self.assertEqual(settings.api_mode, "chat_completions")
        self.assertEqual(settings.max_output_tokens, 20000)

    def test_opencode_go_defaults_to_mimo_v2_pro(self):
        settings = resolve_model_settings(
            provider="opencode-go",
            api_key="secret",
        )

        self.assertEqual(settings.model, "mimo-v2-pro")
        self.assertEqual(settings.api_mode, "chat_completions")

    def test_opencode_go_reads_api_key_from_environment(self):
        with patch.dict(environ, {"OPENCODE_API_KEY": "env-secret"}, clear=True):
            settings = resolve_model_settings(
                provider="opencode-go",
                model="mimo-v2-pro",
            )

        self.assertEqual(settings.api_key, "env-secret")
        self.assertEqual(settings.max_output_tokens, 20000)

    def test_opencode_go_honors_explicit_output_cap(self):
        settings = resolve_model_settings(
            provider="opencode-go",
            model="mimo-v2-pro",
            api_key="secret",
            max_output_tokens=1024,
        )

        self.assertEqual(settings.max_output_tokens, 1024)

    def test_opencode_go_accepts_kimi_as_explicit_override(self):
        settings = resolve_model_settings(
            provider="opencode-go",
            model="kimi-k2.5",
            api_key="secret",
        )

        self.assertEqual(settings.model, "kimi-k2.5")
        self.assertEqual(settings.api_mode, "chat_completions")

    def test_opencode_go_rejects_unsupported_models(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_model_settings(
                provider="opencode-go",
                model="glm-5",
                api_key="secret",
            )

        self.assertIn("Unsupported opencode-go model", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
