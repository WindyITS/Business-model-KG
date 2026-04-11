import json
import tempfile
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunker import read_and_chunk_file
from entity_resolver import resolve_entities
from evaluate_graph import _load_triples_from_json, evaluate
from llm_extractor import KnowledgeGraphExtraction, Triple, aggregate_extraction_audits, audit_knowledge_graph_payload
from main import _mode_name


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

    def test_chunker_handles_local_files_without_tiktoken_network_access(self):
        chunks = read_and_chunk_file("data/microsoft_10k.txt")
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.strip() for chunk in chunks))

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

    def test_mode_name_prefers_explicit_reflection_pipelines(self):
        args = SimpleNamespace(
            chat_two_pass_reflection=True,
            incremental_reflection=True,
            two_pass_reflection=False,
        )
        self.assertEqual(_mode_name(args), "chat_two_pass_reflection")

        args.chat_two_pass_reflection = False
        self.assertEqual(_mode_name(args), "incremental_reflection")

        args.incremental_reflection = False
        args.two_pass_reflection = True
        self.assertEqual(_mode_name(args), "two_pass_reflection")

    def test_mode_name_defaults_to_two_pass_reflection(self):
        args = SimpleNamespace(
            chat_two_pass_reflection=False,
            incremental_reflection=False,
            two_pass_reflection=False,
        )
        self.assertEqual(_mode_name(args), "two_pass_reflection")

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


if __name__ == "__main__":
    unittest.main()
