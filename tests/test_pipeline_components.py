import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunker import read_and_chunk_file
from entity_resolver import resolve_entities
from evaluate_graph import _load_triples_from_json, evaluate
from llm_extractor import LLMExtractor, KnowledgeGraphExtraction, Triple, aggregate_extraction_audits, audit_knowledge_graph_payload
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

    def test_mode_name_is_canonical_pipeline(self):
        args = SimpleNamespace(pipeline="canonical")
        self.assertEqual(_mode_name(args), "canonical_pipeline")

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

    def test_canonical_schema_def_excludes_part_of(self):
        schema_def = LLMExtractor._schema_def(
            "KnowledgeGraphExtraction",
            KnowledgeGraphExtraction,
        )
        relation_enum = schema_def["json_schema"]["schema"]["$defs"]["Triple"]["properties"]["relation"]["enum"]

        self.assertNotIn("PART_OF", relation_enum)

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


if __name__ == "__main__":
    unittest.main()
