import json
import tempfile
import unittest
from pathlib import Path

from graph.evaluate_graph import _load_triples_from_json, evaluate
from graph.neo4j_loader import _merge_node_clause


class GraphComponentTests(unittest.TestCase):
    def test_merge_node_clause_scopes_segments_and_offerings_by_company(self):
        segment_clause = _merge_node_clause("subject", "BusinessSegment", "subject_name", "subject_company_name")
        offering_clause = _merge_node_clause("object", "Offering", "object_name", "object_company_name")
        company_clause = _merge_node_clause("subject", "Company", "subject_name", "subject_company_name")

        self.assertIn("company_name: row.subject_company_name", segment_clause)
        self.assertIn("company_name: row.object_company_name", offering_clause)
        self.assertNotIn("company_name", company_clause)

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


if __name__ == "__main__":
    unittest.main()
