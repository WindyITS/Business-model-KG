import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from graph.evaluate_graph import _load_triples_from_json, evaluate
from graph.neo4j_loader import Neo4jLoader, _merge_node_clause, _orphan_prune_candidates


class _FakeResult:
    def __init__(self, *, single_value=None, data_rows=None):
        self._single_value = single_value
        self._data_rows = data_rows or []

    def single(self):
        return self._single_value

    def data(self):
        return list(self._data_rows)

    def consume(self):
        return None


class GraphComponentTests(unittest.TestCase):
    def test_merge_node_clause_scopes_segments_and_offerings_by_company(self):
        segment_clause = _merge_node_clause("subject", "BusinessSegment", "subject_name", "subject_company_name")
        offering_clause = _merge_node_clause("object", "Offering", "object_name", "object_company_name")
        company_clause = _merge_node_clause("subject", "Company", "subject_name", "subject_company_name")

        self.assertIn("company_name: row.subject_company_name", segment_clause)
        self.assertIn("company_name: row.object_company_name", offering_clause)
        self.assertNotIn("company_name", company_clause)

    def test_orphan_prune_candidates_keep_only_supported_unique_shared_nodes(self):
        candidates = _orphan_prune_candidates(
            [
                {"labels": ["Place"], "name": "Worldwide"},
                {"labels": ["Place"], "name": "Worldwide"},
                {"labels": ["Company"], "name": "Accenture"},
                {"labels": ["BusinessSegment"], "name": "Services"},
                {"labels": ["Channel"], "name": "Direct sales"},
                {"labels": ["RevenueModel"], "name": ""},
            ]
        )

        self.assertEqual(
            candidates,
            [
                {"label": "Place", "name": "Worldwide"},
                {"label": "Company", "name": "Accenture"},
                {"label": "Channel", "name": "Direct sales"},
            ],
        )

    def test_unload_company_removes_company_footprint_and_prunes_orphans(self):
        loader = Neo4jLoader.__new__(Neo4jLoader)
        session = MagicMock()
        session_cm = MagicMock()
        session_cm.__enter__.return_value = session
        session_cm.__exit__.return_value = None
        loader.driver = MagicMock()
        loader.driver.session.return_value = session_cm

        def run_side_effect(query, **params):
            if "RETURN labels(neighbor) AS labels, neighbor.name AS name" in query:
                return _FakeResult(
                    data_rows=[
                        {"labels": ["Place"], "name": "Worldwide"},
                        {"labels": ["Company"], "name": "Accenture"},
                        {"labels": ["BusinessSegment"], "name": "iPhone"},
                    ]
                )
            if "RETURN count(DISTINCT rel) AS relationship_count" in query:
                return _FakeResult(single_value={"relationship_count": 6})
            if "RETURN count(node) AS node_count" in query:
                return _FakeResult(single_value={"node_count": 3})
            if "DETACH DELETE node" in query:
                return _FakeResult()
            if "RETURN count(rel) AS relationship_count" in query:
                return _FakeResult(single_value={"relationship_count": 2})
            if "WHERE type(rel) IN $relation_types" in query and "DELETE rel" in query:
                return _FakeResult()
            if "RETURN size(companies) AS deleted_count" in query:
                return _FakeResult(single_value={"deleted_count": 1})
            if "RETURN size(nodes) AS deleted_count" in query:
                return _FakeResult(single_value={"deleted_count": 2})
            raise AssertionError(f"Unexpected query: {query}")

        session.run.side_effect = run_side_effect

        summary = loader.unload_company("Apple")

        self.assertEqual(
            summary,
            {
                "company_name": "Apple",
                "scoped_nodes_deleted": 3,
                "scoped_relationships_deleted": 6,
                "company_relationships_deleted": 2,
                "company_node_deleted": 1,
                "orphan_nodes_deleted": 2,
            },
        )

        prune_call = next(
            call
            for call in session.run.call_args_list
            if "UNWIND $candidates AS candidate" in call.args[0]
        )
        self.assertEqual(
            prune_call.kwargs["candidates"],
            [
                {"label": "Place", "name": "Worldwide"},
                {"label": "Company", "name": "Accenture"},
            ],
        )

    def test_graph_counts_reports_nodes_and_relationships(self):
        loader = Neo4jLoader.__new__(Neo4jLoader)
        session = MagicMock()
        session_cm = MagicMock()
        session_cm.__enter__.return_value = session
        session_cm.__exit__.return_value = None
        loader.driver = MagicMock()
        loader.driver.session.return_value = session_cm
        session.run.side_effect = [
            _FakeResult(single_value={"node_count": 7}),
            _FakeResult(single_value={"relationship_count": 11}),
        ]

        counts = loader.graph_counts()

        self.assertEqual(counts, {"node_count": 7, "relationship_count": 11})

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
