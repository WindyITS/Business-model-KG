import sys
import unittest
import types
from unittest.mock import patch

if "neo4j" not in sys.modules:
    neo4j_module = types.ModuleType("neo4j")

    class _GraphDatabase:
        @staticmethod
        def driver(*args, **kwargs):
            raise RuntimeError("neo4j GraphDatabase.driver should be patched in tests")

    neo4j_module.GraphDatabase = _GraphDatabase
    sys.modules["neo4j"] = neo4j_module

from text2cypher.validation import SyntheticGraphLoader


class _FakeResult:
    def consume(self) -> None:
        return None


class _FakeSession:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, **params):
        self.calls.append({"query": query, "params": params})
        return _FakeResult()


class _FakeDriver:
    def __init__(self):
        self.calls = []

    def session(self):
        return _FakeSession(self.calls)

    def close(self):
        return None


class ValidateText2CypherDatasetTests(unittest.TestCase):
    def test_load_graph_supports_scoped_duplicates_and_node_id_edges(self):
        fake_driver = _FakeDriver()
        fixture = {
            "graph_id": "fx99_duplicate_scoped_fixture",
            "fixture_id": "FX99_duplicate_scoped_fixture",
            "nodes": [
                {"node_id": "company_a", "label": "Company", "name": "Atlas Dynamics"},
                {"node_id": "company_b", "label": "Company", "name": "Atlas Works"},
                {
                    "node_id": "segment_a",
                    "label": "BusinessSegment",
                    "name": "Atlas",
                    "properties": {"company_name": "Atlas Dynamics", "segment_tag": "primary"},
                },
                {
                    "node_id": "segment_b",
                    "label": "BusinessSegment",
                    "name": "Atlas",
                    "properties": {"company_name": "Atlas Works", "segment_tag": "secondary"},
                },
                {
                    "node_id": "offering_a",
                    "label": "Offering",
                    "name": "Atlas",
                    "properties": {"company_name": "Atlas Dynamics", "offering_tag": "core"},
                },
                {
                    "node_id": "offering_b",
                    "label": "Offering",
                    "name": "Atlas",
                    "properties": {"company_name": "Atlas Works", "offering_tag": "edge"},
                },
                {"node_id": "place_us", "label": "Place", "name": "United States"},
            ],
            "edges": [
                {"from": "company_a", "type": "HAS_SEGMENT", "to": "segment_a"},
                {"from": "company_b", "type": "HAS_SEGMENT", "to": "segment_b"},
                {"from": "segment_a", "type": "OFFERS", "to": "offering_a"},
                {"from": "segment_b", "type": "OFFERS", "to": "offering_b"},
                {"from": "company_a", "type": "OPERATES_IN", "to": "place_us"},
            ],
        }

        with patch("text2cypher.validation.GraphDatabase.driver", return_value=fake_driver):
            loader = SyntheticGraphLoader("bolt://example", "neo4j", "password")
            try:
                loader.setup_constraints()
                loader.load_graph(fixture)
            finally:
                loader.close()

        queries = [call["query"] for call in fake_driver.calls]

        self.assertTrue(
            any(
                "CREATE CONSTRAINT BusinessSegment_name_company IF NOT EXISTS" in query
                and "(node.company_name, node.name) IS UNIQUE" in query
                for query in queries
            )
        )
        self.assertTrue(
            any(
                "CREATE CONSTRAINT Offering_name_company IF NOT EXISTS" in query
                and "(node.company_name, node.name) IS UNIQUE" in query
                for query in queries
            )
        )
        self.assertTrue(any("MERGE (n:BusinessSegment {name: row.name, company_name: row.company_name})" in query for query in queries))
        self.assertTrue(any("MERGE (n:Offering {name: row.name, company_name: row.company_name})" in query for query in queries))
        self.assertTrue(any("MATCH (source:BusinessSegment {node_id: row.from_node_id})" in query for query in queries))
        self.assertTrue(any("MATCH (target:Offering {node_id: row.to_node_id})" in query for query in queries))
        self.assertTrue(
            any(
                "SET place.within_places = row.within_places" in query
                and "place.includes_places = row.includes_places" in query
                for query in queries
            )
        )

        scoped_rows = []
        for call in fake_driver.calls:
            query = call["query"]
            if "MERGE (n:BusinessSegment {name: row.name, company_name: row.company_name})" in query:
                scoped_rows.extend(call["params"]["rows"])

        self.assertEqual(
                {
                    (row["name"], row["company_name"], row["properties"]["segment_tag"])
                    for row in scoped_rows
                },
                {
                    ("Atlas", "Atlas Dynamics", "primary"),
                    ("Atlas", "Atlas Works", "secondary"),
                },
            )

    def test_normalizes_browser_style_neo4j_uri_before_connecting(self):
        recorded = {}

        def _driver(uri, auth):
            recorded["uri"] = uri
            recorded["auth"] = auth
            return _FakeDriver()

        with patch("text2cypher.validation.GraphDatabase.driver", side_effect=_driver):
            loader = SyntheticGraphLoader("http://localhost:7474", "neo4j", "password")
            loader.close()

        self.assertEqual(recorded["uri"], "bolt://localhost:7687")
        self.assertEqual(recorded["auth"], ("neo4j", "password"))


if __name__ == "__main__":
    unittest.main()
