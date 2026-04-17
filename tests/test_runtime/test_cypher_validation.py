import unittest
from os import environ
from unittest.mock import patch

from runtime.cypher_validation import normalize_neo4j_uri, validate_params_match, validate_read_only_cypher


class CypherValidationTests(unittest.TestCase):
    def test_normalize_neo4j_uri_converts_browser_url(self):
        self.assertEqual(normalize_neo4j_uri("http://localhost:7474"), "bolt://localhost:7687")

    def test_normalize_neo4j_uri_converts_host_port_without_scheme(self):
        self.assertEqual(normalize_neo4j_uri("localhost:7474"), "bolt://localhost:7687")

    def test_normalize_neo4j_uri_passes_through_secure_neo4j_scheme(self):
        self.assertEqual(normalize_neo4j_uri("neo4j+s://graph.example.com:7687"), "neo4j+s://graph.example.com:7687")

    def test_normalize_neo4j_uri_reads_environment_fallback(self):
        with patch.dict(environ, {"NEO4J_URI": "http://db.internal:7474"}, clear=True):
            self.assertEqual(normalize_neo4j_uri(None), "bolt://db.internal:7687")

    def test_normalize_neo4j_uri_rejects_invalid_scheme(self):
        with self.assertRaises(ValueError) as ctx:
            normalize_neo4j_uri("ftp://localhost:21")

        self.assertIn("Unsupported Neo4j URI scheme", str(ctx.exception))

    def test_normalize_neo4j_uri_requires_hostname_for_browser_urls(self):
        with self.assertRaises(ValueError) as ctx:
            normalize_neo4j_uri("http://:7474")

        self.assertIn("missing a hostname", str(ctx.exception))

    def test_validate_params_match_reports_missing_params(self):
        failures = validate_params_match(
            "MATCH (company:Company {name: $company}) RETURN company.name AS company",
            {},
        )

        self.assertEqual(
            failures,
            ["Parameter mismatch. Referenced=['company'], provided=[]"],
        )

    def test_validate_read_only_cypher_rejects_write_queries(self):
        failures = validate_read_only_cypher("CREATE (company:Company {name: $company}) RETURN company")

        self.assertEqual(failures, [r"Query contains disallowed clause matching \bCREATE\b"])

    def test_validate_read_only_cypher_rejects_load_csv_queries(self):
        failures = validate_read_only_cypher("LOAD CSV FROM 'https://example.com/export.csv' AS row RETURN row")

        self.assertEqual(failures, [r"Query contains disallowed clause matching \bLOAD\s+CSV\b"])


if __name__ == "__main__":
    unittest.main()
