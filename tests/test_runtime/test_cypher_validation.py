import unittest

from runtime.cypher_validation import normalize_neo4j_uri, validate_params_match, validate_read_only_cypher


class CypherValidationTests(unittest.TestCase):
    def test_normalize_neo4j_uri_converts_browser_url(self):
        self.assertEqual(normalize_neo4j_uri("http://localhost:7474"), "bolt://localhost:7687")

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


if __name__ == "__main__":
    unittest.main()
