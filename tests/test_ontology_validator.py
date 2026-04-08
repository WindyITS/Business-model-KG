import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ontology_validator import validate_payload, validate_triple


class OntologyValidatorTests(unittest.TestCase):
    def test_accepts_valid_canonical_triple(self):
        triple = {
            "subject": "Microsoft",
            "subject_type": "Company",
            "relation": "OPERATES_IN",
            "object": "United States",
            "object_type": "Place",
        }

        result = validate_triple(triple)

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["issues"], [])

    def test_rejects_non_canonical_customer_type(self):
        triple = {
            "subject": "Azure",
            "subject_type": "Offering",
            "relation": "SERVES",
            "object": "enterprise customers",
            "object_type": "CustomerType",
        }

        result = validate_triple(triple)

        self.assertFalse(result["is_valid"])
        self.assertTrue(any(issue["code"] == "non_canonical_label" for issue in result["issues"]))

    def test_rejects_invalid_relation_schema(self):
        triple = {
            "subject": "Azure",
            "subject_type": "Offering",
            "relation": "OPERATES_IN",
            "object": "United States",
            "object_type": "Place",
        }

        result = validate_triple(triple)

        self.assertFalse(result["is_valid"])
        self.assertTrue(any(issue["code"] == "invalid_relation_schema" for issue in result["issues"]))

    def test_deduplicates_valid_payload(self):
        payload = {
            "triples": [
                {
                    "subject": "Microsoft",
                    "subject_type": "Company",
                    "relation": "PARTNERS_WITH",
                    "object": "OpenAI",
                    "object_type": "Company",
                },
                {
                    "subject": "Microsoft",
                    "subject_type": "Company",
                    "relation": "PARTNERS_WITH",
                    "object": "OpenAI",
                    "object_type": "Company",
                },
            ]
        }

        report = validate_payload(payload)

        self.assertEqual(report["summary"]["valid_triple_count"], 1)
        self.assertEqual(report["summary"]["duplicate_triple_count"], 1)


if __name__ == "__main__":
    unittest.main()
