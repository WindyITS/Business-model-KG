import unittest

from ontology.validator import validate_payload, validate_triple, validate_triples


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

    def test_normalizes_curly_quotes_in_entities(self):
        triple = {
            "subject": "“Microsoft”",
            "subject_type": "Company",
            "relation": "OPERATES_IN",
            "object": "“United States”",
            "object_type": "Place",
        }

        result = validate_triple(triple)

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["normalized_triple"]["subject"], "Microsoft")
        self.assertEqual(result["normalized_triple"]["object"], "United States")

    def test_normalizes_place_aliases_deterministically(self):
        triple = {
            "subject": "Microsoft",
            "subject_type": "Company",
            "relation": "OPERATES_IN",
            "object": "U.S.",
            "object_type": "Place",
        }

        result = validate_triple(triple)

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["normalized_triple"]["object"], "United States")

    def test_rejects_non_canonical_customer_type(self):
        triple = {
            "subject": "Intelligent Cloud",
            "subject_type": "BusinessSegment",
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

    def test_accepts_business_segment_serves_relation(self):
        triple = {
            "subject": "Intelligent Cloud",
            "subject_type": "BusinessSegment",
            "relation": "SERVES",
            "object": "large enterprises",
            "object_type": "CustomerType",
        }

        result = validate_triple(triple)

        self.assertTrue(result["is_valid"])

    def test_rejects_offering_serves_relation(self):
        triple = {
            "subject": "Azure",
            "subject_type": "Offering",
            "relation": "SERVES",
            "object": "developers",
            "object_type": "CustomerType",
        }

        result = validate_triple(triple)

        self.assertFalse(result["is_valid"])
        self.assertTrue(any(issue["code"] == "invalid_relation_schema" for issue in result["issues"]))

    def test_rejects_company_serves_relation(self):
        triple = {
            "subject": "Microsoft",
            "subject_type": "Company",
            "relation": "SERVES",
            "object": "developers",
            "object_type": "CustomerType",
        }

        result = validate_triple(triple)

        self.assertFalse(result["is_valid"])
        self.assertTrue(any(issue["code"] == "invalid_relation_schema" for issue in result["issues"]))

    def test_rejects_company_sells_through_relation(self):
        triple = {
            "subject": "Microsoft",
            "subject_type": "Company",
            "relation": "SELLS_THROUGH",
            "object": "resellers",
            "object_type": "Channel",
        }

        result = validate_triple(triple)

        self.assertFalse(result["is_valid"])
        self.assertTrue(any(issue["code"] == "invalid_relation_schema" for issue in result["issues"]))

    def test_rejects_business_segment_monetizes_via_relation(self):
        triple = {
            "subject": "Intelligent Cloud",
            "subject_type": "BusinessSegment",
            "relation": "MONETIZES_VIA",
            "object": "subscription",
            "object_type": "RevenueModel",
        }

        result = validate_triple(triple)

        self.assertFalse(result["is_valid"])
        self.assertTrue(any(issue["code"] == "invalid_relation_schema" for issue in result["issues"]))

    def test_accepts_offering_monetizes_via_relation(self):
        triple = {
            "subject": "Azure",
            "subject_type": "Offering",
            "relation": "MONETIZES_VIA",
            "object": "consumption-based",
            "object_type": "RevenueModel",
        }

        result = validate_triple(triple)

        self.assertTrue(result["is_valid"])

    def test_rejects_child_offering_monetizes_via_relation(self):
        triples = [
            {
                "subject": "Microsoft 365 Commercial",
                "subject_type": "Offering",
                "relation": "OFFERS",
                "object": "Microsoft Teams",
                "object_type": "Offering",
            },
            {
                "subject": "Microsoft Teams",
                "subject_type": "Offering",
                "relation": "MONETIZES_VIA",
                "object": "subscription",
                "object_type": "RevenueModel",
            },
        ]

        report = validate_triples(triples)

        self.assertEqual(report["summary"]["valid_triple_count"], 1)
        self.assertEqual(report["summary"]["invalid_triple_count"], 1)
        self.assertTrue(
            any(
                issue["code"] == "child_offering_monetizes_via"
                for invalid in report["invalid_triples"]
                for issue in invalid["issues"]
            )
        )

    def test_rejects_segment_anchored_offering_sells_through_relation(self):
        triples = [
            {
                "subject": "Intelligent Cloud",
                "subject_type": "BusinessSegment",
                "relation": "OFFERS",
                "object": "Azure",
                "object_type": "Offering",
            },
            {
                "subject": "Azure",
                "subject_type": "Offering",
                "relation": "SELLS_THROUGH",
                "object": "online",
                "object_type": "Channel",
            },
        ]

        report = validate_triples(triples)

        self.assertEqual(report["summary"]["valid_triple_count"], 1)
        self.assertEqual(report["summary"]["invalid_triple_count"], 1)
        self.assertTrue(
            any(
                issue["code"] == "segment_anchored_offering_sells_through"
                for invalid in report["invalid_triples"]
                for issue in invalid["issues"]
            )
        )

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

    def test_rejects_multiple_offering_parents(self):
        triples = [
            {
                "subject": "Microsoft 365 Commercial products and cloud services",
                "subject_type": "Offering",
                "relation": "OFFERS",
                "object": "Office licensed on-premises",
                "object_type": "Offering",
            },
            {
                "subject": "Microsoft 365 Consumer products and cloud services",
                "subject_type": "Offering",
                "relation": "OFFERS",
                "object": "Office licensed on-premises",
                "object_type": "Offering",
            },
        ]

        report = validate_triples(triples)

        self.assertEqual(report["summary"]["valid_triple_count"], 1)
        self.assertEqual(report["summary"]["invalid_triple_count"], 1)
        self.assertTrue(
            any(
                issue["code"] == "multiple_offering_parents"
                for invalid in report["invalid_triples"]
                for issue in invalid["issues"]
            )
        )


if __name__ == "__main__":
    unittest.main()
