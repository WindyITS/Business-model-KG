import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from operates_in_cleanup import filter_operates_in_examples, normalize_operates_in_object


class OperatesInCleanupTests(unittest.TestCase):
    def test_normalize_operates_in_object_keeps_supported_geographies(self):
        self.assertEqual(normalize_operates_in_object("united state"), "United States")
        self.assertEqual(normalize_operates_in_object("u.s ."), "United States")
        self.assertEqual(normalize_operates_in_object("asia-pacific"), "Asia Pacific")
        self.assertEqual(normalize_operates_in_object("asia-pacific region"), "Asia Pacific")
        self.assertEqual(normalize_operates_in_object("apac"), "APAC")
        self.assertEqual(normalize_operates_in_object("americas region"), "Americas")
        self.assertEqual(normalize_operates_in_object("united arab emirate"), "United Arab Emirates")
        self.assertEqual(normalize_operates_in_object("bermuda"), "Bermuda")
        self.assertEqual(normalize_operates_in_object("california"), "California")

    def test_normalize_operates_in_object_drops_removed_or_non_geographic_labels(self):
        self.assertIsNone(normalize_operates_in_object("Middle East and Africa"))
        self.assertIsNone(normalize_operates_in_object("Great China"))
        self.assertIsNone(normalize_operates_in_object("global market"))
        self.assertIsNone(normalize_operates_in_object("u.s. market"))
        self.assertIsNone(normalize_operates_in_object("new york city"))
        self.assertIsNone(normalize_operates_in_object("norwood , massachusetts"))

    def test_filter_examples_removes_only_bad_operates_in_triples(self):
        examples = [
            {
                "instruction": "x",
                "input": "The company operates in the U.S. market and sells Azure through resellers.",
                "output": {
                    "extraction_notes": "",
                    "triples": [
                        {
                            "subject": "Microsoft",
                            "subject_type": "Company",
                            "relation": "OPERATES_IN",
                            "object": "u.s .",
                            "object_type": "Place",
                        },
                        {
                            "subject": "Microsoft",
                            "subject_type": "Company",
                            "relation": "OPERATES_IN",
                            "object": "global market",
                            "object_type": "Place",
                        },
                        {
                            "subject": "Microsoft",
                            "subject_type": "Company",
                            "relation": "SELLS_THROUGH",
                            "object": "resellers",
                            "object_type": "Channel",
                        },
                    ],
                },
                "metadata": {"chunk_key": {"ticker": "MSFT", "chunk_id": "c1"}},
            }
        ]

        filtered_examples, report = filter_operates_in_examples(examples)

        self.assertEqual(len(filtered_examples), 1)
        self.assertEqual(report["dropped_operates_in_triple_count"], 1)
        self.assertEqual(report["normalized_operates_in_triple_count"], 1)
        self.assertEqual(
            filtered_examples[0]["output"]["triples"],
            [
                {
                    "subject": "Microsoft",
                    "subject_type": "Company",
                    "relation": "OPERATES_IN",
                    "object": "United States",
                    "object_type": "Place",
                },
                {
                    "subject": "Microsoft",
                    "subject_type": "Company",
                    "relation": "SELLS_THROUGH",
                    "object": "resellers",
                    "object_type": "Channel",
                },
            ],
        )

    def test_filter_examples_drops_positive_that_becomes_empty(self):
        examples = [
            {
                "instruction": "x",
                "input": "The company operates in the global market.",
                "output": {
                    "extraction_notes": "",
                    "triples": [
                        {
                            "subject": "Microsoft",
                            "subject_type": "Company",
                            "relation": "OPERATES_IN",
                            "object": "global market",
                            "object_type": "Place",
                        }
                    ],
                },
                "metadata": {"chunk_key": {"ticker": "MSFT", "chunk_id": "c2"}, "empty_target": False},
            }
        ]

        filtered_examples, report = filter_operates_in_examples(examples)

        self.assertEqual(filtered_examples, [])
        self.assertEqual(report["dropped_example_count"], 1)

    def test_filter_examples_keeps_original_empty_examples(self):
        examples = [
            {
                "instruction": "x",
                "input": "No business-model relation here.",
                "output": {"extraction_notes": "", "triples": []},
                "metadata": {"chunk_key": {"ticker": "MSFT", "chunk_id": "c3"}, "empty_target": True},
            }
        ]

        filtered_examples, report = filter_operates_in_examples(examples)

        self.assertEqual(len(filtered_examples), 1)
        self.assertEqual(filtered_examples[0]["output"]["triples"], [])
        self.assertEqual(report["preserved_empty_example_count"], 1)


if __name__ == "__main__":
    unittest.main()
