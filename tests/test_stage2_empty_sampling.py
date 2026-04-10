import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finreflectkg_projection import build_empty_example, sample_empty_examples, sample_empty_examples_by_count


class Stage2EmptySamplingTests(unittest.TestCase):
    def test_build_empty_example_returns_empty_target(self):
        rows = [
            {
                "ticker": "msft",
                "year": 2024,
                "source_file": "msft.pdf",
                "page_id": "1",
                "chunk_id": "c1",
                "chunk_text": "Microsoft discusses general accounting policies and financial metrics. " * 12,
                "entity": "Microsoft",
                "entity_type": "ORG",
                "relationship": "discloses",
                "target": "Net Income",
                "target_type": "FIN_METRIC",
            }
        ]

        example = build_empty_example(rows, min_word_count=10, min_char_count=50)

        self.assertIsNotNone(example)
        self.assertEqual(example["output"]["triples"], [])
        self.assertTrue(example["metadata"]["empty_target"])
        self.assertEqual(example["metadata"]["company_name"], "Microsoft")

    def test_sample_empty_examples_respects_target_ratio(self):
        rows = []
        for index in range(10):
            rows.append(
                {
                    "ticker": "msft",
                    "year": 2024,
                    "source_file": "msft.pdf",
                    "page_id": "1",
                    "chunk_id": f"c{index}",
                    "chunk_text": ("This chunk contains finance disclosures only and no ontology-aligned triples. " * 10),
                    "entity": "Microsoft",
                    "entity_type": "ORG",
                    "relationship": "discloses",
                    "target": "Revenue",
                    "target_type": "FIN_METRIC",
                }
            )

        examples, report = sample_empty_examples(
            rows,
            positive_example_count=10,
            empty_ratio=0.3,
            min_word_count=10,
            min_char_count=50,
        )

        self.assertEqual(len(examples), 3)
        self.assertEqual(report["target_empty_count"], 3)
        self.assertEqual(report["sampled_empty_chunk_count"], 3)

    def test_sample_empty_examples_by_count_respects_exclusions(self):
        rows = []
        for index in range(4):
            rows.append(
                {
                    "ticker": "msft",
                    "year": 2024,
                    "source_file": "msft.pdf",
                    "page_id": "1",
                    "chunk_id": f"c{index}",
                    "chunk_text": ("This chunk contains finance disclosures only and no ontology-aligned triples. " * 10),
                    "entity": "Microsoft",
                    "entity_type": "ORG",
                    "relationship": "discloses",
                    "target": "Revenue",
                    "target_type": "FIN_METRIC",
                }
            )

        examples, report = sample_empty_examples_by_count(
            rows,
            target_empty_count=2,
            min_word_count=10,
            min_char_count=50,
            exclude_chunk_keys={'{"chunk_id":"c0","page_id":"1","source_file":"msft.pdf","ticker":"msft","year":2024}'},
        )

        self.assertEqual(len(examples), 2)
        self.assertEqual(report["excluded_chunk_count"], 1)
        self.assertTrue(all(example["metadata"]["chunk_key"]["chunk_id"] != "c0" for example in examples))

    def test_sample_empty_examples_by_count_applies_skip_before_limit(self):
        rows = []
        for index in range(5):
            rows.append(
                {
                    "ticker": "msft",
                    "year": 2024,
                    "source_file": "msft.pdf",
                    "page_id": "1",
                    "chunk_id": f"c{index}",
                    "chunk_text": ("This chunk contains finance disclosures only and no ontology-aligned triples. " * 10),
                    "entity": "Microsoft",
                    "entity_type": "ORG",
                    "relationship": "discloses",
                    "target": "Revenue",
                    "target_type": "FIN_METRIC",
                }
            )

        examples, report = sample_empty_examples_by_count(
            rows,
            target_empty_count=5,
            skip_chunks=2,
            limit_chunks=2,
            min_word_count=10,
            min_char_count=50,
        )

        self.assertEqual({example["metadata"]["chunk_key"]["chunk_id"] for example in examples}, {"c2", "c3"})
        self.assertEqual(report["processed_chunk_count"], 4)
        self.assertEqual(report["skipped_chunk_count"], 2)
        self.assertEqual(report["processed_after_skip_chunk_count"], 2)


if __name__ == "__main__":
    unittest.main()
