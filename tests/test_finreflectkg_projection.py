import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finreflectkg_projection import build_projection_example, iter_grouped_rows, map_row_to_triple


class FinReflectKGProjectionTests(unittest.TestCase):
    def test_maps_safe_finreflectkg_row(self):
        row = {
            "entity": "Microsoft",
            "entity_type": "ORG",
            "relationship": "operates_in",
            "target": "Europe",
            "target_type": "GPE",
        }

        triple, reason = map_row_to_triple(row)

        self.assertIsNone(reason)
        self.assertEqual(
            triple,
            {
                "subject": "Microsoft",
                "subject_type": "Company",
                "relation": "OPERATES_IN",
                "object": "Europe",
                "object_type": "Place",
            },
        )

    def test_drops_unmapped_relation(self):
        row = {
            "entity": "Microsoft",
            "entity_type": "ORG",
            "relationship": "discloses",
            "target": "Net Income",
            "target_type": "FIN_METRIC",
        }

        triple, reason = map_row_to_triple(row)

        self.assertIsNone(triple)
        self.assertEqual(reason, "unmapped_object_type")

    def test_groups_rows_by_chunk_identity(self):
        rows = [
            {"ticker": "msft", "year": 2024, "source_file": "a.pdf", "page_id": "1", "chunk_id": "1", "chunk_text": "A"},
            {"ticker": "msft", "year": 2024, "source_file": "a.pdf", "page_id": "1", "chunk_id": "1", "chunk_text": "A"},
            {"ticker": "msft", "year": 2024, "source_file": "a.pdf", "page_id": "1", "chunk_id": "2", "chunk_text": "B"},
        ]

        groups = list(iter_grouped_rows(rows))

        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]), 2)
        self.assertEqual(len(groups[1]), 1)

    def test_builds_chunk_level_projection_example(self):
        rows = [
            {
                "ticker": "msft",
                "year": 2024,
                "source_file": "microsoft.pdf",
                "page_id": "12",
                "chunk_id": "c1",
                "chunk_text": "Microsoft operates in Europe. Azure is introduced as a cloud platform.",
                "entity": "Microsoft",
                "entity_type": "ORG",
                "relationship": "operates_in",
                "target": "Europe",
                "target_type": "GPE",
            },
            {
                "ticker": "msft",
                "year": 2024,
                "source_file": "microsoft.pdf",
                "page_id": "12",
                "chunk_id": "c1",
                "chunk_text": "Microsoft operates in Europe. Azure is introduced as a cloud platform.",
                "entity": "Microsoft",
                "entity_type": "ORG",
                "relationship": "introduces",
                "target": "Azure",
                "target_type": "PRODUCT",
            },
        ]

        example = build_projection_example(rows)

        self.assertIsNotNone(example)
        self.assertEqual(example["metadata"]["kept_triple_count"], 2)
        self.assertEqual(example["metadata"]["company_name"], "Microsoft")
        self.assertEqual(example["output"]["triples"][0]["subject_type"], "Company")


if __name__ == "__main__":
    unittest.main()
