import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finreflectkg_projection import (
    build_empty_example,
    build_projection_example,
    discover_trusted_segments,
    iter_grouped_rows,
    map_row_to_triple,
    map_row_to_triples,
)


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

    def test_projection_rejects_table_like_chunk_even_with_valid_triple(self):
        rows = [
            {
                "ticker": "msft",
                "year": 2024,
                "source_file": "microsoft.pdf",
                "page_id": "12",
                "chunk_id": "c2",
                "chunk_text": (
                    "| Fiscal Year Ended | 2024 | 2023 |\n"
                    "| Microsoft | Europe | Azure |\n"
                    "| Microsoft | Europe | Azure |"
                ),
                "entity": "Microsoft",
                "entity_type": "ORG",
                "relationship": "operates_in",
                "target": "Europe",
                "target_type": "GPE",
            }
        ]

        example = build_projection_example(rows)

        self.assertIsNone(example)

    def test_maps_direct_has_segment_relation(self):
        row = {
            "entity": "Amazon",
            "entity_type": "ORG",
            "relationship": "has_segment",
            "target": "AWS Segment",
            "target_type": "SEGMENT",
        }

        triples, reason = map_row_to_triples(row)

        self.assertIsNone(reason)
        self.assertEqual(
            triples,
            [
                {
                    "subject": "Amazon",
                    "subject_type": "Company",
                    "relation": "HAS_SEGMENT",
                    "object": "AWS Segment",
                    "object_type": "BusinessSegment",
                }
            ],
        )

    def test_maps_reverse_segment_company_relation_to_has_segment(self):
        row = {
            "entity": "AWS Segment",
            "entity_type": "SEGMENT",
            "relationship": "is_part_of",
            "target": "Amazon",
            "target_type": "ORG",
        }

        triples, reason = map_row_to_triples(row)

        self.assertIsNone(reason)
        self.assertEqual(
            triples,
            [
                {
                    "subject": "Amazon",
                    "subject_type": "Company",
                    "relation": "HAS_SEGMENT",
                    "object": "AWS Segment",
                    "object_type": "BusinessSegment",
                }
            ],
        )

    def test_trusted_segment_product_row_derives_offers_and_part_of(self):
        row = {
            "ticker": "amzn",
            "year": 2024,
            "source_file": "amazon.pdf",
            "entity": "AWS Segment",
            "entity_type": "SEGMENT",
            "relationship": "produce",
            "target": "S3",
            "target_type": "PRODUCT",
        }

        triples, reason = map_row_to_triples(row, trusted_segment_keys={"aws segment"})

        self.assertIsNone(reason)
        self.assertEqual(
            triples,
            [
                {
                    "subject": "AWS Segment",
                    "subject_type": "BusinessSegment",
                    "relation": "OFFERS",
                    "object": "S3",
                    "object_type": "Offering",
                },
                {
                    "subject": "S3",
                    "subject_type": "Offering",
                    "relation": "PART_OF",
                    "object": "AWS Segment",
                    "object_type": "BusinessSegment",
                },
            ],
        )

    def test_product_segment_row_without_trusted_anchor_is_dropped(self):
        row = {
            "entity": "S3",
            "entity_type": "PRODUCT",
            "relationship": "belongs_to",
            "target": "AWS Segment",
            "target_type": "SEGMENT",
        }

        triples, reason = map_row_to_triples(row)

        self.assertEqual(triples, [])
        self.assertEqual(reason, "untrusted_segment_object")

    def test_untrusted_segment_product_row_is_dropped(self):
        row = {
            "entity": "Business Segment",
            "entity_type": "SEGMENT",
            "relationship": "produce",
            "target": "Wireless Service",
            "target_type": "PRODUCT",
        }

        triples, reason = map_row_to_triples(row, trusted_segment_keys={"aws segment"})

        self.assertEqual(triples, [])
        self.assertEqual(reason, "untrusted_segment_subject")

    def test_company_produce_row_still_maps_to_offers(self):
        row = {
            "entity": "Amazon",
            "entity_type": "ORG",
            "relationship": "produce",
            "target": "Kindle",
            "target_type": "PRODUCT",
        }

        triples, reason = map_row_to_triples(row)

        self.assertIsNone(reason)
        self.assertEqual(
            triples,
            [
                {
                    "subject": "Amazon",
                    "subject_type": "Company",
                    "relation": "OFFERS",
                    "object": "Kindle",
                    "object_type": "Offering",
                }
            ],
        )

    def test_company_provide_row_is_dropped_as_noisy_offer(self):
        row = {
            "entity": "Cintas",
            "entity_type": "ORG",
            "relationship": "provide",
            "target": "mentor program",
            "target_type": "PRODUCT",
        }

        triples, reason = map_row_to_triples(row)

        self.assertEqual(triples, [])
        self.assertEqual(reason, "unmapped_relation")

    def test_empty_builder_treats_untrusted_segment_only_chunk_as_empty(self):
        rows = [
            {
                "ticker": "vz",
                "year": 2024,
                "source_file": "verizon.pdf",
                "page_id": "1",
                "chunk_id": "c1",
                "chunk_text": (
                    "The business segment provides wireless service, data service, and security service to enterprise "
                    "customers across the country while discussing strategy and operations in narrative form. "
                    * 5
                ),
                "entity": "business segment",
                "entity_type": "SEGMENT",
                "relationship": "produce",
                "target": "wireless service",
                "target_type": "PRODUCT",
            }
        ]

        example = build_empty_example(
            rows,
            min_word_count=10,
            min_char_count=50,
            trusted_segments_by_filing={("vz", 2024, "verizon.pdf"): {"consumer segment"}},
        )

        self.assertIsNotNone(example)
        self.assertEqual(example["output"]["triples"], [])

    def test_projection_uses_filing_level_trusted_segments_for_part_of(self):
        discovery_rows = [
            {
                "ticker": "pom",
                "year": 2024,
                "source_file": "pom.pdf",
                "page_id": "1",
                "chunk_id": "c1",
                "chunk_text": "POM has reportable segments including Pepco Energy Service.",
                "entity": "POM",
                "entity_type": "ORG",
                "relationship": "has_reportable_segment",
                "target": "Pepco Energy Service",
                "target_type": "SEGMENT",
            }
        ]
        trusted_segments_by_filing, _ = discover_trusted_segments(discovery_rows)

        rows = [
            {
                "ticker": "pom",
                "year": 2024,
                "source_file": "pom.pdf",
                "page_id": "2",
                "chunk_id": "c2",
                "chunk_text": (
                    "Pepco Energy Service produces steam and chill water and develops energy efficiency projects for "
                    "commercial customers across the region in a narrative business description. "
                    * 5
                ),
                "entity": "Pepco Energy Service",
                "entity_type": "SEGMENT",
                "relationship": "produce",
                "target": "steam and chill water",
                "target_type": "PRODUCT",
            }
        ]

        example = build_projection_example(rows, trusted_segments_by_filing=trusted_segments_by_filing)

        self.assertIsNotNone(example)
        self.assertEqual(
            {triple["relation"] for triple in example["output"]["triples"]},
            {"OFFERS", "PART_OF"},
        )


if __name__ == "__main__":
    unittest.main()
