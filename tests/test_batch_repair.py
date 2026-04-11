import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from batch_repair import build_repair_plan, example_chunk_key_text


def make_example(chunk_id: str, triples: list[dict]) -> dict:
    return {
        "instruction": "Extract the graph.",
        "input": f"Chunk {chunk_id} text.",
        "output": {"extraction_notes": "", "triples": triples},
        "metadata": {
            "chunk_key": {
                "ticker": "MSFT",
                "year": 2024,
                "source_file": "msft.pdf",
                "page_id": "1",
                "chunk_id": chunk_id,
            }
        },
    }


class BatchRepairTests(unittest.TestCase):
    def test_build_repair_plan_reuses_unchanged_positives_and_unions_empty_pool(self):
        legacy_projected_a = make_example(
            "a",
            [
                {
                    "subject": "Microsoft",
                    "subject_type": "Company",
                    "relation": "OFFERS",
                    "object": "Azure",
                    "object_type": "Offering",
                }
            ],
        )
        repaired_projected_a = make_example(
            "a",
            [
                {
                    "subject": "Microsoft",
                    "subject_type": "Company",
                    "relation": "OFFERS",
                    "object": "Azure",
                    "object_type": "Offering",
                }
            ],
        )
        repaired_projected_b = make_example(
            "b",
            [
                {
                    "subject": "Microsoft",
                    "subject_type": "Company",
                    "relation": "OFFERS",
                    "object": "Dynamics 365",
                    "object_type": "Offering",
                },
                {
                    "subject": "Dynamics 365",
                    "subject_type": "Offering",
                    "relation": "PART_OF",
                    "object": "Business Applications",
                    "object_type": "BusinessSegment",
                },
            ],
        )
        legacy_projected_b = make_example(
            "b",
            [
                {
                    "subject": "Microsoft",
                    "subject_type": "Company",
                    "relation": "OFFERS",
                    "object": "Dynamics 365",
                    "object_type": "Offering",
                }
            ],
        )
        repaired_projected_c = make_example(
            "c",
            [
                {
                    "subject": "Microsoft",
                    "subject_type": "Company",
                    "relation": "HAS_SEGMENT",
                    "object": "Cloud",
                    "object_type": "BusinessSegment",
                }
            ],
        )

        legacy_augmented_positive_a = make_example(
            "a",
            [
                {
                    "subject": "Microsoft",
                    "subject_type": "Company",
                    "relation": "OFFERS",
                    "object": "Azure",
                    "object_type": "Offering",
                },
                {
                    "subject": "Microsoft",
                    "subject_type": "Company",
                    "relation": "SERVES",
                    "object": "large enterprises",
                    "object_type": "CustomerType",
                },
            ],
        )
        legacy_augmented_positive_b = make_example(
            "b",
            [
                {
                    "subject": "Microsoft",
                    "subject_type": "Company",
                    "relation": "OFFERS",
                    "object": "Dynamics 365",
                    "object_type": "Offering",
                }
            ],
        )
        teacher_log_a = {
            "chunk_key": legacy_augmented_positive_a["metadata"]["chunk_key"],
            "relation_reports": {
                "SERVES": {
                    "valid_triples": [
                        {
                            "subject": "Microsoft",
                            "subject_type": "Company",
                            "relation": "SERVES",
                            "object": "large enterprises",
                            "object_type": "CustomerType",
                        }
                    ],
                    "invalid_triple_count": 0,
                    "duplicate_triple_count": 0,
                    "grounding_rejection_count": 0,
                },
                "SELLS_THROUGH": {
                    "valid_triples": [],
                    "invalid_triple_count": 0,
                    "duplicate_triple_count": 0,
                    "grounding_rejection_count": 0,
                },
                "MONETIZES_VIA": {
                    "valid_triples": [],
                    "invalid_triple_count": 0,
                    "duplicate_triple_count": 0,
                    "grounding_rejection_count": 0,
                },
            },
        }
        teacher_log_b = {
            "chunk_key": legacy_augmented_positive_b["metadata"]["chunk_key"],
            "relation_reports": {
                "SERVES": {
                    "valid_triples": [],
                    "invalid_triple_count": 0,
                    "duplicate_triple_count": 0,
                    "grounding_rejection_count": 0,
                },
                "SELLS_THROUGH": {
                    "valid_triples": [],
                    "invalid_triple_count": 0,
                    "duplicate_triple_count": 0,
                    "grounding_rejection_count": 0,
                },
                "MONETIZES_VIA": {
                    "valid_triples": [],
                    "invalid_triple_count": 0,
                    "duplicate_triple_count": 0,
                    "grounding_rejection_count": 0,
                },
            },
        }

        legacy_empty_c = make_example("c", [])
        legacy_empty_d = make_example("d", [])
        legacy_refill_empty_f = make_example("f", [])
        repaired_empty_d = make_example("d", [])
        repaired_empty_e = make_example("e", [])

        window_chunk_keys = {
            example_chunk_key_text(example)
            for example in [
                repaired_projected_a,
                repaired_projected_b,
                repaired_projected_c,
                repaired_empty_d,
                repaired_empty_e,
                legacy_refill_empty_f,
            ]
        }
        window_chunk_keys.add(example_chunk_key_text(legacy_empty_d))

        plan = build_repair_plan(
            repaired_projected_examples=[repaired_projected_a, repaired_projected_b, repaired_projected_c],
            repaired_sampled_empty_examples=[repaired_empty_d, repaired_empty_e],
            legacy_projected_examples=[legacy_projected_a, legacy_projected_b],
            legacy_empty_origin_examples=[legacy_empty_c, legacy_empty_d, legacy_refill_empty_f],
            legacy_verified_empty_examples=[legacy_empty_d, legacy_refill_empty_f],
            legacy_augmented_positive_examples=[legacy_augmented_positive_a, legacy_augmented_positive_b],
            legacy_teacher_logs=[teacher_log_a, teacher_log_b],
            window_chunk_key_texts=window_chunk_keys,
        )

        self.assertEqual(len(plan["reused_augmented_positive_examples"]), 1)
        self.assertEqual(
            plan["reused_augmented_positive_examples"][0]["output"]["triples"][1]["relation"],
            "SERVES",
        )
        self.assertEqual(len(plan["positive_examples_to_rerun"]), 2)
        self.assertEqual(
            plan["report"]["positive_rerun_reason_counts"],
            {"changed_deterministic_base": 1, "legacy_empty_now_positive": 1},
        )
        # Legacy verified empties (d, f) are reused directly — no teacher rerun.
        self.assertEqual(len(plan["reused_verified_empty_examples"]), 2)
        self.assertIn(
            example_chunk_key_text(legacy_refill_empty_f),
            {example_chunk_key_text(e) for e in plan["reused_verified_empty_examples"]},
        )
        # Only new empty candidate (e) needs a teacher run; d is already in reused_verified.
        self.assertEqual(len(plan["new_empty_candidates_to_run"]), 1)
        self.assertEqual(plan["report"]["reused_verified_empty_count"], 2)
        self.assertEqual(plan["report"]["new_empty_candidate_count"], 1)


if __name__ == "__main__":
    unittest.main()
