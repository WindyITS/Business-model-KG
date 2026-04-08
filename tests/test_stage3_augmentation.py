import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finreflectkg_stage3 import (
    build_relation_trigger_candidate_pool_from_rows,
    build_stage3_prompt,
    filter_relation_triples,
    finalize_stage3_dataset,
    is_narrative_business_prose,
    merge_teacher_reports_into_example,
    relation_trigger_eligibility_report,
)


class Stage3AugmentationTests(unittest.TestCase):
    def test_build_stage3_prompt_includes_existing_graph_context(self):
        example = {
            "instruction": "x",
            "input": "Azure is sold through resellers.",
            "output": {
                "extraction_notes": "",
                "triples": [
                    {
                        "subject": "Microsoft",
                        "subject_type": "Company",
                        "relation": "OFFERS",
                        "object": "Azure",
                        "object_type": "Offering",
                    }
                ],
            },
            "metadata": {
                "company_name": "Microsoft",
                "chunk_key": {
                    "ticker": "msft",
                    "year": 2024,
                    "source_file": "microsoft.pdf",
                    "page_id": "1",
                    "chunk_id": "c0",
                },
            },
        }

        prompt = build_stage3_prompt(example, "SELLS_THROUGH")

        self.assertIn("<company_name>\nMicrosoft\n</company_name>", prompt)
        self.assertIn("<existing_triples>", prompt)
        self.assertIn('"relation":"OFFERS"', prompt)
        self.assertIn("<allowed_subjects>", prompt)
        self.assertIn('"Company":["Microsoft"]', prompt)
        self.assertIn('"Offering":["Azure"]', prompt)

    def test_filter_relation_triples_drops_hallucinated_subjects(self):
        example = {
            "instruction": "x",
            "input": "Azure is sold through resellers.",
            "output": {
                "extraction_notes": "",
                "triples": [
                    {
                        "subject": "Microsoft",
                        "subject_type": "Company",
                        "relation": "OFFERS",
                        "object": "Azure",
                        "object_type": "Offering",
                    }
                ],
            },
            "metadata": {
                "company_name": "Microsoft",
                "chunk_key": {
                    "ticker": "msft",
                    "year": 2024,
                    "source_file": "microsoft.pdf",
                    "page_id": "1",
                    "chunk_id": "c0",
                },
            },
        }

        report = filter_relation_triples(
            example,
            "SELLS_THROUGH",
            [
                {
                    "subject": "Unrelated Corp",
                    "subject_type": "Company",
                    "relation": "SELLS_THROUGH",
                    "object": "resellers",
                    "object_type": "Channel",
                },
                {
                    "subject": "Microsoft",
                    "subject_type": "Company",
                    "relation": "SELLS_THROUGH",
                    "object": "resellers",
                    "object_type": "Channel",
                },
                {
                    "subject": "Azure",
                    "subject_type": "Offering",
                    "relation": "SELLS_THROUGH",
                    "object": "resellers",
                    "object_type": "Channel",
                },
            ],
        )

        self.assertEqual(len(report["valid_triples"]), 2)
        self.assertEqual(report["grounding_rejection_count"], 1)
        self.assertEqual(report["grounding_rejections"][0]["triple"]["subject"], "Unrelated Corp")

    def test_merge_teacher_reports_adds_only_new_valid_triples(self):
        example = {
            "instruction": "x",
            "input": "Microsoft sells through partners and serves developers via Azure subscriptions.",
            "output": {
                "extraction_notes": "",
                "triples": [
                    {
                        "subject": "Microsoft",
                        "subject_type": "Company",
                        "relation": "OFFERS",
                        "object": "Azure",
                        "object_type": "Offering",
                    }
                ],
            },
            "metadata": {
                "chunk_key": {
                    "ticker": "msft",
                    "year": 2024,
                    "source_file": "microsoft.pdf",
                    "page_id": "1",
                    "chunk_id": "c1",
                }
            },
        }

        relation_reports = {
            "SERVES": {
                "valid_triples": [
                    {
                        "subject": "Azure",
                        "subject_type": "Offering",
                        "relation": "SERVES",
                        "object": "developers",
                        "object_type": "CustomerType",
                    }
                ],
                "invalid_triple_count": 0,
                "duplicate_triple_count": 0,
            },
            "SELLS_THROUGH": {
                "valid_triples": [
                    {
                        "subject": "Microsoft",
                        "subject_type": "Company",
                        "relation": "SELLS_THROUGH",
                        "object": "resellers",
                        "object_type": "Channel",
                    }
                ],
                "invalid_triple_count": 0,
                "duplicate_triple_count": 0,
            },
            "MONETIZES_VIA": {
                "valid_triples": [
                    {
                        "subject": "Azure",
                        "subject_type": "Offering",
                        "relation": "MONETIZES_VIA",
                        "object": "subscription",
                        "object_type": "RevenueModel",
                    },
                    {
                        "subject": "Azure",
                        "subject_type": "Offering",
                        "relation": "MONETIZES_VIA",
                        "object": "subscription",
                        "object_type": "RevenueModel",
                    },
                ],
                "invalid_triple_count": 0,
                "duplicate_triple_count": 1,
            },
        }

        augmented, report = merge_teacher_reports_into_example(example, relation_reports)

        self.assertEqual(len(augmented["output"]["triples"]), 4)
        self.assertEqual(report["teacher_added_triple_count"], 3)
        self.assertEqual(augmented["metadata"]["stage3"]["teacher_added_relation_counts"]["SERVES"], 1)
        self.assertEqual(augmented["metadata"]["stage3"]["teacher_added_relation_counts"]["SELLS_THROUGH"], 1)
        self.assertEqual(augmented["metadata"]["stage3"]["teacher_added_relation_counts"]["MONETIZES_VIA"], 1)

    def test_finalize_stage3_dataset_promotes_non_empty_candidates_and_rebalances_empties(self):
        positive_examples = [
            {
                "instruction": "x",
                "input": "A",
                "output": {"extraction_notes": "", "triples": [{"subject": "Microsoft", "subject_type": "Company", "relation": "OFFERS", "object": "Azure", "object_type": "Offering"}]},
                "metadata": {"chunk_key": {"ticker": "msft", "year": 2024, "source_file": "a", "page_id": "1", "chunk_id": "p1"}},
            },
            {
                "instruction": "x",
                "input": "B",
                "output": {"extraction_notes": "", "triples": [{"subject": "Microsoft", "subject_type": "Company", "relation": "HAS_SEGMENT", "object": "More Personal Computing", "object_type": "BusinessSegment"}]},
                "metadata": {"chunk_key": {"ticker": "msft", "year": 2024, "source_file": "a", "page_id": "1", "chunk_id": "p2"}},
            },
        ]
        empty_candidates = [
            {
                "instruction": "x",
                "input": "C",
                "output": {"extraction_notes": "", "triples": []},
                "metadata": {"chunk_key": {"ticker": "msft", "year": 2024, "source_file": "a", "page_id": "1", "chunk_id": "e1"}, "empty_target": True},
            },
            {
                "instruction": "x",
                "input": "D",
                "output": {"extraction_notes": "", "triples": []},
                "metadata": {"chunk_key": {"ticker": "msft", "year": 2024, "source_file": "a", "page_id": "1", "chunk_id": "e2"}, "empty_target": True},
            },
            {
                "instruction": "x",
                "input": "E",
                "output": {
                    "extraction_notes": "",
                    "triples": [
                        {
                            "subject": "Azure",
                            "subject_type": "Offering",
                            "relation": "MONETIZES_VIA",
                            "object": "subscription",
                            "object_type": "RevenueModel",
                        }
                    ],
                },
                "metadata": {"chunk_key": {"ticker": "msft", "year": 2024, "source_file": "a", "page_id": "1", "chunk_id": "e3"}, "empty_target": False},
            },
        ]

        final_positive, final_empty, training, report = finalize_stage3_dataset(
            positive_examples,
            empty_candidates,
            empty_ratio=0.5,
        )

        self.assertEqual(len(final_positive), 3)
        self.assertEqual(len(final_empty), 2)
        self.assertEqual(len(training), 5)
        self.assertEqual(report["promoted_empty_example_count"], 1)
        self.assertEqual(report["target_empty_count"], 2)

    def test_relation_trigger_candidate_pool_selects_non_projected_chunks_with_strong_cues(self):
        rows = [
            {
                "ticker": "msft",
                "year": 2024,
                "source_file": "a",
                "page_id": "1",
                "chunk_id": "c1",
                "chunk_text": (
                    "Microsoft reaches enterprise customers through resellers and distributors, while Azure remains "
                    "available online through the company website. " * 8
                ).strip(),
                "entity": "Microsoft",
                "entity_type": "misc",
                "relationship": "unknown",
                "target": "Azure",
                "target_type": "misc",
            },
            {
                "ticker": "msft",
                "year": 2024,
                "source_file": "a",
                "page_id": "1",
                "chunk_id": "c2",
                "chunk_text": (
                    "Revenue is derived from subscription fees, transaction fees, and licensing revenue across multiple offerings. "
                    * 8
                ).strip(),
                "entity": "Microsoft",
                "entity_type": "misc",
                "relationship": "unknown",
                "target": "Azure",
                "target_type": "misc",
            },
            {
                "ticker": "msft",
                "year": 2024,
                "source_file": "a",
                "page_id": "1",
                "chunk_id": "c3",
                "chunk_text": (
                    "This section contains general administrative language and no relevant business-model trigger phrases. "
                    * 8
                ).strip(),
                "entity": "Microsoft",
                "entity_type": "misc",
                "relationship": "unknown",
                "target": "Azure",
                "target_type": "misc",
            },
        ]

        candidates, report = build_relation_trigger_candidate_pool_from_rows(
            rows,
            target_candidate_count=2,
        )

        self.assertEqual(len(candidates), 2)
        self.assertEqual(report["eligible_relation_trigger_chunk_count"], 2)
        self.assertEqual(report["sampled_relation_trigger_chunk_count"], 2)
        self.assertIn("SERVES", report["matched_relation_counts"])
        self.assertIn("SELLS_THROUGH", report["matched_relation_counts"])
        self.assertIn("MONETIZES_VIA", report["matched_relation_counts"])
        self.assertEqual(candidates[0]["metadata"]["stage3_candidate_source"], "relation_trigger")
        self.assertGreater(candidates[0]["metadata"]["relation_trigger_score"], 0)

    def test_relation_trigger_eligibility_rejects_table_and_disclosure_chunks(self):
        table_text = (
            "| Fiscal Year Ended | 2014 | 2013 | 2012 |\n"
            "| Uniform Direct Sales | 455,485 | 461,328 | 433,994 |"
        )
        legal_text = (
            "Litigation and Other Contingencies. The company is subject to legal proceedings arising in the "
            "ordinary course of business. In the opinion of management, the aggregate liability will not have "
            "a material adverse effect on the consolidated financial statements."
        )
        finance_text = (
            "Net interest expense decreased from fiscal 2012 due to the maturity of senior notes and the "
            "issuance of $250.0 million aggregate principal amount of 3.25% senior notes due 2022."
        )
        business_text = (
            "The company reaches enterprise customers through resellers and distributors. Its business strategy "
            "focuses on providing offerings through channel partners and direct sales teams across North America."
        )

        self.assertFalse(is_narrative_business_prose(table_text))
        self.assertFalse(is_narrative_business_prose(legal_text))
        self.assertFalse(is_narrative_business_prose(finance_text))
        self.assertTrue(is_narrative_business_prose(business_text))

        table_report = relation_trigger_eligibility_report(table_text)
        self.assertTrue(table_report["is_table_like"])
        self.assertFalse(table_report["is_narrative_business_prose"])

    def test_finalize_stage3_dataset_does_not_use_trigger_empties_as_training_empties(self):
        positive_examples = [
            {
                "instruction": "x",
                "input": "A",
                "output": {"extraction_notes": "", "triples": [{"subject": "Microsoft", "subject_type": "Company", "relation": "OFFERS", "object": "Azure", "object_type": "Offering"}]},
                "metadata": {"chunk_key": {"ticker": "msft", "year": 2024, "source_file": "a", "page_id": "1", "chunk_id": "p1"}},
            }
        ]
        empty_candidates = [
            {
                "instruction": "x",
                "input": "B",
                "output": {"extraction_notes": "", "triples": []},
                "metadata": {"chunk_key": {"ticker": "msft", "year": 2024, "source_file": "a", "page_id": "1", "chunk_id": "e1"}, "empty_target": True},
            }
        ]
        trigger_candidates = [
            {
                "instruction": "x",
                "input": "C",
                "output": {"extraction_notes": "", "triples": []},
                "metadata": {
                    "chunk_key": {"ticker": "msft", "year": 2024, "source_file": "a", "page_id": "1", "chunk_id": "t1"},
                    "stage3_candidate_source": "relation_trigger",
                },
            },
            {
                "instruction": "x",
                "input": "D",
                "output": {
                    "extraction_notes": "",
                    "triples": [
                        {
                            "subject": "Microsoft",
                            "subject_type": "Company",
                            "relation": "SELLS_THROUGH",
                            "object": "resellers",
                            "object_type": "Channel",
                        }
                    ],
                },
                "metadata": {
                    "chunk_key": {"ticker": "msft", "year": 2024, "source_file": "a", "page_id": "1", "chunk_id": "t2"},
                    "stage3_candidate_source": "relation_trigger",
                },
            },
        ]

        final_positive, final_empty, training, report = finalize_stage3_dataset(
            positive_examples,
            empty_candidates,
            empty_ratio=1.0,
            augmented_relation_trigger_candidates=trigger_candidates,
        )

        self.assertEqual(len(final_positive), 2)
        self.assertEqual(len(final_empty), 1)
        self.assertEqual(len(training), 3)
        self.assertEqual(report["promoted_relation_trigger_example_count"], 1)
        self.assertEqual(report["rejected_relation_trigger_example_count"], 1)
        self.assertEqual(final_empty[0]["metadata"]["chunk_key"]["chunk_id"], "e1")


if __name__ == "__main__":
    unittest.main()
