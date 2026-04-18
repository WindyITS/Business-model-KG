import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from training.query_planner import (
    build_dataset_manifest,
    build_dataset_splits,
    build_synthetic_company_graphs,
    write_dataset_splits,
)


class QueryPlannerDatasetTests(unittest.TestCase):
    def test_build_synthetic_company_graphs_returns_five_deep_graphs(self):
        companies = build_synthetic_company_graphs()

        self.assertEqual(len(companies), 5)
        self.assertTrue(all(len(company.segments) == 4 for company in companies))

        all_root_offerings = {
            offering.name
            for company in companies
            for segment in company.segments
            for offering in segment.offerings
        }
        self.assertIn("Cloud Platform", all_root_offerings)
        self.assertIn("Marketplace Hub", all_root_offerings)
        self.assertIn("Analytics Studio", all_root_offerings)

    def test_build_dataset_splits_is_deterministic_and_split_scoped(self):
        first = build_dataset_splits(train_size=36, validation_size=18, release_eval_size=24, seed=3)
        second = build_dataset_splits(train_size=36, validation_size=18, release_eval_size=24, seed=3)

        self.assertEqual(first, second)

        allowed_graphs = {
            "train": {"aurora", "redwood", "lattice"},
            "validation": {"nimbus"},
            "release_eval": {"vector"},
        }
        for split_name, examples in first.items():
            self.assertEqual(len(examples), {"train": 36, "validation": 18, "release_eval": 24}[split_name])
            for example in examples:
                self.assertIn(example.route_label, {"local_safe", "strong_model_candidate", "refuse"})
                self.assertTrue(example.case_id)
                self.assertTrue(example.template_id)
                self.assertTrue(example.variant_id)
                self.assertTrue(set(example.metadata["source_graph_ids"]).issubset(allowed_graphs[split_name]))

        local_safe = next(example for example in first["train"] if example.route_label == "local_safe")
        self.assertTrue(local_safe.target["answerable"])
        self.assertIsNotNone(local_safe.gold_cypher)
        self.assertTrue(local_safe.gold_rows)

        non_local = next(example for example in first["train"] if example.route_label != "local_safe")
        self.assertFalse(non_local.target["answerable"])
        self.assertIsNone(non_local.gold_cypher)
        self.assertEqual(non_local.gold_rows, [])

    def test_manifest_tracks_balance_holdouts_and_repetition_stats(self):
        manifest = build_dataset_manifest(train_size=160, validation_size=48, release_eval_size=72, seed=11)

        self.assertEqual(manifest["split_sizes"]["train"], 160)
        self.assertEqual(manifest["graph_assignments"]["train"], ["aurora", "redwood", "lattice"])
        self.assertEqual(manifest["graph_assignments"]["validation"], ["nimbus"])
        self.assertEqual(manifest["graph_assignments"]["release_eval"], ["vector"])
        self.assertEqual(sum(manifest["route_targets"]["train"].values()), 160)
        self.assertEqual(
            sum(manifest["local_safe_family_targets"]["validation"].values()),
            manifest["route_targets"]["validation"]["local_safe"],
        )

        train_stats = manifest["split_stats"]["train"]
        self.assertEqual(train_stats["count"], 160)
        self.assertEqual(train_stats["duplicate_question_count"], 0)
        self.assertEqual(train_stats["duplicate_question_target_count"], 0)
        self.assertEqual(train_stats["unique_question_count"], 160)
        self.assertGreaterEqual(train_stats["unique_target_count"], manifest["route_targets"]["train"]["local_safe"])
        self.assertEqual(
            set(train_stats["refusal_reason_counts"]),
            {
                "ambiguous_closed_label",
                "ambiguous_request",
                "beyond_local_coverage",
                "unsupported_metric",
                "unsupported_schema",
                "unsupported_time",
                "write_request",
            },
        )

    def test_write_dataset_splits_writes_jsonl_and_support_files(self):
        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            written = write_dataset_splits(output_dir, train_size=30, validation_size=15, release_eval_size=21, seed=5)

            self.assertEqual(set(written), {"train", "validation", "release_eval", "synthetic_graphs", "manifest"})
            self.assertTrue(written["train"].exists())
            self.assertTrue(written["validation"].exists())
            self.assertTrue(written["release_eval"].exists())
            self.assertTrue(written["synthetic_graphs"].exists())
            self.assertTrue(written["manifest"].exists())

            first_line = written["train"].read_text(encoding="utf-8").splitlines()[0]
            payload = json.loads(first_line)
            self.assertIn("question", payload)
            self.assertIn("route_label", payload)
            self.assertIn("metadata", payload)
            self.assertIn("case_id", payload)
            self.assertIn("template_id", payload)
            self.assertIn("variant_id", payload)

            manifest = json.loads(written["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["split_sizes"]["release_eval"], 21)


if __name__ == "__main__":
    unittest.main()
