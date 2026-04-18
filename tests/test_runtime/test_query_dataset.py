import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from runtime.query_dataset import (
    build_dataset_manifest,
    build_dataset_splits,
    build_synthetic_company_graphs,
    write_dataset_splits,
)


class QueryDatasetTests(unittest.TestCase):
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

    def test_build_dataset_splits_returns_requested_sizes_and_metadata(self):
        splits = build_dataset_splits(train_size=24, validation_size=12, release_eval_size=18, seed=3)

        self.assertEqual(len(splits["train"]), 24)
        self.assertEqual(len(splits["validation"]), 12)
        self.assertEqual(len(splits["release_eval"]), 18)

        sample = splits["train"][0]
        self.assertIn(sample.route_label, {"local_safe", "strong_model_candidate", "refuse"})
        self.assertIn("family", sample.metadata)
        self.assertIn("source_graph_id", sample.metadata)

        local_safe = next(example for example in splits["train"] if example.route_label == "local_safe")
        self.assertTrue(local_safe.target["answerable"])
        self.assertIsNotNone(local_safe.gold_cypher)
        self.assertTrue(local_safe.gold_rows)

    def test_manifest_tracks_split_policy_and_targets(self):
        manifest = build_dataset_manifest(train_size=80, validation_size=24, release_eval_size=36, seed=11)

        self.assertEqual(manifest["split_sizes"]["train"], 80)
        self.assertEqual(manifest["graph_assignments"]["train"], ["aurora", "redwood", "lattice"])
        self.assertEqual(manifest["graph_assignments"]["validation"], ["nimbus", "vector"])
        self.assertEqual(manifest["graph_assignments"]["release_eval"], ["nimbus", "vector"])
        self.assertEqual(sum(manifest["route_targets"]["train"].values()), 80)
        self.assertEqual(
            sum(manifest["local_safe_family_targets"]["validation"].values()),
            manifest["route_targets"]["validation"]["local_safe"],
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

            manifest = json.loads(written["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["split_sizes"]["release_eval"], 21)


if __name__ == "__main__":
    unittest.main()
