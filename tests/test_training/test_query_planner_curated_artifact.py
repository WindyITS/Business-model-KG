import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from runtime.query_planner import QueryPlanPayload
from training.query_planner import (
    build_curated_artifact,
    freeze_curated_baseline,
    verify_curated_artifact,
)
from training.query_planner import curated_artifact as curated_module


class QueryPlannerCuratedArtifactTests(unittest.TestCase):
    def test_capped_family_targets_is_hash_seed_stable(self):
        script = (
            "import json; "
            "from training.query_planner import dataset as dataset_module; "
            "print(json.dumps(dataset_module._capped_family_targets("
            "{'b': 1, 'a': 1, 'c': 1}, {'a': 1, 'b': 1, 'c': 1}, 2), sort_keys=True))"
        )
        outputs = set()
        for hash_seed in ("1", "2", "3"):
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = hash_seed
            result = subprocess.run(
                [sys.executable, "-c", script],
                check=True,
                cwd=Path(__file__).resolve().parents[2],
                env=env,
                capture_output=True,
                text=True,
            )
            outputs.add(result.stdout.strip())
        self.assertEqual(outputs, {'{"a": 1, "b": 1}'})

    def test_curate_question_rewrites_known_problem_families(self):
        descendant_row = {
            "case_id": "case-1",
            "template_id": "descendant-offerings",
            "variant_id": "v0",
            "question": "List the descendant offerings of Analytics Studio for Vector Industrial.",
            "target": {
                "answerable": True,
                "family": "descendant_offerings_by_root",
                "payload": {"companies": ["Vector Industrial"], "offerings": ["Analytics Studio"]},
            },
            "supervision_target": {
                "route_label": "local_safe",
                "plan": {
                    "answerable": True,
                    "family": "descendant_offerings_by_root",
                    "payload": {"companies": ["Vector Industrial"], "offerings": ["Analytics Studio"]},
                },
            },
            "route_label": "local_safe",
            "family": "descendant_offerings_by_root",
            "gold_cypher": "RETURN 1",
            "gold_params": {},
            "gold_rows": [{"company": "Vector Industrial", "offering": "Analytics Studio"}],
            "metadata": {"source_graph_ids": ["vector"], "source_graph_id": "vector"},
        }
        curated_question, _ = curated_module._curate_question("train", descendant_row)
        self.assertNotRegex(curated_question, curated_module.DESCENDANT_WORDING_PATTERN)
        self.assertIn("Analytics Studio family", curated_question)

        refuse_row = {
            "case_id": "case-2",
            "template_id": "refuse-boundary",
            "variant_id": "v0",
            "question": "Which companies serve developers but not retailers like Aurora Systems?",
            "target": {"answerable": False, "reason": "beyond_local_coverage"},
            "supervision_target": {
                "route_label": "refuse",
                "plan": {"answerable": False, "reason": "beyond_local_coverage"},
            },
            "route_label": "refuse",
            "family": "refuse",
            "gold_cypher": None,
            "gold_params": {},
            "gold_rows": [],
            "metadata": {"source_graph_ids": ["aurora"], "source_graph_id": "aurora"},
        }
        curated_question, _ = curated_module._curate_question("train", refuse_row)
        self.assertNotRegex(curated_question, curated_module.REFUSE_DEV_RETAIL_PATTERN)
        self.assertIn("Aurora Systems", curated_question)

        ranking_row = {
            "case_id": "case-3",
            "template_id": "ranking-topk",
            "variant_id": "v0",
            "question": "Top 3 customer types by company count.",
            "target": {
                "answerable": True,
                "family": "ranking_topk",
                "payload": {
                    "limit": 3,
                    "aggregate_spec": {"kind": "ranking", "ranking_metric": "customer_type_by_company_count"},
                },
            },
            "supervision_target": {
                "route_label": "local_safe",
                "plan": {
                    "answerable": True,
                    "family": "ranking_topk",
                    "payload": {
                        "limit": 3,
                        "aggregate_spec": {"kind": "ranking", "ranking_metric": "customer_type_by_company_count"},
                    },
                },
            },
            "route_label": "local_safe",
            "family": "ranking_topk",
            "gold_cypher": "RETURN 1",
            "gold_params": {},
            "gold_rows": [{"customer_type": "developers", "company_count": 3}],
            "metadata": {"source_graph_ids": ["aurora"], "source_graph_id": "aurora"},
        }
        curated_question, _ = curated_module._curate_question("train", ranking_row)
        self.assertTrue(curated_question.startswith("What are the top 3 customer types"))

    def test_curated_artifact_pipeline_smoke(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            baseline_dir = tmp / "baseline"
            final_dir = tmp / "final"
            freeze_curated_baseline(
                baseline_dir,
                train_size=120,
                validation_size=36,
                release_eval_size=54,
                seed=7,
                shard_size=50,
            )
            build_curated_artifact(baseline_dir, final_dir, seed=7)
            report = verify_curated_artifact(final_dir)
            self.assertTrue(report["ok"], msg="\n".join(report["issues"]))

            self.assertTrue((baseline_dir / "workflow" / "assignments.json").exists())
            self.assertTrue((baseline_dir / "workflow" / "RUBRIC.md").exists())
            self.assertTrue((final_dir / "DATASET_CARD.md").exists())
            self.assertTrue((final_dir / "checksums.txt").exists())

            curation_log = curated_module._load_jsonl(final_dir / "curation_log.jsonl")
            self.assertGreater(len(curation_log), 0)

            manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["split_overlap_stats"]["train__validation"]["local_safe_target_overlap_count"],
                0,
            )
            self.assertEqual(
                manifest["split_overlap_stats"]["train__release_eval"]["local_safe_target_overlap_count"],
                0,
            )
            self.assertEqual(
                manifest["split_overlap_stats"]["validation__release_eval"]["local_safe_target_overlap_count"],
                0,
            )
            for split_name in ("train", "validation", "release_eval"):
                self.assertEqual(
                    manifest["split_stats"][split_name]["duplicate_question_target_count"],
                    0,
                )


if __name__ == "__main__":
    unittest.main()
