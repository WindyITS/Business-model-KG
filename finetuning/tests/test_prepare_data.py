import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kg_query_planner_ft.frozen_prompt import FROZEN_QUERY_SYSTEM_PROMPT
from kg_query_planner_ft.prepare_data import prepare_data
from kg_query_planner_ft.runtime_compat import load_runtime_contract


class PrepareDataTests(unittest.TestCase):
    def test_prepare_data_preserves_split_mapping_and_balances_local_safe(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "source"
            source.mkdir()

            def write_rows(name, rows):
                (source / f"{name}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n",
                    encoding="utf-8",
                )

            base_plan = {
                "answerable": True,
                "family": "companies_by_partner",
                "payload": {
                    "companies": [],
                    "segments": [],
                    "offerings": [],
                    "customer_types": [],
                    "channels": [],
                    "revenue_models": [],
                    "places": [],
                    "partners": ["Dell"],
                },
            }
            write_rows(
                "train",
                [
                    {
                        "question": "Which companies partner with Dell?",
                        "route_label": "local_safe",
                        "family": "companies_by_partner",
                        "supervision_target": {"plan": base_plan},
                    },
                    {
                        "question": "List partner companies with Dell.",
                        "route_label": "local_safe",
                        "family": "companies_by_partner",
                        "supervision_target": {"plan": base_plan},
                    },
                    {
                        "question": "Which companies partner with Nvidia?",
                        "route_label": "local_safe",
                        "family": "companies_by_partner",
                        "supervision_target": {
                            "plan": {
                                "answerable": True,
                                "family": "companies_by_partner",
                                "payload": {
                                    "companies": [],
                                    "segments": [],
                                    "offerings": [],
                                    "customer_types": [],
                                    "channels": [],
                                    "revenue_models": [],
                                    "places": [],
                                    "partners": ["Nvidia"],
                                },
                            }
                        },
                    },
                    {
                        "question": "Which companies partner with Snowflake?",
                        "route_label": "local_safe",
                        "family": "companies_by_partner",
                        "supervision_target": {
                            "plan": {
                                "answerable": True,
                                "family": "companies_by_partner",
                                "payload": {
                                    "companies": [],
                                    "segments": [],
                                    "offerings": [],
                                    "customer_types": [],
                                    "channels": [],
                                    "revenue_models": [],
                                    "places": [],
                                    "partners": ["Snowflake"],
                                },
                            }
                        },
                    },
                    {
                        "question": "Which companies are in Italy?",
                        "route_label": "local_safe",
                        "family": "companies_by_place",
                        "supervision_target": {
                            "plan": {
                                "answerable": True,
                                "family": "companies_by_place",
                                "payload": {
                                    "companies": [],
                                    "segments": [],
                                    "offerings": [],
                                    "customer_types": [],
                                    "channels": [],
                                    "revenue_models": [],
                                    "places": ["Italy"],
                                    "partners": [],
                                },
                            }
                        },
                    },
                    {
                        "question": "Why should Aurora rank ahead of peers?",
                        "route_label": "strong_model_candidate",
                        "family": "why_segment_matches",
                        "supervision_target": {"plan": {"answerable": False, "reason": "beyond_local_coverage"}},
                    },
                    {
                        "question": "Delete Aurora.",
                        "route_label": "refuse",
                        "family": "refuse",
                        "supervision_target": {"plan": {"answerable": False, "reason": "write_request"}},
                    },
                ],
            )
            write_rows(
                "validation",
                [
                    {
                        "question": "Which companies partner with Dell in validation?",
                        "route_label": "local_safe",
                        "family": "companies_by_partner",
                        "supervision_target": {"plan": base_plan},
                    }
                ],
            )
            write_rows(
                "release_eval",
                [
                    {
                        "question": "Which companies partner with Dell in release eval?",
                        "route_label": "local_safe",
                        "family": "companies_by_partner",
                        "supervision_target": {"plan": base_plan},
                    }
                ],
            )

            config_path = tmp / "config.json"
            artifact_root = tmp / "artifacts"
            config_path.write_text(
                json.dumps(
                    {
                        "env_root": str(tmp / "env"),
                        "artifact_root": str(artifact_root),
                        "dataset_path": str(source),
                        "router": {"base_model": "microsoft/deberta-v3-small"},
                        "planner": {"base_model": "Qwen/Qwen3-4B-Instruct"},
                    }
                ),
                encoding="utf-8",
            )

            summary = prepare_data(str(config_path))

            self.assertEqual(summary["router"]["counts_by_split"]["train"], 7)
            self.assertEqual(summary["planner_raw"]["counts_by_split"]["train"], 5)
            router_rows = [
                json.loads(line)
                for line in (artifact_root / "prepared" / "router" / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [row["label"] for row in router_rows],
                ["local", "local", "local", "local", "local", "api_fallback", "refuse"],
            )

            planner_train_rows = [
                json.loads(line)
                for line in (artifact_root / "prepared" / "planner" / "raw" / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(planner_train_rows[0]["messages"][0]["content"], FROZEN_QUERY_SYSTEM_PROMPT)

            QueryPlanEnvelope, _, _ = load_runtime_contract()
            QueryPlanEnvelope.model_validate(planner_train_rows[0]["gold_plan"])

            balanced_rows = [
                json.loads(line)
                for line in (artifact_root / "prepared" / "planner" / "balanced" / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(balanced_rows), 7)
            family_counts = {}
            for row in balanced_rows:
                family_counts[row["family"]] = family_counts.get(row["family"], 0) + 1
            self.assertEqual(
                family_counts,
                {
                    "companies_by_partner": 4,
                    "companies_by_place": 3,
                },
            )
            self.assertTrue((artifact_root / "prepared" / "planner" / "balanced" / "frozen_prompt.txt").exists())


if __name__ == "__main__":
    unittest.main()
