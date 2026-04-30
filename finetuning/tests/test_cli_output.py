import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kg_query_planner_ft.cli_output import (
    render_planner_eval_summary,
    render_planner_training_summary,
    render_prepare_data_summary,
)


class CliOutputTests(unittest.TestCase):
    def test_render_planner_eval_summary_is_human_readable(self):
        summary = {
            "mode": "adapter",
            "backend": "mlx",
            "base_model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
            "artifact_dir": "/tmp/planner/eval",
            "adapter_path": "/tmp/planner/adapter",
            "validation": {
                "count": 10,
                "json_parse_rate": 1.0,
                "contract_valid_rate": 0.6,
                "family_accuracy": 0.5,
                "exact_plan_match_rate": 0.4,
                "correct_output_rate": 0.7,
                "correct_outputs": 7,
                "output_evaluable_count": 10,
                "per_family": {
                    "companies_by_partner": {
                        "count": 4,
                        "contract_valid": 2,
                        "family_correct": 2,
                        "exact_match": 1,
                    },
                    "count_aggregate": {
                        "count": 6,
                        "contract_valid": 1,
                        "family_correct": 1,
                        "exact_match": 0,
                    },
                },
            },
            "release_eval": {
                "count": 12,
                "json_parse_rate": 1.0,
                "contract_valid_rate": 0.5,
                "family_accuracy": 0.4,
                "exact_plan_match_rate": 0.3,
                "correct_output_rate": 0.25,
                "correct_outputs": 3,
                "output_evaluable_count": 12,
                "per_family": {
                    "companies_by_partner": {
                        "count": 5,
                        "contract_valid": 3,
                        "family_correct": 3,
                        "exact_match": 2,
                    },
                    "count_aggregate": {
                        "count": 7,
                        "contract_valid": 0,
                        "family_correct": 0,
                        "exact_match": 0,
                    },
                },
            },
        }

        rendered = render_planner_eval_summary(summary)

        self.assertIn("Planner Evaluation Summary", rendered)
        self.assertIn("Validation split:", rendered)
        self.assertIn("Release eval split:", rendered)
        self.assertIn("correct_output=70.00% (7/10)", rendered)
        self.assertIn("weakest families by contract validity", rendered)
        self.assertIn("count_aggregate", rendered)

    def test_render_planner_training_summary_mentions_fresh_start(self):
        summary = {
            "data_dir": "/tmp/prepared/planner/balanced",
            "adapter_dir": "/tmp/planner/adapter",
            "train_examples": 5000,
            "steps_per_epoch": 1250,
            "total_iters": 3750,
            "checkpoint_every": 500,
            "resume_adapter_file": None,
            "effective_batch_size": 16,
            "config_path": "/tmp/planner/adapter/train_config.yaml",
        }

        rendered = render_planner_training_summary(summary)

        self.assertIn("Planner Training Summary", rendered)
        self.assertIn("Resume adapter file: none (training from scratch)", rendered)

    def test_render_prepare_data_summary_is_human_readable(self):
        summary = {
            "source_root": "/tmp/source",
            "router": {
                "output_dir": "/tmp/router",
                "counts_by_split": {"train": 8, "valid": 2, "test": 3},
                "label_counts_by_split": {
                    "train": {"local": 5, "api_fallback": 2, "refuse": 1},
                    "valid": {"local": 1, "api_fallback": 1},
                    "test": {"local": 2, "refuse": 1},
                },
            },
            "planner_raw": {
                "output_dir": "/tmp/planner/raw",
                "counts_by_split": {"train": 5, "valid": 2, "test": 2},
                "family_counts_by_split": {
                    "train": {"companies_by_partner": 3, "count_aggregate": 2},
                    "valid": {"companies_by_partner": 2},
                    "test": {"count_aggregate": 2},
                },
            },
            "planner_balanced": {
                "output_dir": "/tmp/planner/balanced",
                "counts_by_split": {"train": 7, "valid": 2, "test": 2},
                "family_counts_by_split": {
                    "train": {"companies_by_partner": 4, "count_aggregate": 3},
                    "valid": {"companies_by_partner": 2},
                    "test": {"count_aggregate": 2},
                },
            },
        }

        rendered = render_prepare_data_summary(summary)

        self.assertIn("Prepare Data Summary", rendered)
        self.assertIn("Router dataset:", rendered)
        self.assertIn("Planner raw dataset:", rendered)
        self.assertIn("Planner balanced dataset:", rendered)
        self.assertIn("train:", rendered)


if __name__ == "__main__":
    unittest.main()
