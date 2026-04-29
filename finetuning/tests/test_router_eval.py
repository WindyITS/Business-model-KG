import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kg_query_planner_ft.router_eval import decide_router_outcome, planner_gate_is_open, predict_router_probabilities
from kg_query_planner_ft.router_metrics import (
    LOCAL_DECISION_THRESHOLD,
    apply_router_policy,
    metrics_for_label,
    summarize_predictions,
)


class RouterMetricTests(unittest.TestCase):
    def test_metrics_for_label_zero_predictions_have_zero_precision(self):
        metrics = metrics_for_label(
            ["local", "api_fallback"],
            ["api_fallback", "api_fallback"],
            "local",
        )
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)

    def test_apply_router_policy_requires_confident_local_then_chooses_best_nonlocal_label(self):
        probs = np.array(
            [
                [0.01, 0.98, 0.01],
                [0.05, 0.20, 0.80],
                [0.70, 0.20, 0.10],
                [0.10, LOCAL_DECISION_THRESHOLD - 0.01, 0.12],
                [0.01, 0.00009, 0.98],
            ]
        )
        decisions = apply_router_policy(probs)
        self.assertEqual(decisions, ["local", "refuse", "api_fallback", "refuse", "refuse"])
        summary = summarize_predictions(["local", "refuse", "api_fallback", "refuse", "refuse"], decisions)
        self.assertAlmostEqual(summary["accuracy"], 1.0)

    def test_decide_router_outcome_uses_fixed_local_gate_then_best_nonlocal_label(self):
        thresholds = {
            "local_threshold": {"threshold": 0.00005},
        }
        self.assertEqual(
            decide_router_outcome({"local": 0.96, "refuse": 0.91, "api_fallback": 0.01}, thresholds),
            "local",
        )
        self.assertEqual(
            decide_router_outcome({"local": 0.12, "refuse": 0.91, "api_fallback": 0.20}, thresholds),
            "refuse",
        )
        self.assertEqual(
            decide_router_outcome({"local": 0.94, "refuse": 0.30, "api_fallback": 0.40}, thresholds),
            "api_fallback",
        )
        self.assertEqual(
            decide_router_outcome({"local": 0.00009, "refuse": 0.98, "api_fallback": 0.01}, thresholds),
            "refuse",
        )

    def test_predict_router_probabilities_passes_progress_desc(self):
        captured: dict[str, object] = {}

        def fake_collect(model_dir, rows, *, max_length, batch_size, desc):
            captured["model_dir"] = model_dir
            captured["rows"] = rows
            captured["max_length"] = max_length
            captured["batch_size"] = batch_size
            captured["desc"] = desc
            return np.array([[0.1, 0.8, 0.1]])

        with patch("kg_query_planner_ft.router_eval.collect_router_logits", side_effect=fake_collect):
            probabilities = predict_router_probabilities(
                "Which companies partner with Dell?",
                model_dir=Path("/tmp/router"),
                max_length=256,
                temperature=1.0,
            )

        self.assertEqual(captured["desc"], "score router question")
        self.assertEqual(captured["batch_size"], 1)
        self.assertEqual(len(captured["rows"]), 1)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_planner_gate_requires_local_predictions_and_precision(self):
        closed_policy_metrics = {
            "counts": {"api_fallback": 4},
            "per_label": {"local": {"precision": 1.0}},
        }
        open_policy_metrics = {
            "counts": {"local": 3, "api_fallback": 1},
            "per_label": {"local": {"precision": 0.98}},
        }

        self.assertFalse(planner_gate_is_open(closed_policy_metrics, min_local_precision=0.97))
        self.assertTrue(planner_gate_is_open(open_policy_metrics, min_local_precision=0.97))


if __name__ == "__main__":
    unittest.main()
