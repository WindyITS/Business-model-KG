import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kg_query_planner_ft.router_eval import decide_router_outcome, planner_gate_is_open, predict_router_probabilities
from kg_query_planner_ft.router_metrics import (
    apply_router_policy,
    choose_binary_threshold,
    metrics_for_label,
    summarize_predictions,
)


class RouterMetricTests(unittest.TestCase):
    def test_choose_binary_threshold_respects_precision_floor(self):
        scores = np.array([0.99, 0.95, 0.91, 0.65, 0.20])
        truth = np.array([True, True, False, False, False])
        threshold = choose_binary_threshold(scores, truth, 0.95)
        self.assertGreaterEqual(threshold["precision"], 0.95)
        self.assertAlmostEqual(threshold["threshold"], 0.95)

    def test_metrics_for_label_zero_predictions_have_zero_precision(self):
        metrics = metrics_for_label(
            ["local", "api_fallback"],
            ["api_fallback", "api_fallback"],
            "local",
        )
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)

    def test_apply_router_policy_uses_local_then_refuse_then_fallback(self):
        probs = np.array(
            [
                [0.01, 0.98, 0.01],
                [0.05, 0.20, 0.80],
                [0.70, 0.20, 0.10],
            ]
        )
        decisions = apply_router_policy(probs, local_threshold=0.97, refuse_threshold=0.75)
        self.assertEqual(decisions, ["local", "refuse", "api_fallback"])
        summary = summarize_predictions(["local", "refuse", "api_fallback"], decisions)
        self.assertAlmostEqual(summary["accuracy"], 1.0)

    def test_decide_router_outcome_matches_threshold_policy(self):
        thresholds = {
            "local_threshold": {"threshold": 0.97},
            "refuse_threshold": {"threshold": 0.90},
        }
        self.assertEqual(
            decide_router_outcome({"local": 0.98, "refuse": 0.91, "api_fallback": 0.01}, thresholds),
            "local",
        )
        self.assertEqual(
            decide_router_outcome({"local": 0.12, "refuse": 0.91, "api_fallback": 0.20}, thresholds),
            "refuse",
        )
        self.assertEqual(
            decide_router_outcome({"local": 0.50, "refuse": 0.30, "api_fallback": 0.20}, thresholds),
            "api_fallback",
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
