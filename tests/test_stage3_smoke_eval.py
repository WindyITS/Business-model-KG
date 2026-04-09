import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from evaluate_stage3_smoke import evaluate_case_report
from stage3_smoke_cases import Stage3SmokeCase


class Stage3SmokeEvalTests(unittest.TestCase):
    def test_case_requires_clean_output_not_just_filtered_match(self):
        case = Stage3SmokeCase(
            case_id="cleanliness_required",
            relation="SERVES",
            text="x",
            expected_triples=(
                {
                    "subject": "acme",
                    "subject_type": "Company",
                    "relation": "SERVES",
                    "object": "developers",
                    "object_type": "CustomerType",
                },
            ),
        )
        report = {
            "valid_triples": [
                {
                    "subject": "acme",
                    "subject_type": "Company",
                    "relation": "SERVES",
                    "object": "developers",
                    "object_type": "CustomerType",
                }
            ],
            "invalid_triple_count": 1,
            "duplicate_triple_count": 0,
            "grounding_rejection_count": 0,
            "error": None,
        }

        evaluation = evaluate_case_report(case, report)

        self.assertTrue(evaluation["exact_match"])
        self.assertFalse(evaluation["passed"])
        self.assertIn("invalid_triples_emitted", evaluation["failure_reasons"])

    def test_case_fails_when_teacher_call_errors_even_if_expected_empty(self):
        case = Stage3SmokeCase(
            case_id="error_not_a_pass",
            relation="MONETIZES_VIA",
            text="x",
            expected_triples=(),
        )
        report = {
            "valid_triples": [],
            "invalid_triple_count": 0,
            "duplicate_triple_count": 0,
            "grounding_rejection_count": 0,
            "error": "connection refused",
        }

        evaluation = evaluate_case_report(case, report)

        self.assertTrue(evaluation["exact_match"])
        self.assertFalse(evaluation["passed"])
        self.assertIn("teacher_call_error", evaluation["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
