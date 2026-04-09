import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finreflectkg_stage3 import build_stage3_prompt, relation_system_prompt
from ontology_validator import validate_triples
from stage3_prompt_profiles import DEFAULT_PROMPT_PROFILE, list_prompt_profiles
from stage3_smoke_cases import SMOKE_CASES, smoke_case_example


class Stage3SmokeCaseTests(unittest.TestCase):
    def test_smoke_cases_have_valid_expected_triples(self):
        self.assertGreaterEqual(len(SMOKE_CASES), 5)
        for case in SMOKE_CASES:
            report = validate_triples(list(case.expected_triples), dedupe=True)
            self.assertEqual(
                report["summary"]["invalid_triple_count"],
                0,
                msg=f"Case {case.case_id} has invalid expected triples",
            )

    def test_smoke_case_examples_include_company_anchor(self):
        for case in SMOKE_CASES:
            example = smoke_case_example(case)
            self.assertEqual(example["metadata"]["company_name"], case.company_name)
            self.assertEqual(example["metadata"]["chunk_key"]["chunk_id"], case.case_id)

    def test_default_prompt_profile_is_available(self):
        profiles = list_prompt_profiles()
        self.assertIn(DEFAULT_PROMPT_PROFILE, profiles)
        self.assertEqual(DEFAULT_PROMPT_PROFILE, "default")

    def test_system_prompt_contains_extraction_rules(self):
        system = relation_system_prompt("SERVES")
        self.assertIn("strict Information Extraction system", system)
        self.assertIn("ALLOWED OBJECT LABELS", system)
        self.assertIn("EXAMPLE OUTPUT", system)
        self.assertIn("Sales offices", relation_system_prompt("SELLS_THROUGH"))
        self.assertIn("organizational structure", relation_system_prompt("SELLS_THROUGH"))
        self.assertIn("activity description", relation_system_prompt("MONETIZES_VIA"))

    def test_user_prompt_contains_constraints(self):
        example = smoke_case_example(SMOKE_CASES[0])
        user = build_stage3_prompt(example, "SERVES")
        self.assertIn("<constraints>", user)
        self.assertIn("Do not output markdown code blocks", user)
        self.assertIn("SUBJECT RULE", user)


if __name__ == "__main__":
    unittest.main()
