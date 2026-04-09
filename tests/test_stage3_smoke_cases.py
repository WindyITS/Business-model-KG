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

    def test_prompt_profiles_are_listed_and_switchable(self):
        profiles = list_prompt_profiles()
        self.assertIn(DEFAULT_PROMPT_PROFILE, profiles)
        self.assertIn("relaxed_v1", profiles)
        self.assertIn("balanced_v1", profiles)
        self.assertIn("balanced_v2", profiles)
        self.assertIn("balanced_v3", profiles)
        self.assertIn("naive_v1", profiles)

        example = smoke_case_example(SMOKE_CASES[0])
        strict_system = relation_system_prompt("SERVES", prompt_profile=DEFAULT_PROMPT_PROFILE)
        relaxed_system = relation_system_prompt("SERVES", prompt_profile="relaxed_v1")
        relaxed_user = build_stage3_prompt(example, "SERVES", prompt_profile="relaxed_v1")
        balanced_system = relation_system_prompt("SERVES", prompt_profile="balanced_v1")
        balanced_v2_system = relation_system_prompt("SERVES", prompt_profile="balanced_v2")
        balanced_v2_user = build_stage3_prompt(example, "SERVES", prompt_profile="balanced_v2")
        balanced_v3_system = relation_system_prompt("SERVES", prompt_profile="balanced_v3")
        balanced_v3_user = build_stage3_prompt(example, "SERVES", prompt_profile="balanced_v3")
        naive_system = relation_system_prompt("SERVES", prompt_profile="naive_v1")
        naive_user = build_stage3_prompt(example, "SERVES", prompt_profile="naive_v1")

        self.assertNotEqual(strict_system, relaxed_system)
        self.assertNotEqual(strict_system, balanced_system)
        self.assertNotEqual(strict_system, balanced_v2_system)
        self.assertNotEqual(strict_system, naive_system)
        self.assertIn("strongly grounded", relaxed_system)
        self.assertIn("grounds it so strongly", relaxed_user)
        self.assertIn("ordinary business prose", balanced_system)
        self.assertIn("careful annotation job", balanced_v2_system)
        self.assertIn("FOLLOW THIS REASONING STEPS", balanced_v2_system)
        self.assertIn("CONSIDER THE FOLLOWING", balanced_v2_system)
        self.assertIn("If the wording directly supports two labels, extract two triples.", balanced_v2_system)
        self.assertIn("If the chunk clearly supports more than one triple", balanced_v2_user)
        self.assertIn("strict Information Extraction system", balanced_v3_system)
        self.assertIn("ALLOWED OBJECT LABELS", balanced_v3_system)
        self.assertIn("EXAMPLE OUTPUT", balanced_v3_system)
        self.assertIn("<constraints>", balanced_v3_user)
        self.assertIn("Do not output markdown code blocks", balanced_v3_user)
        self.assertIn("straightforward way", naive_system)
        self.assertIn("should count", naive_user)


if __name__ == "__main__":
    unittest.main()
