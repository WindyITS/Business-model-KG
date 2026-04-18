import unittest

from runtime.query_prompt import QUERY_REPAIR_SYSTEM_PROMPT, QUERY_SYSTEM_PROMPT


class QueryPromptTests(unittest.TestCase):
    def test_system_prompt_describes_family_plan_contract(self):
        prompt = QUERY_SYSTEM_PROMPT.casefold()
        self.assertIn("do not write cypher", prompt)
        self.assertIn('"family"', prompt)
        self.assertIn("companies_by_cross_segment_filters", prompt)
        self.assertIn("ranking_topk", prompt)
        self.assertIn("same_segment", prompt)
        self.assertIn("customer_type_by_company_count", prompt)
        self.assertIn("ambiguous_closed_label", prompt)

    def test_repair_prompt_keeps_repairs_in_plan_space(self):
        prompt = QUERY_REPAIR_SYSTEM_PROMPT.casefold()
        self.assertIn("return a corrected json plan only", prompt)
        self.assertIn("do not write cypher", prompt)
        self.assertIn("beyond_local_coverage", prompt)


if __name__ == "__main__":
    unittest.main()
