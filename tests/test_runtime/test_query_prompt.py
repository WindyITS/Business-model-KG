import unittest

from runtime.query_prompt import HOSTED_QUERY_SYSTEM_PROMPT, LOCAL_QUERY_SYSTEM_PROMPT


class QueryPromptTests(unittest.TestCase):
    def test_local_prompt_describes_family_plan_contract(self):
        prompt = LOCAL_QUERY_SYSTEM_PROMPT.casefold()
        self.assertIn("do not write cypher", prompt)
        self.assertIn("do not return refusals", prompt)
        self.assertIn('"family"', prompt)
        self.assertIn("companies_by_cross_segment_filters", prompt)
        self.assertIn("ranking_topk", prompt)
        self.assertIn("same_segment", prompt)
        self.assertIn("customer_type_by_company_count", prompt)
        self.assertIn("answerable must always be true", prompt)
        self.assertIn("router, not this planner, owns refusals", prompt)
        self.assertNotIn("valid refusal reasons", prompt)

    def test_hosted_prompt_describes_full_cypher_contract(self):
        prompt = HOSTED_QUERY_SYSTEM_PROMPT.casefold()
        self.assertIn('"cypher"', prompt)
        self.assertIn('"params"', prompt)
        self.assertIn("write the full cypher yourself", prompt)
        self.assertIn("read-only", prompt)
        self.assertIn("company-scoped", prompt)
        self.assertIn("within_places", prompt)
        self.assertIn("government agencies", prompt)
        self.assertIn("never use create", prompt)


if __name__ == "__main__":
    unittest.main()
