import unittest

from runtime.query_prompt import QUERY_REPAIR_SYSTEM_PROMPT, QUERY_SYSTEM_PROMPT


class QueryPromptTests(unittest.TestCase):
    def test_system_prompt_requires_closed_label_normalization(self):
        prompt = QUERY_SYSTEM_PROMPT.casefold()
        self.assertIn("always normalize the user's wording to the exact canonical label", prompt)
        self.assertIn("government agencies", prompt)
        self.assertIn("healthcare organizations", prompt)
        self.assertIn("return an unsupported response instead of inventing a new label", prompt)
        self.assertIn("preserve boolean semantics", prompt)
        self.assertIn("customer_type_1", prompt)
        self.assertIn("different segments or offerings of the same company", prompt)
        self.assertIn("database architecture", prompt)
        self.assertIn("geography-plus-channel segment query", prompt)
        self.assertIn("includes_places", prompt)
        self.assertIn("within_places", prompt)

    def test_repair_prompt_is_short_and_schema_grounded(self):
        prompt = QUERY_REPAIR_SYSTEM_PROMPT.casefold()
        self.assertIn("make the smallest possible fix", prompt)
        self.assertIn("compact json only", prompt)
        self.assertIn("place-rollup pattern", prompt)
        self.assertIn("customer_type_1", prompt)


if __name__ == "__main__":
    unittest.main()
