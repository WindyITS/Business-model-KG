import unittest

from text2cypher.prompting import TEXT2CYPHER_SYSTEM_PROMPT


class Text2CypherPromptingTests(unittest.TestCase):
    def test_system_prompt_requires_closed_label_normalization(self):
        prompt = TEXT2CYPHER_SYSTEM_PROMPT.casefold()
        self.assertIn("always normalize the user's wording to the exact canonical label", prompt)
        self.assertIn("government agencies", prompt)
        self.assertIn("healthcare organizations", prompt)
        self.assertIn("return an unsupported response instead of inventing a new label", prompt)
        self.assertIn("preserve boolean semantics", prompt)
        self.assertIn("customer_type_1", prompt)
        self.assertIn("different segments or offerings of the same company", prompt)


if __name__ == "__main__":
    unittest.main()
