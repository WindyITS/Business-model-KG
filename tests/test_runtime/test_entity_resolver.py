import unittest

from llm_extraction.models import KnowledgeGraphExtraction, Triple
from runtime.entity_resolver import clean_entity_name as clean_resolved_entity_name, resolve_entities


class EntityResolverTests(unittest.TestCase):
    def test_entity_resolver_preserves_best_surface_form(self):
        extractions = [
            KnowledgeGraphExtraction(
                extraction_notes="ok",
                triples=[
                    Triple(
                        subject="OpenAI",
                        subject_type="Company",
                        relation="PARTNERS_WITH",
                        object="NASA",
                        object_type="Company",
                    ),
                    Triple(
                        subject="openai",
                        subject_type="Company",
                        relation="PARTNERS_WITH",
                        object="NASA",
                        object_type="Company",
                    ),
                ],
            )
        ]

        resolved = resolve_entities(extractions)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].subject, "OpenAI")

    def test_entity_resolver_strips_curly_quotes(self):
        self.assertEqual(clean_resolved_entity_name('  “Apollo”  '), "Apollo")

        extractions = [
            KnowledgeGraphExtraction(
                extraction_notes="ok",
                triples=[
                    Triple(
                        subject="Palantir",
                        subject_type="Company",
                        relation="OFFERS",
                        object="“Apollo”",
                        object_type="Offering",
                    ),
                    Triple(
                        subject="Palantir",
                        subject_type="Company",
                        relation="OFFERS",
                        object="Apollo",
                        object_type="Offering",
                    ),
                ],
            )
        ]

        resolved = resolve_entities(extractions)

        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].object, "Apollo")


if __name__ == "__main__":
    unittest.main()
