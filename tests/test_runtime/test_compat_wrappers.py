import importlib
import importlib.util
import sys
import unittest

from runtime import query as query_module
from runtime import query_cypher as runtime_query_cypher


REMOVED_LEGACY_MODULES = (
    "main",
    "llm_extractor",
    "entity_resolver",
    "model_provider",
    "neo4j_loader",
    "ontology_config",
    "ontology_validator",
    "place_hierarchy",
    "query_cypher",
)


class ModuleSurfaceTests(unittest.TestCase):
    def test_runtime_query_cypher_exposes_canonical_entrypoint(self):
        self.assertIs(runtime_query_cypher.main, query_module.main_query_cypher)

    def test_legacy_top_level_modules_are_not_importable(self):
        for legacy_name in REMOVED_LEGACY_MODULES:
            with self.subTest(legacy_name=legacy_name):
                sys.modules.pop(legacy_name, None)
                self.assertIsNone(importlib.util.find_spec(legacy_name))
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(legacy_name)


if __name__ == "__main__":
    unittest.main()
