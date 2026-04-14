import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from place_hierarchy import (
    BROADER_PLACE_MATCH,
    EXACT_PLACE_MATCH,
    PLACE_PARENTS,
    classify_place_match,
    expand_place_query,
    normalize_place_name,
    place_ancestors,
    place_hierarchy_edges,
)


class PlaceHierarchyTests(unittest.TestCase):
    def test_normalize_place_name_applies_documented_aliases(self):
        self.assertEqual(normalize_place_name("U.S."), "United States")
        self.assertEqual(normalize_place_name("asia-pacific"), "Asia Pacific")

    def test_place_ancestors_include_broader_regions_for_italy(self):
        ancestors = place_ancestors("Italy")

        self.assertIn("Europe", ancestors)
        self.assertIn("European Union", ancestors)
        self.assertIn("EMEA", ancestors)

    def test_expand_place_query_keeps_requested_place_first(self):
        expanded = expand_place_query("Italy")

        self.assertEqual(expanded[0], "Italy")
        self.assertIn("Europe", expanded[1:])

    def test_classify_place_match_distinguishes_exact_and_broader_matches(self):
        self.assertEqual(classify_place_match("Italy", "Italy"), EXACT_PLACE_MATCH)
        self.assertEqual(classify_place_match("Italy", "Europe"), BROADER_PLACE_MATCH)
        self.assertIsNone(classify_place_match("Italy", "Japan"))

    def test_place_hierarchy_edges_include_parent_edges_for_query_expansion(self):
        edges = set(place_hierarchy_edges(["Italy", "United States"]))

        self.assertIn(("Italy", "Europe"), edges)
        self.assertIn(("Italy", "European Union"), edges)
        self.assertIn(("United States", "North America"), edges)
        self.assertIn(("Europe", "EMEA"), edges)

    def test_aliases_cover_common_country_variants(self):
        self.assertEqual(normalize_place_name("Czechia"), "Czech Republic")
        self.assertEqual(normalize_place_name("Ivory Coast"), "Cote d'Ivoire")
        self.assertEqual(normalize_place_name("Washington, DC"), "District of Columbia")

    def test_us_state_rolls_up_to_country_and_continents(self):
        ancestors = place_ancestors("California")

        self.assertIn("United States", ancestors)
        self.assertIn("North America", ancestors)
        self.assertIn("Americas", ancestors)

    def test_representative_regions_are_covered_across_world_areas(self):
        self.assertIn("Africa", place_ancestors("Kenya"))
        self.assertIn("Middle East", place_ancestors("Jordan"))
        self.assertIn("Southeast Asia", place_ancestors("Thailand"))
        self.assertIn("Asia Pacific", place_ancestors("Australia"))
        self.assertIn("Latin America", place_ancestors("Mexico"))
        self.assertIn("Eastern Europe", place_ancestors("Poland"))

    def test_taxonomy_is_broad_not_just_seed_examples(self):
        self.assertGreaterEqual(len(PLACE_PARENTS), 250)


if __name__ == "__main__":
    unittest.main()
