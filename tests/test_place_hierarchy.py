import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from place_hierarchy import (
    BROADER_PLACE_MATCH,
    COMPANY_PLACE_PROPERTY_MATCH_CYPHER,
    EXACT_PLACE_MATCH,
    NARROWER_PLACE_MATCH,
    PLACE_INCLUDES_PROPERTY,
    PLACE_WITHIN_PROPERTY,
    classify_place_match,
    normalize_place_name,
    place_ancestors,
    place_descendants,
    place_query_properties,
    place_query_property_rows,
)


class PlaceHierarchyTests(unittest.TestCase):
    def test_normalize_place_name_applies_documented_aliases(self):
        self.assertEqual(normalize_place_name("U.S."), "United States")
        self.assertEqual(normalize_place_name("asia-pacific"), "Asia Pacific")

    def test_aliases_cover_common_country_variants(self):
        self.assertEqual(normalize_place_name("Czechia"), "Czech Republic")
        self.assertEqual(normalize_place_name("Ivory Coast"), "Cote d'Ivoire")
        self.assertEqual(normalize_place_name("Washington, DC"), "District of Columbia")

    def test_macro_regions_and_countries_normalize_to_canonical_forms(self):
        self.assertEqual(normalize_place_name("Italy"), "Italy")
        self.assertEqual(normalize_place_name("EMEA"), "EMEA")
        self.assertEqual(normalize_place_name("European Union"), "European Union")

    def test_place_ancestors_capture_broader_areas_for_italy(self):
        ancestors = place_ancestors("Italy")

        self.assertIn("Europe", ancestors)
        self.assertIn("European Union", ancestors)
        self.assertIn("EMEA", ancestors)

    def test_place_descendants_capture_narrower_members_for_europe(self):
        descendants = place_descendants("Europe")

        self.assertIn("Italy", descendants)
        self.assertIn("Germany", descendants)
        self.assertIn("Western Europe", descendants)

    def test_place_query_properties_use_within_and_includes_arrays(self):
        italy = place_query_properties("Italy")
        europe = place_query_properties("Europe")

        self.assertEqual(italy["name"], "Italy")
        self.assertIn("Europe", italy[PLACE_WITHIN_PROPERTY])
        self.assertEqual(italy[PLACE_INCLUDES_PROPERTY], [])

        self.assertEqual(europe["name"], "Europe")
        self.assertIn("Italy", europe[PLACE_INCLUDES_PROPERTY])
        self.assertIn("EMEA", europe[PLACE_WITHIN_PROPERTY])

    def test_place_query_property_rows_normalize_and_deduplicate_inputs(self):
        rows = place_query_property_rows(["U.S.", "United States", "Italy"])

        names = [row["name"] for row in rows]
        self.assertEqual(names, ["Italy", "United States"])
        united_states = next(row for row in rows if row["name"] == "United States")
        self.assertIn("California", united_states[PLACE_INCLUDES_PROPERTY])
        self.assertIn("North America", united_states[PLACE_WITHIN_PROPERTY])

    def test_classify_place_match_distinguishes_exact_narrower_and_broader(self):
        self.assertEqual(classify_place_match("Italy", "Italy"), EXACT_PLACE_MATCH)
        self.assertEqual(classify_place_match("Europe", "Italy"), BROADER_PLACE_MATCH)
        self.assertEqual(classify_place_match("Italy", "Europe"), NARROWER_PLACE_MATCH)
        self.assertIsNone(classify_place_match("Japan", "Italy"))

    def test_company_place_property_match_cypher_uses_place_properties(self):
        self.assertIn(f"place.{PLACE_INCLUDES_PROPERTY}", COMPANY_PLACE_PROPERTY_MATCH_CYPHER)
        self.assertIn(f"place.{PLACE_WITHIN_PROPERTY}", COMPANY_PLACE_PROPERTY_MATCH_CYPHER)
        self.assertIn("'narrower_place'", COMPANY_PLACE_PROPERTY_MATCH_CYPHER)
        self.assertIn("'broader_region'", COMPANY_PLACE_PROPERTY_MATCH_CYPHER)


if __name__ == "__main__":
    unittest.main()
