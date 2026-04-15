import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from place_hierarchy import (
    normalize_place_name,
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


if __name__ == "__main__":
    unittest.main()
