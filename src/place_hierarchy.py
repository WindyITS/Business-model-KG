import re
import unicodedata
from collections import deque
from functools import lru_cache
from typing import Iterable


WHITESPACE_RE = re.compile(r"\s+")
QUOTE_CHARS = "\"'`“”‘’ "
PLACE_WITHIN_PROPERTY = "within_places"
PLACE_INCLUDES_PROPERTY = "includes_places"
EXACT_PLACE_MATCH = "exact"
NARROWER_PLACE_MATCH = "narrower_place"
BROADER_PLACE_MATCH = "broader_region"

# Place normalization only. This covers the ontology's macro-regions, U.S. states plus D.C.,
# and a broad sovereign-country set so aliases resolve to canonical place labels.
_MACRO_REGION_PARENTS: dict[str, tuple[str, ...]] = {
    "Africa": ("EMEA",),
    "Asia Pacific": ("APAC",),
    "Caribbean": ("Americas",),
    "Central America": ("Latin America", "Americas"),
    "Eastern Europe": ("Europe", "EMEA"),
    "Europe": ("EMEA",),
    "European Union": ("Europe", "EMEA"),
    "Latin America": ("Americas",),
    "Middle East": ("EMEA",),
    "North America": ("Americas",),
    "South America": ("Americas",),
    "Southeast Asia": ("Asia", "Asia Pacific", "APAC"),
    "Western Europe": ("Europe", "EMEA"),
}

AFRICA_COUNTRIES = (
    "Algeria",
    "Angola",
    "Benin",
    "Botswana",
    "Burkina Faso",
    "Burundi",
    "Cameroon",
    "Cape Verde",
    "Central African Republic",
    "Chad",
    "Comoros",
    "Cote d'Ivoire",
    "Democratic Republic of the Congo",
    "Djibouti",
    "Egypt",
    "Equatorial Guinea",
    "Eritrea",
    "Eswatini",
    "Ethiopia",
    "Gabon",
    "Gambia",
    "Ghana",
    "Guinea",
    "Guinea-Bissau",
    "Kenya",
    "Lesotho",
    "Liberia",
    "Libya",
    "Madagascar",
    "Malawi",
    "Mali",
    "Mauritania",
    "Mauritius",
    "Morocco",
    "Mozambique",
    "Namibia",
    "Niger",
    "Nigeria",
    "Republic of the Congo",
    "Rwanda",
    "Sao Tome and Principe",
    "Senegal",
    "Seychelles",
    "Sierra Leone",
    "Somalia",
    "South Africa",
    "South Sudan",
    "Sudan",
    "Tanzania",
    "Togo",
    "Tunisia",
    "Uganda",
    "Zambia",
    "Zimbabwe",
)

MIDDLE_EAST_COUNTRIES = (
    "Bahrain",
    "Iran",
    "Iraq",
    "Israel",
    "Jordan",
    "Kuwait",
    "Lebanon",
    "Oman",
    "Palestine",
    "Qatar",
    "Saudi Arabia",
    "Syria",
    "United Arab Emirates",
    "Yemen",
)

SOUTHEAST_ASIA_COUNTRIES = (
    "Brunei",
    "Cambodia",
    "Indonesia",
    "Laos",
    "Malaysia",
    "Myanmar",
    "Philippines",
    "Singapore",
    "Thailand",
    "Timor-Leste",
    "Vietnam",
)

SOUTH_ASIA_COUNTRIES = (
    "Afghanistan",
    "Bangladesh",
    "Bhutan",
    "India",
    "Maldives",
    "Nepal",
    "Pakistan",
    "Sri Lanka",
)

EAST_ASIA_COUNTRIES = (
    "China",
    "Japan",
    "Mongolia",
    "North Korea",
    "South Korea",
)

CENTRAL_ASIA_COUNTRIES = (
    "Kazakhstan",
    "Kyrgyzstan",
    "Tajikistan",
    "Turkmenistan",
    "Uzbekistan",
)

TRANSCONTINENTAL_ASIA_COUNTRIES = (
    "Armenia",
    "Azerbaijan",
    "Georgia",
    "Russia",
)

OCEANIA_COUNTRIES = (
    "Australia",
    "Fiji",
    "Kiribati",
    "Marshall Islands",
    "Micronesia",
    "Nauru",
    "New Zealand",
    "Palau",
    "Papua New Guinea",
    "Samoa",
    "Solomon Islands",
    "Tonga",
    "Tuvalu",
    "Vanuatu",
)

NORTH_AMERICA_COUNTRIES = (
    "Canada",
    "Mexico",
    "United States",
)

CENTRAL_AMERICA_COUNTRIES = (
    "Belize",
    "Costa Rica",
    "El Salvador",
    "Guatemala",
    "Honduras",
    "Nicaragua",
    "Panama",
)

CARIBBEAN_COUNTRIES = (
    "Antigua and Barbuda",
    "Bahamas",
    "Barbados",
    "Cuba",
    "Dominica",
    "Dominican Republic",
    "Grenada",
    "Haiti",
    "Jamaica",
    "Saint Kitts and Nevis",
    "Saint Lucia",
    "Saint Vincent and the Grenadines",
    "Trinidad and Tobago",
)

SOUTH_AMERICA_COUNTRIES = (
    "Argentina",
    "Bolivia",
    "Brazil",
    "Chile",
    "Colombia",
    "Ecuador",
    "Guyana",
    "Paraguay",
    "Peru",
    "Suriname",
    "Uruguay",
    "Venezuela",
)

LATIN_AMERICA_COUNTRIES = (
    "Belize",
    "Bolivia",
    "Brazil",
    "Chile",
    "Colombia",
    "Costa Rica",
    "Cuba",
    "Dominican Republic",
    "Ecuador",
    "El Salvador",
    "Guatemala",
    "Haiti",
    "Honduras",
    "Mexico",
    "Nicaragua",
    "Panama",
    "Paraguay",
    "Peru",
    "Uruguay",
    "Venezuela",
    "Argentina",
)

WESTERN_EUROPE_COUNTRIES = (
    "Andorra",
    "Austria",
    "Belgium",
    "Cyprus",
    "Denmark",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Iceland",
    "Ireland",
    "Italy",
    "Liechtenstein",
    "Luxembourg",
    "Malta",
    "Monaco",
    "Netherlands",
    "Norway",
    "Portugal",
    "San Marino",
    "Spain",
    "Sweden",
    "Switzerland",
    "United Kingdom",
    "Vatican City",
)

EASTERN_EUROPE_COUNTRIES = (
    "Albania",
    "Armenia",
    "Azerbaijan",
    "Belarus",
    "Bosnia and Herzegovina",
    "Bulgaria",
    "Croatia",
    "Czech Republic",
    "Estonia",
    "Georgia",
    "Hungary",
    "Latvia",
    "Lithuania",
    "Moldova",
    "Montenegro",
    "North Macedonia",
    "Poland",
    "Romania",
    "Russia",
    "Serbia",
    "Slovakia",
    "Slovenia",
    "Turkey",
    "Ukraine",
)

EU_COUNTRIES = (
    "Austria",
    "Belgium",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Ireland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "Poland",
    "Portugal",
    "Romania",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
)

US_STATE_NAMES = (
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "District of Columbia",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
)


def _merge_parent(existing: tuple[str, ...] | None, parents: tuple[str, ...]) -> tuple[str, ...]:
    ordered: list[str] = list(existing or ())
    for parent in parents:
        if parent not in ordered:
            ordered.append(parent)
    return tuple(ordered)


def _build_raw_place_parents() -> dict[str, tuple[str, ...]]:
    raw_place_parents = dict(_MACRO_REGION_PARENTS)

    def assign_many(places: Iterable[str], *parents: str) -> None:
        for place in places:
            raw_place_parents[place] = _merge_parent(raw_place_parents.get(place), parents)

    assign_many(AFRICA_COUNTRIES, "Africa", "EMEA")
    assign_many(MIDDLE_EAST_COUNTRIES, "Middle East", "EMEA")
    assign_many(SOUTHEAST_ASIA_COUNTRIES, "Southeast Asia", "Asia", "Asia Pacific", "APAC")
    assign_many(SOUTH_ASIA_COUNTRIES, "Asia", "Asia Pacific", "APAC")
    assign_many(EAST_ASIA_COUNTRIES, "Asia", "Asia Pacific", "APAC")
    assign_many(CENTRAL_ASIA_COUNTRIES, "Asia", "Asia Pacific", "APAC")
    assign_many(TRANSCONTINENTAL_ASIA_COUNTRIES, "Asia")
    assign_many(OCEANIA_COUNTRIES, "Asia Pacific", "APAC")
    assign_many(NORTH_AMERICA_COUNTRIES, "North America", "Americas")
    assign_many(CENTRAL_AMERICA_COUNTRIES, "Central America", "Latin America", "Americas")
    assign_many(CARIBBEAN_COUNTRIES, "Caribbean", "Americas")
    assign_many(SOUTH_AMERICA_COUNTRIES, "South America", "Americas")
    assign_many(LATIN_AMERICA_COUNTRIES, "Latin America")
    assign_many(WESTERN_EUROPE_COUNTRIES, "Europe", "Western Europe", "EMEA")
    assign_many(EASTERN_EUROPE_COUNTRIES, "Europe", "Eastern Europe", "EMEA")
    assign_many(EU_COUNTRIES, "European Union")
    assign_many(US_STATE_NAMES, "United States")

    # Common transcontinental or business-region overlaps.
    assign_many(("Cyprus", "Egypt", "Turkey"), "Middle East")
    assign_many(("Armenia", "Azerbaijan", "Georgia", "Kazakhstan", "Russia", "Turkey"), "Asia")

    return raw_place_parents


_RAW_PLACE_PARENTS = _build_raw_place_parents()

_RAW_PLACE_ALIASES = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Brunei Darussalam": "Brunei",
    "Burma": "Myanmar",
    "Cabo Verde": "Cape Verde",
    "Congo Republic": "Republic of the Congo",
    "Congo-Brazzaville": "Republic of the Congo",
    "Congo-Kinshasa": "Democratic Republic of the Congo",
    "Czechia": "Czech Republic",
    "D.C.": "District of Columbia",
    "DC": "District of Columbia",
    "Democratic Republic of Congo": "Democratic Republic of the Congo",
    "DR Congo": "Democratic Republic of the Congo",
    "DRC": "Democratic Republic of the Congo",
    "East Timor": "Timor-Leste",
    "Great Britain": "United Kingdom",
    "Holy See": "Vatican City",
    "Iran, Islamic Republic of": "Iran",
    "Ivory Coast": "Cote d'Ivoire",
    "Korea, Democratic People's Republic of": "North Korea",
    "Korea, Republic of": "South Korea",
    "Lao PDR": "Laos",
    "Lao People's Democratic Republic": "Laos",
    "Macedonia": "North Macedonia",
    "Micronesia, Federated States of": "Micronesia",
    "Moldova, Republic of": "Moldova",
    "Palestine, State of": "Palestine",
    "Palestinian Territories": "Palestine",
    "Republic of Ireland": "Ireland",
    "Republic of Korea": "South Korea",
    "Republic of Moldova": "Moldova",
    "Russian Federation": "Russia",
    "South Korea": "South Korea",
    "State of Palestine": "Palestine",
    "Swaziland": "Eswatini",
    "Syrian Arab Republic": "Syria",
    "The Bahamas": "Bahamas",
    "The Gambia": "Gambia",
    "The Netherlands": "Netherlands",
    "Timor Leste": "Timor-Leste",
    "Turkiye": "Turkey",
    "U.K.": "United Kingdom",
    "U.S.": "United States",
    "U.S.A.": "United States",
    "UAE": "United Arab Emirates",
    "UK": "United Kingdom",
    "US": "United States",
    "USA": "United States",
    "United Republic of Tanzania": "Tanzania",
    "United States of America": "United States",
    "Vatican": "Vatican City",
    "Venezuela, Bolivarian Republic of": "Venezuela",
    "Viet Nam": "Vietnam",
    "Washington, D.C.": "District of Columbia",
    "Washington, DC": "District of Columbia",
    "asia-pacific": "Asia Pacific",
    "emea": "EMEA",
}


def _clean_place_name(name: str) -> str:
    cleaned = unicodedata.normalize("NFKC", name).strip()
    cleaned = cleaned.strip(QUOTE_CHARS)
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    return cleaned


def _canonical_place_key(name: str) -> str:
    cleaned = _clean_place_name(name)
    cleaned = cleaned.casefold()
    cleaned = cleaned.replace("’", "'")
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    return cleaned


_PLACE_ALIASES = {
    _canonical_place_key(alias): canonical
    for alias, canonical in _RAW_PLACE_ALIASES.items()
}

_PLACE_NAMES = {
    _clean_place_name(place)
    for place in set(_RAW_PLACE_PARENTS)
    | {parent for parents in _RAW_PLACE_PARENTS.values() for parent in parents}
    | set(_RAW_PLACE_ALIASES.values())
}

_CANONICAL_PLACE_NAMES = {_canonical_place_key(place): place for place in _PLACE_NAMES}


def normalize_place_name(name: str) -> str:
    cleaned = _clean_place_name(name)
    key = _canonical_place_key(cleaned)
    if key in _PLACE_ALIASES:
        return _PLACE_ALIASES[key]
    return _CANONICAL_PLACE_NAMES.get(key, cleaned)


PLACE_PARENTS = {
    normalize_place_name(place): tuple(normalize_place_name(parent) for parent in parents)
    for place, parents in _RAW_PLACE_PARENTS.items()
}

PLACE_CHILDREN: dict[str, tuple[str, ...]] = {}
_children_builder: dict[str, list[str]] = {}
for place, parents in PLACE_PARENTS.items():
    for parent in parents:
        _children_builder.setdefault(parent, []).append(place)
PLACE_CHILDREN = {
    parent: tuple(sorted(children))
    for parent, children in _children_builder.items()
}


def place_parents(place: str) -> tuple[str, ...]:
    return PLACE_PARENTS.get(normalize_place_name(place), ())


def place_children(place: str) -> tuple[str, ...]:
    return PLACE_CHILDREN.get(normalize_place_name(place), ())


@lru_cache(maxsize=None)
def place_ancestors(place: str) -> tuple[str, ...]:
    normalized_place = normalize_place_name(place)
    queue = deque(place_parents(normalized_place))
    ancestors: list[str] = []
    seen: set[str] = set()

    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        ancestors.append(current)
        queue.extend(place_parents(current))

    return tuple(ancestors)


@lru_cache(maxsize=None)
def place_descendants(place: str) -> tuple[str, ...]:
    normalized_place = normalize_place_name(place)
    queue = deque(place_children(normalized_place))
    descendants: list[str] = []
    seen: set[str] = set()

    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        descendants.append(current)
        queue.extend(place_children(current))

    return tuple(descendants)


def place_query_properties(place: str) -> dict[str, object]:
    normalized_place = normalize_place_name(place)
    return {
        "name": normalized_place,
        PLACE_WITHIN_PROPERTY: list(place_ancestors(normalized_place)),
        PLACE_INCLUDES_PROPERTY: list(place_descendants(normalized_place)),
    }


def place_query_property_rows(place_names: Iterable[str]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    for place_name in place_names:
        normalized_place = normalize_place_name(place_name)
        if not normalized_place or normalized_place in seen:
            continue
        seen.add(normalized_place)
        rows.append(place_query_properties(normalized_place))

    rows.sort(key=lambda row: str(row["name"]))
    return tuple(rows)


def classify_place_match(requested_place: str, company_place: str) -> str | None:
    requested = normalize_place_name(requested_place)
    company = normalize_place_name(company_place)
    if requested == company:
        return EXACT_PLACE_MATCH
    if requested in place_descendants(company):
        return NARROWER_PLACE_MATCH
    if requested in place_ancestors(company):
        return BROADER_PLACE_MATCH
    return None


COMPANY_PLACE_PROPERTY_MATCH_CYPHER = f"""
MATCH (company:Company)-[:OPERATES_IN]->(place:Place)
WITH company, place,
     CASE
       WHEN place.name = $place THEN 0
       WHEN $place IN coalesce(place.{PLACE_INCLUDES_PROPERTY}, []) THEN 1
       WHEN $place IN coalesce(place.{PLACE_WITHIN_PROPERTY}, []) THEN 2
       ELSE NULL
     END AS match_rank
WHERE match_rank IS NOT NULL
RETURN company.name AS company,
       CASE match_rank
         WHEN 0 THEN '{EXACT_PLACE_MATCH}'
         WHEN 1 THEN '{NARROWER_PLACE_MATCH}'
         ELSE '{BROADER_PLACE_MATCH}'
       END AS geography_match
ORDER BY match_rank, company
""".strip()
