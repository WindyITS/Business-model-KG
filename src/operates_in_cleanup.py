import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ontology_validator import canonical_entity_key, clean_entity_name, validate_triples


CANONICAL_REGIONS = (
    "Africa",
    "APAC",
    "Americas",
    "Asia",
    "Asia Pacific",
    "Caribbean",
    "Central America",
    "EMEA",
    "Eastern Europe",
    "Europe",
    "European Union",
    "Latin America",
    "Middle East",
    "North America",
    "South America",
    "Southeast Asia",
    "Western Europe",
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
    "District of Columbia",
)

COUNTRY_NAMES = (
    "Afghanistan",
    "Albania",
    "Algeria",
    "Andorra",
    "Angola",
    "Antigua and Barbuda",
    "Argentina",
    "Armenia",
    "Australia",
    "Austria",
    "Azerbaijan",
    "Bahamas",
    "Bahrain",
    "Bangladesh",
    "Barbados",
    "Belarus",
    "Belgium",
    "Bermuda",
    "Belize",
    "Benin",
    "Bhutan",
    "Bolivia",
    "Bosnia and Herzegovina",
    "Botswana",
    "Brazil",
    "Brunei",
    "Bulgaria",
    "Burkina Faso",
    "Burundi",
    "Cambodia",
    "Cameroon",
    "Canada",
    "Cape Verde",
    "Central African Republic",
    "Chad",
    "Chile",
    "China",
    "Colombia",
    "Comoros",
    "Costa Rica",
    "Cote d'Ivoire",
    "Croatia",
    "Cuba",
    "Cyprus",
    "Czech Republic",
    "Democratic Republic of the Congo",
    "Denmark",
    "Djibouti",
    "Dominica",
    "Dominican Republic",
    "Ecuador",
    "Egypt",
    "El Salvador",
    "Equatorial Guinea",
    "Eritrea",
    "Estonia",
    "Eswatini",
    "Ethiopia",
    "Fiji",
    "Finland",
    "France",
    "Gabon",
    "Gambia",
    "Georgia",
    "Germany",
    "Ghana",
    "Greece",
    "Grenada",
    "Guatemala",
    "Guinea",
    "Guinea-Bissau",
    "Guyana",
    "Haiti",
    "Honduras",
    "Hong Kong",
    "Hungary",
    "Iceland",
    "India",
    "Indonesia",
    "Iran",
    "Iraq",
    "Ireland",
    "Israel",
    "Italy",
    "Jamaica",
    "Japan",
    "Jordan",
    "Kazakhstan",
    "Kenya",
    "Kiribati",
    "Kuwait",
    "Kyrgyzstan",
    "Laos",
    "Latvia",
    "Lebanon",
    "Lesotho",
    "Liberia",
    "Libya",
    "Liechtenstein",
    "Lithuania",
    "Luxembourg",
    "Macau",
    "Madagascar",
    "Malawi",
    "Malaysia",
    "Maldives",
    "Mali",
    "Malta",
    "Marshall Islands",
    "Mauritania",
    "Mauritius",
    "Mexico",
    "Micronesia",
    "Moldova",
    "Monaco",
    "Mongolia",
    "Montenegro",
    "Morocco",
    "Mozambique",
    "Myanmar",
    "Namibia",
    "Nauru",
    "Nepal",
    "Netherlands",
    "New Zealand",
    "Nicaragua",
    "Niger",
    "Nigeria",
    "North Korea",
    "North Macedonia",
    "Norway",
    "Oman",
    "Pakistan",
    "Palau",
    "Panama",
    "Papua New Guinea",
    "Paraguay",
    "Peru",
    "Philippines",
    "Poland",
    "Portugal",
    "Puerto Rico",
    "Qatar",
    "Republic of the Congo",
    "Romania",
    "Russia",
    "Rwanda",
    "Saint Kitts and Nevis",
    "Saint Lucia",
    "Saint Vincent and the Grenadines",
    "Samoa",
    "San Marino",
    "Sao Tome and Principe",
    "Saudi Arabia",
    "Senegal",
    "Serbia",
    "Seychelles",
    "Sierra Leone",
    "Singapore",
    "Slovakia",
    "Slovenia",
    "Solomon Islands",
    "Somalia",
    "South Africa",
    "South Korea",
    "South Sudan",
    "Spain",
    "Sri Lanka",
    "Sudan",
    "Suriname",
    "Sweden",
    "Switzerland",
    "Syria",
    "Taiwan",
    "Tajikistan",
    "Tanzania",
    "Thailand",
    "Timor-Leste",
    "Togo",
    "Tonga",
    "Trinidad and Tobago",
    "Tunisia",
    "Turkey",
    "Turkmenistan",
    "Tuvalu",
    "Uganda",
    "Ukraine",
    "United Arab Emirates",
    "United Kingdom",
    "United States",
    "Uruguay",
    "Uzbekistan",
    "Vanuatu",
    "Vatican City",
    "Venezuela",
    "Vietnam",
    "Yemen",
    "Zambia",
    "Zimbabwe",
)

CANONICAL_PLACE_LABELS = tuple(sorted(set(CANONICAL_REGIONS + US_STATE_NAMES + COUNTRY_NAMES)))

PLACE_ALIAS_TO_CANONICAL = {
    canonical_entity_key(label): label for label in CANONICAL_PLACE_LABELS
}
PLACE_ALIAS_TO_CANONICAL.update(
    {
        canonical_entity_key("u.s."): "United States",
        canonical_entity_key("u.s"): "United States",
        canonical_entity_key("u.s ."): "United States",
        canonical_entity_key("us"): "United States",
        canonical_entity_key("usa"): "United States",
        canonical_entity_key("united state"): "United States",
        canonical_entity_key("u.k."): "United Kingdom",
        canonical_entity_key("u.k"): "United Kingdom",
        canonical_entity_key("u.k ."): "United Kingdom",
        canonical_entity_key("uk"): "United Kingdom",
        canonical_entity_key("uae"): "United Arab Emirates",
        canonical_entity_key("u.a.e."): "United Arab Emirates",
        canonical_entity_key("united arab emirate"): "United Arab Emirates",
        canonical_entity_key("asia-pacific"): "Asia Pacific",
        canonical_entity_key("asia/pacific"): "Asia Pacific",
        canonical_entity_key("asia pacific region"): "Asia Pacific",
        canonical_entity_key("asia-pacific region"): "Asia Pacific",
        canonical_entity_key("apac"): "APAC",
        canonical_entity_key("americas"): "Americas",
        canonical_entity_key("americas region"): "Americas",
        canonical_entity_key("emea"): "EMEA",
        canonical_entity_key("eame"): "EMEA",
        canonical_entity_key("korea"): "South Korea",
        canonical_entity_key("south korea"): "South Korea",
        canonical_entity_key("philippine"): "Philippines",
        canonical_entity_key("ukraine"): "Ukraine",
        canonical_entity_key("e.u."): "European Union",
        canonical_entity_key("eu"): "European Union",
        canonical_entity_key("district of columbia"): "District of Columbia",
        canonical_entity_key("washington d.c."): "District of Columbia",
        canonical_entity_key("washington dc"): "District of Columbia",
        canonical_entity_key("d.c."): "District of Columbia",
        canonical_entity_key("dc"): "District of Columbia",
    }
)


def normalize_operates_in_object(value: str) -> str | None:
    cleaned = clean_entity_name(value)
    if not cleaned:
        return None
    return PLACE_ALIAS_TO_CANONICAL.get(canonical_entity_key(cleaned))


def filter_operates_in_example(example: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    triples = list((example.get("output") or {}).get("triples") or [])
    originally_empty = not triples

    kept_triples: list[dict[str, Any]] = []
    removed_operates_in_labels: list[str] = []
    normalized_operates_in_labels: list[tuple[str, str]] = []
    operates_in_before = 0
    operates_in_after = 0

    for triple in triples:
        if triple.get("relation") != "OPERATES_IN":
            kept_triples.append(copy.deepcopy(triple))
            continue

        operates_in_before += 1
        original_object = str(triple.get("object", ""))
        normalized_object = normalize_operates_in_object(original_object)
        if normalized_object is None:
            removed_operates_in_labels.append(original_object)
            continue

        normalized_triple = copy.deepcopy(triple)
        normalized_triple["object"] = normalized_object
        kept_triples.append(normalized_triple)
        operates_in_after += 1
        if clean_entity_name(original_object) != normalized_object:
            normalized_operates_in_labels.append((original_object, normalized_object))

    validation = validate_triples(kept_triples, dedupe=True)
    filtered_triples = validation["valid_triples"]

    if not originally_empty and not filtered_triples:
        return None, {
            "originally_empty": False,
            "dropped_example": True,
            "operates_in_before": operates_in_before,
            "operates_in_after": 0,
            "removed_operates_in_labels": removed_operates_in_labels,
            "normalized_operates_in_labels": normalized_operates_in_labels,
            "deduped_or_invalid_removed_count": len(kept_triples) - len(filtered_triples),
            "remaining_triple_count": 0,
        }

    filtered_example = copy.deepcopy(example)
    filtered_example.setdefault("output", {})["triples"] = filtered_triples
    filtered_example.setdefault("metadata", {})["empty_target"] = len(filtered_triples) == 0

    return filtered_example, {
        "originally_empty": originally_empty,
        "dropped_example": False,
        "operates_in_before": operates_in_before,
        "operates_in_after": sum(1 for triple in filtered_triples if triple.get("relation") == "OPERATES_IN"),
        "removed_operates_in_labels": removed_operates_in_labels,
        "normalized_operates_in_labels": normalized_operates_in_labels,
        "deduped_or_invalid_removed_count": len(kept_triples) - len(filtered_triples),
        "remaining_triple_count": len(filtered_triples),
    }


def filter_operates_in_examples(examples: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    filtered_examples: list[dict[str, Any]] = []
    report = build_empty_cleanup_report()

    for example in examples:
        original_triples = list((example.get("output") or {}).get("triples") or [])
        report["input_example_count"] += 1
        report["input_triple_count"] += len(original_triples)
        report["input_operates_in_triple_count"] += sum(
            1 for triple in original_triples if triple.get("relation") == "OPERATES_IN"
        )

        filtered_example, example_report = filter_operates_in_example(example)
        report["normalized_operates_in_triple_count"] += len(example_report["normalized_operates_in_labels"])
        report["dropped_operates_in_triple_count"] += len(example_report["removed_operates_in_labels"])
        report["deduped_or_invalid_removed_count"] += int(example_report["deduped_or_invalid_removed_count"])

        for label in example_report["removed_operates_in_labels"]:
            report["dropped_operates_in_labels"][clean_entity_name(label)] += 1
        for original, normalized in example_report["normalized_operates_in_labels"]:
            report["normalized_operates_in_labels"][f"{clean_entity_name(original)} -> {normalized}"] += 1

        if filtered_example is None:
            report["dropped_example_count"] += 1
            continue

        if example_report["originally_empty"]:
            report["preserved_empty_example_count"] += 1
        elif example_report["operates_in_before"] > example_report["operates_in_after"]:
            report["modified_example_count"] += 1
        elif example_report["operates_in_before"] > 0 and example_report["normalized_operates_in_labels"]:
            report["modified_example_count"] += 1

        remaining_triples = list((filtered_example.get("output") or {}).get("triples") or [])
        report["output_example_count"] += 1
        report["output_triple_count"] += len(remaining_triples)
        report["output_operates_in_triple_count"] += sum(
            1 for triple in remaining_triples if triple.get("relation") == "OPERATES_IN"
        )
        if remaining_triples:
            report["output_positive_example_count"] += 1
        else:
            report["output_empty_example_count"] += 1
        filtered_examples.append(filtered_example)

    finalize_cleanup_report(report)
    return filtered_examples, report


def filter_operates_in_jsonl(input_jsonl: Path, output_jsonl: Path, report_path: Path) -> dict[str, Any]:
    examples: list[dict[str, Any]] = []
    with input_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    filtered_examples, report = filter_operates_in_examples(examples)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for example in filtered_examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    report_payload = {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        **report,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_payload


def build_empty_cleanup_report() -> dict[str, Any]:
    return {
        "input_example_count": 0,
        "input_triple_count": 0,
        "input_operates_in_triple_count": 0,
        "output_example_count": 0,
        "output_positive_example_count": 0,
        "output_empty_example_count": 0,
        "output_triple_count": 0,
        "output_operates_in_triple_count": 0,
        "modified_example_count": 0,
        "dropped_example_count": 0,
        "preserved_empty_example_count": 0,
        "normalized_operates_in_triple_count": 0,
        "dropped_operates_in_triple_count": 0,
        "deduped_or_invalid_removed_count": 0,
        "dropped_operates_in_labels": Counter(),
        "normalized_operates_in_labels": Counter(),
    }


def finalize_cleanup_report(report: dict[str, Any]) -> None:
    report["dropped_operates_in_labels"] = dict(
        sorted(report["dropped_operates_in_labels"].items(), key=lambda item: (-item[1], item[0]))
    )
    report["normalized_operates_in_labels"] = dict(
        sorted(report["normalized_operates_in_labels"].items(), key=lambda item: (-item[1], item[0]))
    )
