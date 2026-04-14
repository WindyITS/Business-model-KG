import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from ontology_config import canonical_labels, is_valid_relation_schema, node_type_names, relation_names
from place_hierarchy import normalize_place_name


WHITESPACE_RE = re.compile(r"\s+")
CANONICAL_LABEL_NODE_TYPES = {"CustomerType", "Channel", "RevenueModel"}
QUOTE_CHARS = "\"'`“”‘’ "


def clean_entity_name(name: str) -> str:
    cleaned = unicodedata.normalize("NFKC", name).strip()
    cleaned = cleaned.strip(QUOTE_CHARS)
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    return cleaned


def canonical_entity_key(name: str) -> str:
    cleaned = clean_entity_name(name)
    cleaned = cleaned.casefold()
    cleaned = cleaned.replace("’", "'")
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    return cleaned


def _extract_triples(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        triples = payload
    elif isinstance(payload, dict):
        if "triples" in payload:
            triples = payload["triples"]
        elif "resolved_triples" in payload:
            triples = payload["resolved_triples"]
        elif "valid_triples" in payload:
            triples = payload["valid_triples"]
        else:
            triples = []
    else:
        triples = []

    extracted: list[dict[str, Any]] = []
    for triple in triples:
        if isinstance(triple, dict):
            extracted.append(
                {
                    "subject": triple.get("subject", ""),
                    "subject_type": triple.get("subject_type", ""),
                    "relation": triple.get("relation", ""),
                    "object": triple.get("object", ""),
                    "object_type": triple.get("object_type", ""),
                }
            )
    return extracted


def _grounded_in_text(value: str, source_text: str) -> bool:
    return canonical_entity_key(value) in canonical_entity_key(source_text)


def validate_triple(
    triple: dict[str, Any],
    source_text: str | None = None,
    require_text_grounding: bool = False,
    ontology_version: str = "canonical",
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    normalized = {
        "subject": clean_entity_name(str(triple.get("subject", ""))),
        "subject_type": str(triple.get("subject_type", "")).strip(),
        "relation": str(triple.get("relation", "")).strip(),
        "object": clean_entity_name(str(triple.get("object", ""))),
        "object_type": str(triple.get("object_type", "")).strip(),
    }

    for node_field, type_field in (("subject", "subject_type"), ("object", "object_type")):
        if normalized[type_field] == "Place" and normalized[node_field]:
            normalized[node_field] = normalize_place_name(normalized[node_field])

    if not normalized["subject"]:
        issues.append({"code": "empty_subject", "message": "Subject is empty after normalization."})
    if not normalized["object"]:
        issues.append({"code": "empty_object", "message": "Object is empty after normalization."})

    valid_node_types = set(node_type_names(ontology_version))
    valid_relations = set(relation_names(ontology_version))

    if normalized["subject_type"] not in valid_node_types:
        issues.append({"code": "invalid_subject_type", "message": f"Invalid subject_type: {normalized['subject_type']}"})
    if normalized["object_type"] not in valid_node_types:
        issues.append({"code": "invalid_object_type", "message": f"Invalid object_type: {normalized['object_type']}"})
    if normalized["relation"] not in valid_relations:
        issues.append({"code": "invalid_relation", "message": f"Invalid relation: {normalized['relation']}"})

    if not issues and not is_valid_relation_schema(
        normalized["subject_type"],
        normalized["relation"],
        normalized["object_type"],
        ontology_version,
    ):
        issues.append(
            {
                "code": "invalid_relation_schema",
                "message": (
                    f"Relation {normalized['relation']} does not allow "
                    f"{normalized['subject_type']} -> {normalized['object_type']}."
                ),
            }
        )

    for node_field, type_field in (("subject", "subject_type"), ("object", "object_type")):
        node_type = normalized[type_field]
        if node_type in CANONICAL_LABEL_NODE_TYPES:
            allowed = set(canonical_labels(node_type, ontology_version))
            if normalized[node_field] not in allowed:
                issues.append(
                    {
                        "code": "non_canonical_label",
                        "message": f"{node_field}={normalized[node_field]!r} is not an approved {node_type} label.",
                    }
                )

    if require_text_grounding and source_text:
        for node_field, type_field in (("subject", "subject_type"), ("object", "object_type")):
            node_type = normalized[type_field]
            if node_type in CANONICAL_LABEL_NODE_TYPES:
                continue
            if normalized[node_field] and not _grounded_in_text(normalized[node_field], source_text):
                issues.append(
                    {
                        "code": "not_grounded_in_text",
                        "message": f"{node_field}={normalized[node_field]!r} does not appear in the source text.",
                    }
                )

    return {
        "is_valid": not issues,
        "normalized_triple": normalized,
        "issues": issues,
    }


def validate_triples(
    triples: list[dict[str, Any]],
    source_text: str | None = None,
    require_text_grounding: bool = False,
    dedupe: bool = True,
    ontology_version: str = "canonical",
) -> dict[str, Any]:
    provisional_valid_records: list[dict[str, Any]] = []
    invalid_triples: list[dict[str, Any]] = []
    duplicate_triples: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    offering_parents: dict[str, tuple[int, dict[str, str]]] = {}

    for index, triple in enumerate(triples):
        result = validate_triple(
            triple,
            source_text=source_text,
            require_text_grounding=require_text_grounding,
            ontology_version=ontology_version,
        )
        normalized = result["normalized_triple"]
        if not result["is_valid"]:
            invalid_triples.append(
                {
                    "index": index,
                    "triple": triple,
                    "normalized_triple": normalized,
                    "issues": result["issues"],
                }
            )
            continue

        triple_key = (
            canonical_entity_key(normalized["subject"]),
            normalized["subject_type"],
            normalized["relation"],
            canonical_entity_key(normalized["object"]),
            normalized["object_type"],
        )
        if dedupe and triple_key in seen:
            duplicate_triples.append(
                {
                    "index": index,
                    "triple": triple,
                    "normalized_triple": normalized,
                    "issues": [{"code": "duplicate", "message": "Triple duplicates an earlier valid triple."}],
                }
            )
            continue

        if (
            normalized["relation"] == "OFFERS"
            and normalized["subject_type"] == "Offering"
            and normalized["object_type"] == "Offering"
        ):
            child_key = canonical_entity_key(normalized["object"])
            existing_parent = offering_parents.get(child_key)
            if existing_parent is not None and canonical_entity_key(existing_parent[1]["subject"]) != canonical_entity_key(normalized["subject"]):
                invalid_triples.append(
                    {
                        "index": index,
                        "triple": triple,
                        "normalized_triple": normalized,
                        "issues": [
                            {
                                "code": "multiple_offering_parents",
                                "message": (
                                    f"Offering {normalized['object']!r} already has offering parent "
                                    f"{existing_parent[1]['subject']!r}; a child offering may have at most one offering parent."
                                ),
                            }
                        ],
                    }
                )
                continue
            offering_parents[child_key] = (index, normalized)

        seen.add(triple_key)
        provisional_valid_records.append(
            {
                "index": index,
                "triple": triple,
                "normalized_triple": normalized,
            }
        )

    offering_parent_children = {
        canonical_entity_key(record["normalized_triple"]["object"])
        for record in provisional_valid_records
        if record["normalized_triple"]["relation"] == "OFFERS"
        and record["normalized_triple"]["subject_type"] == "Offering"
        and record["normalized_triple"]["object_type"] == "Offering"
    }

    direct_segment_children = {
        canonical_entity_key(record["normalized_triple"]["object"])
        for record in provisional_valid_records
        if record["normalized_triple"]["relation"] == "OFFERS"
        and record["normalized_triple"]["subject_type"] == "BusinessSegment"
        and record["normalized_triple"]["object_type"] == "Offering"
    }

    kept_records: list[dict[str, Any]] = []
    for record in provisional_valid_records:
        normalized = record["normalized_triple"]
        subject_key = canonical_entity_key(normalized["subject"])
        if normalized["relation"] == "MONETIZES_VIA" and subject_key in offering_parent_children:
            invalid_triples.append(
                {
                    "index": record["index"],
                    "triple": record["triple"],
                    "normalized_triple": normalized,
                    "issues": [
                        {
                            "code": "child_offering_monetizes_via",
                            "message": (
                                f"Offering {normalized['subject']!r} has an explicit offering parent and cannot carry "
                                "MONETIZES_VIA in the canonical ontology."
                            ),
                        }
                    ],
                }
            )
            continue
        if normalized["relation"] == "SELLS_THROUGH" and normalized["subject_type"] == "Offering" and subject_key in direct_segment_children:
            invalid_triples.append(
                {
                    "index": record["index"],
                    "triple": record["triple"],
                    "normalized_triple": normalized,
                    "issues": [
                        {
                            "code": "segment_anchored_offering_sells_through",
                            "message": (
                                f"Offering {normalized['subject']!r} has a BusinessSegment anchor and cannot carry "
                                "SELLS_THROUGH in the canonical ontology."
                            ),
                        }
                    ],
                }
            )
            continue
        kept_records.append(record)
    provisional_valid_records = kept_records

    valid_triples = [record["normalized_triple"] for record in provisional_valid_records]

    return {
        "summary": {
            "input_triple_count": len(triples),
            "valid_triple_count": len(valid_triples),
            "invalid_triple_count": len(invalid_triples),
            "duplicate_triple_count": len(duplicate_triples),
            "output_triple_count": len(valid_triples),
            "require_text_grounding": require_text_grounding,
            "ontology_version": ontology_version,
        },
        "valid_triples": valid_triples,
        "invalid_triples": invalid_triples,
        "duplicate_triples": duplicate_triples,
    }


def validate_payload(
    payload: Any,
    source_text: str | None = None,
    require_text_grounding: bool = False,
    dedupe: bool = True,
    ontology_version: str = "canonical",
) -> dict[str, Any]:
    return validate_triples(
        _extract_triples(payload),
        source_text=source_text,
        require_text_grounding=require_text_grounding,
        dedupe=dedupe,
        ontology_version=ontology_version,
    )


def validate_file(
    triples_path: Path,
    source_text_path: Path | None = None,
    require_text_grounding: bool = False,
    dedupe: bool = True,
    ontology_version: str = "canonical",
) -> dict[str, Any]:
    payload = json.loads(triples_path.read_text(encoding="utf-8"))
    source_text = source_text_path.read_text(encoding="utf-8") if source_text_path else None
    report = validate_payload(
        payload,
        source_text=source_text,
        require_text_grounding=require_text_grounding,
        dedupe=dedupe,
        ontology_version=ontology_version,
    )
    report["source"] = {
        "triples_path": str(triples_path),
        "source_text_path": str(source_text_path) if source_text_path else None,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate triples against the business-model ontology.")
    parser.add_argument("triples_path", type=Path, help="Path to a triples JSON payload.")
    parser.add_argument("--source-text-path", type=Path, default=None, help="Optional path to the source filing text.")
    parser.add_argument("--require-text-grounding", action="store_true", help="Require non-canonical entity names to appear in the source text.")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable duplicate filtering.")
    parser.add_argument("--ontology-version", choices=["canonical", "default"], default="canonical", help="Ontology version to validate against.")
    parser.add_argument("--report-path", type=Path, default=None, help="Optional path to write the full JSON validation report.")
    parser.add_argument("--show-invalid", type=int, default=5, help="How many invalid triples to print in the CLI summary.")
    args = parser.parse_args()

    report = validate_file(
        args.triples_path,
        source_text_path=args.source_text_path,
        require_text_grounding=args.require_text_grounding,
        dedupe=not args.no_dedupe,
        ontology_version=args.ontology_version,
    )

    summary = report["summary"]
    print(f"Input triples:     {summary['input_triple_count']}")
    print(f"Valid triples:     {summary['valid_triple_count']}")
    print(f"Invalid triples:   {summary['invalid_triple_count']}")
    print(f"Duplicate triples: {summary['duplicate_triple_count']}")
    print(f"Output triples:    {summary['output_triple_count']}")

    if args.show_invalid:
        print("\nInvalid triples:")
        for record in report["invalid_triples"][: args.show_invalid]:
            issue_codes = ", ".join(issue["code"] for issue in record["issues"])
            print(f"- index {record['index']}: {issue_codes} :: {record['triple']}")

    if args.report_path:
        args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return 0 if summary["invalid_triple_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
