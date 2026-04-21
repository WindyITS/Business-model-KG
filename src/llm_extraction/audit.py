from collections import Counter
from typing import Any

from ontology.validator import validate_triples


TRIPLE_REQUIRED_KEYS = ("subject", "subject_type", "relation", "object", "object_type")
FORMAT_ISSUE_CODES = {"empty_subject", "empty_object"}


def normalize_lenient_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        return {"triples": payload}
    if isinstance(payload, dict):
        if "triples" in payload:
            return payload
        if all(key in payload for key in TRIPLE_REQUIRED_KEYS):
            return {"triples": [payload]}
        return payload
    return {}


def audit_knowledge_graph_payload(
    payload: Any,
    *,
    payload_parse_recovered: bool = False,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    normalized_payload = normalize_lenient_payload(payload)
    raw_triples = normalized_payload.get("triples", [])
    payload_triples_is_list = isinstance(raw_triples, list)
    if not payload_triples_is_list:
        raw_triples = []

    candidate_triples: list[dict[str, Any]] = []
    non_dict_triple_count = 0
    missing_key_triple_count = 0

    for triple in raw_triples:
        if not isinstance(triple, dict):
            non_dict_triple_count += 1
            continue
        if any(key not in triple for key in TRIPLE_REQUIRED_KEYS):
            missing_key_triple_count += 1
            continue
        candidate_triples.append({key: triple.get(key) for key in TRIPLE_REQUIRED_KEYS})

    validation_report = validate_triples(candidate_triples, dedupe=True)
    invalid_issue_counts: Counter[str] = Counter()
    malformed_from_validation = 0
    ontology_rejected_triple_count = 0

    for invalid in validation_report["invalid_triples"]:
        issue_codes = {issue["code"] for issue in invalid["issues"]}
        invalid_issue_counts.update(issue_codes)
        if issue_codes & FORMAT_ISSUE_CODES:
            malformed_from_validation += 1
        else:
            ontology_rejected_triple_count += 1

    malformed_triple_count = non_dict_triple_count + missing_key_triple_count + malformed_from_validation
    audit = {
        "payload_parse_recovered": payload_parse_recovered,
        "payload_triples_is_list": payload_triples_is_list,
        "raw_triple_count": len(raw_triples),
        "non_dict_triple_count": non_dict_triple_count,
        "missing_key_triple_count": missing_key_triple_count,
        "malformed_triple_count": malformed_triple_count,
        "ontology_rejected_triple_count": ontology_rejected_triple_count,
        "invalid_triple_count": validation_report["summary"]["invalid_triple_count"],
        "duplicate_triple_count": validation_report["summary"]["duplicate_triple_count"],
        "kept_triple_count": len(validation_report["valid_triples"]),
        "invalid_issue_counts": dict(sorted(invalid_issue_counts.items())),
        "ontology_version": "canonical",
    }
    return validation_report["valid_triples"], audit


def aggregate_extraction_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    aggregated = {
        "raw_triple_count": 0,
        "non_dict_triple_count": 0,
        "missing_key_triple_count": 0,
        "malformed_triple_count": 0,
        "ontology_rejected_triple_count": 0,
        "invalid_triple_count": 0,
        "duplicate_triple_count": 0,
        "kept_triple_count": 0,
        "payload_parse_recovered_count": 0,
        "invalid_issue_counts": {},
    }
    issue_counts: Counter[str] = Counter()

    for audit in audits:
        aggregated["raw_triple_count"] += int(audit.get("raw_triple_count", 0))
        aggregated["non_dict_triple_count"] += int(audit.get("non_dict_triple_count", 0))
        aggregated["missing_key_triple_count"] += int(audit.get("missing_key_triple_count", 0))
        aggregated["malformed_triple_count"] += int(audit.get("malformed_triple_count", 0))
        aggregated["ontology_rejected_triple_count"] += int(audit.get("ontology_rejected_triple_count", 0))
        aggregated["invalid_triple_count"] += int(audit.get("invalid_triple_count", 0))
        aggregated["duplicate_triple_count"] += int(audit.get("duplicate_triple_count", 0))
        aggregated["kept_triple_count"] += int(audit.get("kept_triple_count", 0))
        aggregated["payload_parse_recovered_count"] += 1 if audit.get("payload_parse_recovered") else 0
        issue_counts.update(audit.get("invalid_issue_counts", {}))

    aggregated["invalid_issue_counts"] = dict(sorted(issue_counts.items()))
    return aggregated
