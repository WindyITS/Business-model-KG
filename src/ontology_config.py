import json
from functools import lru_cache
from pathlib import Path
from typing import Any


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
ONTOLOGY_PATH = CONFIG_DIR / "ontology.json"


@lru_cache(maxsize=1)
def load_ontology_config() -> dict[str, Any]:
    return json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))


def canonical_labels(label_group: str) -> list[str]:
    ontology = load_ontology_config()
    labels = ontology["canonical_labels"][label_group]
    return list(labels.keys())


def relation_names() -> list[str]:
    ontology = load_ontology_config()
    return list(ontology["relations"].keys())


def node_type_names() -> list[str]:
    ontology = load_ontology_config()
    return list(ontology["node_types"].keys())


def allowed_subject_types(relation: str) -> list[str]:
    ontology = load_ontology_config()
    return list(ontology["relations"][relation]["subject_types"])


def allowed_object_types(relation: str) -> list[str]:
    ontology = load_ontology_config()
    return list(ontology["relations"][relation]["object_types"])


def is_valid_relation_schema(subject_type: str, relation: str, object_type: str) -> bool:
    ontology = load_ontology_config()
    if relation not in ontology["relations"]:
        return False
    relation_payload = ontology["relations"][relation]
    return subject_type in relation_payload["subject_types"] and object_type in relation_payload["object_types"]


def build_strict_ontology_prompt() -> str:
    ontology = load_ontology_config()

    lines: list[str] = ["=== STRICT BUSINESS-MODEL ONTOLOGY (Closed Schema, Closed Labels) ===", ""]
    lines.append("NODE TYPE DEFINITIONS:")
    for node_type, payload in ontology["node_types"].items():
        lines.append(f'- "{node_type}": {payload["definition"]}')

    lines.extend(["", "ALLOWED RELATIONS (subject → object):"])
    for relation, payload in ontology["relations"].items():
        subject_types = " | ".join(payload["subject_types"])
        object_types = " | ".join(payload["object_types"])
        lines.append(f"- {relation}: {subject_types} → {object_types}")

    lines.extend(
        [
            "",
            "CANONICAL CUSTOMER LABELS (exact match only):",
            json.dumps(canonical_labels("CustomerType"), ensure_ascii=False),
            "",
            "CANONICAL CHANNEL LABELS (exact match only):",
            json.dumps(canonical_labels("Channel"), ensure_ascii=False),
            "",
            "CANONICAL REVENUE MODEL LABELS (exact match only):",
            json.dumps(canonical_labels("RevenueModel"), ensure_ascii=False),
            "",
            "GLOBAL RULES:",
        ]
    )
    for rule in ontology["global_rules"]:
        lines.append(f"- {rule}")

    return "\n".join(lines)
