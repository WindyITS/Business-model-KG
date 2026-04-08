import logging
import re
import unicodedata
from typing import Dict, List, Tuple

from llm_extractor import KnowledgeGraphExtraction, Triple

logger = logging.getLogger(__name__)


WHITESPACE_RE = re.compile(r"\s+")


def clean_entity_name(name: str) -> str:
    cleaned = unicodedata.normalize("NFKC", name).strip()
    cleaned = cleaned.strip("\"'` ")
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    return cleaned


def canonical_entity_key(name: str) -> str:
    cleaned = clean_entity_name(name)
    cleaned = cleaned.casefold()
    cleaned = cleaned.replace("’", "'")
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    return cleaned


def _surface_score(name: str) -> Tuple[int, int, int]:
    letters = [char for char in name if char.isalpha()]
    has_upper = any(char.isupper() for char in letters)
    has_lower = any(char.islower() for char in letters)
    mixed_case = has_upper and has_lower
    is_long_all_caps = bool(letters) and all(char.isupper() for char in letters) and len(letters) > 4
    return (0 if is_long_all_caps else 1, 1 if mixed_case else 0, len(name))


def _choose_surface(current: str, candidate: str) -> str:
    if _surface_score(candidate) > _surface_score(current):
        return candidate
    return current


def resolve_entities(extractions: List[KnowledgeGraphExtraction]) -> List[Triple]:
    """Flatten and lightly normalize triples without destroying casing. Deduplication is handled downstream by the ontology validator."""
    all_triples: List[Triple] = []

    logger.info("Flattening extractions to a single list of triples...")
    for extraction in extractions:
        if extraction and extraction.triples:
            all_triples.extend(extraction.triples)

    logger.info("Total raw triples extracted: %s", len(all_triples))

    canonical_names: Dict[tuple[str, str], str] = {}
    resolved_records = []

    logger.info("Normalizing entity surface forms...")
    for triple in all_triples:
        subject_clean = clean_entity_name(triple.subject)
        object_clean = clean_entity_name(triple.object)
        if not subject_clean or not object_clean:
            continue

        subject_key = (triple.subject_type, canonical_entity_key(subject_clean))
        object_key = (triple.object_type, canonical_entity_key(object_clean))
        canonical_names[subject_key] = _choose_surface(canonical_names.get(subject_key, subject_clean), subject_clean)
        canonical_names[object_key] = _choose_surface(canonical_names.get(object_key, object_clean), object_clean)

        resolved_records.append(
            (
                subject_key,
                triple.subject_type,
                triple.relation,
                object_key,
                triple.object_type,
            )
        )

    resolved_triples: List[Triple] = []
    seen_records = set()
    for subject_key, subject_type, relation, object_key, object_type in resolved_records:
        resolved_triple = Triple(
            subject=canonical_names[subject_key],
            subject_type=subject_type,
            relation=relation,
            object=canonical_names[object_key],
            object_type=object_type,
        )
        triple_key = (
            resolved_triple.subject,
            resolved_triple.subject_type,
            resolved_triple.relation,
            resolved_triple.object,
            resolved_triple.object_type,
        )
        if triple_key in seen_records:
            continue
        seen_records.add(triple_key)
        resolved_triples.append(resolved_triple)

    logger.info("Total resolved triples: %s", len(resolved_triples))
    return resolved_triples
