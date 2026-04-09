import re
from typing import Any

from ontology_validator import canonical_entity_key

NON_NARRATIVE_SECTION_PHRASES = (
    "litigation and other contingencies",
    "subject to legal proceedings",
    "in the opinion of management",
    "consolidated financial statements",
    "notes to consolidated financial statements",
    "net interest expense",
    "interest expense less interest income",
    "income before income taxes",
    "aggregate principal amount",
    "fair value",
    "maturity of the",
    "senior notes",
    "deconsolidation",
    "gain on",
    "gain from",
    "fiscal year ended",
    "principal executive office",
    "principal executive offices",
    "securities registered pursuant",
    "commission washington d.c.",
)

BUSINESS_PROSE_CUES = (
    "business strategy",
    "customers",
    "customer",
    "operates",
    "operates in",
    "provides",
    "offerings",
    "offering",
    "introduced",
    "sold through",
    "sells through",
    "distributed through",
    "revenue derived from",
    "revenue consists",
    "revenue generated from",
    "we bring value",
    "market",
    "channel",
    "subscription",
)

TABLE_CELL_PATTERN = re.compile(r"\|\s*[^|]+\s*\|")
NUMERIC_TOKEN_PATTERN = re.compile(
    r"(?:\$?\d[\d,]*(?:\.\d+)?%?)|(?:\b\d{4}\b)|(?:\b\d+\.\d+\b)"
)
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")


def _sentence_count(text: str) -> int:
    return len([part for part in SENTENCE_SPLIT_PATTERN.split(text) if part.strip()])


def _numeric_token_count(text: str) -> int:
    return len(NUMERIC_TOKEN_PATTERN.findall(text))


def _alpha_token_count(text: str) -> int:
    return sum(1 for token in text.split() if any(char.isalpha() for char in token))


def is_table_like_chunk(chunk_text: str) -> bool:
    text = str(chunk_text or "")
    stripped = text.strip()
    if not stripped:
        return True

    pipe_count = text.count("|")
    if pipe_count >= 8 or len(TABLE_CELL_PATTERN.findall(text)) >= 4:
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True

    short_lines = sum(1 for line in lines if len(line.split()) <= 6)
    if len(lines) >= 6 and short_lines / len(lines) >= 0.6:
        return True

    return False


def contains_non_narrative_section_cues(chunk_text: str) -> bool:
    normalized = canonical_entity_key(str(chunk_text or ""))
    return any(canonical_entity_key(phrase) in normalized for phrase in NON_NARRATIVE_SECTION_PHRASES)


def is_numeric_dense_chunk(chunk_text: str) -> bool:
    text = str(chunk_text or "")
    alpha_tokens = _alpha_token_count(text)
    if alpha_tokens <= 0:
        return True

    numeric_tokens = _numeric_token_count(text)
    return numeric_tokens >= 8 and (numeric_tokens / max(alpha_tokens, 1)) >= 0.18


def is_narrative_business_prose(chunk_text: str) -> bool:
    text = str(chunk_text or "").strip()
    if not text:
        return False
    if is_table_like_chunk(text):
        return False
    if contains_non_narrative_section_cues(text):
        return False
    if is_numeric_dense_chunk(text):
        return False

    sentence_count = _sentence_count(text)
    if sentence_count < 2:
        return False

    normalized = canonical_entity_key(text)
    return any(canonical_entity_key(cue) in normalized for cue in BUSINESS_PROSE_CUES)


def chunk_quality_report(chunk_text: str) -> dict[str, Any]:
    return {
        "is_table_like": is_table_like_chunk(chunk_text),
        "has_non_narrative_section_cues": contains_non_narrative_section_cues(chunk_text),
        "is_numeric_dense": is_numeric_dense_chunk(chunk_text),
        "is_narrative_business_prose": is_narrative_business_prose(chunk_text),
    }
