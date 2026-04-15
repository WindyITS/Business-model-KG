from __future__ import annotations

from collections.abc import Iterable


def dedupe_question_variants(canonical: str, paraphrases: Iterable[str]) -> list[str]:
    """Return stable question variants with duplicates and empty rows removed."""

    variants: list[str] = [canonical]
    seen = {canonical}
    for question in paraphrases:
        cleaned = question.strip()
        if not cleaned or cleaned in seen:
            continue
        variants.append(cleaned)
        seen.add(cleaned)
    return variants

