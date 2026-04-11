import hashlib
import json
import heapq
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from tqdm import tqdm

from chunk_quality import chunk_quality_report
from ontology_validator import canonical_entity_key, clean_entity_name, validate_triples


DEFAULT_INSTRUCTION = (
    "Extract the business-model knowledge graph from the following SEC 10-K text "
    "using the strict ontology of this repo. Output ONLY valid JSON."
)

SAFE_ENTITY_TYPE_MAP = {
    "org": "Company",
    "comp": "Company",
    "segment": "BusinessSegment",
    "product": "Offering",
    "gpe": "Place",
}

DIRECT_RELATION_MAP = {
    "operates_in": "OPERATES_IN",
    "partners_with": "PARTNERS_WITH",
    "produce": "OFFERS",
    "produces": "OFFERS",
    "offer": "OFFERS",
    "offers": "OFFERS",
    "provide": "OFFERS",
    "provides": "OFFERS",
    "introduce": "OFFERS",
    "introduces": "OFFERS",
}

FILE_KEY_FIELDS = ("ticker", "year", "source_file")
CHUNK_KEY_FIELDS = ("ticker", "year", "source_file", "page_id", "chunk_id")
SEGMENT_ANCHOR_RELATIONS = {"has_segment", "has_reportable_segment"}
SEGMENT_REVERSE_RELATIONS = {"part_of", "is_part_of"}
SEGMENT_OFFERING_RELATIONS = {"produce", "produces", "offer", "offers", "provide", "provides", "introduce", "introduces"}
PRODUCT_SEGMENT_RELATIONS = {"part_of", "is_part_of", "belongs_to"}
EXCLUDED_SEGMENT_KEYS = {
    "corporate facility",
    "corporate facilities",
    "peer group",
    "peer groups",
    "non-segment/corporate",
    "non segment/corporate",
}
EXCLUDED_SEGMENT_SUBSTRINGS = (
    "stock index",
    "shared facility",
    "shared facilities",
)


def normalize_source_label(value: Any) -> str:
    text = str(value or "").strip()
    text = text.replace("-", "_").replace(" ", "_")
    return text.casefold()


def chunk_key_from_row(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in CHUNK_KEY_FIELDS)


def filing_key_from_row(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in FILE_KEY_FIELDS)


def map_entity_type(source_type: Any) -> str | None:
    return SAFE_ENTITY_TYPE_MAP.get(normalize_source_label(source_type))


def map_relation(source_relation: Any) -> str | None:
    return DIRECT_RELATION_MAP.get(normalize_source_label(source_relation))


def is_plausible_business_segment(name: Any) -> bool:
    normalized = canonical_entity_key(str(name or ""))
    if not normalized:
        return False
    if normalized in EXCLUDED_SEGMENT_KEYS:
        return False
    if any(fragment in normalized for fragment in EXCLUDED_SEGMENT_SUBSTRINGS):
        return False
    return True


def chunk_key_text_from_payload(chunk_key: dict[str, Any]) -> str:
    return json.dumps(chunk_key, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def discover_trusted_segments(
    rows: Iterable[dict[str, Any]],
    *,
    limit_chunks: int | None = None,
    skip_chunks: int = 0,
) -> tuple[dict[tuple[Any, ...], set[str]], dict[str, Any]]:
    trusted_segments_by_filing: dict[tuple[Any, ...], set[str]] = {}
    processed_chunks = 0
    anchor_row_count = 0
    window_chunk_key_count = 0

    total = (skip_chunks + limit_chunks) if limit_chunks is not None else None
    progress = tqdm(iter_grouped_rows(rows), desc="Structural prepass", unit="chunk", total=total)
    for chunk_rows in progress:
        processed_chunks += 1
        if processed_chunks <= skip_chunks:
            continue

        window_chunk_key_count += 1
        filing_key = filing_key_from_row(chunk_rows[0])
        for row in chunk_rows:
            relation = normalize_source_label(row.get("relationship"))
            subject_type = map_entity_type(row.get("entity_type"))
            object_type = map_entity_type(row.get("target_type"))
            subject = str(row.get("entity") or "").strip()
            object_name = str(row.get("target") or "").strip()

            if (
                relation in SEGMENT_ANCHOR_RELATIONS
                and subject_type == "Company"
                and object_type == "BusinessSegment"
                and is_plausible_business_segment(object_name)
            ):
                trusted_segments_by_filing.setdefault(filing_key, set()).add(canonical_entity_key(object_name))
                anchor_row_count += 1
            elif (
                relation in SEGMENT_REVERSE_RELATIONS
                and subject_type == "BusinessSegment"
                and object_type == "Company"
                and is_plausible_business_segment(subject)
            ):
                trusted_segments_by_filing.setdefault(filing_key, set()).add(canonical_entity_key(subject))
                anchor_row_count += 1

        if limit_chunks is not None and (processed_chunks - skip_chunks) >= limit_chunks:
            break
    progress.close()

    skipped_chunks = min(skip_chunks, processed_chunks)
    processed_after_skip_chunks = max(0, processed_chunks - skipped_chunks)
    report = {
        "processed_chunk_count": processed_chunks,
        "skipped_chunk_count": skipped_chunks,
        "processed_after_skip_chunk_count": processed_after_skip_chunks,
        "filings_with_trusted_segments": len(trusted_segments_by_filing),
        "trusted_segment_count": sum(len(names) for names in trusted_segments_by_filing.values()),
        "anchor_row_count": anchor_row_count,
        "window_chunk_key_count": window_chunk_key_count,
    }
    return trusted_segments_by_filing, report


def map_row_to_triples(
    row: dict[str, Any],
    *,
    trusted_segment_keys: set[str] | None = None,
) -> tuple[list[dict[str, str]], str | None]:
    trusted_segment_keys = trusted_segment_keys or set()

    subject_type = map_entity_type(row.get("entity_type"))
    if not subject_type:
        return [], "unmapped_subject_type"

    object_type = map_entity_type(row.get("target_type"))
    if not object_type:
        return [], "unmapped_object_type"

    relation = normalize_source_label(row.get("relationship"))
    subject = str(row.get("entity") or "").strip()
    object_name = str(row.get("target") or "").strip()
    if not subject:
        return [], "empty_subject"
    if not object_name:
        return [], "empty_object"

    subject_key = canonical_entity_key(subject)
    object_key = canonical_entity_key(object_name)

    if relation in SEGMENT_ANCHOR_RELATIONS and subject_type == "Company" and object_type == "BusinessSegment":
        if not is_plausible_business_segment(object_name):
            return [], "filtered_segment_label"
        return [
            {
                "subject": subject,
                "subject_type": "Company",
                "relation": "HAS_SEGMENT",
                "object": object_name,
                "object_type": "BusinessSegment",
            }
        ], None

    if relation in SEGMENT_REVERSE_RELATIONS and subject_type == "BusinessSegment" and object_type == "Company":
        if not is_plausible_business_segment(subject):
            return [], "filtered_segment_label"
        return [
            {
                "subject": object_name,
                "subject_type": "Company",
                "relation": "HAS_SEGMENT",
                "object": subject,
                "object_type": "BusinessSegment",
            }
        ], None

    if relation in PRODUCT_SEGMENT_RELATIONS and subject_type == "Offering" and object_type == "BusinessSegment":
        if not is_plausible_business_segment(object_name):
            return [], "filtered_segment_label"
        if trusted_segment_keys and object_key not in trusted_segment_keys:
            return [], "untrusted_segment_object"
        return [
            {
                "subject": object_name,
                "subject_type": "BusinessSegment",
                "relation": "OFFERS",
                "object": subject,
                "object_type": "Offering",
            },
            {
                "subject": subject,
                "subject_type": "Offering",
                "relation": "PART_OF",
                "object": object_name,
                "object_type": "BusinessSegment",
            },
        ], None

    mapped_relation = map_relation(row.get("relationship"))
    if not mapped_relation:
        return [], "unmapped_relation"

    if subject_type == "BusinessSegment" and mapped_relation in {"OFFERS", "OPERATES_IN"}:
        if not is_plausible_business_segment(subject):
            return [], "filtered_segment_label"
        if subject_key not in trusted_segment_keys:
            return [], "untrusted_segment_subject"

    if (
        subject_type == "BusinessSegment"
        and object_type == "Offering"
        and relation in SEGMENT_OFFERING_RELATIONS
        and mapped_relation == "OFFERS"
    ):
        return [
            {
                "subject": subject,
                "subject_type": "BusinessSegment",
                "relation": "OFFERS",
                "object": object_name,
                "object_type": "Offering",
            },
            {
                "subject": object_name,
                "subject_type": "Offering",
                "relation": "PART_OF",
                "object": subject,
                "object_type": "BusinessSegment",
            },
        ], None

    return [
        {
            "subject": subject,
            "subject_type": subject_type,
            "relation": mapped_relation,
            "object": object_name,
            "object_type": object_type,
        }
    ], None


def map_row_to_triple(row: dict[str, Any]) -> tuple[dict[str, str] | None, str | None]:
    triples, reason = map_row_to_triples(row)
    if not triples:
        return None, reason
    return triples[0], None


def confidence_tier(mapped_count: int, source_count: int) -> str:
    if source_count <= 0:
        return "empty"
    mapped_ratio = mapped_count / source_count
    if mapped_count >= 4 and mapped_ratio >= 0.4:
        return "high"
    if mapped_count >= 2:
        return "medium"
    return "low"


def _chunk_key_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {field: rows[0].get(field) for field in CHUNK_KEY_FIELDS}


def _rank_entity_surfaces(values: Iterable[Any]) -> list[str]:
    counts: Counter[str] = Counter()
    surfaces: dict[str, str] = {}

    for value in values:
        cleaned = clean_entity_name(str(value or ""))
        if not cleaned:
            continue

        key = canonical_entity_key(cleaned)
        counts[key] += 1
        current = surfaces.get(key)
        if current is None or len(cleaned) > len(current):
            surfaces[key] = cleaned

    ranked_keys = sorted(
        counts,
        key=lambda key: (-counts[key], -len(surfaces[key]), surfaces[key].casefold()),
    )
    return [surfaces[key] for key in ranked_keys]


def _subject_company_candidates(rows: list[dict[str, Any]]) -> list[str]:
    return _rank_entity_surfaces(
        row.get("entity")
        for row in rows
        if map_entity_type(row.get("entity_type")) == "Company"
    )


def _analyze_chunk_rows(
    rows: list[dict[str, Any]],
    *,
    trusted_segments_by_filing: dict[tuple[Any, ...], set[str]] | None = None,
) -> dict[str, Any] | None:
    if not rows:
        return None

    source_triple_count = len(rows)
    chunk_text = str(rows[0].get("chunk_text") or "").strip()
    if not chunk_text:
        return None

    mapped_triples: list[dict[str, str]] = []
    dropped_reason_counts: Counter[str] = Counter()
    trusted_segment_keys = (trusted_segments_by_filing or {}).get(filing_key_from_row(rows[0]), set())

    for row in rows:
        triples, reason = map_row_to_triples(row, trusted_segment_keys=trusted_segment_keys)
        if not triples:
            if reason:
                dropped_reason_counts[reason] += 1
            continue
        mapped_triples.extend(triples)

    validation_report = validate_triples(mapped_triples, dedupe=True)
    valid_triples = validation_report["valid_triples"]
    company_candidates = _subject_company_candidates(rows)
    deterministic_relation_counts: Counter[str] = Counter(
        str(triple.get("relation", ""))
        for triple in valid_triples
    )

    metadata = {
        "source_dataset": "domyn/FinReflectKG",
        "chunk_key": _chunk_key_payload(rows),
        "source_triple_count": source_triple_count,
        "mapped_triple_count": len(mapped_triples),
        "kept_triple_count": len(valid_triples),
        "dropped_triple_count": source_triple_count - len(valid_triples),
        "dropped_reason_counts": dict(sorted(dropped_reason_counts.items())),
        "validation_summary": validation_report["summary"],
        "confidence": confidence_tier(len(valid_triples), source_triple_count),
        "source_relations_seen": sorted(
            {normalize_source_label(row.get("relationship")) for row in rows if row.get("relationship")}
        ),
        "source_entity_types_seen": sorted(
            {normalize_source_label(row.get("entity_type")) for row in rows if row.get("entity_type")}
        ),
        "source_target_types_seen": sorted(
            {normalize_source_label(row.get("target_type")) for row in rows if row.get("target_type")}
        ),
        "chunk_quality": chunk_quality_report(chunk_text),
        "deterministic_relation_counts": dict(sorted(deterministic_relation_counts.items())),
    }
    if company_candidates:
        metadata["company_name"] = company_candidates[0]
        metadata["company_name_candidates"] = company_candidates[:5]

    return {
        "chunk_text": chunk_text,
        "valid_triples": valid_triples,
        "metadata": metadata,
    }


def build_projection_example(
    rows: list[dict[str, Any]],
    instruction: str = DEFAULT_INSTRUCTION,
    *,
    trusted_segments_by_filing: dict[tuple[Any, ...], set[str]] | None = None,
) -> dict[str, Any] | None:
    analysis = _analyze_chunk_rows(rows, trusted_segments_by_filing=trusted_segments_by_filing)
    if analysis is None or not analysis["valid_triples"]:
        return None
    if not analysis["metadata"]["chunk_quality"]["is_narrative_business_prose"]:
        return None

    return {
        "instruction": instruction,
        "input": analysis["chunk_text"],
        "output": {
            "extraction_notes": "Projected deterministically from FinReflectKG into the repo ontology.",
            "triples": analysis["valid_triples"],
        },
        "metadata": analysis["metadata"],
    }


def build_empty_example(
    rows: list[dict[str, Any]],
    *,
    instruction: str = DEFAULT_INSTRUCTION,
    min_word_count: int = 80,
    min_char_count: int = 400,
    trusted_segments_by_filing: dict[tuple[Any, ...], set[str]] | None = None,
) -> dict[str, Any] | None:
    analysis = _analyze_chunk_rows(rows, trusted_segments_by_filing=trusted_segments_by_filing)
    if analysis is None:
        return None
    if analysis["valid_triples"]:
        return None
    if not analysis["metadata"]["chunk_quality"]["is_narrative_business_prose"]:
        return None

    chunk_text = analysis["chunk_text"]
    word_count = len(chunk_text.split())
    char_count = len(chunk_text)
    if word_count < min_word_count or char_count < min_char_count:
        return None

    metadata = dict(analysis["metadata"])
    metadata["empty_target"] = True
    metadata["chunk_text_word_count"] = word_count
    metadata["chunk_text_char_count"] = char_count
    metadata["confidence"] = "empty"

    return {
        "instruction": instruction,
        "input": chunk_text,
        "output": {
            "extraction_notes": "No ontology-aligned business-model triples projected from this chunk.",
            "triples": [],
        },
        "metadata": metadata,
    }


def iter_grouped_rows(rows: Iterable[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    current_key: tuple[Any, ...] | None = None
    current_rows: list[dict[str, Any]] = []

    for row in rows:
        row_key = chunk_key_from_row(row)
        if current_key is None:
            current_key = row_key
        if row_key != current_key:
            if current_rows:
                yield current_rows
            current_rows = [row]
            current_key = row_key
            continue
        current_rows.append(row)

    if current_rows:
        yield current_rows


def _iter_with_limit(rows: Iterable[dict[str, Any]], limit_rows: int | None = None) -> Iterator[dict[str, Any]]:
    for index, row in enumerate(rows):
        if limit_rows is not None and index >= limit_rows:
            break
        yield row


def load_finreflectkg_rows(
    *,
    hf_dataset: str | None = None,
    split: str = "train",
    cache_dir: str | None = None,
    parquet_files: list[str] | None = None,
    streaming: bool = True,
    limit_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only when deps are missing.
        raise RuntimeError("The 'datasets' package is required for FinReflectKG projection.") from exc

    if parquet_files:
        dataset = load_dataset(
            "parquet",
            data_files=parquet_files,
            split="train",
            streaming=streaming,
            cache_dir=cache_dir,
        )
    elif hf_dataset:
        dataset = load_dataset(hf_dataset, split=split, streaming=streaming, cache_dir=cache_dir)
    else:
        raise ValueError("Either hf_dataset or parquet_files must be provided.")

    return _iter_with_limit(dataset, limit_rows=limit_rows)


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _stable_chunk_score(chunk_key: dict[str, Any]) -> int:
    payload = json.dumps(chunk_key, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return int(hashlib.sha1(payload.encode("utf-8")).hexdigest(), 16)


def project_dataset_rows(
    rows: Iterable[dict[str, Any]],
    *,
    instruction: str = DEFAULT_INSTRUCTION,
    limit_chunks: int | None = None,
    skip_chunks: int = 0,
    trusted_segments_by_filing: dict[tuple[Any, ...], set[str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    processed_chunks = 0
    kept_chunks = 0
    total_source_rows = 0
    total_kept_triples = 0
    confidence_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()

    total = (skip_chunks + limit_chunks) if limit_chunks is not None else None
    progress = tqdm(iter_grouped_rows(rows), desc="Stage 1 projection", unit="chunk", total=total)
    for chunk_rows in progress:
        processed_chunks += 1

        if processed_chunks <= skip_chunks:
            continue

        total_source_rows += len(chunk_rows)
        example = build_projection_example(
            chunk_rows,
            instruction=instruction,
            trusted_segments_by_filing=trusted_segments_by_filing,
        )
        if example is not None:
            examples.append(example)
            kept_chunks += 1
            kept_triples = int(example["metadata"]["kept_triple_count"])
            total_kept_triples += kept_triples
            confidence_counts[example["metadata"]["confidence"]] += 1
            relation_counts.update(
                str(triple.get("relation", ""))
                for triple in example.get("output", {}).get("triples", [])
            )

        if limit_chunks is not None and (processed_chunks - skip_chunks) >= limit_chunks:
            break
    progress.close()

    skipped_chunks = min(skip_chunks, processed_chunks)
    processed_after_skip_chunks = max(0, processed_chunks - skipped_chunks)
    report = {
        "source_dataset": "domyn/FinReflectKG",
        "processed_chunk_count": processed_chunks,
        "skipped_chunk_count": skipped_chunks,
        "processed_after_skip_chunk_count": processed_after_skip_chunks,
        "kept_chunk_count": kept_chunks,
        "dropped_chunk_count": processed_after_skip_chunks - kept_chunks,
        "total_source_row_count": total_source_rows,
        "total_kept_triple_count": total_kept_triples,
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
    }
    return examples, report


def sample_empty_examples(
    rows: Iterable[dict[str, Any]],
    *,
    positive_example_count: int,
    empty_ratio: float = 0.3,
    instruction: str = DEFAULT_INSTRUCTION,
    limit_chunks: int | None = None,
    skip_chunks: int = 0,
    min_word_count: int = 80,
    min_char_count: int = 400,
    trusted_segments_by_filing: dict[tuple[Any, ...], set[str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_empty_count = max(0, int(round(positive_example_count * empty_ratio)))
    return sample_empty_examples_by_count(
        rows,
        target_empty_count=target_empty_count,
        positive_example_count=positive_example_count,
        empty_ratio=empty_ratio,
        instruction=instruction,
        limit_chunks=limit_chunks,
        skip_chunks=skip_chunks,
        min_word_count=min_word_count,
        min_char_count=min_char_count,
        trusted_segments_by_filing=trusted_segments_by_filing,
    )


def sample_empty_examples_by_count(
    rows: Iterable[dict[str, Any]],
    *,
    target_empty_count: int,
    positive_example_count: int | None = None,
    empty_ratio: float | None = None,
    instruction: str = DEFAULT_INSTRUCTION,
    limit_chunks: int | None = None,
    skip_chunks: int = 0,
    min_word_count: int = 80,
    min_char_count: int = 400,
    exclude_chunk_keys: set[str] | None = None,
    trusted_segments_by_filing: dict[tuple[Any, ...], set[str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    max_heap: list[tuple[int, dict[str, Any]]] = []
    processed_chunks = 0
    eligible_empty_chunk_count = 0
    excluded_chunk_count = 0
    excluded_chunk_keys = exclude_chunk_keys or set()

    total = (skip_chunks + limit_chunks) if limit_chunks is not None else None
    progress = tqdm(iter_grouped_rows(rows), desc="Stage 2 empty sampling", unit="chunk", total=total)
    for chunk_rows in progress:
        processed_chunks += 1
        if processed_chunks <= skip_chunks:
            continue

        example = build_empty_example(
            chunk_rows,
            instruction=instruction,
            min_word_count=min_word_count,
            min_char_count=min_char_count,
            trusted_segments_by_filing=trusted_segments_by_filing,
        )
        if example is not None:
            chunk_key_payload = example["metadata"]["chunk_key"]
            chunk_key_text = json.dumps(chunk_key_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            if chunk_key_text in excluded_chunk_keys:
                excluded_chunk_count += 1
                if limit_chunks is not None and (processed_chunks - skip_chunks) >= limit_chunks:
                    break
                continue
            eligible_empty_chunk_count += 1
            score = _stable_chunk_score(chunk_key_payload)
            heap_item = (-score, example)
            if len(max_heap) < target_empty_count:
                heapq.heappush(max_heap, heap_item)
            elif target_empty_count > 0 and score < -max_heap[0][0]:
                heapq.heapreplace(max_heap, heap_item)

        if limit_chunks is not None and (processed_chunks - skip_chunks) >= limit_chunks:
            break
    progress.close()

    sampled_items = sorted(((-score, example) for score, example in max_heap), key=lambda item: item[0])
    empty_examples = [example for _, example in sampled_items]
    skipped_chunks = min(skip_chunks, processed_chunks)
    processed_after_skip_chunks = max(0, processed_chunks - skipped_chunks)
    report = {
        "processed_chunk_count": processed_chunks,
        "skipped_chunk_count": skipped_chunks,
        "processed_after_skip_chunk_count": processed_after_skip_chunks,
        "positive_example_count": positive_example_count,
        "eligible_empty_chunk_count": eligible_empty_chunk_count,
        "excluded_chunk_count": excluded_chunk_count,
        "sampled_empty_chunk_count": len(empty_examples),
        "target_empty_ratio": empty_ratio,
        "target_empty_count": target_empty_count,
        "min_word_count": min_word_count,
        "min_char_count": min_char_count,
    }
    return empty_examples, report


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
