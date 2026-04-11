import json
from collections import Counter
from typing import Any

from finreflectkg_projection import CHUNK_KEY_FIELDS, chunk_key_text_from_payload, iter_grouped_rows
from finreflectkg_stage3 import merge_teacher_reports_into_example
from ontology_validator import canonical_entity_key, validate_triples


def example_chunk_key_text(example: dict[str, Any]) -> str:
    metadata = example.get("metadata") or {}
    return chunk_key_text_from_payload(metadata.get("chunk_key") or {})


def teacher_log_chunk_key_text(call_log: dict[str, Any]) -> str:
    return chunk_key_text_from_payload(call_log.get("chunk_key") or {})


def filter_examples_to_window(
    examples: list[dict[str, Any]],
    *,
    window_chunk_key_texts: set[str],
) -> list[dict[str, Any]]:
    return [example for example in examples if example_chunk_key_text(example) in window_chunk_key_texts]


def filter_teacher_logs_to_window(
    teacher_logs: list[dict[str, Any]],
    *,
    window_chunk_key_texts: set[str],
) -> list[dict[str, Any]]:
    return [call_log for call_log in teacher_logs if teacher_log_chunk_key_text(call_log) in window_chunk_key_texts]


def index_examples_by_chunk_key(examples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for example in examples:
        indexed[example_chunk_key_text(example)] = example
    return indexed


def index_teacher_logs_by_chunk_key(teacher_logs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for call_log in teacher_logs:
        indexed[teacher_log_chunk_key_text(call_log)] = call_log
    return indexed


def normalized_triple_signatures(example: dict[str, Any]) -> set[tuple[str, str, str, str, str]]:
    validation = validate_triples(example.get("output", {}).get("triples", []), dedupe=True)
    signatures: set[tuple[str, str, str, str, str]] = set()
    for triple in validation["valid_triples"]:
        signatures.add(
            (
                canonical_entity_key(str(triple.get("subject", ""))),
                str(triple.get("subject_type", "")),
                str(triple.get("relation", "")),
                canonical_entity_key(str(triple.get("object", ""))),
                str(triple.get("object_type", "")),
            )
        )
    return signatures


def examples_have_same_triples(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return normalized_triple_signatures(left) == normalized_triple_signatures(right)


def rebuild_augmented_positive_from_log(
    base_example: dict[str, Any],
    teacher_log: dict[str, Any],
) -> dict[str, Any]:
    augmented_example, _ = merge_teacher_reports_into_example(
        base_example,
        teacher_log.get("relation_reports", {}),
    )
    return augmented_example


def collect_window_chunk_key_texts(
    rows: list[dict[str, Any]] | Any,
    *,
    limit_chunks: int | None = None,
    skip_chunks: int = 0,
) -> set[str]:
    chunk_key_texts: set[str] = set()
    processed_chunks = 0
    for chunk_rows in iter_grouped_rows(rows):
        processed_chunks += 1
        if processed_chunks <= skip_chunks:
            continue
        chunk_key = {field: chunk_rows[0].get(field) for field in CHUNK_KEY_FIELDS}
        chunk_key_texts.add(chunk_key_text_from_payload(chunk_key))
        if limit_chunks is not None and (processed_chunks - skip_chunks) >= limit_chunks:
            break
    return chunk_key_texts


def build_repair_plan(
    *,
    repaired_projected_examples: list[dict[str, Any]],
    repaired_sampled_empty_examples: list[dict[str, Any]],
    legacy_projected_examples: list[dict[str, Any]],
    legacy_empty_examples: list[dict[str, Any]],
    legacy_augmented_positive_examples: list[dict[str, Any]],
    legacy_teacher_logs: list[dict[str, Any]],
    window_chunk_key_texts: set[str],
) -> dict[str, Any]:
    filtered_legacy_projected = filter_examples_to_window(
        legacy_projected_examples,
        window_chunk_key_texts=window_chunk_key_texts,
    )
    filtered_legacy_empty = filter_examples_to_window(
        legacy_empty_examples,
        window_chunk_key_texts=window_chunk_key_texts,
    )
    filtered_legacy_augmented_positive = filter_examples_to_window(
        legacy_augmented_positive_examples,
        window_chunk_key_texts=window_chunk_key_texts,
    )
    filtered_legacy_teacher_logs = filter_teacher_logs_to_window(
        legacy_teacher_logs,
        window_chunk_key_texts=window_chunk_key_texts,
    )

    legacy_projected_map = index_examples_by_chunk_key(filtered_legacy_projected)
    legacy_empty_map = index_examples_by_chunk_key(filtered_legacy_empty)
    legacy_augmented_positive_map = index_examples_by_chunk_key(filtered_legacy_augmented_positive)
    legacy_teacher_log_map = index_teacher_logs_by_chunk_key(filtered_legacy_teacher_logs)

    repaired_positive_keys = {example_chunk_key_text(example) for example in repaired_projected_examples}
    legacy_empty_keys = set(legacy_empty_map)

    reused_augmented_positive_examples: list[dict[str, Any]] = []
    reused_teacher_logs: list[dict[str, Any]] = []
    positive_examples_to_rerun: list[dict[str, Any]] = []
    positive_rerun_reason_counts: Counter[str] = Counter()

    for example in repaired_projected_examples:
        chunk_key_text = example_chunk_key_text(example)
        legacy_projected = legacy_projected_map.get(chunk_key_text)
        legacy_augmented_positive = legacy_augmented_positive_map.get(chunk_key_text)
        legacy_teacher_log = legacy_teacher_log_map.get(chunk_key_text)

        if (
            legacy_projected is not None
            and legacy_augmented_positive is not None
            and legacy_teacher_log is not None
            and examples_have_same_triples(legacy_projected, example)
        ):
            reused_augmented_positive_examples.append(
                rebuild_augmented_positive_from_log(example, legacy_teacher_log)
            )
            reused_teacher_logs.append(legacy_teacher_log)
            continue

        if legacy_projected is None:
            reason = "legacy_empty_now_positive" if chunk_key_text in legacy_empty_keys else "new_positive"
        elif legacy_augmented_positive is None or legacy_teacher_log is None:
            reason = "missing_reusable_stage3_positive"
        else:
            reason = "changed_deterministic_base"

        positive_examples_to_rerun.append(example)
        positive_rerun_reason_counts[reason] += 1

    new_empty_keys: set[str] = set()
    legacy_recheck_empty_keys: set[str] = set()
    candidate_empty_map: dict[str, dict[str, Any]] = {}
    for example in repaired_sampled_empty_examples:
        chunk_key_text = example_chunk_key_text(example)
        if chunk_key_text in repaired_positive_keys:
            continue
        new_empty_keys.add(chunk_key_text)
        candidate_empty_map[chunk_key_text] = example

    for chunk_key_text, example in legacy_empty_map.items():
        if chunk_key_text in repaired_positive_keys:
            continue
        legacy_recheck_empty_keys.add(chunk_key_text)
        candidate_empty_map.setdefault(chunk_key_text, example)

    empty_examples_to_rerun = [candidate_empty_map[key] for key in sorted(candidate_empty_map)]

    filtered_out_legacy_positive_keys = set(legacy_projected_map) - repaired_positive_keys
    report = {
        "window_chunk_key_count": len(window_chunk_key_texts),
        "repaired_positive_count": len(repaired_projected_examples),
        "repaired_sampled_empty_count": len(repaired_sampled_empty_examples),
        "legacy_positive_count_in_window": len(filtered_legacy_projected),
        "legacy_empty_count_in_window": len(filtered_legacy_empty),
        "legacy_augmented_positive_count_in_window": len(filtered_legacy_augmented_positive),
        "legacy_teacher_log_count_in_window": len(filtered_legacy_teacher_logs),
        "reused_positive_count": len(reused_augmented_positive_examples),
        "rerun_positive_count": len(positive_examples_to_rerun),
        "positive_rerun_reason_counts": dict(sorted(positive_rerun_reason_counts.items())),
        "rerun_empty_count": len(empty_examples_to_rerun),
        "legacy_empty_recheck_count": len(legacy_recheck_empty_keys),
        "new_empty_recheck_count": len(new_empty_keys - legacy_recheck_empty_keys),
        "empty_pool_overlap_count": len(new_empty_keys & legacy_recheck_empty_keys),
        "legacy_positive_filtered_out_count": len(filtered_out_legacy_positive_keys),
    }
    return {
        "reused_augmented_positive_examples": reused_augmented_positive_examples,
        "reused_teacher_logs": reused_teacher_logs,
        "positive_examples_to_rerun": positive_examples_to_rerun,
        "empty_examples_to_rerun": empty_examples_to_rerun,
        "report": report,
    }
