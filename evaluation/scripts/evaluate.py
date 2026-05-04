"""Evaluate extraction pipeline outputs against clean gold benchmark triples."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import unicodedata
from dataclasses import dataclass, replace
from collections import defaultdict, deque
from pathlib import Path
from typing import Any
from uuid import uuid4

from runtime.output_layout import slugify_company_name


QUOTE_CHARS = "\"'`“”‘’ "
TRIPLE_FIELDS: tuple[str, ...] = ("subject", "subject_type", "relation", "object", "object_type")
EDGE_FIELDS: tuple[str, ...] = ("subject", "relation", "object")
SPLITS: tuple[str, ...] = ("dev", "test")


Triple = dict[str, str]
TripleKey = tuple[str, str, str, str, str]
EdgeKey = tuple[str, str, str]
WeightedMatch = dict[str, Any]


HIERARCHY_RELATIONS: tuple[str, ...] = ("HAS_SEGMENT", "OFFERS")
PARTIAL_HIERARCHY_WEIGHT = 0.75
PARTIAL_SEGMENT_ROLLUP_WEIGHT = 0.5
PARTIAL_COMPANY_ALIAS_WEIGHT = 0.9
RELATIONS: tuple[str, ...] = (
    "HAS_SEGMENT",
    "OFFERS",
    "SERVES",
    "SELLS_THROUGH",
    "OPERATES_IN",
    "PARTNERS_WITH",
    "MONETIZES_VIA",
)
CORPORATE_SUFFIXES: tuple[str, ...] = (
    "inc",
    "incorporated",
    "corporation",
    "corp",
    "company",
    "co",
    "group",
    "holdings",
    "holding",
    "plc",
    "ltd",
    "limited",
)


@dataclass(frozen=True)
class EvaluationPaths:
    gold_path: Path
    prediction_path: Path
    output_dir: Path
    company: str
    company_slug: str
    pipeline: str
    split: str | None = None


def clean_entity_name(name: str) -> str:
    cleaned = unicodedata.normalize("NFKC", name).strip()
    cleaned = cleaned.strip(QUOTE_CHARS)
    return " ".join(cleaned.split())


def entity_key(name: str) -> str:
    cleaned = clean_entity_name(name).casefold()
    cleaned = cleaned.replace("’", "'")
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    return cleaned


def compact_entity_key(name: str) -> str:
    return "".join(character for character in entity_key(name) if character.isalnum())


def company_alias_key(name: str) -> str:
    cleaned = entity_key(name)
    tokens: list[str] = []
    for token in cleaned.split():
        normalized = "".join(character for character in token if character.isalnum())
        if normalized and normalized not in CORPORATE_SUFFIXES:
            tokens.append(normalized)
    if tokens:
        return "".join(tokens)
    compact = compact_entity_key(name)
    for suffix in CORPORATE_SUFFIXES:
        if compact.endswith(suffix) and len(compact) > len(suffix) + 3:
            return compact[: -len(suffix)]
    return compact


def company_names_compatible(left: str, right: str) -> bool:
    left_key = company_alias_key(left)
    right_key = company_alias_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    shorter, longer = sorted((left_key, right_key), key=len)
    return len(shorter) >= 5 and longer.startswith(shorter)


def entity_values_match(left: str, left_type: str, right: str, right_type: str) -> bool:
    if left_type != right_type:
        return False
    if entity_key(left) == entity_key(right):
        return True
    if left_type == "Company":
        return company_names_compatible(left, right)
    return False


def triple_key(triple: Triple) -> TripleKey:
    subject_type = triple["subject_type"].strip()
    object_type = triple["object_type"].strip()
    return (
        entity_key(triple["subject"]),
        subject_type,
        triple["relation"].strip(),
        entity_key(triple["object"]),
        object_type,
    )


def edge_key(triple: Triple) -> EdgeKey:
    return (
        entity_key(triple["subject"]),
        triple["relation"].strip(),
        entity_key(triple["object"]),
    )


def cleaned_triple(triple: dict[str, Any]) -> Triple:
    return {
        "subject": clean_entity_name(str(triple.get("subject", ""))),
        "subject_type": str(triple.get("subject_type", "")).strip(),
        "relation": str(triple.get("relation", "")).strip(),
        "object": clean_entity_name(str(triple.get("object", ""))),
        "object_type": str(triple.get("object_type", "")).strip(),
    }


def read_jsonl(path: Path) -> list[Triple]:
    rows: list[Triple] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object.")
        rows.append(cleaned_triple(payload))
    return rows


def read_prediction_triples(path: Path) -> list[Triple]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    triples = payload.get("triples") if isinstance(payload, dict) else None
    if not isinstance(triples, list):
        raise ValueError(f"{path} does not contain a top-level 'triples' list.")
    return [cleaned_triple(triple) for triple in triples if isinstance(triple, dict)]


def unique_by_key(triples: list[Triple]) -> dict[TripleKey, Triple]:
    records: dict[TripleKey, Triple] = {}
    for triple in triples:
        records.setdefault(triple_key(triple), triple)
    return records


def unique_by_edge_key(triples: list[Triple]) -> dict[EdgeKey, Triple]:
    records: dict[EdgeKey, Triple] = {}
    for triple in triples:
        records.setdefault(edge_key(triple), triple)
    return records


def metric_payload(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def weighted_metric_payload(tp: float, fp: float, fn: float) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def macro_average(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {
            "averaged_count": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    return {
        "averaged_count": len(metrics),
        "precision": sum(float(metric["precision"]) for metric in metrics) / len(metrics),
        "recall": sum(float(metric["recall"]) for metric in metrics) / len(metrics),
        "f1": sum(float(metric["f1"]) for metric in metrics) / len(metrics),
    }


@dataclass
class MatchingContext:
    children: dict[tuple[str, str], set[tuple[str, str]]]
    segment_offerings: dict[str, set[str]]
    segment_companies: dict[str, set[str]]


def entity_node_key(name: str, node_type: str) -> tuple[str, str]:
    return (node_type, entity_key(name))


def build_matching_context(triples: list[Triple]) -> MatchingContext:
    children: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    segment_offerings: dict[str, set[str]] = defaultdict(set)
    segment_companies: dict[str, set[str]] = defaultdict(set)

    for triple in triples:
        relation = triple["relation"]
        subject = entity_node_key(triple["subject"], triple["subject_type"])
        obj = entity_node_key(triple["object"], triple["object_type"])
        if relation in HIERARCHY_RELATIONS:
            children[subject].add(obj)
        if relation == "HAS_SEGMENT" and triple["object_type"] == "BusinessSegment":
            segment_companies[entity_key(triple["object"])].add(entity_key(triple["subject"]))
        if (
            relation == "OFFERS"
            and triple["subject_type"] == "BusinessSegment"
            and triple["object_type"] == "Offering"
        ):
            segment_offerings[entity_key(triple["subject"])].add(entity_key(triple["object"]))

    return MatchingContext(
        children=dict(children),
        segment_offerings=dict(segment_offerings),
        segment_companies=dict(segment_companies),
    )


def has_hierarchy_path(
    context: MatchingContext,
    source_name: str,
    source_type: str,
    target_name: str,
    target_type: str,
) -> bool:
    source = entity_node_key(source_name, source_type)
    target = entity_node_key(target_name, target_type)
    if source == target:
        return True

    queue: deque[tuple[str, str]] = deque([source])
    seen = {source}
    while queue:
        current = queue.popleft()
        for child in context.children.get(current, set()):
            if child == target:
                return True
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return False


def hierarchy_related(
    context: MatchingContext,
    left_name: str,
    left_type: str,
    right_name: str,
    right_type: str,
) -> bool:
    if left_type != right_type:
        return False
    return has_hierarchy_path(context, left_name, left_type, right_name, right_type) or has_hierarchy_path(
        context,
        right_name,
        right_type,
        left_name,
        left_type,
    )


def segment_rollup_related(context: MatchingContext, left: Triple, right: Triple) -> bool:
    if left["subject_type"] != "BusinessSegment" or right["subject_type"] != "BusinessSegment":
        return False
    left_segment = entity_key(left["subject"])
    right_segment = entity_key(right["subject"])
    if left_segment == right_segment:
        return True

    if hierarchy_related(context, left["subject"], "BusinessSegment", right["subject"], "BusinessSegment"):
        return True

    shared_offerings = context.segment_offerings.get(left_segment, set()) & context.segment_offerings.get(right_segment, set())
    if shared_offerings:
        return True

    left_companies = context.segment_companies.get(left_segment, set())
    right_companies = context.segment_companies.get(right_segment, set())
    return bool(left_companies & right_companies) and entity_values_match(
        left["object"],
        left["object_type"],
        right["object"],
        right["object_type"],
    )


def relaxed_match_score(gold: Triple, predicted: Triple, context: MatchingContext) -> tuple[float, str | None]:
    if triple_key(gold) == triple_key(predicted):
        return 1.0, "exact"
    if (
        gold["relation"] != predicted["relation"]
        or gold["subject_type"] != predicted["subject_type"]
        or gold["object_type"] != predicted["object_type"]
    ):
        return 0.0, None

    subject_matches = entity_values_match(
        gold["subject"],
        gold["subject_type"],
        predicted["subject"],
        predicted["subject_type"],
    )
    object_matches = entity_values_match(
        gold["object"],
        gold["object_type"],
        predicted["object"],
        predicted["object_type"],
    )
    if subject_matches and object_matches:
        return PARTIAL_COMPANY_ALIAS_WEIGHT, "company_alias_or_lexical_normalization"

    if object_matches and hierarchy_related(
        context,
        gold["subject"],
        gold["subject_type"],
        predicted["subject"],
        predicted["subject_type"],
    ):
        return PARTIAL_HIERARCHY_WEIGHT, "subject_parent_child"

    if subject_matches and hierarchy_related(
        context,
        gold["object"],
        gold["object_type"],
        predicted["object"],
        predicted["object_type"],
    ):
        return PARTIAL_HIERARCHY_WEIGHT, "object_parent_child"

    if object_matches and segment_rollup_related(context, gold, predicted):
        return PARTIAL_SEGMENT_ROLLUP_WEIGHT, "subject_segment_rollup"

    if gold["relation"] == "HAS_SEGMENT" and subject_matches:
        gold_as_subject = {
            **gold,
            "subject": gold["object"],
            "subject_type": gold["object_type"],
            "object": predicted["object"],
            "object_type": predicted["object_type"],
        }
        predicted_as_subject = {
            **predicted,
            "subject": predicted["object"],
            "subject_type": predicted["object_type"],
            "object": gold["object"],
            "object_type": gold["object_type"],
        }
        if segment_rollup_related(context, gold_as_subject, predicted_as_subject):
            return PARTIAL_SEGMENT_ROLLUP_WEIGHT, "segment_rollup"

    return 0.0, None


def greedy_weighted_matches(
    gold_by_key: dict[TripleKey, Triple],
    predicted_by_key: dict[TripleKey, Triple],
    *,
    context: MatchingContext,
) -> list[WeightedMatch]:
    candidates: list[tuple[float, str, TripleKey, TripleKey]] = []
    for gold_key, gold in gold_by_key.items():
        for predicted_key, predicted in predicted_by_key.items():
            score, reason = relaxed_match_score(gold, predicted, context)
            if score > 0 and reason:
                candidates.append((score, reason, gold_key, predicted_key))

    candidates.sort(
        key=lambda item: (
            -item[0],
            item[1],
            tuple(str(part) for part in item[2]),
            tuple(str(part) for part in item[3]),
        )
    )
    used_gold: set[TripleKey] = set()
    used_predicted: set[TripleKey] = set()
    matches: list[WeightedMatch] = []
    for score, reason, gold_key, predicted_key in candidates:
        if gold_key in used_gold or predicted_key in used_predicted:
            continue
        used_gold.add(gold_key)
        used_predicted.add(predicted_key)
        matches.append(
            {
                "weight": score,
                "reason": reason,
                "gold": gold_by_key[gold_key],
                "predicted": predicted_by_key[predicted_key],
            }
        )
    return matches


def weighted_evaluate_triples(
    gold_triples: list[Triple],
    predicted_triples: list[Triple],
    *,
    context: MatchingContext | None = None,
) -> dict[str, Any]:
    gold_by_key = unique_by_key(gold_triples)
    predicted_by_key = unique_by_key(predicted_triples)
    if context is None:
        context = build_matching_context([*gold_by_key.values(), *predicted_by_key.values()])
    matches = greedy_weighted_matches(gold_by_key, predicted_by_key, context=context)
    weighted_tp = sum(float(match["weight"]) for match in matches)
    return {
        "metrics": weighted_metric_payload(
            tp=weighted_tp,
            fp=len(predicted_by_key) - weighted_tp,
            fn=len(gold_by_key) - weighted_tp,
        ),
        "matches": matches,
    }


def sort_triples(triples: list[Triple]) -> list[Triple]:
    return sorted(triples, key=lambda triple: tuple(triple[field] for field in TRIPLE_FIELDS))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[Triple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def write_unmatched_review_csv(path: Path, *, false_positives: list[Triple], false_negatives: list[Triple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("row_id", "match_id", "source", *TRIPLE_FIELDS)
    rows: list[dict[str, str]] = []
    for index, triple in enumerate(false_negatives, start=1):
        rows.append(
            {
                "row_id": f"gold_{index:04d}",
                "match_id": "",
                "source": "gold",
                **triple,
            }
        )
    for index, triple in enumerate(false_positives, start=1):
        rows.append(
            {
                "row_id": f"predicted_{index:04d}",
                "match_id": "",
                "source": "predicted",
                **triple,
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def result_folder_has_files(path: Path) -> bool:
    return path.exists() and any(candidate.is_file() for candidate in path.rglob("*"))


def confirm_overwrite(path: Path) -> bool:
    response = input(
        f"There are already files in the results folder {path}. "
        "Proceeding with a new evaluation is going to overwrite them. Do you want to proceed? [Y/n] "
    ).strip()
    return response.casefold() not in {"n", "no"}


def prepare_result_folder(path: Path, *, assume_yes: bool = False) -> bool:
    if not result_folder_has_files(path):
        return True

    if not assume_yes and not confirm_overwrite(path):
        return False

    return True


def staging_result_folder(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.parent / f".{path.name}.staging-{uuid4().hex[:8]}"


def remap_evaluation_paths(
    paths: list[EvaluationPaths],
    *,
    original_root: Path,
    staging_root: Path,
) -> list[EvaluationPaths]:
    remapped: list[EvaluationPaths] = []
    for path in paths:
        relative_output_dir = path.output_dir.relative_to(original_root)
        remapped.append(replace(path, output_dir=staging_root / relative_output_dir))
    return remapped


def finalize_result_folder(staging_root: Path, final_root: Path) -> None:
    if final_root.exists():
        shutil.rmtree(final_root)
    shutil.move(str(staging_root), str(final_root))


def cleanup_staging_folder(staging_root: Path) -> None:
    if staging_root.exists():
        shutil.rmtree(staging_root)


def evaluate_triples(gold_triples: list[Triple], predicted_triples: list[Triple]) -> dict[str, Any]:
    gold_by_key = unique_by_key(gold_triples)
    predicted_by_key = unique_by_key(predicted_triples)
    edge_result = evaluate_edges(gold_triples, predicted_triples)

    gold_keys = set(gold_by_key)
    predicted_keys = set(predicted_by_key)
    matched_keys = gold_keys & predicted_keys
    false_positive_keys = predicted_keys - gold_keys
    false_negative_keys = gold_keys - predicted_keys

    strict_metrics = metric_payload(
        tp=len(matched_keys),
        fp=len(false_positive_keys),
        fn=len(false_negative_keys),
    )
    full_context = build_matching_context([*gold_by_key.values(), *predicted_by_key.values()])
    relaxed_result = weighted_evaluate_triples(gold_triples, predicted_triples, context=full_context)

    by_relation: dict[str, dict[str, Any]] = {}
    for relation in RELATIONS:
        relation_gold = [triple for triple in gold_triples if triple["relation"] == relation]
        relation_predicted = [triple for triple in predicted_triples if triple["relation"] == relation]
        if not relation_gold and not relation_predicted:
            continue
        strict_relation = evaluate_triples_strict_only(relation_gold, relation_predicted)
        edge_relation = evaluate_edges_strict_only(relation_gold, relation_predicted)
        relaxed_relation = weighted_evaluate_triples(relation_gold, relation_predicted, context=full_context)
        by_relation[relation] = {
            "edge": edge_relation["metrics"],
            "strict": strict_relation["metrics"],
            "relaxed": relaxed_relation["metrics"],
        }

    return {
        "metrics": strict_metrics,
        "edge": edge_result["metrics"],
        "strict": strict_metrics,
        "relaxed": relaxed_result["metrics"],
        "by_relation": by_relation,
        "counts": {
            "gold_triples": len(gold_triples),
            "gold_unique_triples": len(gold_by_key),
            "predicted_triples": len(predicted_triples),
            "predicted_unique_triples": len(predicted_by_key),
            **edge_result["counts"],
        },
        "matched": sort_triples([gold_by_key[key] for key in matched_keys]),
        "false_positives": sort_triples([predicted_by_key[key] for key in false_positive_keys]),
        "false_negatives": sort_triples([gold_by_key[key] for key in false_negative_keys]),
        "edge_matched": edge_result["matched"],
        "edge_false_positives": edge_result["false_positives"],
        "edge_false_negatives": edge_result["false_negatives"],
        "relaxed_matches": relaxed_result["matches"],
    }


def evaluate_triples_strict_only(gold_triples: list[Triple], predicted_triples: list[Triple]) -> dict[str, Any]:
    gold_by_key = unique_by_key(gold_triples)
    predicted_by_key = unique_by_key(predicted_triples)
    gold_keys = set(gold_by_key)
    predicted_keys = set(predicted_by_key)
    matched_keys = gold_keys & predicted_keys
    return {
        "metrics": metric_payload(
            tp=len(matched_keys),
            fp=len(predicted_keys - gold_keys),
            fn=len(gold_keys - predicted_keys),
        )
    }


def evaluate_edges(gold_triples: list[Triple], predicted_triples: list[Triple]) -> dict[str, Any]:
    gold_by_key = unique_by_edge_key(gold_triples)
    predicted_by_key = unique_by_edge_key(predicted_triples)
    gold_keys = set(gold_by_key)
    predicted_keys = set(predicted_by_key)
    matched_keys = gold_keys & predicted_keys
    false_positive_keys = predicted_keys - gold_keys
    false_negative_keys = gold_keys - predicted_keys
    return {
        "metrics": metric_payload(
            tp=len(matched_keys),
            fp=len(false_positive_keys),
            fn=len(false_negative_keys),
        ),
        "counts": {
            "gold_unique_edges": len(gold_by_key),
            "predicted_unique_edges": len(predicted_by_key),
        },
        "matched": sort_triples([gold_by_key[key] for key in matched_keys]),
        "false_positives": sort_triples([predicted_by_key[key] for key in false_positive_keys]),
        "false_negatives": sort_triples([gold_by_key[key] for key in false_negative_keys]),
    }


def evaluate_edges_strict_only(gold_triples: list[Triple], predicted_triples: list[Triple]) -> dict[str, Any]:
    gold_by_key = unique_by_edge_key(gold_triples)
    predicted_by_key = unique_by_edge_key(predicted_triples)
    gold_keys = set(gold_by_key)
    predicted_keys = set(predicted_by_key)
    matched_keys = gold_keys & predicted_keys
    return {
        "metrics": metric_payload(
            tp=len(matched_keys),
            fp=len(predicted_keys - gold_keys),
            fn=len(gold_keys - predicted_keys),
        )
    }


def evaluate_company(paths: EvaluationPaths) -> dict[str, Any]:
    if not paths.prediction_path.is_file():
        result = {
            "company": paths.company,
            "company_slug": paths.company_slug,
            "pipeline": paths.pipeline,
            "split": paths.split,
            "status": "missing_prediction",
            "gold_path": str(paths.gold_path),
            "prediction_path": str(paths.prediction_path),
        }
        write_json(paths.output_dir / "metrics.json", result)
        return result

    gold_triples = read_jsonl(paths.gold_path)
    predicted_triples = read_prediction_triples(paths.prediction_path)
    result = evaluate_triples(gold_triples, predicted_triples)
    summary = {
        "company": paths.company,
        "company_slug": paths.company_slug,
        "pipeline": paths.pipeline,
        "split": paths.split,
        "status": "evaluated",
        "gold_path": str(paths.gold_path),
        "prediction_path": str(paths.prediction_path),
        **result["counts"],
        **result["metrics"],
        "edge": result["edge"],
        "strict": result["strict"],
        "relaxed": result["relaxed"],
        "by_relation": result["by_relation"],
    }

    write_json(paths.output_dir / "metrics.json", summary)
    write_jsonl(paths.output_dir / "matched.jsonl", result["matched"])
    write_jsonl(paths.output_dir / "false_positives.jsonl", result["false_positives"])
    write_jsonl(paths.output_dir / "false_negatives.jsonl", result["false_negatives"])
    write_jsonl(paths.output_dir / "edge_matched.jsonl", result["edge_matched"])
    write_jsonl(paths.output_dir / "edge_false_positives.jsonl", result["edge_false_positives"])
    write_jsonl(paths.output_dir / "edge_false_negatives.jsonl", result["edge_false_negatives"])
    write_jsonl(paths.output_dir / "relaxed_matches.jsonl", result["relaxed_matches"])
    write_unmatched_review_csv(
        paths.output_dir / "unmatched_for_review.csv",
        false_positives=result["false_positives"],
        false_negatives=result["false_negatives"],
    )
    return summary


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [result for result in results if result.get("status") == "evaluated"]
    edge_tp = sum(int(result["edge"]["true_positives"]) for result in evaluated)
    edge_fp = sum(int(result["edge"]["false_positives"]) for result in evaluated)
    edge_fn = sum(int(result["edge"]["false_negatives"]) for result in evaluated)
    strict_tp = sum(int(result["strict"]["true_positives"]) for result in evaluated)
    strict_fp = sum(int(result["strict"]["false_positives"]) for result in evaluated)
    strict_fn = sum(int(result["strict"]["false_negatives"]) for result in evaluated)
    relaxed_tp = sum(float(result["relaxed"]["true_positives"]) for result in evaluated)
    relaxed_fp = sum(float(result["relaxed"]["false_positives"]) for result in evaluated)
    relaxed_fn = sum(float(result["relaxed"]["false_negatives"]) for result in evaluated)

    edge_company_metrics = [result["edge"] for result in evaluated]
    strict_company_metrics = [result["strict"] for result in evaluated]
    relaxed_company_metrics = [result["relaxed"] for result in evaluated]
    edge_relation_metrics = [
        relation_payload["edge"]
        for result in evaluated
        for relation_payload in result.get("by_relation", {}).values()
    ]
    strict_relation_metrics = [
        relation_payload["strict"]
        for result in evaluated
        for relation_payload in result.get("by_relation", {}).values()
    ]
    relaxed_relation_metrics = [
        relation_payload["relaxed"]
        for result in evaluated
        for relation_payload in result.get("by_relation", {}).values()
    ]
    edge_macro = macro_average(edge_company_metrics)
    relaxed_macro = macro_average(relaxed_company_metrics)
    return {
        "evaluated_company_count": len(evaluated),
        "missing_prediction_count": sum(1 for result in results if result.get("status") == "missing_prediction"),
        "primary_metric": "edge_macro_by_company",
        "edge_micro": metric_payload(edge_tp, edge_fp, edge_fn),
        "edge_macro_by_company": edge_macro,
        "edge_macro_by_company_relation": macro_average(edge_relation_metrics),
        "strict_micro": metric_payload(strict_tp, strict_fp, strict_fn),
        "strict_macro_by_company": macro_average(strict_company_metrics),
        "strict_macro_by_company_relation": macro_average(strict_relation_metrics),
        "relaxed_micro": weighted_metric_payload(relaxed_tp, relaxed_fp, relaxed_fn),
        "relaxed_macro_by_company": relaxed_macro,
        "relaxed_macro_by_company_relation": macro_average(relaxed_relation_metrics),
        **edge_macro,
        "true_positives": relaxed_tp,
        "false_positives": relaxed_fp,
        "false_negatives": relaxed_fn,
    }


def prediction_path(outputs_root: Path, company_slug: str, pipeline: str) -> Path:
    return outputs_root / company_slug / pipeline / "latest" / "resolved_triples.json"


def company_name_from_gold_path(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").title()


def build_split_evaluation_paths(
    *,
    root: Path,
    outputs_root: Path,
    pipeline: str,
    split: str,
) -> list[EvaluationPaths]:
    clean_dir = root / "benchmarks" / split / "clean"
    result_root = root / "results" / pipeline / split
    gold_paths = sorted(path for path in clean_dir.glob("*.jsonl") if path.is_file())
    paths: list[EvaluationPaths] = []
    for gold_path in gold_paths:
        company_slug = slugify_company_name(gold_path.stem)
        company = company_name_from_gold_path(gold_path)
        paths.append(
            EvaluationPaths(
                gold_path=gold_path,
                prediction_path=prediction_path(outputs_root, company_slug, pipeline),
                output_dir=result_root / "companies" / company_slug,
                company=company,
                company_slug=company_slug,
                pipeline=pipeline,
                split=split,
            )
        )
    return paths


def find_cherry_pick_gold_path(root: Path, company: str) -> tuple[Path, str]:
    company_slug = slugify_company_name(company)
    matches: list[tuple[Path, str]] = []
    for split in SPLITS:
        candidate = root / "benchmarks" / split / "clean" / f"{company_slug}.jsonl"
        if candidate.is_file():
            matches.append((candidate, split))

    if not matches:
        raise FileNotFoundError(f"No clean gold benchmark found for company {company!r}. Expected {company_slug}.jsonl.")
    if len(matches) > 1:
        locations = ", ".join(str(path) for path, _split in matches)
        raise ValueError(f"Company {company!r} appears in multiple benchmark splits: {locations}")
    return matches[0]


def build_cherry_pick_evaluation_path(
    *,
    root: Path,
    outputs_root: Path,
    pipeline: str,
    company: str,
) -> EvaluationPaths:
    company_slug = slugify_company_name(company)
    gold_path, _split = find_cherry_pick_gold_path(root, company)
    return EvaluationPaths(
        gold_path=gold_path,
        prediction_path=prediction_path(outputs_root, company_slug, pipeline),
        output_dir=root / "results" / "cherry_picked" / pipeline / company_slug,
        company=company,
        company_slug=company_slug,
        pipeline=pipeline,
        split=None,
    )


def evaluate_paths(paths: list[EvaluationPaths], *, output_root: Path) -> dict[str, Any]:
    results = [evaluate_company(path) for path in paths]
    summary = {
        "result_count": len(results),
        "results": results,
        "aggregate": aggregate_metrics(results),
    }
    write_json(output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate extraction outputs against clean gold benchmark triples.")
    parser.add_argument("--pipeline", required=True, help="Extraction pipeline to evaluate, for example zero-shot or analyst.")
    parser.add_argument("--split", choices=SPLITS, default=None, help="Benchmark split for all-company evaluation.")
    parser.add_argument("--company", default=None, help="Single company to evaluate in cherry-picked mode.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Evaluation folder root. Defaults to the repo's evaluation/ folder.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "outputs",
        help="Pipeline outputs root. Defaults to the repo's outputs/ folder.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Overwrite existing result files without prompting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    outputs_root = args.outputs_root.resolve()

    if args.company and args.split:
        raise SystemExit("--company and --split are separate modes. Use --company for cherry-picked evaluation or --split for all-company evaluation.")
    if not args.company and not args.split:
        raise SystemExit("Provide either --split for all-company evaluation or --company for cherry-picked evaluation.")

    if args.company:
        path = build_cherry_pick_evaluation_path(
            root=root,
            outputs_root=outputs_root,
            pipeline=args.pipeline,
            company=args.company,
        )
        output_root = path.output_dir
        if not prepare_result_folder(output_root, assume_yes=args.yes):
            print("Evaluation cancelled. Existing results were left unchanged.")
            return 0
        staging_root = staging_result_folder(output_root)
        try:
            summary = evaluate_paths([replace(path, output_dir=staging_root)], output_root=staging_root)
            finalize_result_folder(staging_root, output_root)
        except Exception:
            cleanup_staging_folder(staging_root)
            raise
    else:
        paths = build_split_evaluation_paths(
            root=root,
            outputs_root=outputs_root,
            pipeline=args.pipeline,
            split=args.split,
        )
        output_root = root / "results" / args.pipeline / args.split
        if not prepare_result_folder(output_root, assume_yes=args.yes):
            print("Evaluation cancelled. Existing results were left unchanged.")
            return 0
        staging_root = staging_result_folder(output_root)
        try:
            staging_paths = remap_evaluation_paths(paths, original_root=output_root, staging_root=staging_root)
            summary = evaluate_paths(staging_paths, output_root=staging_root)
            finalize_result_folder(staging_root, output_root)
        except Exception:
            cleanup_staging_folder(staging_root)
            raise

    aggregate = summary["aggregate"]
    edge_micro = aggregate["edge_micro"]
    strict_micro = aggregate["strict_micro"]
    relaxed_micro = aggregate["relaxed_micro"]
    print(
        "Evaluated "
        f"{aggregate['evaluated_company_count']} companies "
        f"(missing predictions: {aggregate['missing_prediction_count']}). "
        f"3-field Macro-F1={aggregate['f1']:.3f}, "
        f"precision={aggregate['precision']:.3f}, recall={aggregate['recall']:.3f}. "
        f"3-field Micro-F1={edge_micro['f1']:.3f}; "
        f"Strict Micro-F1={strict_micro['f1']:.3f}; "
        f"Relaxed Micro-F1={relaxed_micro['f1']:.3f}"
    )
    print(f"Results: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
