"""Evaluate extraction pipeline outputs against clean gold benchmark triples."""

from __future__ import annotations

import argparse
import json
import random
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
DEFAULT_PIPELINES: tuple[str, ...] = ("zero-shot", "memo_graph_only", "analyst")
BOOTSTRAP_SEED = 20260430


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
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def weighted_metric_payload(tp: float, fp: float, fn: float) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def macro_average(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    return {
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
    weighted_fp = len(predicted_by_key) - weighted_tp
    weighted_fn = len(gold_by_key) - weighted_tp
    return {
        "metrics": weighted_metric_payload(tp=weighted_tp, fp=weighted_fp, fn=weighted_fn),
        "counts": {
            "true_positives": weighted_tp,
            "false_positives": weighted_fp,
            "false_negatives": weighted_fn,
        },
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
    edge_result = evaluate_edges(gold_triples, predicted_triples)
    full_context = build_matching_context([*gold_triples, *predicted_triples])
    relaxed_result = weighted_evaluate_triples(gold_triples, predicted_triples, context=full_context)

    return {
        "metrics": {
            **edge_result["metrics"],
            "relaxed_f1": relaxed_result["metrics"]["f1"],
        },
        "counts": {
            "gold_triples": len(gold_triples),
            "predicted_triples": len(predicted_triples),
            **edge_result["counts"],
        },
        "exact_counts": edge_result["metric_counts"],
        "relaxed_counts": relaxed_result["counts"],
        "matched": edge_result["matched"],
        "false_positives": edge_result["false_positives"],
        "false_negatives": edge_result["false_negatives"],
        "relaxed_matches": relaxed_result["matches"],
    }


def evaluate_edges(gold_triples: list[Triple], predicted_triples: list[Triple]) -> dict[str, Any]:
    gold_by_key = unique_by_edge_key(gold_triples)
    predicted_by_key = unique_by_edge_key(predicted_triples)
    gold_keys = set(gold_by_key)
    predicted_keys = set(predicted_by_key)
    matched_keys = gold_keys & predicted_keys
    false_positive_keys = predicted_keys - gold_keys
    false_negative_keys = gold_keys - predicted_keys
    tp = len(matched_keys)
    fp = len(false_positive_keys)
    fn = len(false_negative_keys)
    return {
        "metrics": metric_payload(tp=tp, fp=fp, fn=fn),
        "metric_counts": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        },
        "counts": {
            "gold_unique_edges": len(gold_by_key),
            "predicted_unique_edges": len(predicted_by_key),
        },
        "matched": sort_triples([gold_by_key[key] for key in matched_keys]),
        "false_positives": sort_triples([predicted_by_key[key] for key in false_positive_keys]),
        "false_negatives": sort_triples([gold_by_key[key] for key in false_negative_keys]),
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
    }

    write_json(paths.output_dir / "metrics.json", summary)
    write_jsonl(paths.output_dir / "matched.jsonl", result["matched"])
    write_jsonl(paths.output_dir / "false_positives.jsonl", result["false_positives"])
    write_jsonl(paths.output_dir / "false_negatives.jsonl", result["false_negatives"])
    write_jsonl(paths.output_dir / "relaxed_matches.jsonl", result["relaxed_matches"])
    summary["_exact_counts"] = result["exact_counts"]
    summary["_relaxed_counts"] = result["relaxed_counts"]
    return summary


def evaluate_company_metrics(paths: EvaluationPaths) -> dict[str, Any]:
    if not paths.prediction_path.is_file():
        return {
            "company": paths.company,
            "company_slug": paths.company_slug,
            "pipeline": paths.pipeline,
            "split": paths.split,
            "status": "missing_prediction",
            "gold_path": str(paths.gold_path),
            "prediction_path": str(paths.prediction_path),
        }

    gold_triples = read_jsonl(paths.gold_path)
    predicted_triples = read_prediction_triples(paths.prediction_path)
    result = evaluate_triples(gold_triples, predicted_triples)
    return {
        "company": paths.company,
        "company_slug": paths.company_slug,
        "pipeline": paths.pipeline,
        "split": paths.split,
        "status": "evaluated",
        "gold_path": str(paths.gold_path),
        "prediction_path": str(paths.prediction_path),
        **result["counts"],
        **result["metrics"],
        "_exact_counts": result["exact_counts"],
        "_relaxed_counts": result["relaxed_counts"],
    }


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [result for result in results if result.get("status") == "evaluated"]
    exact_tp = sum(int(result["_exact_counts"]["true_positives"]) for result in evaluated)
    exact_fp = sum(int(result["_exact_counts"]["false_positives"]) for result in evaluated)
    exact_fn = sum(int(result["_exact_counts"]["false_negatives"]) for result in evaluated)
    relaxed_tp = sum(float(result["_relaxed_counts"]["true_positives"]) for result in evaluated)
    relaxed_fp = sum(float(result["_relaxed_counts"]["false_positives"]) for result in evaluated)
    relaxed_fn = sum(float(result["_relaxed_counts"]["false_negatives"]) for result in evaluated)

    exact_metrics = metric_payload(exact_tp, exact_fp, exact_fn)
    exact_macro = macro_average(evaluated)
    relaxed_metrics = weighted_metric_payload(relaxed_tp, relaxed_fp, relaxed_fn)
    return {
        "evaluated_company_count": len(evaluated),
        "missing_prediction_count": sum(1 for result in results if result.get("status") == "missing_prediction"),
        "precision": exact_metrics["precision"],
        "recall": exact_metrics["recall"],
        "f1": exact_metrics["f1"],
        "macro_f1": exact_macro["f1"],
        "relaxed_f1": relaxed_metrics["f1"],
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
    public_results = [
        {key: value for key, value in result.items() if not key.startswith("_")}
        for result in results
    ]
    summary = {
        "result_count": len(public_results),
        "results": public_results,
        "aggregate": aggregate_metrics(results),
    }
    write_json(output_root / "summary.json", summary)
    return summary


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = quantile * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def metric_ci(samples: list[dict[str, Any]], metric: str) -> dict[str, float]:
    values = [float(sample[metric]) for sample in samples]
    return {
        "mean": sum(values) / len(values) if values else 0.0,
        "low": percentile(values, 0.025),
        "high": percentile(values, 0.975),
    }


def bootstrap_metrics(
    *,
    root: Path,
    outputs_root: Path,
    split: str,
    pipelines: list[str],
    companies: list[str],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "method": "company-level bootstrap",
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "split": split,
        "companies": companies,
        "point_estimates": {},
        "confidence_intervals": {},
    }

    for pipeline in pipelines:
        rng = random.Random(seed)
        paths = build_split_evaluation_paths(root=root, outputs_root=outputs_root, pipeline=pipeline, split=split)
        selected_paths = [path for path in paths if path.company_slug in companies]
        company_results = [evaluate_company_metrics(path) for path in selected_paths]
        evaluated = [result for result in company_results if result.get("status") == "evaluated"]
        point_estimate = aggregate_metrics(evaluated)
        payload["point_estimates"][pipeline] = {
            "precision": point_estimate["precision"],
            "recall": point_estimate["recall"],
            "f1": point_estimate["f1"],
            "macro_f1": point_estimate["macro_f1"],
            "relaxed_f1": point_estimate["relaxed_f1"],
        }
        samples = [
            aggregate_metrics([rng.choice(evaluated) for _ in evaluated])
            for _sample_index in range(n_bootstrap)
        ] if evaluated else []
        payload["confidence_intervals"][pipeline] = {
            metric: metric_ci(samples, metric)
            for metric in ("precision", "recall", "f1", "macro_f1", "relaxed_f1")
        }

    return payload


def bootstrap_scope_payloads(
    *,
    root: Path,
    outputs_root: Path,
    split: str,
    pipelines: list[str],
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    clean_dir = root / "benchmarks" / split / "clean"
    all_companies = sorted(path.stem for path in clean_dir.glob("*.jsonl") if path.is_file())
    non_berkshire = [company for company in all_companies if company != "berkshire"]
    return {
        "luca_full_test_1000_bootstrap.json": bootstrap_metrics(
            root=root,
            outputs_root=outputs_root,
            split=split,
            pipelines=pipelines,
            companies=all_companies,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "luca_non_berkshire_1000_bootstrap.json": bootstrap_metrics(
            root=root,
            outputs_root=outputs_root,
            split=split,
            pipelines=pipelines,
            companies=non_berkshire,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
    }


def edge_record_key(row: dict[str, Any]) -> EdgeKey:
    return (
        entity_key(str(row["subject"])),
        str(row["relation"]).strip(),
        entity_key(str(row["object"])),
    )


def read_annotation_edges(path: Path) -> dict[str, set[EdgeKey]]:
    annotators: dict[str, set[EdgeKey]] = defaultdict(set)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or "annotator" not in row:
            raise ValueError(f"{path}:{line_number} is not an annotation edge object.")
        annotators[str(row["annotator"])].add(edge_record_key(row))
    return dict(annotators)


def jaccard(left: set[EdgeKey], right: set[EdgeKey]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def pairwise_f1(left: set[EdgeKey], right: set[EdgeKey]) -> float:
    overlap = len(left & right)
    denominator = len(left) + len(right)
    return 2 * overlap / denominator if denominator else 1.0


def average_pairwise(values: dict[str, set[EdgeKey]], fn) -> float:
    labels = sorted(values)
    pairs = [
        fn(values[left], values[right])
        for index, left in enumerate(labels)
        for right in labels[index + 1 :]
    ]
    return sum(pairs) / len(pairs) if pairs else 1.0


def edge_support_counts(annotator_edges: dict[str, set[EdgeKey]]) -> dict[str, int]:
    union = set().union(*annotator_edges.values()) if annotator_edges else set()
    counts = {"unanimous": 0, "majority_only": 0, "single_annotator": 0}
    annotator_count = len(annotator_edges)
    for edge in union:
        support = sum(edge in edges for edges in annotator_edges.values())
        if support == annotator_count:
            counts["unanimous"] += 1
        elif support > annotator_count / 2:
            counts["majority_only"] += 1
        else:
            counts["single_annotator"] += 1
    return counts


def annotation_reliability_payload(root: Path) -> dict[str, Any]:
    source_dir = root / "benchmarks" / "annotation_reliability"
    annotator_edges = read_annotation_edges(source_dir / "amazon_inter_annotator_edges.jsonl")
    required = {"official", "luca", "zhong"}
    if set(annotator_edges) != required:
        raise ValueError(f"Expected annotators {sorted(required)}, found {sorted(annotator_edges)}.")

    official = annotator_edges["official"]
    luca = annotator_edges["luca"]
    zhong = annotator_edges["zhong"]
    union = official | luca | zhong
    intersection = official & luca & zhong
    support_counts = edge_support_counts(annotator_edges)
    relation_rows = []
    for relation in RELATIONS:
        relation_edges = {
            annotator: {edge for edge in edges if edge[1] == relation}
            for annotator, edges in annotator_edges.items()
        }
        relation_union = set().union(*relation_edges.values()) if relation_edges else set()
        if not relation_union:
            continue
        relation_rows.append(
            {
                "relation": relation,
                "krippendorffs_alpha": average_pairwise(relation_edges, pairwise_f1),
                "three_way_jaccard": (
                    len(set.intersection(*relation_edges.values())) / len(relation_union)
                    if relation_union
                    else 1.0
                ),
            }
        )

    intra_rows = [
        json.loads(line)
        for line in (source_dir / "intra_annotator_counts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in intra_rows:
        row["metric_set"] = "intra_annotator"
        row["scope"] = str(row["label"]).casefold().replace(" ", "_")

    inter_summary = {
        "metric_set": "inter_annotator_amazon",
        "official_edges": len(official),
        "luca_edges": len(luca),
        "zhong_edges": len(zhong),
        "official_vs_luca_jaccard": jaccard(official, luca),
        "official_vs_zhong_jaccard": jaccard(official, zhong),
        "luca_vs_zhong_jaccard": jaccard(luca, zhong),
        "average_pairwise_jaccard": average_pairwise(annotator_edges, jaccard),
        "three_way_jaccard": len(intersection) / len(union) if union else 1.0,
        "krippendorffs_alpha": average_pairwise(annotator_edges, pairwise_f1),
        "candidate_edges": len(union),
        **support_counts,
    }
    return {
        "summary": {
            "inter_annotator_amazon": inter_summary,
            "per_relation_amazon": relation_rows,
            "intra_annotator": {
                "combined_micro": next(row for row in intra_rows if row["scope"] == "combined_micro"),
                "macro_average": next(row for row in intra_rows if row["scope"] == "macro_average"),
            },
            "notes": [
                "Amazon inter-annotator metrics use normalized 3-field edges.",
                "krippendorffs_alpha is computed as average pairwise edge-set F1 over the induced candidate-edge universe.",
                "Intra-annotator metrics are normalized from repeat-annotation counts.",
            ],
        },
        "inter_rows": [
            {
                "metric_set": "inter_annotator_amazon",
                "scope": "overall",
                **inter_summary,
            }
        ],
        "relation_rows": relation_rows,
        "intra_rows": intra_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate extraction outputs against clean gold benchmark triples.")
    parser.add_argument("--pipeline", default=None, help="Extraction pipeline to evaluate, for example zero-shot or analyst.")
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=list(DEFAULT_PIPELINES),
        help="Pipelines for reporting modes such as bootstrap.",
    )
    parser.add_argument("--split", choices=SPLITS, default=None, help="Benchmark split for all-company evaluation.")
    parser.add_argument("--company", default=None, help="Single company to evaluate in cherry-picked mode.")
    parser.add_argument("--bootstrap", action="store_true", help="Compute paper bootstrap confidence intervals.")
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Number of bootstrap samples.")
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED, help="Bootstrap random seed.")
    parser.add_argument(
        "--annotation-reliability",
        action="store_true",
        help="Compute Amazon inter-annotator and intra-annotator reliability artifacts.",
    )
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

    if args.bootstrap:
        if args.company:
            raise SystemExit("--bootstrap runs on split-level company samples; do not combine it with --company.")
        split = args.split or "test"
        output_root = root / "results" / "bootstrap"
        if not prepare_result_folder(output_root, assume_yes=args.yes):
            print("Bootstrap cancelled. Existing results were left unchanged.")
            return 0
        staging_root = staging_result_folder(output_root)
        try:
            payloads = bootstrap_scope_payloads(
                root=root,
                outputs_root=outputs_root,
                split=split,
                pipelines=args.pipelines,
                n_bootstrap=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            )
            for filename, payload in payloads.items():
                write_json(staging_root / filename, payload)
            finalize_result_folder(staging_root, output_root)
        except Exception:
            cleanup_staging_folder(staging_root)
            raise
        print(f"Bootstrap results: {output_root}")
        return 0

    if args.annotation_reliability:
        output_root = root / "results" / "annotation_reliability"
        if not prepare_result_folder(output_root, assume_yes=args.yes):
            print("Annotation reliability cancelled. Existing results were left unchanged.")
            return 0
        staging_root = staging_result_folder(output_root)
        try:
            payload = annotation_reliability_payload(root)
            write_json(staging_root / "summary.json", payload["summary"])
            write_jsonl(staging_root / "inter_annotator_amazon.jsonl", payload["inter_rows"])
            write_jsonl(staging_root / "per_relation_amazon_reliability.jsonl", payload["relation_rows"])
            write_jsonl(staging_root / "intra_annotator.jsonl", payload["intra_rows"])
            finalize_result_folder(staging_root, output_root)
        except Exception:
            cleanup_staging_folder(staging_root)
            raise
        print(f"Annotation reliability results: {output_root}")
        return 0

    if args.company and args.split:
        raise SystemExit("--company and --split are separate modes. Use --company for cherry-picked evaluation or --split for all-company evaluation.")
    if not args.company and not args.split:
        raise SystemExit("Provide either --split for all-company evaluation or --company for cherry-picked evaluation.")
    if not args.pipeline:
        raise SystemExit("Provide --pipeline for extraction evaluation.")

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
    print(
        "Evaluated "
        f"{aggregate['evaluated_company_count']} companies "
        f"(missing predictions: {aggregate['missing_prediction_count']}). "
        f"precision={aggregate['precision']:.3f}, "
        f"recall={aggregate['recall']:.3f}, "
        f"F1={aggregate['f1']:.3f}, "
        f"Macro-F1={aggregate['macro_f1']:.3f}, "
        f"Relaxed F1={aggregate['relaxed_f1']:.3f}"
    )
    print(f"Results: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
