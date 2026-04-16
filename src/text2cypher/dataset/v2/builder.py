from __future__ import annotations

import importlib
import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .models import (
    DatasetSpec,
    FixtureEdgeSpec,
    FixtureNodeSpec,
    FixtureSpec,
    ResultColumnSpec,
    SourceExampleSpec,
)
from .paraphrases import dedupe_question_variants

DEFAULT_SPEC_MODULES = (
    "text2cypher.dataset.v2.spec_core",
    "text2cypher.dataset.v2.spec_rollups",
    "text2cypher.dataset.v2.spec_negative",
    "text2cypher.dataset.v2.spec_hard_train",
    "text2cypher.dataset.v2.spec_heldout_test",
)
DEFAULT_OUTPUT_ROOT = Path("datasets/text2cypher/v3")
TRAINING_POOL_SPLITS = ("train", "dev", "test")
HELDOUT_SPLIT = "heldout_test"
VALIDATION_SPLIT = "valid"
VALIDATION_TARGET_ROWS = 100
VALIDATION_CANDIDATE_SPLITS = ("dev",)
SOURCE_SPLIT_ORDER = TRAINING_POOL_SPLITS + (HELDOUT_SPLIT,)
PARAM_PATTERN = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
SFT_SYSTEM_PROMPT = (
    "Translate the user request into read-only Cypher for a business-model knowledge graph built from "
    "company business descriptions. The graph is about how companies are structured, what they offer, "
    "who they serve, how they sell, how they monetize, where they operate, and which companies they "
    "partner with. Use only these exact node labels: Company, BusinessSegment, Offering, CustomerType, "
    "Channel, RevenueModel, and Place. Use only these exact relationship types and exact casing: "
    "HAS_SEGMENT, OFFERS, SERVES, SELLS_THROUGH, MONETIZES_VIA, OPERATES_IN, and PARTNERS_WITH. If a "
    "label or relationship is not listed here, it does not exist in the KG. Never invent labels, "
    "properties, relationship names, wildcard edges, anonymous relationship patterns, arrows such as "
    "[:]-->>, or casing variants such as HAS_segment. The ontology is segment-centered. The default "
    "structure is Company-[:HAS_SEGMENT]->BusinessSegment-[:OFFERS]->Offering. SERVES and "
    "SELLS_THROUGH live only on BusinessSegment. MONETIZES_VIA lives only on Offering. OPERATES_IN and "
    "PARTNERS_WITH stay company-level. Company-[:OFFERS]->Offering is only for explicit company-level "
    "offering questions. Offering families use Offering-[:OFFERS]->Offering recursively. "
    "BusinessSegment and Offering are company-scoped by company_name, so when matching them under a "
    "company use {company_name: company.name} or {company_name: $company} as appropriate. Channel, "
    "CustomerType, RevenueModel, and Place are global by name. Geography is canonical at "
    "Company-[:OPERATES_IN]->Place. Place may use within_places and includes_places helper arrays "
    "instead of hierarchy edges. Use Cypher parameters for user-provided values and do not rename the "
    "canonical parameter keys: company, segment, offering, customer_type, channel, revenue_model, and "
    "place. Every user-provided value must appear only in params and must be referenced in Cypher with "
    "a $parameter. Never inline user-provided strings, numbers, company names, offering names, "
    "customer types, channel names, revenue models, or place names directly in the Cypher text. Do not "
    "hardcode a literal value in Cypher and also repeat that value in params. For example, use "
    ":CustomerType {name: $customer_type}, :Channel {name: $channel}, :Offering {name: $offering}, "
    "and :Place {name: $place}, not literal values in quotes. Return requested scalar "
    "columns, not whole nodes. Use stable aliases such as company, "
    "segment, offering, customer_type, channel, revenue_model, place, boolean aliases like is_match, "
    "and count aliases like segment_count or offering_count. For list queries prefer RETURN DISTINCT "
    "... ORDER BY .... Canonical idioms matter. For direct segment-offering membership queries, use "
    "direct membership only: MATCH (s)-[:OFFERS]->(o:Offering {name: $offering}) with "
    "WHERE s.company_name = c.name AND o.company_name = c.name. Do not replace direct segment-offering "
    "membership with a root offering plus [:OFFERS*0..] traversal. Use descendant traversal only for "
    "offering family, descendant-offering, or monetization queries. For descendant or monetization "
    "queries, start from a company-scoped root offering and use "
    "(root)-[:OFFERS*0..]->(o:Offering {company_name: company.name}). Never attach MONETIZES_VIA "
    "directly to BusinessSegment. For place-plus-revenue company queries, use the pattern MATCH "
    "(company:Company)-[:OPERATES_IN]->(place:Place {name: $place}) MATCH "
    "(company)-[:HAS_SEGMENT]->(:BusinessSegment {company_name: company.name})-[:OFFERS]->"
    "(root:Offering {company_name: company.name}) MATCH (root)-[:OFFERS*0..]->"
    "(o:Offering {company_name: company.name})-[:MONETIZES_VIA]->"
    "(r:RevenueModel {name: $revenue_model}) RETURN DISTINCT company.name AS company ORDER BY company. "
    "For segment intersection queries over customer type, channel, and offering, use the pattern MATCH "
    "(c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment {company_name: c.name})-[:SERVES]->"
    "(:CustomerType {name: $customer_type}) MATCH (s)-[:SELLS_THROUGH]->(:Channel {name: $channel}) "
    "MATCH (s)-[:OFFERS]->(o:Offering {name: $offering}) WHERE s.company_name = c.name AND "
    "o.company_name = c.name RETURN DISTINCT c.name AS company, s.name AS segment ORDER BY company, "
    "segment. Do not invent suppliers, named customers, employees, prices, revenue amounts, time "
    "series, or unsupported relations. Output compact JSON only. "
    'For answerable requests return {"answerable": true, "cypher": "...", "params": {...}}. '
    'For unsupported or ambiguous requests return {"answerable": false, "reason": "..."}.' 
)


class DatasetBuildError(ValueError):
    pass


def _resolve_output_split(
    intent_id: str,
    intent_split_map: Dict[str, str],
    *,
    validation_intents: set[str],
    has_heldout: bool,
) -> str:
    split = intent_split_map[intent_id]
    if not has_heldout:
        return split
    if split == HELDOUT_SPLIT:
        return HELDOUT_SPLIT
    if intent_id in validation_intents:
        return VALIDATION_SPLIT
    return "train"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _coerce_result_shape(result_shape: Sequence[ResultColumnSpec] | Sequence[Dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if result_shape is None:
        return None
    coerced: list[dict[str, Any]] = []
    for column in result_shape:
        if isinstance(column, ResultColumnSpec):
            coerced.append({k: v for k, v in asdict(column).items() if v is not None})
        else:
            item = dict(column)
            if item.get("description") is None:
                item.pop("description", None)
            coerced.append(item)
    return coerced


def _normalize_example_params(
    *,
    example_id: str,
    gold_cypher: str | None,
    params: Dict[str, Any],
    answerable: bool,
) -> dict[str, Any]:
    normalized = _jsonable(dict(params))
    if not answerable or not gold_cypher:
        return normalized

    referenced = sorted(set(PARAM_PATTERN.findall(gold_cypher)))
    missing = [name for name in referenced if name not in normalized]
    if missing:
        raise DatasetBuildError(
            f"Example {example_id} is missing params required by gold_cypher: {missing}"
        )
    return {name: normalized[name] for name in referenced}


def _normalize_fixture_node(node: FixtureNodeSpec | Dict[str, Any]) -> dict[str, Any]:
    if isinstance(node, FixtureNodeSpec):
        row = {
            "node_id": node.node_id,
            "label": node.label,
            "name": node.name,
        }
        if node.properties:
            row["properties"] = _jsonable(dict(node.properties))
        return row
    row = dict(node)
    if "properties" in row and row["properties"] is not None:
        row["properties"] = _jsonable(row["properties"])
    return row


def _normalize_fixture_edge(edge: FixtureEdgeSpec | Dict[str, Any]) -> dict[str, Any]:
    if isinstance(edge, FixtureEdgeSpec):
        return {"from": edge.source, "type": edge.type, "to": edge.target}
    row = dict(edge)
    if "source" in row and "from" not in row:
        row["from"] = row.pop("source")
    return row


def _normalize_fixture(fixture: FixtureSpec | Dict[str, Any]) -> dict[str, Any]:
    if isinstance(fixture, FixtureSpec):
        row = {
            "fixture_id": fixture.fixture_id,
            "graph_id": fixture.graph_id,
            "graph_purpose": fixture.graph_purpose,
            "covered_families": list(fixture.covered_families),
            "nodes": [_normalize_fixture_node(node) for node in fixture.nodes],
            "edges": [_normalize_fixture_edge(edge) for edge in fixture.edges],
            "invariants_satisfied": list(fixture.invariants_satisfied),
            "authoring_notes": list(fixture.authoring_notes),
        }
        return row
    row = dict(fixture)
    row["covered_families"] = list(row.get("covered_families", []))
    row["nodes"] = [_normalize_fixture_node(node) for node in row.get("nodes", [])]
    row["edges"] = [_normalize_fixture_edge(edge) for edge in row.get("edges", [])]
    row["invariants_satisfied"] = list(row.get("invariants_satisfied", []))
    row["authoring_notes"] = list(row.get("authoring_notes", []))
    return row


def _normalize_source_example(source: SourceExampleSpec | Dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, SourceExampleSpec):
        row = {
            "example_id": source.example_id,
            "intent_id": source.intent_id,
            "family_id": source.family_id,
            "fixture_id": source.fixture_id,
            "graph_id": source.graph_id,
            "binding_id": source.binding_id,
            "question_canonical": source.question_canonical,
            "gold_cypher": source.gold_cypher,
            "params": _jsonable(dict(source.params)),
            "answerable": source.answerable,
            "refusal_reason": source.refusal_reason,
            "result_shape": _coerce_result_shape(source.result_shape),
            "difficulty": source.difficulty,
            "split": source.split,
            "paraphrases": list(source.paraphrases),
        }
        return row
    row = dict(source)
    row["params"] = _jsonable(dict(row.get("params", {})))
    if "result_shape" in row:
        row["result_shape"] = _coerce_result_shape(row["result_shape"])
        if row.get("result_shape") == [] and not row.get("answerable", True):
            row["result_shape"] = None
    row["paraphrases"] = list(row.get("paraphrases", []))
    row["params"] = _normalize_example_params(
        example_id=row["example_id"],
        gold_cypher=row.get("gold_cypher"),
        params=row["params"],
        answerable=row.get("answerable", True),
    )
    return row


def _assistant_output_payload(row: Dict[str, Any]) -> dict[str, Any]:
    if row["answerable"]:
        return {
            "answerable": True,
            "cypher": row["gold_cypher"],
            "params": _jsonable(row["params"]),
        }
    return {
        "answerable": False,
        "reason": row["refusal_reason"],
    }


def _assistant_output_text(row: Dict[str, Any]) -> str:
    return json.dumps(
        _assistant_output_payload(row),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    )


def _normalize_question_text(question: str) -> str:
    return " ".join(question.split()).casefold()


def _load_module_spec(module_name: str) -> DatasetSpec:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise DatasetBuildError(f"Could not import spec module {module_name!r}: {exc}") from exc
    if not hasattr(module, "build_spec"):
        raise DatasetBuildError(f"Spec module {module_name!r} must expose build_spec()")
    spec = module.build_spec()
    if not isinstance(spec, DatasetSpec):
        raise DatasetBuildError(
            f"Spec module {module_name!r} returned {type(spec).__name__}, expected DatasetSpec"
        )
    return spec


def load_dataset_specs(module_names: Sequence[str] = DEFAULT_SPEC_MODULES) -> DatasetSpec:
    fixtures: list[dict[str, Any]] = []
    source_examples: list[dict[str, Any]] = []
    seen_fixture_ids: set[str] = set()
    seen_graph_ids: set[str] = set()
    seen_example_ids: set[str] = set()

    for module_name in module_names:
        spec = _load_module_spec(module_name)
        for fixture in spec.fixtures:
            row = _normalize_fixture(fixture)
            fixture_id = row["fixture_id"]
            graph_id = row["graph_id"]
            if fixture_id in seen_fixture_ids:
                raise DatasetBuildError(f"Duplicate fixture_id detected: {fixture_id}")
            if graph_id in seen_graph_ids:
                raise DatasetBuildError(f"Duplicate graph_id detected: {graph_id}")
            seen_fixture_ids.add(fixture_id)
            seen_graph_ids.add(graph_id)
            fixtures.append(row)

        for source in spec.source_examples:
            row = _normalize_source_example(source)
            example_id = row["example_id"]
            if example_id in seen_example_ids:
                raise DatasetBuildError(f"Duplicate example_id detected: {example_id}")
            seen_example_ids.add(example_id)
            source_examples.append(row)

    _validate_spec_shapes(fixtures, source_examples)
    return DatasetSpec(fixtures=fixtures, source_examples=source_examples)


def _validate_spec_shapes(fixtures: Sequence[Dict[str, Any]], source_examples: Sequence[Dict[str, Any]]) -> None:
    fixture_ids = {fixture["fixture_id"] for fixture in fixtures}
    graph_ids = {fixture["graph_id"] for fixture in fixtures}

    for fixture in fixtures:
        if not fixture["fixture_id"] or not fixture["graph_id"]:
            raise DatasetBuildError("Fixture rows must include fixture_id and graph_id")
        if not isinstance(fixture["graph_purpose"], str) or not fixture["graph_purpose"].strip():
            raise DatasetBuildError(f"Fixture {fixture['fixture_id']} must include graph_purpose")
        if not fixture["covered_families"]:
            raise DatasetBuildError(f"Fixture {fixture['fixture_id']} must declare covered_families")
        if not fixture["invariants_satisfied"]:
            raise DatasetBuildError(f"Fixture {fixture['fixture_id']} must declare invariants_satisfied")
        if not fixture["authoring_notes"]:
            raise DatasetBuildError(f"Fixture {fixture['fixture_id']} must declare authoring_notes")
        if not fixture["nodes"]:
            raise DatasetBuildError(f"Fixture {fixture['fixture_id']} must include at least one node")
        node_ids: set[str] = set()
        for node in fixture["nodes"]:
            for key in ("node_id", "label", "name"):
                if key not in node or node[key] in (None, ""):
                    raise DatasetBuildError(f"Fixture {fixture['fixture_id']} node missing {key}")
            if node["node_id"] in node_ids:
                raise DatasetBuildError(
                    f"Duplicate node_id {node['node_id']!r} in fixture {fixture['fixture_id']}"
                )
            node_ids.add(node["node_id"])
        for edge in fixture["edges"]:
            for key in ("from", "type", "to"):
                if key not in edge or edge[key] in (None, ""):
                    raise DatasetBuildError(f"Fixture {fixture['fixture_id']} edge missing {key}")
            if edge["from"] not in node_ids or edge["to"] not in node_ids:
                raise DatasetBuildError(
                    f"Fixture {fixture['fixture_id']} has edge referencing unknown node ids"
                )

    intent_splits: dict[str, str] = {}
    for example in source_examples:
        if example["fixture_id"] is not None and example["fixture_id"] not in fixture_ids:
            raise DatasetBuildError(
                f"Example {example['example_id']} references unknown fixture_id {example['fixture_id']!r}"
            )
        if example["graph_id"] is not None and example["graph_id"] not in graph_ids:
            raise DatasetBuildError(
                f"Example {example['example_id']} references unknown graph_id {example['graph_id']!r}"
            )
        if not example["example_id"] or not example["intent_id"]:
            raise DatasetBuildError("Source examples must include example_id and intent_id")
        if not isinstance(example["question_canonical"], str) or not example["question_canonical"].strip():
            raise DatasetBuildError(f"Example {example['example_id']} must include question_canonical")
        if example["split"] not in SOURCE_SPLIT_ORDER:
            raise DatasetBuildError(
                f"Example {example['example_id']} has unsupported split {example['split']!r}"
            )
        if not isinstance(example["params"], dict):
            raise DatasetBuildError(f"Example {example['example_id']} must provide params as an object")
        if not isinstance(example["answerable"], bool):
            raise DatasetBuildError(f"Example {example['example_id']} must provide boolean answerable")
        if example["answerable"]:
            if not isinstance(example["result_shape"], list):
                raise DatasetBuildError(
                    f"Answerable example {example['example_id']} must provide result_shape as a list"
                )
            if example["refusal_reason"] is not None:
                raise DatasetBuildError(
                    f"Answerable example {example['example_id']} must not provide refusal_reason"
                )
            for column in example["result_shape"]:
                if not isinstance(column, dict):
                    raise DatasetBuildError(
                        f"Answerable example {example['example_id']} has invalid result_shape row"
                    )
                if not column.get("column") or not column.get("type"):
                    raise DatasetBuildError(
                        f"Answerable example {example['example_id']} has incomplete result_shape row"
                    )
        elif example["result_shape"] is not None:
            raise DatasetBuildError(f"Refusal example {example['example_id']} must omit result_shape")
        if example["answerable"]:
            if not example["gold_cypher"]:
                raise DatasetBuildError(f"Answerable example {example['example_id']} must include gold_cypher")
        else:
            if example["gold_cypher"] is not None:
                raise DatasetBuildError(f"Refusal example {example['example_id']} must omit gold_cypher")
            if not isinstance(example["refusal_reason"], str) or not example["refusal_reason"].strip():
                raise DatasetBuildError(
                    f"Refusal example {example['example_id']} must include refusal_reason"
                )
        previous_split = intent_splits.setdefault(example["intent_id"], example["split"])
        if previous_split != example["split"]:
            raise DatasetBuildError(
                f"Intent {example['intent_id']} is assigned to both {previous_split!r} and {example['split']!r}"
            )
        variants = dedupe_question_variants(example["question_canonical"], example["paraphrases"])
        if len(variants) != len(set(variants)):
            raise DatasetBuildError(f"Duplicate question variants found for {example['example_id']}")


def _serialize_fixture_row(fixture: Dict[str, Any]) -> Dict[str, Any]:
    row = {k: v for k, v in fixture.items() if k not in {"source", "target"}}
    return _jsonable(row)


def _serialize_source_row(source: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "example_id": source["example_id"],
        "intent_id": source["intent_id"],
        "family_id": source["family_id"],
        "fixture_id": source["fixture_id"],
        "graph_id": source["graph_id"],
        "binding_id": source["binding_id"],
        "question_canonical": source["question_canonical"],
        "gold_cypher": source["gold_cypher"],
        "params": _jsonable(source["params"]),
        "answerable": source["answerable"],
        "refusal_reason": source["refusal_reason"],
        "result_shape": source["result_shape"],
        "difficulty": source["difficulty"],
        "split": source["split"],
    }
    return _jsonable(row)


def _build_training_rows(source_examples: Sequence[Dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in source_examples:
        variants = dedupe_question_variants(source["question_canonical"], source["paraphrases"])
        for variant_index, question in enumerate(variants):
            variant_kind = "canonical" if variant_index == 0 else "paraphrase"
            rows.append(
                _jsonable(
                    {
                        "training_example_id": f"{source['example_id']}__v{variant_index:02d}",
                        "source_example_id": source["example_id"],
                        "intent_id": source["intent_id"],
                        "family_id": source["family_id"],
                        "fixture_id": source["fixture_id"],
                        "graph_id": source["graph_id"],
                        "binding_id": source["binding_id"],
                        "variant_index": variant_index,
                        "variant_kind": variant_kind,
                        "question": question,
                        "gold_cypher": source["gold_cypher"],
                        "params": _jsonable(source["params"]),
                        "answerable": source["answerable"],
                        "refusal_reason": source["refusal_reason"],
                        "result_shape": source["result_shape"],
                        "difficulty": source["difficulty"],
                    }
                )
            )
    return rows


def _build_split_manifest(source_examples: Sequence[Dict[str, Any]], training_rows: Sequence[Dict[str, Any]], dataset_path: Path) -> Dict[str, Any]:
    split_to_sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    split_to_intents: dict[str, set[str]] = defaultdict(set)
    family_split_intents: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for source in source_examples:
        split = source["split"]
        split_to_sources[split].append(source)
        split_to_intents[split].add(source["intent_id"])
        family_split_intents[source["family_id"]][split].add(source["intent_id"])

    split_counts: dict[str, dict[str, int]] = {}
    for split in SOURCE_SPLIT_ORDER:
        split_rows = [row for row in training_rows if row["intent_id"] in split_to_intents.get(split, set())]
        split_counts[split] = {
            "rows": len(split_rows),
            "intents": len(split_to_intents.get(split, set())),
            "source_examples": len(split_to_sources.get(split, [])),
        }

    family_intent_split_counts: dict[str, dict[str, int]] = {}
    for family_id, split_map in sorted(family_split_intents.items()):
        family_intent_split_counts[family_id] = {
            split: len(intents) for split, intents in sorted(split_map.items())
        }

    intent_split_map = {
        intent_id: split
        for split, source_rows in split_to_sources.items()
        for intent_id in sorted({row["intent_id"] for row in source_rows})
        if split
    }

    return {
        "dataset_path": str(dataset_path),
        "split_unit": "intent_id",
        "split_counts": split_counts,
        "family_intent_split_counts": family_intent_split_counts,
        "intent_split_map": dict(sorted(intent_split_map.items())),
    }


def _validate_train_family_coverage(manifest: Dict[str, Any]) -> None:
    missing_train_families = [
        family_id
        for family_id, split_counts in sorted(manifest["family_intent_split_counts"].items())
        if split_counts.get("train", 0) < 1
    ]
    if missing_train_families:
        raise DatasetBuildError(
            "Every query family must have train coverage. Missing from train: "
            + ", ".join(missing_train_families)
        )


def _validate_training_question_contracts(
    training_rows: Sequence[Dict[str, Any]],
    intent_split_map: Dict[str, str],
    *,
    validation_intents: set[str] | None = None,
) -> None:
    validation_intents = validation_intents or set()
    has_heldout = any(split == HELDOUT_SPLIT for split in intent_split_map.values())
    question_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in training_rows:
        question_to_rows[_normalize_question_text(row["question"])].append(row)

    conflicting_questions: list[str] = []
    leaked_questions: list[str] = []
    for normalized_question, rows in sorted(question_to_rows.items()):
        completions = {_assistant_output_text(row) for row in rows}
        if len(completions) > 1:
            conflicting_questions.append(rows[0]["question"])
        splits = {
            _resolve_output_split(
                row["intent_id"],
                intent_split_map,
                validation_intents=validation_intents,
                has_heldout=has_heldout,
            )
            for row in rows
        }
        if len(splits) > 1:
            leaked_questions.append(rows[0]["question"])

    if conflicting_questions:
        sample = ", ".join(repr(question) for question in conflicting_questions[:5])
        raise DatasetBuildError(
            "duplicate question conflicts map to different assistant targets. "
            f"Sample conflicts: {sample}"
        )
    if leaked_questions:
        sample = ", ".join(repr(question) for question in leaked_questions[:5])
        raise DatasetBuildError(
            "duplicate question leakage across splits detected. "
            f"Sample leaked prompts: {sample}"
        )


def _build_leakage_report(
    training_rows: Sequence[Dict[str, Any]],
    intent_split_map: Dict[str, str],
) -> dict[str, Any] | None:
    if not any(split == HELDOUT_SPLIT for split in intent_split_map.values()):
        return None

    train_rows = [
        row for row in training_rows if intent_split_map[row["intent_id"]] in TRAINING_POOL_SPLITS
    ]
    heldout_rows = [
        row for row in training_rows if intent_split_map[row["intent_id"]] == HELDOUT_SPLIT
    ]

    normalized_train_questions = {_normalize_question_text(row["question"]) for row in train_rows}
    normalized_heldout_questions = {_normalize_question_text(row["question"]) for row in heldout_rows}
    overlapping_questions = sorted(normalized_train_questions & normalized_heldout_questions)

    train_fixture_ids = {row["fixture_id"] for row in train_rows if row["fixture_id"]}
    heldout_fixture_ids = {row["fixture_id"] for row in heldout_rows if row["fixture_id"]}
    overlapping_fixture_ids = sorted(train_fixture_ids & heldout_fixture_ids)

    train_graph_ids = {row["graph_id"] for row in train_rows if row["graph_id"]}
    heldout_graph_ids = {row["graph_id"] for row in heldout_rows if row["graph_id"]}
    overlapping_graph_ids = sorted(train_graph_ids & heldout_graph_ids)

    report = {
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "normalized_question_overlap_count": len(overlapping_questions),
        "normalized_question_overlap_examples": overlapping_questions[:10],
        "fixture_overlap_count": len(overlapping_fixture_ids),
        "fixture_overlap_examples": overlapping_fixture_ids[:10],
        "graph_overlap_count": len(overlapping_graph_ids),
        "graph_overlap_examples": overlapping_graph_ids[:10],
    }
    if overlapping_questions or overlapping_fixture_ids or overlapping_graph_ids:
        raise DatasetBuildError(
            "held-out evaluation set leaks into the training pool; inspect leakage_report.json details"
        )
    return report


def _build_sft_rows(
    training_rows: Sequence[Dict[str, Any]],
    intent_split_map: Dict[str, str],
    *,
    validation_intents: set[str] | None = None,
) -> list[dict[str, Any]]:
    validation_intents = validation_intents or set()
    has_heldout = any(split == HELDOUT_SPLIT for split in intent_split_map.values())
    grouped_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in training_rows:
        split = _resolve_output_split(
            row["intent_id"],
            intent_split_map,
            validation_intents=validation_intents,
            has_heldout=has_heldout,
        )
        completion = _assistant_output_text(row)
        grouped_rows[(split, row["question"], completion)].append(row)

    sft_rows: list[dict[str, Any]] = []
    for (split, question, completion), rows in sorted(grouped_rows.items()):
        first_row = rows[0]
        training_example_ids = sorted(row["training_example_id"] for row in rows)
        source_example_ids = sorted({row["source_example_id"] for row in rows})
        intent_ids = sorted({row["intent_id"] for row in rows})
        family_ids = sorted({row["family_id"] for row in rows})
        fixture_ids = sorted({row["fixture_id"] for row in rows if row["fixture_id"] is not None})
        graph_ids = sorted({row["graph_id"] for row in rows if row["graph_id"] is not None})
        binding_ids = sorted({row["binding_id"] for row in rows if row["binding_id"] is not None})
        difficulties = sorted({row["difficulty"] for row in rows})
        variant_kinds = sorted({row["variant_kind"] for row in rows})

        sft_rows.append(
            {
                "sft_example_id": training_example_ids[0],
                "training_example_ids": training_example_ids,
                "split": split,
                "prompt": question,
                "completion": completion,
                "messages": [
                    {"role": "system", "content": SFT_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": completion},
                ],
                "metadata": {
                    "example_id": first_row["source_example_id"],
                    "training_example_id": first_row["training_example_id"],
                    "intent_id": first_row["intent_id"],
                    "family_id": first_row["family_id"],
                    "source_example_ids": source_example_ids,
                    "intent_ids": intent_ids,
                    "family_ids": family_ids,
                    "fixture_ids": fixture_ids,
                    "graph_ids": graph_ids,
                    "binding_ids": binding_ids,
                    "variant_kinds": variant_kinds,
                    "difficulty": difficulties,
                    "answerable": first_row["answerable"],
                    "refusal_reason": first_row["refusal_reason"],
                    "result_shape": first_row["result_shape"],
                },
            }
        )
    return sft_rows


def _build_sft_manifest(
    sft_rows: Sequence[Dict[str, Any]],
    training_rows: Sequence[Dict[str, Any]],
    dataset_path: Path,
    split_order: Sequence[str],
    *,
    validation_selection: dict[str, Any] | None = None,
) -> Dict[str, Any]:
    split_counts: dict[str, dict[str, int]] = {}
    for split in split_order:
        rows = [row for row in sft_rows if row["split"] == split]
        split_counts[split] = {
            "rows": len(rows),
            "answerable_rows": sum(row["metadata"]["answerable"] for row in rows),
            "refusal_rows": sum(not row["metadata"]["answerable"] for row in rows),
        }

    return {
        "dataset_path": str(dataset_path),
        "system_prompt": SFT_SYSTEM_PROMPT,
        "assistant_contract": {
            "answerable": {"keys": ["answerable", "cypher", "params"]},
            "refusal": {"keys": ["answerable", "reason"]},
        },
        "counts": {
            "source_training_rows": len(training_rows),
            "sft_rows": len(sft_rows),
            "duplicate_prompt_rows_merged": len(training_rows) - len(sft_rows),
            "answerable_rows": sum(row["metadata"]["answerable"] for row in sft_rows),
            "refusal_rows": sum(not row["metadata"]["answerable"] for row in sft_rows),
        },
        "split_counts": split_counts,
        "validation_selection": validation_selection,
    }

def _select_validation_intents(
    sft_rows: Sequence[Dict[str, Any]],
    intent_split_map: Dict[str, str],
    *,
    target_rows: int = VALIDATION_TARGET_ROWS,
    candidate_splits: Sequence[str] = VALIDATION_CANDIDATE_SPLITS,
) -> dict[str, Any] | None:
    if target_rows <= 0:
        return None

    candidate_counts: dict[str, int] = defaultdict(int)
    candidate_families: dict[str, str] = {}
    for row in sft_rows:
        intent_id = row["metadata"]["intent_id"]
        if intent_split_map.get(intent_id) not in candidate_splits:
            continue
        candidate_counts[intent_id] += 1
        candidate_families[intent_id] = row["metadata"]["family_id"]

    if not candidate_counts:
        return None

    items = sorted(candidate_counts.items(), key=lambda item: (item[1], item[0]))
    subsets: dict[int, tuple[str, ...]] = {0: ()}
    for intent_id, count in items:
        next_subsets = dict(subsets)
        for total, chosen in subsets.items():
            new_total = total + count
            candidate = chosen + (intent_id,)
            existing = next_subsets.get(new_total)
            if existing is None or candidate < existing:
                next_subsets[new_total] = candidate
        subsets = next_subsets

    best_total, best_intents = min(
        subsets.items(),
        key=lambda item: (
            abs(item[0] - target_rows),
            item[0] < target_rows,
            item[0],
            len(item[1]),
            item[1],
        ),
    )
    selected_intents = sorted(best_intents)
    return {
        "target_rows": target_rows,
        "selected_rows": best_total,
        "candidate_splits": list(candidate_splits),
        "selected_intent_ids": selected_intents,
        "selected_family_ids": sorted({candidate_families[intent_id] for intent_id in selected_intents}),
    }


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=False))
            handle.write("\n")


def build_dataset(spec: DatasetSpec, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    fixtures = [_serialize_fixture_row(_normalize_fixture(fixture)) for fixture in spec.fixtures]
    normalized_source_examples = [_normalize_source_example(source) for source in spec.source_examples]
    source_examples = [_serialize_source_row(source) for source in normalized_source_examples]
    all_training_rows = _build_training_rows(normalized_source_examples)
    for row in all_training_rows:
        if row["answerable"] and row["gold_cypher"] is None:
            raise DatasetBuildError(f"Training row {row['training_example_id']} is answerable but missing gold_cypher")
        if not row["answerable"] and row["gold_cypher"] is not None:
            raise DatasetBuildError(f"Training row {row['training_example_id']} is refusal but has gold_cypher")
        if not row["answerable"] and not row["refusal_reason"]:
            raise DatasetBuildError(
                f"Training row {row['training_example_id']} is refusal but missing refusal_reason"
            )

    seen_training_ids = set()
    for row in all_training_rows:
        training_id = row["training_example_id"]
        if training_id in seen_training_ids:
            raise DatasetBuildError(f"Duplicate training_example_id detected: {training_id}")
        seen_training_ids.add(training_id)

    all_training_rows.sort(key=lambda row: (row["intent_id"], row["source_example_id"], row["variant_index"]))
    source_examples.sort(key=lambda row: (row["fixture_id"] or "", row["intent_id"], row["example_id"]))
    fixtures.sort(key=lambda row: (row["fixture_id"], row["graph_id"]))

    manifest = _build_split_manifest(
        normalized_source_examples,
        all_training_rows,
        output_root / "training" / "training_examples.jsonl",
    )
    leakage_report = _build_leakage_report(all_training_rows, manifest["intent_split_map"])
    has_heldout = leakage_report is not None
    provisional_sft_rows = _build_sft_rows(all_training_rows, manifest["intent_split_map"])
    validation_selection = (
        _select_validation_intents(provisional_sft_rows, manifest["intent_split_map"])
        if has_heldout
        else None
    )
    validation_intents = set(validation_selection["selected_intent_ids"]) if validation_selection else set()

    _validate_training_question_contracts(
        all_training_rows,
        manifest["intent_split_map"],
        validation_intents=validation_intents,
    )
    _validate_train_family_coverage(manifest)
    sft_rows = _build_sft_rows(
        all_training_rows,
        manifest["intent_split_map"],
        validation_intents=validation_intents,
    )

    rows_with_output_split = [
        {
            **row,
            "output_split": _resolve_output_split(
                row["intent_id"],
                manifest["intent_split_map"],
                validation_intents=validation_intents,
                has_heldout=has_heldout,
            ),
        }
        for row in all_training_rows
    ]
    training_pool_rows = [row for row in rows_with_output_split if row["output_split"] in {"train", VALIDATION_SPLIT}]
    train_rows = [row for row in rows_with_output_split if row["output_split"] == "train"]
    validation_rows = [row for row in rows_with_output_split if row["output_split"] == VALIDATION_SPLIT]
    heldout_rows = [row for row in rows_with_output_split if row["output_split"] == HELDOUT_SPLIT]

    training_rows = [{k: v for k, v in row.items() if k != "output_split"} for row in training_pool_rows]
    train_rows = [{k: v for k, v in row.items() if k != "output_split"} for row in train_rows]
    validation_rows = [{k: v for k, v in row.items() if k != "output_split"} for row in validation_rows]
    heldout_rows = [{k: v for k, v in row.items() if k != "output_split"} for row in heldout_rows]

    training_sft_rows = [row for row in sft_rows if row["split"] in {"train", VALIDATION_SPLIT}]
    train_sft_rows = [row for row in sft_rows if row["split"] == "train"]
    validation_sft_rows = [row for row in sft_rows if row["split"] == VALIDATION_SPLIT]
    heldout_sft_rows = [row for row in sft_rows if row["split"] == HELDOUT_SPLIT]
    sft_manifest = _build_sft_manifest(
        training_sft_rows if has_heldout else sft_rows,
        training_rows if has_heldout else all_training_rows,
        output_root / "training" / "messages.jsonl",
        split_order=("train", VALIDATION_SPLIT) if has_heldout and validation_sft_rows else ("train",) if has_heldout else TRAINING_POOL_SPLITS,
        validation_selection=validation_selection,
    )
    heldout_sft_manifest = None
    if heldout_sft_rows:
        heldout_sft_manifest = _build_sft_manifest(
            heldout_sft_rows,
            heldout_rows,
            output_root / "evaluation" / "test_messages.jsonl",
            split_order=(HELDOUT_SPLIT,),
        )
    return {
        "fixtures": fixtures,
        "source_examples": source_examples,
        "training_examples": training_rows,
        "train_examples": train_rows,
        "validation_examples": validation_rows,
        "split_manifest": manifest,
        "heldout_examples": heldout_rows,
        "sft_examples": training_sft_rows,
        "train_sft_examples": train_sft_rows,
        "validation_sft_examples": validation_sft_rows,
        "heldout_sft_examples": heldout_sft_rows,
        "sft_manifest": sft_manifest,
        "heldout_sft_manifest": heldout_sft_manifest,
        "leakage_report": leakage_report,
        "validation_selection": validation_selection,
    }


def write_dataset(dataset: dict[str, Any], output_root: Path = DEFAULT_OUTPUT_ROOT) -> None:
    source_root = output_root / "source"
    training_root = output_root / "training"
    evaluation_root = output_root / "evaluation"
    reports_root = output_root / "reports"

    _write_jsonl(source_root / "fixture_instances.jsonl", dataset["fixtures"])
    _write_jsonl(source_root / "bound_seed_examples.jsonl", dataset["source_examples"])
    _write_jsonl(training_root / "training_examples.jsonl", dataset["training_examples"])
    _write_jsonl(training_root / "messages.jsonl", dataset["sft_examples"])

    has_heldout = bool(dataset.get("heldout_sft_examples"))
    if has_heldout:
        _write_jsonl(training_root / "train_examples.jsonl", dataset["train_examples"])
        _write_jsonl(training_root / "train_messages.jsonl", dataset["train_sft_examples"])
        if dataset.get("validation_examples"):
            _write_jsonl(training_root / "valid_examples.jsonl", dataset["validation_examples"])
        if dataset.get("validation_sft_examples"):
            _write_jsonl(training_root / "valid_messages.jsonl", dataset["validation_sft_examples"])
        _write_jsonl(evaluation_root / "test_examples.jsonl", dataset["heldout_examples"])
        _write_jsonl(evaluation_root / "test_messages.jsonl", dataset["heldout_sft_examples"])
    else:
        split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in TRAINING_POOL_SPLITS}
        for row in dataset["training_examples"]:
            source_split = dataset["split_manifest"]["intent_split_map"][row["intent_id"]]
            split_rows[source_split].append(row)
        for split in TRAINING_POOL_SPLITS:
            _write_jsonl(training_root / f"{split}.jsonl", split_rows[split])

        sft_split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in TRAINING_POOL_SPLITS}
        for row in dataset["sft_examples"]:
            sft_split_rows[row["split"]].append(row)
        for split in TRAINING_POOL_SPLITS:
            _write_jsonl(training_root / f"{split}_messages.jsonl", sft_split_rows[split])

    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / "training_split_manifest.json").write_text(
        json.dumps(dataset["split_manifest"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (reports_root / "sft_manifest.json").write_text(
        json.dumps(dataset["sft_manifest"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if dataset.get("heldout_sft_manifest") is not None:
        (reports_root / "heldout_test_manifest.json").write_text(
            json.dumps(dataset["heldout_sft_manifest"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if dataset.get("leakage_report") is not None:
        (reports_root / "leakage_report.json").write_text(
            json.dumps(dataset["leakage_report"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build the active text2cypher dataset release from spec modules.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Local dataset root to write.",
    )
    parser.add_argument(
        "--spec-module",
        action="append",
        dest="spec_modules",
        help="Spec module to load. May be passed multiple times; defaults to the active dataset spec modules.",
    )
    args = parser.parse_args(argv)
    spec_modules = args.spec_modules or list(DEFAULT_SPEC_MODULES)

    spec = load_dataset_specs(spec_modules)
    dataset = build_dataset(spec, args.output_root)
    write_dataset(dataset, args.output_root)

    summary = {
        "output_root": str(args.output_root),
        "fixtures": len(dataset["fixtures"]),
        "source_examples": len(dataset["source_examples"]),
        "training_examples": len(dataset["training_examples"]),
        "sft_examples": len(dataset["sft_examples"]),
    }
    if dataset.get("validation_examples") is not None:
        summary["validation_examples"] = len(dataset["validation_examples"])
    if dataset.get("validation_sft_examples") is not None:
        summary["validation_sft_examples"] = len(dataset["validation_sft_examples"])
    if dataset.get("heldout_examples"):
        summary["heldout_examples"] = len(dataset["heldout_examples"])
    if dataset.get("heldout_sft_examples"):
        summary["heldout_sft_examples"] = len(dataset["heldout_sft_examples"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0
