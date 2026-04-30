from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .cli_output import render_planner_eval_summary
from .config import load_config
from .json_utils import compact_json, extract_first_json_object, read_jsonl, write_json, write_jsonl
from .offline_contract import normalize_query_plan_contract
from .paths import dataset_root, planner_adapter_dir, planner_eval_dir, prepared_planner_raw_dir
from .planner_worker import LMStudioPlannerGenerator, PlannerGenerator
from .progress import StepProgress, track

try:
    from runtime.query_planner import (
        LOOKUP_FAMILIES,
        QueryPlanEnvelope,
        QueryPlanPayload,
        compile_query_plan,
    )
    from runtime.query_planner import _normalize_payload as _runtime_normalize_payload
except ModuleNotFoundError:
    repo_src = Path(__file__).resolve().parents[3] / "src"
    if repo_src.exists():
        sys.path.insert(0, str(repo_src))
    from runtime.query_planner import (
        LOOKUP_FAMILIES,
        QueryPlanEnvelope,
        QueryPlanPayload,
        compile_query_plan,
    )
    from runtime.query_planner import _normalize_payload as _runtime_normalize_payload


class OutputEvaluationError(ValueError):
    pass


_LOOKUP_SORT_KEYS: dict[str, tuple[str, ...]] = {
    "companies_list": ("company",),
    "segments_by_company": ("company", "segment"),
    "offerings_by_company": ("company", "offering"),
    "offerings_by_segment": ("company", "segment", "offering"),
    "companies_by_segment_filters": ("company",),
    "segments_by_segment_filters": ("company", "segment"),
    "companies_by_cross_segment_filters": ("company",),
    "descendant_offerings_by_root": ("company", "offering"),
    "companies_by_descendant_revenue": ("company",),
    "companies_by_place": ("company",),
    "segments_by_place_and_segment_filters": ("company", "segment"),
    "companies_by_partner": ("company",),
}


def _load_synthetic_graph_index(source_root: Path, source_split_name: str) -> dict[str, dict[str, Any]]:
    graph_path = source_root / f"{source_split_name}_synthetic_graphs.json"
    if not graph_path.exists():
        return {}
    graphs = json.loads(graph_path.read_text(encoding="utf-8"))
    if not isinstance(graphs, list):
        return {}
    return {
        graph["graph_id"]: graph
        for graph in graphs
        if isinstance(graph, dict) and isinstance(graph.get("graph_id"), str)
    }


def _row_source_graph_ids(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    ids = row.get("source_graph_ids", metadata.get("source_graph_ids"))
    if isinstance(ids, list):
        return [graph_id for graph_id in ids if isinstance(graph_id, str)]
    graph_id = row.get("source_graph_id", metadata.get("source_graph_id"))
    return [graph_id] if isinstance(graph_id, str) else []


def _row_graphs(row: dict[str, Any], graphs_by_id: dict[str, dict[str, Any]] | None) -> list[dict[str, Any]]:
    inline_graphs = row.get("source_graphs", row.get("synthetic_graphs"))
    if isinstance(inline_graphs, list):
        graphs = [graph for graph in inline_graphs if isinstance(graph, dict)]
        if graphs:
            return graphs

    graph_ids = _row_source_graph_ids(row)
    if not graph_ids:
        raise OutputEvaluationError("missing source graph metadata")
    if graphs_by_id is None:
        raise OutputEvaluationError("missing synthetic graph index")

    missing = [graph_id for graph_id in graph_ids if graph_id not in graphs_by_id]
    if missing:
        raise OutputEvaluationError(f"missing synthetic graph(s): {', '.join(missing)}")
    return [graphs_by_id[graph_id] for graph_id in graph_ids]


def _companies(graphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    companies: list[dict[str, Any]] = []
    for graph in graphs:
        graph_companies = graph.get("companies")
        if isinstance(graph_companies, list):
            companies.extend(company for company in graph_companies if isinstance(company, dict))
        else:
            companies.append(graph)
    return companies


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = _canonical_row(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _canonical_row(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"))


def _sort_and_limit_rows(
    rows: list[dict[str, Any]],
    family: str,
    payload: QueryPlanPayload,
    *,
    apply_limit: bool,
) -> list[dict[str, Any]]:
    keys = _LOOKUP_SORT_KEYS[family]
    sorted_rows = sorted(_dedupe_rows(rows), key=lambda row: tuple(row.get(key) for key in keys))
    if apply_limit and payload.limit is not None:
        return sorted_rows[: payload.limit]
    return sorted_rows


def _company_name(company: dict[str, Any]) -> str:
    return str(company.get("name", ""))


def _company_matches_filters(
    company: dict[str, Any],
    payload: QueryPlanPayload,
    *,
    use_places: bool,
) -> bool:
    if payload.companies and _company_name(company) not in payload.companies:
        return False
    if use_places and payload.places and not _company_matches_places(company, payload.places):
        return False
    return True


def _company_matches_places(company: dict[str, Any], places: list[str]) -> bool:
    company_places = set(company.get("places") or [])
    return any(place in company_places for place in places)


def _segments(company: dict[str, Any]) -> list[dict[str, Any]]:
    segments = company.get("segments")
    if not isinstance(segments, list):
        return []
    return [segment for segment in segments if isinstance(segment, dict)]


def _segment_matches_name(segment: dict[str, Any], payload: QueryPlanPayload) -> bool:
    return not payload.segments or segment.get("name") in payload.segments


def _direct_offerings(segment: dict[str, Any]) -> list[dict[str, Any]]:
    offerings = segment.get("offerings")
    if not isinstance(offerings, list):
        return []
    return [offering for offering in offerings if isinstance(offering, dict)]


def _walk_offering_tree(offering: dict[str, Any]) -> list[dict[str, Any]]:
    offerings = [offering]
    children = offering.get("children")
    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict):
                offerings.extend(_walk_offering_tree(child))
    return offerings


def _segment_offerings(segment: dict[str, Any], hierarchy_mode: str) -> list[dict[str, Any]]:
    direct = _direct_offerings(segment)
    if hierarchy_mode == "descendant":
        descendants: list[dict[str, Any]] = []
        for offering in direct:
            descendants.extend(_walk_offering_tree(offering))
        return descendants
    return direct


def _segment_has_offering(segment: dict[str, Any], offering_name: str, hierarchy_mode: str) -> bool:
    return any(offering.get("name") == offering_name for offering in _segment_offerings(segment, hierarchy_mode))


def _segment_has_revenue_model(segment: dict[str, Any], revenue_model: str, hierarchy_mode: str) -> bool:
    return any(
        revenue_model in (offering.get("revenue_models") or [])
        for offering in _segment_offerings(segment, hierarchy_mode)
    )


def _descendant_offerings_by_root(segment: dict[str, Any], root_names: list[str]) -> list[dict[str, Any]]:
    descendants: list[dict[str, Any]] = []
    for root in _direct_offerings(segment):
        if root.get("name") in root_names:
            descendants.extend(_walk_offering_tree(root))
    return descendants


def _filter_atoms(payload: QueryPlanPayload) -> list[tuple[str, str]]:
    atoms: list[tuple[str, str]] = []
    atoms.extend(("customer_type", value) for value in payload.customer_types)
    atoms.extend(("channel", value) for value in payload.channels)
    atoms.extend(("offering", value) for value in payload.offerings)
    atoms.extend(("revenue_model", value) for value in payload.revenue_models)
    return atoms


def _filter_hierarchy_mode(payload: QueryPlanPayload) -> str:
    return payload.hierarchy_mode or ("descendant" if payload.revenue_models else "direct")


def _segment_matches_atom(
    segment: dict[str, Any],
    atom: tuple[str, str],
    *,
    hierarchy_mode: str,
) -> bool:
    kind, value = atom
    if kind == "customer_type":
        return value in (segment.get("customer_types") or [])
    if kind == "channel":
        return value in (segment.get("channels") or [])
    if kind == "offering":
        return _segment_has_offering(segment, value, hierarchy_mode)
    return _segment_has_revenue_model(segment, value, hierarchy_mode)


def _segment_matches_all_filters(segment: dict[str, Any], payload: QueryPlanPayload) -> bool:
    hierarchy_mode = _filter_hierarchy_mode(payload)
    return all(
        _segment_matches_atom(segment, atom, hierarchy_mode=hierarchy_mode)
        for atom in _filter_atoms(payload)
    )


def _matching_segments(
    payload: QueryPlanPayload,
    graphs: list[dict[str, Any]],
    *,
    use_places: bool,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for company in _companies(graphs):
        if not _company_matches_filters(company, payload, use_places=use_places):
            continue
        for segment in _segments(company):
            if not _segment_matches_name(segment, payload):
                continue
            if _segment_matches_all_filters(segment, payload):
                matches.append((company, segment))
    return matches


def _company_matches_cross_segment_filters(company: dict[str, Any], payload: QueryPlanPayload) -> bool:
    if not _company_matches_filters(company, payload, use_places=False):
        return False
    hierarchy_mode = _filter_hierarchy_mode(payload)
    candidate_segments = [segment for segment in _segments(company) if _segment_matches_name(segment, payload)]
    for atom in _filter_atoms(payload):
        if not any(
            _segment_matches_atom(segment, atom, hierarchy_mode=hierarchy_mode)
            for segment in candidate_segments
        ):
            return False
    return True


def _lookup_rows(
    family: str,
    payload: QueryPlanPayload,
    graphs: list[dict[str, Any]],
    *,
    apply_limit: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]]
    if family == "companies_list":
        rows = [{"company": _company_name(company)} for company in _companies(graphs)]
    elif family == "segments_by_company":
        rows = [
            {"company": _company_name(company), "segment": str(segment.get("name", ""))}
            for company in _companies(graphs)
            if _company_matches_filters(company, payload, use_places=False)
            for segment in _segments(company)
        ]
    elif family == "offerings_by_company":
        rows = [
            {"company": _company_name(company), "offering": str(offering.get("name", ""))}
            for company in _companies(graphs)
            if _company_matches_filters(company, payload, use_places=False)
            for segment in _segments(company)
            for offering in _direct_offerings(segment)
        ]
    elif family == "offerings_by_segment":
        rows = [
            {
                "company": _company_name(company),
                "segment": str(segment.get("name", "")),
                "offering": str(offering.get("name", "")),
            }
            for company in _companies(graphs)
            if _company_matches_filters(company, payload, use_places=False)
            for segment in _segments(company)
            if _segment_matches_name(segment, payload)
            for offering in _direct_offerings(segment)
        ]
    elif family == "companies_by_segment_filters":
        rows = [
            {"company": _company_name(company)}
            for company, _segment in _matching_segments(payload, graphs, use_places=False)
        ]
    elif family == "segments_by_segment_filters":
        rows = [
            {"company": _company_name(company), "segment": str(segment.get("name", ""))}
            for company, segment in _matching_segments(payload, graphs, use_places=False)
        ]
    elif family == "companies_by_cross_segment_filters":
        rows = [
            {"company": _company_name(company)}
            for company in _companies(graphs)
            if _company_matches_cross_segment_filters(company, payload)
        ]
    elif family == "descendant_offerings_by_root":
        rows = [
            {"company": _company_name(company), "offering": str(offering.get("name", ""))}
            for company in _companies(graphs)
            if _company_matches_filters(company, payload, use_places=False)
            for segment in _segments(company)
            for offering in _descendant_offerings_by_root(segment, payload.offerings)
        ]
    elif family == "companies_by_descendant_revenue":
        rows = [
            {"company": _company_name(company)}
            for company in _companies(graphs)
            if _company_matches_filters(company, payload, use_places=bool(payload.places))
            if _company_has_descendant_revenue(company, payload)
        ]
    elif family == "companies_by_place":
        rows = [
            {"company": _company_name(company)}
            for company in _companies(graphs)
            if _company_matches_filters(company, payload, use_places=True)
        ]
    elif family == "segments_by_place_and_segment_filters":
        rows = [
            {"company": _company_name(company), "segment": str(segment.get("name", ""))}
            for company, segment in _matching_segments(payload, graphs, use_places=True)
        ]
    elif family == "companies_by_partner":
        rows = [
            {"company": _company_name(company)}
            for company in _companies(graphs)
            if _company_matches_filters(company, payload, use_places=False)
            if any(partner in (company.get("partners") or []) for partner in payload.partners)
        ]
    else:
        raise OutputEvaluationError(f"unsupported lookup family: {family!r}")
    return _sort_and_limit_rows(rows, family, payload, apply_limit=apply_limit)


def _company_has_descendant_revenue(company: dict[str, Any], payload: QueryPlanPayload) -> bool:
    for segment in _segments(company):
        for offering in _descendant_offerings_by_root(segment, payload.offerings):
            revenue_models = offering.get("revenue_models") or []
            if all(revenue_model in revenue_models for revenue_model in payload.revenue_models):
                return True
    return False


def _count_value(row: dict[str, Any], count_target: str) -> Any:
    if count_target == "company":
        return row.get("company")
    if count_target == "segment":
        return (row.get("company"), row.get("segment"))
    return (row.get("company"), row.get("offering"))


def _count_rows(payload: QueryPlanPayload, graphs: list[dict[str, Any]]) -> list[dict[str, int]]:
    spec = payload.aggregate_spec or {}
    base_family = str(spec.get("base_family"))
    count_target = str(spec.get("count_target"))
    rows = _lookup_rows(base_family, payload, graphs, apply_limit=False)
    return [{f"{count_target}_count": len({_count_value(row, count_target) for row in rows})}]


def _boolean_rows(payload: QueryPlanPayload, graphs: list[dict[str, Any]]) -> list[dict[str, bool]]:
    if payload.base_family is None:
        raise OutputEvaluationError("boolean_exists missing base_family")
    return [{"is_match": bool(_lookup_rows(payload.base_family, payload, graphs, apply_limit=False))}]


def _ranking_limit(payload: QueryPlanPayload) -> int:
    spec = payload.aggregate_spec or {}
    limit = payload.limit if payload.limit is not None else spec.get("limit", 5)
    if not isinstance(limit, int) or limit <= 0:
        raise OutputEvaluationError("ranking_topk requires a positive limit")
    return limit


def _ranking_rows(payload: QueryPlanPayload, graphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spec = payload.aggregate_spec or {}
    metric = spec.get("ranking_metric")
    limit = _ranking_limit(payload)
    selected_companies = [
        company
        for company in _companies(graphs)
        if _company_matches_filters(company, payload, use_places=bool(payload.places))
    ]

    if metric == "customer_type_by_company_count":
        counts: dict[str, set[str]] = defaultdict(set)
        for company in selected_companies:
            for segment in _segments(company):
                for customer_type in segment.get("customer_types") or []:
                    counts[str(customer_type)].add(_company_name(company))
        rows = [
            {"customer_type": customer_type, "company_count": len(companies)}
            for customer_type, companies in counts.items()
        ]
        return sorted(rows, key=lambda row: (-row["company_count"], row["customer_type"]))[:limit]

    if metric == "channel_by_segment_count":
        counts: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for company in selected_companies:
            for segment in _segments(company):
                for channel in segment.get("channels") or []:
                    counts[str(channel)].add((_company_name(company), str(segment.get("name", ""))))
        rows = [
            {"channel": channel, "segment_count": len(segments)}
            for channel, segments in counts.items()
        ]
        return sorted(rows, key=lambda row: (-row["segment_count"], row["channel"]))[:limit]

    if metric == "revenue_model_by_company_count":
        counts: dict[str, set[str]] = defaultdict(set)
        for company in selected_companies:
            for segment in _segments(company):
                for root in _direct_offerings(segment):
                    for offering in _walk_offering_tree(root):
                        for revenue_model in offering.get("revenue_models") or []:
                            counts[str(revenue_model)].add(_company_name(company))
        rows = [
            {"revenue_model": revenue_model, "company_count": len(companies)}
            for revenue_model, companies in counts.items()
        ]
        return sorted(rows, key=lambda row: (-row["company_count"], row["revenue_model"]))[:limit]

    if metric == "company_by_matched_segment_count":
        counts: dict[str, set[str]] = defaultdict(set)
        for company, segment in _matching_segments(payload, graphs, use_places=bool(payload.places)):
            counts[_company_name(company)].add(str(segment.get("name", "")))
        rows = [
            {"company": company, "segment_count": len(segments)}
            for company, segments in counts.items()
        ]
        return sorted(rows, key=lambda row: (-row["segment_count"], row["company"]))[:limit]

    raise OutputEvaluationError(f"unsupported ranking metric: {metric!r}")


def _execute_plan_rows(plan: dict[str, Any], graphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        envelope = QueryPlanEnvelope.model_validate(plan)
    except Exception as exc:  # noqa: BLE001
        raise OutputEvaluationError(f"runtime plan validation failed: {exc}") from exc

    compiled = compile_query_plan(envelope)
    if not compiled.answerable:
        raise OutputEvaluationError(f"runtime plan rejected: {compiled.reason}")

    payload = _runtime_normalize_payload(envelope.payload)
    family = envelope.family
    if family in LOOKUP_FAMILIES:
        return _lookup_rows(family, payload, graphs, apply_limit=True)
    if family == "boolean_exists":
        return _boolean_rows(payload, graphs)
    if family == "count_aggregate":
        return _count_rows(payload, graphs)
    if family == "ranking_topk":
        return _ranking_rows(payload, graphs)
    raise OutputEvaluationError(f"unsupported family: {family!r}")


def _rows_equal(
    predicted_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    *,
    order_matters: bool,
) -> bool:
    if order_matters:
        return [_canonical_row(row) for row in predicted_rows] == [_canonical_row(row) for row in gold_rows]
    return Counter(_canonical_row(row) for row in predicted_rows) == Counter(_canonical_row(row) for row in gold_rows)


def _family_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    bucket: dict[str, dict[str, int]] = defaultdict(lambda: Counter())  # type: ignore[assignment]
    for row in rows:
        family = row["family"]
        bucket[family]["count"] += 1
        if row["json_parse_ok"]:
            bucket[family]["json_parse_ok"] += 1
        if row["contract_valid"]:
            bucket[family]["contract_valid"] += 1
        if row["family_correct"]:
            bucket[family]["family_correct"] += 1
        if row["exact_match"]:
            bucket[family]["exact_match"] += 1
    return {family: dict(sorted(counts.items())) for family, counts in sorted(bucket.items())}


def _evaluate_split(
    rows: list[dict[str, Any]],
    generator: PlannerGenerator,
    *,
    max_tokens: int,
    graphs_by_id: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    for row in track(rows, total=len(rows), desc="planner eval", unit="row"):
        generated_text = generator.generate(row["question"], max_tokens=max_tokens)
        parsed_json: Any | None = None
        json_parse_ok = False
        contract_valid = False
        family_correct = False
        exact_match = False
        normalized_predicted_plan: dict[str, Any] | None = None
        contract_error: str | None = None
        output_evaluable = isinstance(row.get("gold_rows"), list)
        correct_output: bool | None = False if output_evaluable else None
        predicted_rows: list[dict[str, Any]] | None = None
        output_error: str | None = None
        try:
            parsed_json = extract_first_json_object(generated_text)
            json_parse_ok = True
            normalized_predicted_plan = normalize_query_plan_contract(parsed_json)
            contract_valid = normalized_predicted_plan is not None
            if not contract_valid:
                raise ValueError("generated JSON does not match the expected planner contract")
            family_correct = normalized_predicted_plan.get("family") == row["gold_plan"].get("family")
            exact_match = compact_json(normalized_predicted_plan) == compact_json(row["gold_plan"])
        except Exception as exc:  # noqa: BLE001
            contract_error = str(exc)

        if output_evaluable:
            if normalized_predicted_plan is None:
                output_error = contract_error or "generated JSON does not match the expected planner contract"
            else:
                try:
                    graphs = _row_graphs(row, graphs_by_id)
                    predicted_rows = _execute_plan_rows(normalized_predicted_plan, graphs)
                    correct_output = _rows_equal(
                        predicted_rows,
                        row["gold_rows"],
                        order_matters=row["gold_plan"].get("family") == "ranking_topk",
                    )
                except Exception as exc:  # noqa: BLE001
                    output_error = str(exc)

        predictions.append(
            {
                "question": row["question"],
                "family": row["family"],
                "gold_plan": row["gold_plan"],
                "generated_text": generated_text,
                "parsed_json": parsed_json,
                "json_parse_ok": json_parse_ok,
                "contract_valid": contract_valid,
                "family_correct": family_correct,
                "exact_match": exact_match,
                "contract_error": contract_error,
                "output_evaluable": output_evaluable,
                "correct_output": correct_output,
                "predicted_rows": predicted_rows,
                "output_error": output_error,
            }
        )

    count = len(predictions)
    output_evaluable_count = sum(1 for row in predictions if row["output_evaluable"])
    correct_outputs = sum(1 for row in predictions if row["correct_output"] is True)
    metrics = {
        "count": count,
        "json_parse_rate": sum(1 for row in predictions if row["json_parse_ok"]) / count if count else 0.0,
        "contract_valid_rate": sum(1 for row in predictions if row["contract_valid"]) / count if count else 0.0,
        "family_accuracy": sum(1 for row in predictions if row["family_correct"]) / count if count else 0.0,
        "exact_plan_match_rate": sum(1 for row in predictions if row["exact_match"]) / count if count else 0.0,
        "correct_output_rate": correct_outputs / output_evaluable_count if output_evaluable_count else 0.0,
        "correct_outputs": correct_outputs,
        "output_evaluable_count": output_evaluable_count,
        "per_family": _family_summary(predictions),
    }
    return metrics, predictions


def evaluate_planner(
    config_path: str | None = None,
    *,
    base_only: bool = False,
    backend: str = "mlx",
    lmstudio_model: str | None = None,
    lmstudio_base_url: str = "http://localhost:1234/v1",
    lmstudio_api_key: str | None = None,
) -> dict[str, Any]:
    if backend != "mlx" and base_only:
        raise ValueError("--base-only is supported only for the mlx backend.")

    with StepProgress(total=4, desc="eval-planner") as progress:
        config = load_config(config_path)
        if backend == "mlx":
            generator = PlannerGenerator(
                model_path=config.planner.base_model,
                adapter_path=None if base_only else str(planner_adapter_dir(config)),
            )
            eval_dir = planner_eval_dir(config, base_only=base_only, backend=backend)
            summary = {
                "backend": "mlx",
                "mode": "base_model" if base_only else "adapter",
                "base_model": config.planner.base_model,
                "adapter_path": None if base_only else str(planner_adapter_dir(config)),
                "artifact_dir": str(eval_dir),
            }
        elif backend == "lmstudio":
            resolved_lmstudio_model = lmstudio_model or config.planner.base_model
            generator = LMStudioPlannerGenerator(
                model_name=resolved_lmstudio_model,
                base_url=lmstudio_base_url,
                api_key=lmstudio_api_key,
            )
            eval_dir = planner_eval_dir(
                config,
                backend=backend,
                model_name=resolved_lmstudio_model,
            )
            summary = {
                "backend": "lmstudio",
                "mode": "lmstudio_model",
                "base_model": config.planner.base_model,
                "adapter_path": None,
                "served_model": resolved_lmstudio_model,
                "lmstudio_base_url": lmstudio_base_url,
                "artifact_dir": str(eval_dir),
            }
        else:
            raise ValueError(f"Unsupported planner eval backend: {backend}")

        prepared_dir = prepared_planner_raw_dir(config)
        source_root = dataset_root(config)
        validation_graphs = _load_synthetic_graph_index(source_root, "validation")
        test_graphs = _load_synthetic_graph_index(source_root, "release_eval")
        validation_rows = read_jsonl(prepared_dir / "valid.jsonl")
        test_rows = read_jsonl(prepared_dir / "test.jsonl")
        progress.advance("loaded planner evaluation splits")

        validation_metrics, validation_predictions = _evaluate_split(
            validation_rows,
            generator,
            max_tokens=config.planner.max_tokens,
            graphs_by_id=validation_graphs,
        )
        progress.advance("evaluated validation split")
        test_metrics, test_predictions = _evaluate_split(
            test_rows,
            generator,
            max_tokens=config.planner.max_tokens,
            graphs_by_id=test_graphs,
        )
        progress.advance("evaluated release eval split")
        eval_dir.mkdir(parents=True, exist_ok=True)
        write_json(eval_dir / "validation_metrics.json", validation_metrics)
        write_json(eval_dir / "release_eval_metrics.json", test_metrics)
        write_jsonl(eval_dir / "validation_predictions.jsonl", validation_predictions)
        write_jsonl(eval_dir / "release_eval_predictions.jsonl", test_predictions)
        summary["validation"] = validation_metrics
        summary["release_eval"] = test_metrics
        write_json(eval_dir / "summary.json", summary)
        progress.advance("wrote planner evaluation artifacts")
        return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the local planner with either the MLX fine-tuning stack or an LM Studio-served model."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to the fine-tuning JSON config.")
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Evaluate the base planner model without loading any adapter weights.",
    )
    parser.add_argument(
        "--backend",
        choices=("mlx", "lmstudio"),
        default="mlx",
        help="Planner generation backend: the MLX fine-tuning stack or an LM Studio-served model.",
    )
    parser.add_argument(
        "--lmstudio-model",
        type=str,
        default=None,
        help="LM Studio model name/ID to evaluate when --backend lmstudio.",
    )
    parser.add_argument(
        "--lmstudio-base-url",
        type=str,
        default="http://localhost:1234/v1",
        help="Base URL for LM Studio's OpenAI-compatible API when --backend lmstudio.",
    )
    parser.add_argument(
        "--lmstudio-api-key",
        type=str,
        default=None,
        help="API key for LM Studio's OpenAI-compatible API when --backend lmstudio.",
    )
    parser.add_argument("--json", action="store_true", help="Print the final summary as compact JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = evaluate_planner(
        args.config,
        base_only=args.base_only,
        backend=args.backend,
        lmstudio_model=args.lmstudio_model,
        lmstudio_base_url=args.lmstudio_base_url,
        lmstudio_api_key=args.lmstudio_api_key,
    )
    print(compact_json(summary) if args.json else render_planner_eval_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
