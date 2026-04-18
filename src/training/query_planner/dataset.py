from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from runtime.query_planner import QueryPlanEnvelope, QueryPlanPayload, compile_query_plan

from .graphs import (
    SyntheticCompany,
    all_offerings,
    build_synthetic_company_graphs,
    descendant_offering_names,
    evaluate_query_plan,
    root_offerings_with_children,
    segment_revenue_models,
)

RouteLabel = Literal["local_safe", "strong_model_candidate", "refuse"]
SplitName = Literal["train", "validation", "release_eval"]

DEFAULT_ROUTE_TARGETS_BY_SPLIT: dict[SplitName, dict[RouteLabel, int]] = {
    "train": {
        "local_safe": 5000,
        "strong_model_candidate": 1500,
        "refuse": 1500,
    },
    "validation": {
        "local_safe": 750,
        "strong_model_candidate": 225,
        "refuse": 225,
    },
    "release_eval": {
        "local_safe": 900,
        "strong_model_candidate": 450,
        "refuse": 450,
    },
}

LOCAL_SAFE_FAMILY_TARGETS_TRAIN: dict[str, int] = {
    "companies_list": 40,
    "segments_by_company": 210,
    "offerings_by_company": 180,
    "offerings_by_segment": 270,
    "companies_by_segment_filters": 600,
    "segments_by_segment_filters": 600,
    "companies_by_cross_segment_filters": 700,
    "descendant_offerings_by_root": 350,
    "companies_by_descendant_revenue": 350,
    "companies_by_place": 220,
    "segments_by_place_and_segment_filters": 380,
    "companies_by_partner": 200,
    "boolean_exists": 400,
    "count_aggregate": 350,
    "ranking_topk": 150,
}

LOCAL_SAFE_BUCKET_TARGETS_TRAIN: dict[str, int] = {
    "inventory": 700,
    "same_segment": 1200,
    "cross_segment": 700,
    "hierarchy": 700,
    "geography": 600,
    "partner": 200,
    "boolean": 400,
    "count": 350,
    "ranking": 150,
}

STRONG_MODEL_CANDIDATE_FAMILY_WEIGHTS: dict[str, int] = {
    "common_offerings_between_segments": 3,
    "unique_offerings_to_segment": 3,
    "compare_segments_by_customer_types": 2,
    "compare_segments_by_channels": 2,
    "compare_segments_by_offering_count": 2,
    "why_segment_matches": 2,
    "compare_companies_by_customer_types": 2,
    "compare_companies_by_channels": 2,
    "weighted_ranking_request": 1,
}

REFUSAL_REASON_WEIGHTS: dict[str, int] = {
    "unsupported_schema": 1,
    "unsupported_metric": 1,
    "unsupported_time": 1,
    "ambiguous_closed_label": 1,
    "ambiguous_request": 1,
    "write_request": 1,
    "beyond_local_coverage": 1,
}

GRAPH_IDS_BY_SPLIT: dict[SplitName, tuple[str, ...]] = {
    "train": ("aurora", "redwood", "lattice"),
    "validation": ("nimbus",),
    "release_eval": ("vector",),
}

SPLIT_SEED_OFFSETS: dict[SplitName, int] = {
    "train": 0,
    "validation": 101,
    "release_eval": 202,
}

CUSTOMER_TYPE_SURFACES: dict[str, tuple[str, ...]] = {
    "consumers": ("consumers", "end users", "retail consumers"),
    "small businesses": ("small businesses", "SMBs", "smaller companies"),
    "mid-market companies": ("mid-market companies", "mid-market firms", "midsize companies"),
    "large enterprises": ("large enterprises", "enterprise customers", "big enterprises"),
    "developers": ("developers", "software builders", "app teams"),
    "IT professionals": ("IT professionals", "IT teams", "technology administrators"),
    "government agencies": ("government agencies", "public-sector bodies", "government offices"),
    "educational institutions": ("educational institutions", "schools and universities", "education organizations"),
    "healthcare organizations": ("healthcare organizations", "healthcare firms", "health systems"),
    "financial services firms": ("financial services firms", "financial institutions", "banks and insurers"),
    "manufacturers": ("manufacturers", "industrial companies", "production businesses"),
    "retailers": ("retailers", "merchant businesses", "retail companies"),
}

CHANNEL_SURFACES: dict[str, tuple[str, ...]] = {
    "direct sales": ("direct sales", "direct account teams", "its own sales force"),
    "online": ("online", "digital channels", "self-service web channels"),
    "retail": ("retail", "retail storefronts", "store networks"),
    "distributors": ("distributors", "distribution partners", "wholesale distributors"),
    "resellers": ("resellers", "reseller partners", "channel resellers"),
    "OEMs": ("OEMs", "device makers", "original equipment manufacturers"),
    "system integrators": ("system integrators", "integration partners", "SI partners"),
    "managed service providers": ("managed service providers", "MSPs", "managed-service partners"),
    "marketplaces": ("marketplaces", "digital marketplaces", "third-party marketplaces"),
}

REVENUE_MODEL_SURFACES: dict[str, tuple[str, ...]] = {
    "subscription": ("subscription", "recurring subscriptions", "recurring access fees"),
    "advertising": ("advertising", "ad sales", "advertiser spend"),
    "licensing": ("licensing", "license fees", "software licensing"),
    "consumption-based": ("consumption-based", "usage-based pricing", "pay-as-you-go pricing"),
    "hardware sales": ("hardware sales", "device sales", "equipment sales"),
    "service fees": ("service fees", "services revenue", "professional-services fees"),
    "transaction fees": ("transaction fees", "take rates", "transaction-based fees"),
}

PLACE_SURFACES: dict[str, tuple[str, ...]] = {
    "Worldwide": ("Worldwide", "global", "the world"),
    "United States": ("United States", "the U.S.", "America"),
    "United Kingdom": ("United Kingdom", "the UK", "Britain"),
    "Europe": ("Europe", "the European market", "European geographies"),
    "EMEA": ("EMEA", "Europe, the Middle East, and Africa", "the EMEA region"),
    "APAC": ("APAC", "Asia Pacific", "the Asia-Pacific region"),
    "Latin America": ("Latin America", "LatAm", "the Latin American market"),
    "Italy": ("Italy", "the Italian market", "Italian operations"),
    "Germany": ("Germany", "the German market", "German operations"),
    "Japan": ("Japan", "the Japanese market", "Japanese operations"),
    "Canada": ("Canada", "Canadian operations", "the Canadian market"),
    "Australia": ("Australia", "the Australian market", "Australian operations"),
    "France": ("France", "the French market", "French operations"),
    "Mexico": ("Mexico", "the Mexican market", "Mexican operations"),
}

SURFACE_WRAPPERS_BY_SPLIT: dict[SplitName, tuple[tuple[str, str | None], ...]] = {
    "train": (
        ("direct", None),
        ("graph", "In the graph, {body}"),
        ("dataset", "Within this dataset, {body}"),
        ("kg", "From the knowledge graph, {body}"),
        ("available", "Using the available graph, {body}"),
        ("snapshot", "From the KG snapshot, {body}"),
        ("scope", "Looking at this graph only, {body}"),
    ),
    "validation": (
        ("direct", None),
        ("dataset", "Within this dataset, {body}"),
        ("kg", "From the available knowledge graph, {body}"),
        ("snapshot", "From this KG snapshot, {body}"),
        ("scope", "Looking only at the graph, {body}"),
    ),
    "release_eval": (
        ("scope", "Using only this graph, {body}"),
        ("alone", "Based only on the available knowledge graph, {body}"),
        ("snapshot", "From this KG snapshot alone, {body}"),
        ("outside", "Without bringing in outside information, {body}"),
        ("provided", "Using just the provided graph, {body}"),
        ("supplied", "Looking only at the supplied dataset, {body}"),
    ),
}


@dataclass(frozen=True)
class CanonicalCase:
    case_id: str
    route_label: RouteLabel
    family: str
    bucket: str
    plan: QueryPlanEnvelope
    source_graph_ids: tuple[str, ...]
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetExample:
    case_id: str
    template_id: str
    variant_id: str
    question: str
    target: dict[str, Any]
    route_label: RouteLabel
    family: str
    gold_cypher: str | None
    gold_params: dict[str, Any]
    gold_rows: list[dict[str, Any]]
    metadata: dict[str, Any]


def _stable_id(prefix: str, *parts: str) -> str:
    payload = "|".join(parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _scale_targets(targets: dict[str, int], total: int) -> dict[str, int]:
    base_total = sum(targets.values())
    if total <= 0 or base_total <= 0:
        return {label: 0 for label in targets}

    exact = {label: (value * total) / base_total for label, value in targets.items()}
    scaled = {label: int(amount) for label, amount in exact.items()}
    remainder = total - sum(scaled.values())
    if remainder > 0:
        ranked = sorted(targets, key=lambda label: (exact[label] - scaled[label], targets[label]), reverse=True)
        for label in ranked[:remainder]:
            scaled[label] += 1
    return scaled


def _natural_join(values: Iterable[str]) -> str:
    items = [value for value in values if value]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _graph_source_id(graph_ids: Iterable[str]) -> str:
    return "+".join(sorted(set(graph_ids)))


def _style_indices(split: SplitName) -> tuple[int, ...]:
    if split == "train":
        return (0,)
    if split == "validation":
        return (0, 1)
    return (1, 2)


def _surface_value(value: str, surfaces: dict[str, tuple[str, ...]], style_index: int) -> str:
    options = surfaces.get(value, (value,))
    return options[min(style_index, len(options) - 1)]


def _surface_join(values: list[str], surfaces: dict[str, tuple[str, ...]], style_index: int) -> str:
    return _natural_join(_surface_value(value, surfaces, style_index) for value in values)


def _normalized_target_key(plan: QueryPlanEnvelope) -> str:
    return json.dumps(plan.model_dump(mode="json", exclude_none=True), sort_keys=True, separators=(",", ":"))


def _lower_initial(text: str) -> str:
    if not text:
        return text
    return text[:1].lower() + text[1:]


def _expand_surface_variants(base_id: str, templates: tuple[str, ...], split: SplitName) -> list[tuple[str, str, str]]:
    expanded: list[tuple[str, str, str]] = []
    for template_index, question in enumerate(templates, start=1):
        body = _lower_initial(question.strip())
        for wrapper_id, wrapper in SURFACE_WRAPPERS_BY_SPLIT[split]:
            rendered = question if wrapper is None else wrapper.format(body=body)
            expanded.append((f"{base_id}-t{template_index:02d}", wrapper_id, rendered))
    return _dedupe_surfaces(expanded)


def _selection_key(seed: int, case_id: str, template_id: str, variant_id: str, question: str) -> str:
    payload = f"{seed}|{case_id}|{template_id}|{variant_id}|{question}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _payload_filter_count(payload: QueryPlanPayload) -> int:
    return sum(
        len(values)
        for values in (
            payload.companies,
            payload.segments,
            payload.offerings,
            payload.customer_types,
            payload.channels,
            payload.revenue_models,
            payload.places,
            payload.partners,
        )
    )


def _payload_boolean_depth(payload: QueryPlanPayload) -> int:
    semantic_count = len(payload.customer_types) + len(payload.channels) + len(payload.offerings) + len(payload.revenue_models)
    return max(0, semantic_count - 1)


def _base_metadata(case: CanonicalCase) -> dict[str, Any]:
    payload = case.plan.payload or QueryPlanPayload()
    metadata = {
        "route_label": case.route_label,
        "family": case.family,
        "bucket": case.bucket,
        "filter_count": _payload_filter_count(payload),
        "boolean_depth": _payload_boolean_depth(payload),
        "has_geography": bool(payload.places),
        "has_descendant_offering": payload.hierarchy_mode == "descendant" or case.family in {
            "descendant_offerings_by_root",
            "companies_by_descendant_revenue",
        },
        "has_aggregation": case.family in {"count_aggregate", "ranking_topk"},
        "source_graph_id": _graph_source_id(case.source_graph_ids),
        "source_graph_ids": list(case.source_graph_ids),
    }
    if not case.plan.answerable and case.plan.reason is not None:
        metadata["refusal_reason"] = case.plan.reason
    metadata.update(case.context.get("metadata", {}))
    return metadata


def _make_local_safe_case(
    *,
    family: str,
    bucket: str,
    payload: QueryPlanPayload,
    source_graph_ids: Iterable[str],
    context: dict[str, Any] | None = None,
) -> CanonicalCase | None:
    plan = QueryPlanEnvelope(answerable=True, family=family, payload=payload)
    compiled = compile_query_plan(plan)
    if not compiled.answerable:
        return None
    normalized_target = _normalized_target_key(plan)
    return CanonicalCase(
        case_id=_stable_id("case", "local_safe", family, normalized_target),
        route_label="local_safe",
        family=family,
        bucket=bucket,
        plan=plan,
        source_graph_ids=tuple(sorted(set(source_graph_ids))),
        context=context or {},
    )


def _make_refusal_case(
    *,
    family: str,
    bucket: str,
    reason: str,
    source_graph_ids: Iterable[str],
    context: dict[str, Any],
) -> CanonicalCase:
    context_key = json.dumps(context, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return CanonicalCase(
        case_id=_stable_id("case", "refuse", family, reason, context_key),
        route_label="refuse",
        family=family,
        bucket=bucket,
        plan=QueryPlanEnvelope(answerable=False, reason=reason),
        source_graph_ids=tuple(sorted(set(source_graph_ids))),
        context=context,
    )


def _make_strong_candidate_case(
    *,
    family: str,
    source_graph_ids: Iterable[str],
    context: dict[str, Any],
) -> CanonicalCase:
    context_key = json.dumps(context, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return CanonicalCase(
        case_id=_stable_id("case", "strong_model_candidate", family, context_key),
        route_label="strong_model_candidate",
        family=family,
        bucket="strong_model_candidate",
        plan=QueryPlanEnvelope(answerable=False, reason="beyond_local_coverage"),
        source_graph_ids=tuple(sorted(set(source_graph_ids))),
        context=context,
    )


def _dedupe_local_safe_cases(cases: list[CanonicalCase]) -> list[CanonicalCase]:
    unique_by_target: dict[str, CanonicalCase] = {}
    for case in cases:
        target_key = _normalized_target_key(case.plan)
        unique_by_target.setdefault(target_key, case)
    return [unique_by_target[key] for key in sorted(unique_by_target)]


def _dedupe_case_ids(cases: list[CanonicalCase]) -> list[CanonicalCase]:
    unique_by_case_id: dict[str, CanonicalCase] = {}
    for case in cases:
        unique_by_case_id.setdefault(case.case_id, case)
    return [unique_by_case_id[key] for key in sorted(unique_by_case_id)]


def _company_subsets(companies: tuple[SyntheticCompany, ...], *, max_size: int = 3) -> list[tuple[SyntheticCompany, ...]]:
    ordered = sorted(companies, key=lambda company: company.name)
    subsets: list[tuple[SyntheticCompany, ...]] = []
    for size in range(1, min(max_size, len(ordered)) + 1):
        if size == 1:
            subsets.extend((company,) for company in ordered)
            continue
        for index, company in enumerate(ordered):
            for other_index in range(index + 1, len(ordered)):
                subset = [company, ordered[other_index]]
                if size == 2:
                    subsets.append(tuple(subset))
        if size == 3 and len(ordered) >= 3:
            subsets.append(tuple(ordered[:3]))
    return subsets


def _all_places(companies: tuple[SyntheticCompany, ...]) -> list[str]:
    return sorted({place for company in companies for place in company.places})


def _all_partners(companies: tuple[SyntheticCompany, ...]) -> list[str]:
    return sorted({partner for company in companies for partner in company.partners})


def _render_filter_clause(payload: QueryPlanPayload, split: SplitName, style_index: int) -> str:
    parts: list[str] = []
    if payload.customer_types:
        parts.append(f"serve {_surface_join(payload.customer_types, CUSTOMER_TYPE_SURFACES, style_index)}")
    if payload.channels:
        parts.append(f"sell through {_surface_join(payload.channels, CHANNEL_SURFACES, style_index)}")
    if payload.offerings:
        parts.append(f"include {_natural_join(payload.offerings)}")
    if payload.revenue_models:
        parts.append(f"monetize via {_surface_join(payload.revenue_models, REVENUE_MODEL_SURFACES, style_index)}")
    if payload.places:
        parts.append(f"operate in {_surface_join(payload.places, PLACE_SURFACES, style_index)}")
    if payload.partners:
        parts.append(f"partner with {_natural_join(payload.partners)}")
    return " and ".join(parts)


def _render_base_predicate(case: CanonicalCase, split: SplitName) -> list[tuple[str, str, str]]:
    payload = case.plan.payload or QueryPlanPayload()
    base_family = payload.base_family
    if base_family is None:
        return []
    nested_payload = QueryPlanPayload(
        companies=payload.companies,
        segments=payload.segments,
        offerings=payload.offerings,
        customer_types=payload.customer_types,
        channels=payload.channels,
        revenue_models=payload.revenue_models,
        places=payload.places,
        partners=payload.partners,
        binding_scope=payload.binding_scope,
        hierarchy_mode=payload.hierarchy_mode,
        limit=payload.limit,
    )
    nested_case = CanonicalCase(
        case_id=case.case_id,
        route_label="local_safe",
        family=base_family,
        bucket=case.bucket,
        plan=QueryPlanEnvelope(answerable=True, family=base_family, payload=nested_payload),
        source_graph_ids=case.source_graph_ids,
        context=case.context,
    )
    return _render_local_safe_surfaces(nested_case, split)


def _render_local_safe_surfaces(case: CanonicalCase, split: SplitName) -> list[tuple[str, str, str]]:
    payload = case.plan.payload or QueryPlanPayload()
    family = case.family
    surfaces: list[tuple[str, str, str]] = []
    company_names = _natural_join(payload.companies)
    for style_index in _style_indices(split):
        style_id = f"s{style_index}"
        if family == "companies_list":
            if payload.limit:
                templates = (
                    f"Name up to {payload.limit} companies in the graph.",
                    f"List as many as {payload.limit} companies represented in the knowledge graph.",
                    f"Give me {payload.limit} companies from the dataset.",
                    f"What are {payload.limit} companies currently represented here?",
                    f"Identify {payload.limit} companies in the graph.",
                    f"Show {payload.limit} companies that appear in the knowledge graph.",
                    f"Return {payload.limit} companies from the graph.",
                    f"Which {payload.limit} companies are represented in the dataset?",
                )
            else:
                templates = (
                    "Which companies are in the graph?",
                    "List the companies represented in the knowledge graph.",
                    "Name the companies in the dataset.",
                    "What companies appear in the graph?",
                    "Identify the companies represented here.",
                    "Which companies are currently represented in the knowledge graph?",
                    "Show the companies present in the graph.",
                    "What companies are available in this graph?",
                )
            surfaces.extend(_expand_surface_variants(f"{family}-{style_id}", templates, split))
            continue

        if family == "segments_by_company":
            noun = "business segments" if split != "release_eval" else "segment portfolio"
            templates = (
                f"What are the {noun} of {company_names}?",
                f"List the {noun} for {company_names}.",
                f"Which {noun} belong to {company_names}?",
                f"Identify the {noun} associated with {company_names}.",
                f"Show the {noun} tied to {company_names}.",
                f"Name the {noun} that sit under {company_names}.",
            )
            if payload.limit:
                templates += (
                    f"List up to {payload.limit} business segments for {company_names}.",
                    f"Name as many as {payload.limit} segments belonging to {company_names}.",
                )
            surfaces.extend(_expand_surface_variants(f"{family}-{style_id}", templates, split))
            continue

        if family == "offerings_by_company":
            templates = (
                f"Which offerings does {company_names} have?",
                f"List the offerings for {company_names}.",
                f"What offerings belong to {company_names}?",
                f"Identify the offerings associated with {company_names}.",
                f"Show the offering inventory for {company_names}.",
                f"Name the offerings attached to {company_names}.",
            )
            if payload.limit:
                templates += (
                    f"List up to {payload.limit} offerings for {company_names}.",
                    f"Name as many as {payload.limit} offerings owned by {company_names}.",
                )
            surfaces.extend(_expand_surface_variants(f"{family}-{style_id}", templates, split))
            continue

        if family == "offerings_by_segment":
            segment_names = _natural_join(payload.segments)
            templates = (
                f"What offerings sit in the {segment_names} segment of {company_names}?",
                f"List the offerings in {company_names}'s {segment_names} segment.",
                f"Which offerings belong to the {segment_names} segment at {company_names}?",
                f"Show the offering inventory for the {segment_names} segment of {company_names}.",
                f"Name the offerings associated with the {segment_names} segment at {company_names}.",
                f"Identify the offerings that sit under {company_names}'s {segment_names} segment.",
            )
            if payload.limit:
                templates += (
                    f"List up to {payload.limit} offerings in the {segment_names} segment of {company_names}.",
                    f"Give me as many as {payload.limit} offerings for {company_names}'s {segment_names} segment.",
                )
            surfaces.extend(_expand_surface_variants(f"{family}-{style_id}", templates, split))
            continue

        if family == "companies_by_segment_filters":
            clause = _render_filter_clause(payload, split, style_index)
            templates = (
                f"Which companies {clause}?",
                f"List the companies that {clause}.",
                f"Name the companies that {clause}.",
                f"Identify companies that {clause}.",
                f"What companies {clause}?",
            )
            surfaces.extend(_expand_surface_variants(f"{family}-{style_id}", templates, split))
            continue

        if family == "segments_by_segment_filters":
            clause = _render_filter_clause(payload, split, style_index)
            templates = (
                f"Which business segments {clause}?",
                f"List the business segments that {clause}.",
                f"Name the segments that {clause}.",
                f"Identify business segments that {clause}.",
                f"What segments {clause}?",
            )
            surfaces.extend(_expand_surface_variants(f"{family}-{style_id}", templates, split))
            continue

        if family == "companies_by_cross_segment_filters":
            clause = _render_filter_clause(payload, split, style_index)
            templates = (
                f"Which companies {clause} across different segments?",
                f"List the companies that, across their portfolio, {clause}.",
                f"Name the companies that {clause} across multiple segments.",
                f"Which companies satisfy {clause} at the business-wide level?",
                f"Identify companies that {clause} even if the evidence is split across segments.",
            )
            surfaces.extend(_expand_surface_variants(f"{family}-{style_id}", templates, split))
            continue

        if family == "descendant_offerings_by_root":
            root_name = payload.offerings[0]
            company_phrase = company_names if company_names else "the companies in scope"
            if payload.limit:
                templates = (
                    f"Which offerings sit under the {root_name} family at {company_phrase}, up to {payload.limit} results?",
                    f"List up to {payload.limit} descendant offerings of {root_name} for {company_phrase}.",
                    f"What offerings descend from {root_name} at {company_phrase}, limited to {payload.limit}?",
                    f"Name as many as {payload.limit} offerings in the {root_name} family at {company_phrase}.",
                    f"Identify up to {payload.limit} descendant offerings under {root_name} for {company_phrase}.",
                )
            else:
                templates = (
                    f"Which offerings sit under the {root_name} family at {company_phrase}?",
                    f"List the descendant offerings of {root_name} for {company_phrase}.",
                    f"What offerings descend from {root_name} at {company_phrase}?",
                    f"Name the offerings in the {root_name} family at {company_phrase}.",
                    f"Identify the descendant offerings under {root_name} for {company_phrase}.",
                )
            surfaces.extend(_expand_surface_variants(f"{family}-{style_id}", templates, split))
            continue

        if family == "companies_by_descendant_revenue":
            root_name = payload.offerings[0]
            revenue_phrase = _surface_join(payload.revenue_models, REVENUE_MODEL_SURFACES, style_index)
            if payload.limit:
                templates = (
                    f"Which companies monetize descendant offerings of {root_name} via {revenue_phrase}, up to {payload.limit} results?",
                    f"List up to {payload.limit} companies whose {root_name} family uses {revenue_phrase}.",
                    f"What companies monetize the {root_name} offering family through {revenue_phrase}, limited to {payload.limit}?",
                    f"Identify up to {payload.limit} companies whose descendants of {root_name} use {revenue_phrase}.",
                    f"Name as many as {payload.limit} companies that monetize {root_name} descendants with {revenue_phrase}.",
                )
            else:
                templates = (
                    f"Which companies monetize descendant offerings of {root_name} via {revenue_phrase}?",
                    f"List the companies whose {root_name} family uses {revenue_phrase}.",
                    f"What companies monetize the {root_name} offering family through {revenue_phrase}?",
                    f"Identify companies whose descendants of {root_name} use {revenue_phrase}.",
                    f"Name the companies that monetize {root_name} descendants with {revenue_phrase}.",
                )
            surfaces.extend(_expand_surface_variants(f"{family}-{style_id}", templates, split))
            continue

        if family == "companies_by_place":
            place_phrase = _surface_join(payload.places, PLACE_SURFACES, style_index)
            templates = (
                f"Which companies operate in {place_phrase}?",
                f"List the companies with business presence in {place_phrase}.",
                f"What companies have an operating footprint in {place_phrase}?",
                f"Name the companies active in {place_phrase}.",
                f"Identify companies operating in {place_phrase}.",
            )
            if payload.limit:
                templates += (
                    f"List up to {payload.limit} companies operating in {place_phrase}.",
                    f"Name as many as {payload.limit} companies with presence in {place_phrase}.",
                )
            surfaces.extend(_expand_surface_variants(f"{family}-{style_id}", templates, split))
            continue

        if family == "segments_by_place_and_segment_filters":
            place_phrase = _surface_join(payload.places, PLACE_SURFACES, style_index)
            clause = _render_filter_clause(
                QueryPlanPayload(
                    companies=payload.companies,
                    segments=payload.segments,
                    offerings=payload.offerings,
                    customer_types=payload.customer_types,
                    channels=payload.channels,
                    revenue_models=payload.revenue_models,
                    hierarchy_mode=payload.hierarchy_mode,
                ),
                split,
                style_index,
            )
            templates = (
                f"Which business segments of companies operating in {place_phrase} {clause}?",
                f"List the segments for companies in {place_phrase} that {clause}.",
                f"Name the segments at companies with presence in {place_phrase} that {clause}.",
                f"Identify business segments operating in {place_phrase} that {clause}.",
                f"What segments belong to companies in {place_phrase} and {clause}?",
            )
            surfaces.extend(_expand_surface_variants(f"{family}-{style_id}", templates, split))
            continue

        if family == "companies_by_partner":
            partner_phrase = _natural_join(payload.partners)
            templates = (
                f"Which companies partner with {partner_phrase}?",
                f"List the companies that have named partnerships with {partner_phrase}.",
                f"What companies are partnered with {partner_phrase}?",
                f"Identify the companies that partner with {partner_phrase}.",
                f"Name the companies tied to {partner_phrase} through partnerships.",
            )
            if payload.limit:
                templates += (
                    f"List up to {payload.limit} companies that partner with {partner_phrase}.",
                    f"Name as many as {payload.limit} companies partnered with {partner_phrase}.",
                )
            surfaces.extend(_expand_surface_variants(f"{family}-{style_id}", templates, split))
            continue

        if family == "boolean_exists":
            predicate_surfaces = _render_base_predicate(case, split)
            for template_id, variant_id, question in predicate_surfaces:
                lowered = question.rstrip("?").rstrip(".")
                rewritten = lowered
                rewritten = rewritten.replace("Which companies", "Are there companies", 1)
                rewritten = rewritten.replace("List the companies that", "Are there companies that", 1)
                rewritten = rewritten.replace("Name the companies that", "Are there companies that", 1)
                rewritten = rewritten.replace("Which business segments", "Are there business segments", 1)
                rewritten = rewritten.replace("List the business segments that", "Are there business segments that", 1)
                rewritten = rewritten.replace("What companies", "Are there companies that", 1)
                if rewritten == lowered:
                    rewritten = f"Is there a match where {lowered[0].lower()}{lowered[1:]}"
                surfaces.append((f"{family}-{template_id}", f"{style_id}-{variant_id}", rewritten + "?"))
            continue

        if family == "count_aggregate":
            aggregate_spec = payload.aggregate_spec or {}
            subject = aggregate_spec.get("count_target", "company")
            noun = {"company": "companies", "segment": "business segments", "offering": "offerings"}[subject]
            base_family = payload.base_family or aggregate_spec.get("base_family")
            if base_family == "offerings_by_company":
                templates = (
                    f"How many {noun} does {company_names} have?",
                    f"Count the {noun} for {company_names}.",
                    f"What is the {noun} count for {company_names}?",
                    f"How many {noun} belong to {company_names}?",
                )
                surfaces.extend(_expand_surface_variants(f"{family}-{style_id}", templates, split))
                continue

            clause = _render_filter_clause(payload, split, style_index)
            templates = (
                f"How many {noun} {clause}?",
                f"Count the {noun} that {clause}.",
                f"What is the {noun} count for cases that {clause}?",
                f"How many {noun} match the condition to {clause}?",
            )
            surfaces.extend(_expand_surface_variants(f"{family}-{style_id}", templates, split))
            continue

        if family == "ranking_topk":
            aggregate_spec = payload.aggregate_spec or {}
            metric = aggregate_spec.get("ranking_metric")
            limit = payload.limit or 5
            if metric == "customer_type_by_company_count":
                templates = (
                    f"Top {limit} customer types by company count.",
                    f"Which {limit} customer types appear across the most companies?",
                    f"Name the top {limit} customer types by company coverage.",
                    f"What are the top {limit} customer types by company count?",
                )
            elif metric == "channel_by_segment_count":
                templates = (
                    f"Top {limit} channels by segment count.",
                    f"Which {limit} channels appear across the most segments?",
                    f"Name the top {limit} channels by segment usage.",
                    f"What are the top {limit} channels by segment count?",
                )
            elif metric == "revenue_model_by_company_count":
                templates = (
                    f"Top {limit} revenue models by company count.",
                    f"Which {limit} revenue models show up across the most companies?",
                    f"Name the top {limit} revenue models by company footprint.",
                    f"What are the top {limit} revenue models by company count?",
                )
            else:
                clause = _render_filter_clause(payload, split, style_index)
                templates = (
                    f"Top {limit} companies by matched segment count for segments that {clause}.",
                    f"Which {limit} companies have the most matching segments that {clause}?",
                    f"Name the top {limit} companies by segment-match count where segments {clause}.",
                    f"What are the top {limit} companies by matched segments that {clause}?",
                )
            surfaces.extend(_expand_surface_variants(f"{family}-{style_id}", templates, split))
            continue

    return _dedupe_surfaces(surfaces)


def _render_strong_candidate_surfaces(case: CanonicalCase, split: SplitName) -> list[tuple[str, str, str]]:
    context = case.context
    family = case.family
    templates: tuple[str, ...]
    if family == "common_offerings_between_segments":
        templates = (
            f"What offerings do {context['segment_a']} and {context['segment_b']} have in common at {context['company']}?",
            f"Which offerings appear in both {context['segment_a']} and {context['segment_b']} at {context['company']}?",
            f"List the shared offerings between {context['segment_a']} and {context['segment_b']} at {context['company']}.",
            f"Identify the overlapping offerings for {context['segment_a']} and {context['segment_b']} at {context['company']}.",
            f"Show the offerings common to {context['segment_a']} and {context['segment_b']} at {context['company']}.",
            f"Name the offerings that both {context['segment_a']} and {context['segment_b']} carry at {context['company']}.",
        )
    elif family == "unique_offerings_to_segment":
        templates = (
            f"Which offerings are unique to {context['segment_a']} compared with {context['segment_b']} at {context['company']}?",
            f"List the offerings that only {context['segment_a']} has, compared with {context['segment_b']}, at {context['company']}.",
            f"What offerings belong to {context['segment_a']} but not {context['segment_b']} at {context['company']}?",
            f"Identify the offerings unique to {context['segment_a']} versus {context['segment_b']} at {context['company']}.",
            f"Show the offering difference between {context['segment_a']} and {context['segment_b']} at {context['company']}.",
            f"Name the offerings exclusive to {context['segment_a']} relative to {context['segment_b']} at {context['company']}.",
        )
    elif family == "compare_segments_by_customer_types":
        templates = (
            f"Compare {context['segment_a']} and {context['segment_b']} at {context['company']} by the customer types they serve.",
            f"How do {context['segment_a']} and {context['segment_b']} differ in the customer types they serve at {context['company']}?",
            f"Show a comparison of {context['segment_a']} and {context['segment_b']} at {context['company']} by customer type coverage.",
            f"Contrast {context['segment_a']} with {context['segment_b']} at {context['company']} on served customer types.",
            f"What is the customer-type comparison between {context['segment_a']} and {context['segment_b']} at {context['company']}?",
            f"Compare the customer audiences of {context['segment_a']} and {context['segment_b']} at {context['company']}.",
        )
    elif family == "compare_segments_by_channels":
        templates = (
            f"Compare {context['segment_a']} and {context['segment_b']} at {context['company']} by the channels they use.",
            f"How do {context['segment_a']} and {context['segment_b']} differ in their sales channels at {context['company']}?",
            f"Show a channel comparison for {context['segment_a']} versus {context['segment_b']} at {context['company']}.",
            f"Contrast {context['segment_a']} with {context['segment_b']} at {context['company']} on go-to-market channels.",
            f"What is the channel mix comparison between {context['segment_a']} and {context['segment_b']} at {context['company']}?",
            f"Compare the selling channels of {context['segment_a']} and {context['segment_b']} at {context['company']}.",
        )
    elif family == "compare_segments_by_offering_count":
        templates = (
            f"Which of {context['segment_a']} or {context['segment_b']} at {context['company']} has more offerings?",
            f"Compare {context['segment_a']} and {context['segment_b']} at {context['company']} by offering count.",
            f"Show which segment has more offerings between {context['segment_a']} and {context['segment_b']} at {context['company']}.",
            f"What is the offering-count comparison between {context['segment_a']} and {context['segment_b']} at {context['company']}?",
            f"Tell me whether {context['segment_a']} or {context['segment_b']} carries more offerings at {context['company']}.",
            f"Which segment has the larger offering inventory: {context['segment_a']} or {context['segment_b']} at {context['company']}?",
        )
    elif family == "why_segment_matches":
        templates = (
            f"Why does {context['segment']} at {context['company']} match a request about {context['focus']}?",
            f"Explain why {context['segment']} at {context['company']} is a match for {context['focus']}.",
            f"What makes {context['segment']} at {context['company']} relevant to {context['focus']}?",
            f"Why should {context['segment']} at {context['company']} count as a match for {context['focus']}?",
            f"Give the rationale for matching {context['segment']} at {context['company']} to {context['focus']}.",
            f"Explain the match between {context['segment']} at {context['company']} and {context['focus']}.",
        )
    elif family == "compare_companies_by_customer_types":
        templates = (
            f"Compare {context['company_a']} and {context['company_b']} by the customer types they serve.",
            f"How do {context['company_a']} and {context['company_b']} differ in served customer types?",
            f"Show a customer-type comparison for {context['company_a']} versus {context['company_b']}.",
            f"Contrast {context['company_a']} with {context['company_b']} on customer coverage.",
            f"What is the customer-type comparison between {context['company_a']} and {context['company_b']}?",
            f"Compare the served audiences of {context['company_a']} and {context['company_b']}.",
            f"Which customer types separate {context['company_a']} from {context['company_b']}?",
            f"Map the overlap and differences in served customer types for {context['company_a']} and {context['company_b']}.",
            f"Lay out the customer-type coverage of {context['company_a']} versus {context['company_b']}.",
        )
    elif family == "compare_companies_by_channels":
        templates = (
            f"Compare {context['company_a']} and {context['company_b']} by the channels they use.",
            f"How do {context['company_a']} and {context['company_b']} differ in sales channels?",
            f"Show a channel comparison for {context['company_a']} versus {context['company_b']}.",
            f"Contrast {context['company_a']} with {context['company_b']} on go-to-market channels.",
            f"What is the channel comparison between {context['company_a']} and {context['company_b']}?",
            f"Compare the selling channels of {context['company_a']} and {context['company_b']}.",
            f"Which channels distinguish {context['company_a']} from {context['company_b']}?",
            f"Map the overlap and differences in go-to-market channels for {context['company_a']} and {context['company_b']}.",
            f"Lay out the channel mix of {context['company_a']} versus {context['company_b']}.",
        )
    else:
        templates = (
            f"Rank the segments of {context['company']} using a custom weighted score that prioritizes {context['focus']}.",
            f"Give me a weighted ranking of {context['company']}'s segments, giving extra importance to {context['focus']}.",
            f"Order {context['company']}'s segments by a custom score that overweights {context['focus']}.",
            f"Produce a weighted ranking for {context['company']}'s segments with extra weight on {context['focus']}.",
            f"Rank the segments of {context['company']} using a custom formula that emphasizes {context['focus']}.",
            f"Show a custom weighted leaderboard for {context['company']}'s segments based on {context['focus']}.",
        )
    return _expand_surface_variants(family, templates, split)


def _render_refusal_surfaces(case: CanonicalCase, split: SplitName) -> list[tuple[str, str, str]]:
    reason = case.plan.reason or "beyond_local_coverage"
    context = case.context
    templates = tuple(context["templates"])
    return _expand_surface_variants(reason, templates, split)


def _dedupe_surfaces(surfaces: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    seen_questions: set[str] = set()
    unique: list[tuple[str, str, str]] = []
    for template_id, variant_id, question in surfaces:
        normalized_question = question.strip().casefold()
        if normalized_question in seen_questions:
            continue
        seen_questions.add(normalized_question)
        unique.append((template_id, variant_id, question))
    return unique


def _surface_pool_for_case(case: CanonicalCase, split: SplitName) -> list[tuple[str, str, str]]:
    if case.route_label == "local_safe":
        return _render_local_safe_surfaces(case, split)
    if case.route_label == "strong_model_candidate":
        return _render_strong_candidate_surfaces(case, split)
    return _render_refusal_surfaces(case, split)


def _build_example(companies: tuple[SyntheticCompany, ...], case: CanonicalCase, template_id: str, variant_id: str, question: str) -> DatasetExample:
    compiled = compile_query_plan(case.plan)
    if case.route_label == "local_safe" and not compiled.answerable:
        raise ValueError(f"Local-safe example compiled to refusal for question: {question}")
    if case.route_label != "local_safe" and compiled.answerable:
        raise ValueError(f"Non-local example compiled to an answer for question: {question}")
    rows = evaluate_query_plan(companies, case.plan) if compiled.answerable else []
    return DatasetExample(
        case_id=case.case_id,
        template_id=template_id,
        variant_id=variant_id,
        question=question,
        target=case.plan.model_dump(mode="json", exclude_none=True),
        route_label=case.route_label,
        family=case.family,
        gold_cypher=compiled.cypher if compiled.answerable else None,
        gold_params=compiled.params if compiled.answerable else {},
        gold_rows=rows,
        metadata=_base_metadata(case),
    )


def _family_targets_for_split(train_targets: dict[str, int], total: int) -> dict[str, int]:
    return _scale_targets(train_targets, total)


def _materialize_family_examples(
    *,
    companies: tuple[SyntheticCompany, ...],
    family_cases: dict[str, list[CanonicalCase]],
    family_targets: dict[str, int],
    split: SplitName,
    seed: int,
    seen_questions: set[str],
    seen_pairs: set[str],
) -> list[DatasetExample]:
    selected: list[DatasetExample] = []
    for family, target_count in sorted(family_targets.items()):
        if target_count <= 0:
            continue
        pool: list[tuple[str, DatasetExample]] = []
        for case in family_cases.get(family, []):
            for template_id, variant_id, question in _surface_pool_for_case(case, split):
                example = _build_example(companies, case, template_id, variant_id, question)
                sort_key = _selection_key(seed, case.case_id, template_id, variant_id, question)
                pool.append((sort_key, example))
        pool.sort(key=lambda item: item[0])

        family_selected = 0
        for _, example in pool:
            normalized_question = example.question.strip().casefold()
            normalized_pair = f"{normalized_question}|{json.dumps(example.target, sort_keys=True, separators=(',', ':'))}"
            if normalized_question in seen_questions or normalized_pair in seen_pairs:
                continue
            selected.append(example)
            seen_questions.add(normalized_question)
            seen_pairs.add(normalized_pair)
            family_selected += 1
            if family_selected >= target_count:
                break
        if family_selected < target_count:
            raise ValueError(
                f"Could not materialize enough unique examples for {family!r} in split {split}: "
                f"needed {target_count}, got {family_selected}."
            )
    return selected


def _inventory_cases(companies: tuple[SyntheticCompany, ...]) -> dict[str, list[CanonicalCase]]:
    cases: dict[str, list[CanonicalCase]] = defaultdict(list)
    subsets = _company_subsets(companies)
    company_map = {company.name: company for company in companies}

    for limit in (None, 1, 2, 3, 5):
        case = _make_local_safe_case(
            family="companies_list",
            bucket="inventory",
            payload=QueryPlanPayload(limit=limit),
            source_graph_ids=[company.graph_id for company in companies],
        )
        if case is not None:
            cases["companies_list"].append(case)

    for subset in subsets:
        company_names = [company.name for company in subset]
        source_ids = [company.graph_id for company in subset]
        for limit in (None, 2, 4, 6):
            case = _make_local_safe_case(
                family="segments_by_company",
                bucket="inventory",
                payload=QueryPlanPayload(companies=company_names, limit=limit),
                source_graph_ids=source_ids,
            )
            if case is not None:
                cases["segments_by_company"].append(case)
            case = _make_local_safe_case(
                family="offerings_by_company",
                bucket="inventory",
                payload=QueryPlanPayload(companies=company_names, limit=limit),
                source_graph_ids=source_ids,
            )
            if case is not None:
                cases["offerings_by_company"].append(case)

    for company in companies:
        for segment in company.segments:
            for limit in (None, 2, 4, 6):
                case = _make_local_safe_case(
                    family="offerings_by_segment",
                    bucket="inventory",
                    payload=QueryPlanPayload(companies=[company.name], segments=[segment.name], limit=limit),
                    source_graph_ids=[company.graph_id],
                )
                if case is not None:
                    cases["offerings_by_segment"].append(case)

    return {family: _dedupe_local_safe_cases(materialized) for family, materialized in cases.items()}


def _same_segment_cases(companies: tuple[SyntheticCompany, ...]) -> dict[str, list[CanonicalCase]]:
    cases: dict[str, list[CanonicalCase]] = defaultdict(list)
    families = ("companies_by_segment_filters", "segments_by_segment_filters")
    for company in companies:
        for segment in company.segments:
            revenue_models = sorted(segment_revenue_models(segment, descendant=True))
            for family in families:
                for customer_type in segment.customer_types:
                    case = _make_local_safe_case(
                        family=family,
                        bucket="same_segment",
                        payload=QueryPlanPayload(customer_types=[customer_type], binding_scope="same_segment"),
                        source_graph_ids=[company.graph_id],
                    )
                    if case is not None:
                        cases[family].append(case)
                for channel in segment.channels:
                    case = _make_local_safe_case(
                        family=family,
                        bucket="same_segment",
                        payload=QueryPlanPayload(channels=[channel], binding_scope="same_segment"),
                        source_graph_ids=[company.graph_id],
                    )
                    if case is not None:
                        cases[family].append(case)
                for offering in segment.offerings[:3]:
                    case = _make_local_safe_case(
                        family=family,
                        bucket="same_segment",
                        payload=QueryPlanPayload(offerings=[offering.name], binding_scope="same_segment"),
                        source_graph_ids=[company.graph_id],
                    )
                    if case is not None:
                        cases[family].append(case)
                for revenue_model in revenue_models[:3]:
                    case = _make_local_safe_case(
                        family=family,
                        bucket="same_segment",
                        payload=QueryPlanPayload(revenue_models=[revenue_model], binding_scope="same_segment", hierarchy_mode="descendant"),
                        source_graph_ids=[company.graph_id],
                    )
                    if case is not None:
                        cases[family].append(case)
                for customer_type in segment.customer_types:
                    for channel in segment.channels:
                        global_case = _make_local_safe_case(
                            family=family,
                            bucket="same_segment",
                            payload=QueryPlanPayload(
                                customer_types=[customer_type],
                                channels=[channel],
                                binding_scope="same_segment",
                            ),
                            source_graph_ids=[company.graph_id],
                        )
                        if global_case is not None:
                            cases[family].append(global_case)
                        scoped_case = _make_local_safe_case(
                            family=family,
                            bucket="same_segment",
                            payload=QueryPlanPayload(
                                companies=[company.name],
                                segments=[segment.name],
                                customer_types=[customer_type],
                                channels=[channel],
                                binding_scope="same_segment",
                            ),
                            source_graph_ids=[company.graph_id],
                        )
                        if scoped_case is not None:
                            cases[family].append(scoped_case)
    return {family: _dedupe_local_safe_cases(materialized) for family, materialized in cases.items()}


def _cross_segment_cases(companies: tuple[SyntheticCompany, ...]) -> dict[str, list[CanonicalCase]]:
    cases: list[CanonicalCase] = []
    for company in companies:
        customer_types = sorted({customer_type for segment in company.segments for customer_type in segment.customer_types})
        channels = sorted({channel for segment in company.segments for channel in segment.channels})
        for index, first in enumerate(customer_types):
            for second in customer_types[index + 1 :]:
                for scoped in (False, True):
                    payload = QueryPlanPayload(
                        companies=[company.name] if scoped else [],
                        customer_types=[first, second],
                        binding_scope="across_segments",
                    )
                    case = _make_local_safe_case(
                        family="companies_by_cross_segment_filters",
                        bucket="cross_segment",
                        payload=payload,
                        source_graph_ids=[company.graph_id],
                    )
                    if case is not None:
                        cases.append(case)
        for index, first in enumerate(channels):
            for second in channels[index + 1 :]:
                payload = QueryPlanPayload(
                    companies=[company.name],
                    channels=[first, second],
                    binding_scope="across_segments",
                )
                case = _make_local_safe_case(
                    family="companies_by_cross_segment_filters",
                    bucket="cross_segment",
                    payload=payload,
                    source_graph_ids=[company.graph_id],
                )
                if case is not None:
                    cases.append(case)
        for customer_type in customer_types[:6]:
            for channel in channels[:4]:
                payload = QueryPlanPayload(
                    companies=[company.name],
                    customer_types=[customer_type],
                    channels=[channel],
                    binding_scope="across_segments",
                )
                case = _make_local_safe_case(
                    family="companies_by_cross_segment_filters",
                    bucket="cross_segment",
                    payload=payload,
                    source_graph_ids=[company.graph_id],
                )
                if case is not None:
                    cases.append(case)
    return {"companies_by_cross_segment_filters": _dedupe_local_safe_cases(cases)}


def _hierarchy_cases(companies: tuple[SyntheticCompany, ...]) -> dict[str, list[CanonicalCase]]:
    descendants_cases: list[CanonicalCase] = []
    revenue_cases: list[CanonicalCase] = []
    for company in companies:
        for segment in company.segments:
            for root in root_offerings_with_children(segment):
                for company_names in ([company.name], []):
                    for limit in (None, 2, 4, 6):
                        case = _make_local_safe_case(
                            family="descendant_offerings_by_root",
                            bucket="hierarchy",
                            payload=QueryPlanPayload(
                                companies=company_names,
                                offerings=[root.name],
                                hierarchy_mode="descendant",
                                limit=limit,
                            ),
                            source_graph_ids=[company.graph_id],
                        )
                        if case is not None:
                            descendants_cases.append(case)
                descendant_names = descendant_offering_names(segment, root.name)
                revenue_models = sorted(
                    {
                        revenue_model
                        for offering in all_offerings(segment)
                        if offering.name in descendant_names
                        for revenue_model in offering.revenue_models
                    }
                )
                for revenue_model in revenue_models:
                    for limit in (None, 2, 4):
                        for scoped_company in (False, True):
                            case = _make_local_safe_case(
                                family="companies_by_descendant_revenue",
                                bucket="hierarchy",
                                payload=QueryPlanPayload(
                                    companies=[company.name] if scoped_company else [],
                                    offerings=[root.name],
                                    revenue_models=[revenue_model],
                                    hierarchy_mode="descendant",
                                    limit=limit,
                                ),
                                source_graph_ids=[company.graph_id],
                            )
                            if case is not None:
                                revenue_cases.append(case)
                        case = _make_local_safe_case(
                            family="companies_by_descendant_revenue",
                            bucket="hierarchy",
                            payload=QueryPlanPayload(
                                offerings=[root.name],
                                revenue_models=[revenue_model],
                                places=[company.places[0]],
                                hierarchy_mode="descendant",
                                limit=limit,
                            ),
                            source_graph_ids=[company.graph_id],
                        )
                        if case is not None:
                            revenue_cases.append(case)
                if len(revenue_models) >= 2:
                    case = _make_local_safe_case(
                        family="companies_by_descendant_revenue",
                        bucket="hierarchy",
                        payload=QueryPlanPayload(
                            companies=[company.name],
                            offerings=[root.name],
                            revenue_models=revenue_models[:2],
                            hierarchy_mode="descendant",
                            limit=3,
                        ),
                        source_graph_ids=[company.graph_id],
                    )
                    if case is not None:
                        revenue_cases.append(case)
    return {
        "descendant_offerings_by_root": _dedupe_local_safe_cases(descendants_cases),
        "companies_by_descendant_revenue": _dedupe_local_safe_cases(revenue_cases),
    }


def _geography_cases(companies: tuple[SyntheticCompany, ...]) -> dict[str, list[CanonicalCase]]:
    place_cases: list[CanonicalCase] = []
    segment_cases: list[CanonicalCase] = []
    for place in _all_places(companies):
        for limit in (None, 2, 3, 5):
            case = _make_local_safe_case(
                family="companies_by_place",
                bucket="geography",
                payload=QueryPlanPayload(places=[place], limit=limit),
                source_graph_ids=[company.graph_id for company in companies if place in company.places],
            )
            if case is not None:
                place_cases.append(case)
    for company in companies:
        for place in company.places:
            for segment in company.segments:
                for payload in (
                    QueryPlanPayload(places=[place], customer_types=[segment.customer_types[0]], binding_scope="same_segment"),
                    QueryPlanPayload(places=[place], channels=[segment.channels[0]], binding_scope="same_segment"),
                    QueryPlanPayload(
                        companies=[company.name],
                        places=[place],
                        customer_types=[segment.customer_types[0]],
                        channels=[segment.channels[0]],
                        binding_scope="same_segment",
                    ),
                ):
                    case = _make_local_safe_case(
                        family="segments_by_place_and_segment_filters",
                        bucket="geography",
                        payload=payload,
                        source_graph_ids=[company.graph_id],
                    )
                    if case is not None:
                        segment_cases.append(case)
    return {
        "companies_by_place": _dedupe_local_safe_cases(place_cases),
        "segments_by_place_and_segment_filters": _dedupe_local_safe_cases(segment_cases),
    }


def _partner_cases(companies: tuple[SyntheticCompany, ...]) -> dict[str, list[CanonicalCase]]:
    cases: list[CanonicalCase] = []
    for partner in _all_partners(companies):
        for limit in (None, 2, 3):
            case = _make_local_safe_case(
                family="companies_by_partner",
                bucket="partner",
                payload=QueryPlanPayload(partners=[partner], limit=limit),
                source_graph_ids=[company.graph_id for company in companies if partner in company.partners],
            )
            if case is not None:
                cases.append(case)
    for company in companies:
        for partner in company.partners:
            case = _make_local_safe_case(
                family="companies_by_partner",
                bucket="partner",
                payload=QueryPlanPayload(companies=[company.name], partners=[partner]),
                source_graph_ids=[company.graph_id],
            )
            if case is not None:
                cases.append(case)
    return {"companies_by_partner": _dedupe_local_safe_cases(cases)}


def _boolean_cases(companies: tuple[SyntheticCompany, ...]) -> dict[str, list[CanonicalCase]]:
    cases: list[CanonicalCase] = []
    for company in companies:
        for partner in company.partners[:2]:
            for payload in (
                QueryPlanPayload(base_family="companies_by_partner", partners=[partner]),
                QueryPlanPayload(base_family="companies_by_partner", companies=[company.name], partners=[partner]),
            ):
                case = _make_local_safe_case(
                    family="boolean_exists",
                    bucket="boolean",
                    payload=payload,
                    source_graph_ids=[company.graph_id],
                )
                if case is not None:
                    cases.append(case)
        for place in company.places[:3]:
            for payload in (
                QueryPlanPayload(base_family="companies_by_place", places=[place]),
                QueryPlanPayload(base_family="companies_by_place", companies=[company.name], places=[place]),
            ):
                case = _make_local_safe_case(
                    family="boolean_exists",
                    bucket="boolean",
                    payload=payload,
                    source_graph_ids=[company.graph_id],
                )
                if case is not None:
                    cases.append(case)
        for segment in company.segments:
            cases_to_add = (
                QueryPlanPayload(
                    base_family="companies_by_segment_filters",
                    companies=[company.name],
                    customer_types=[segment.customer_types[0]],
                ),
                QueryPlanPayload(
                    base_family="companies_by_segment_filters",
                    companies=[company.name],
                    channels=[segment.channels[0]],
                ),
                QueryPlanPayload(
                    base_family="companies_by_segment_filters",
                    companies=[company.name],
                    customer_types=[segment.customer_types[0]],
                    channels=[segment.channels[0]],
                ),
                QueryPlanPayload(
                    base_family="segments_by_segment_filters",
                    companies=[company.name],
                    customer_types=[segment.customer_types[0]],
                ),
                QueryPlanPayload(
                    base_family="segments_by_segment_filters",
                    companies=[company.name],
                    channels=[segment.channels[0]],
                ),
                QueryPlanPayload(
                    base_family="companies_by_partner",
                    companies=[company.name],
                    partners=[company.partners[0]],
                ),
                QueryPlanPayload(
                    base_family="companies_by_place",
                    companies=[company.name],
                    places=[company.places[0]],
                ),
            )
            for payload in cases_to_add:
                case = _make_local_safe_case(
                    family="boolean_exists",
                    bucket="boolean",
                    payload=payload,
                    source_graph_ids=[company.graph_id],
                )
                if case is not None:
                    cases.append(case)
            if len(segment.customer_types) >= 2:
                case = _make_local_safe_case(
                    family="boolean_exists",
                    bucket="boolean",
                    payload=QueryPlanPayload(
                        base_family="companies_by_cross_segment_filters",
                        companies=[company.name],
                        customer_types=list(segment.customer_types[:2]),
                        binding_scope="across_segments",
                    ),
                    source_graph_ids=[company.graph_id],
                )
                if case is not None:
                    cases.append(case)
            for root in root_offerings_with_children(segment):
                case = _make_local_safe_case(
                    family="boolean_exists",
                    bucket="boolean",
                    payload=QueryPlanPayload(
                        base_family="descendant_offerings_by_root",
                        companies=[company.name],
                        offerings=[root.name],
                        hierarchy_mode="descendant",
                    ),
                    source_graph_ids=[company.graph_id],
                )
                if case is not None:
                    cases.append(case)
                descendant_models = sorted(
                    {
                        revenue_model
                        for offering in all_offerings(segment)
                        if offering.name in descendant_offering_names(segment, root.name)
                        for revenue_model in offering.revenue_models
                    }
                )
                if descendant_models:
                    case = _make_local_safe_case(
                        family="boolean_exists",
                        bucket="boolean",
                        payload=QueryPlanPayload(
                            base_family="companies_by_descendant_revenue",
                            companies=[company.name],
                            offerings=[root.name],
                            revenue_models=[descendant_models[0]],
                            hierarchy_mode="descendant",
                        ),
                        source_graph_ids=[company.graph_id],
                    )
                    if case is not None:
                        cases.append(case)
    return {"boolean_exists": _dedupe_local_safe_cases(cases)}


def _count_cases(companies: tuple[SyntheticCompany, ...]) -> dict[str, list[CanonicalCase]]:
    cases: list[CanonicalCase] = []
    for company in companies:
        for segment in company.segments:
            payloads = (
                QueryPlanPayload(
                    customer_types=[segment.customer_types[0]],
                    base_family="companies_by_segment_filters",
                    aggregate_spec={
                        "kind": "count",
                        "base_family": "companies_by_segment_filters",
                        "count_target": "company",
                    },
                ),
                QueryPlanPayload(
                    channels=[segment.channels[0]],
                    base_family="segments_by_segment_filters",
                    aggregate_spec={
                        "kind": "count",
                        "base_family": "segments_by_segment_filters",
                        "count_target": "segment",
                    },
                ),
                QueryPlanPayload(
                    companies=[company.name],
                    base_family="offerings_by_company",
                    aggregate_spec={
                        "kind": "count",
                        "base_family": "offerings_by_company",
                        "count_target": "offering",
                    },
                ),
            )
            for payload in payloads:
                case = _make_local_safe_case(
                    family="count_aggregate",
                    bucket="count",
                    payload=payload,
                    source_graph_ids=[company.graph_id],
                )
                if case is not None:
                    cases.append(case)
            for root in root_offerings_with_children(segment):
                case = _make_local_safe_case(
                    family="count_aggregate",
                    bucket="count",
                    payload=QueryPlanPayload(
                        companies=[company.name],
                        offerings=[root.name],
                        hierarchy_mode="descendant",
                        base_family="descendant_offerings_by_root",
                        aggregate_spec={
                            "kind": "count",
                            "base_family": "descendant_offerings_by_root",
                            "count_target": "offering",
                        },
                    ),
                    source_graph_ids=[company.graph_id],
                )
                if case is not None:
                    cases.append(case)
    return {"count_aggregate": _dedupe_local_safe_cases(cases)}


def _ranking_cases(companies: tuple[SyntheticCompany, ...]) -> dict[str, list[CanonicalCase]]:
    cases: list[CanonicalCase] = []
    unique_places = _all_places(companies)
    metrics = (
        "customer_type_by_company_count",
        "channel_by_segment_count",
        "revenue_model_by_company_count",
        "company_by_matched_segment_count",
    )
    for metric in metrics:
        for limit in (3, 5):
            base_payload = QueryPlanPayload(
                aggregate_spec={"kind": "ranking", "ranking_metric": metric},
                limit=limit,
            )
            if metric == "company_by_matched_segment_count":
                base_payload = QueryPlanPayload(
                    customer_types=["developers"],
                    aggregate_spec={"kind": "ranking", "ranking_metric": metric},
                    limit=limit,
                )
            case = _make_local_safe_case(
                family="ranking_topk",
                bucket="ranking",
                payload=base_payload,
                source_graph_ids=[company.graph_id for company in companies],
            )
            if case is not None:
                cases.append(case)
            for company in companies:
                scoped_payload = QueryPlanPayload(
                    companies=[company.name],
                    aggregate_spec={"kind": "ranking", "ranking_metric": metric},
                    limit=limit,
                )
                if metric == "company_by_matched_segment_count":
                    scoped_payload = QueryPlanPayload(
                        companies=[company.name],
                        customer_types=["developers"],
                        aggregate_spec={"kind": "ranking", "ranking_metric": metric},
                        limit=limit,
                    )
                case = _make_local_safe_case(
                    family="ranking_topk",
                    bucket="ranking",
                    payload=scoped_payload,
                    source_graph_ids=[company.graph_id],
                )
                if case is not None:
                    cases.append(case)
            for place in unique_places[:4]:
                place_payload = QueryPlanPayload(
                    places=[place],
                    aggregate_spec={"kind": "ranking", "ranking_metric": metric},
                    limit=limit,
                )
                if metric == "company_by_matched_segment_count":
                    place_payload = QueryPlanPayload(
                        customer_types=["developers"],
                        places=[place],
                        aggregate_spec={"kind": "ranking", "ranking_metric": metric},
                        limit=limit,
                    )
                case = _make_local_safe_case(
                    family="ranking_topk",
                    bucket="ranking",
                    payload=place_payload,
                    source_graph_ids=[company.graph_id for company in companies if place in company.places],
                )
                if case is not None:
                    cases.append(case)
    return {"ranking_topk": _dedupe_local_safe_cases(cases)}


def _canonical_local_safe_cases(companies: tuple[SyntheticCompany, ...]) -> dict[str, list[CanonicalCase]]:
    cases: dict[str, list[CanonicalCase]] = defaultdict(list)
    for family_cases in (
        _inventory_cases(companies),
        _same_segment_cases(companies),
        _cross_segment_cases(companies),
        _hierarchy_cases(companies),
        _geography_cases(companies),
        _partner_cases(companies),
        _boolean_cases(companies),
        _count_cases(companies),
        _ranking_cases(companies),
    ):
        for family, materialized in family_cases.items():
            cases[family].extend(materialized)
    return {family: _dedupe_local_safe_cases(materialized) for family, materialized in cases.items()}


def _canonical_strong_candidate_cases(companies: tuple[SyntheticCompany, ...]) -> dict[str, list[CanonicalCase]]:
    cases: dict[str, list[CanonicalCase]] = defaultdict(list)
    for company in companies:
        for segment_a in company.segments:
            for segment_b in company.segments:
                if segment_a.name == segment_b.name:
                    continue
                offerings_a = {offering.name for offering in segment_a.offerings}
                offerings_b = {offering.name for offering in segment_b.offerings}
                if offerings_a.intersection(offerings_b):
                    cases["common_offerings_between_segments"].append(
                        _make_strong_candidate_case(
                            family="common_offerings_between_segments",
                            source_graph_ids=[company.graph_id],
                            context={"company": company.name, "segment_a": segment_a.name, "segment_b": segment_b.name},
                        )
                    )
                if offerings_a.difference(offerings_b):
                    cases["unique_offerings_to_segment"].append(
                        _make_strong_candidate_case(
                            family="unique_offerings_to_segment",
                            source_graph_ids=[company.graph_id],
                            context={"company": company.name, "segment_a": segment_a.name, "segment_b": segment_b.name},
                        )
                    )
                cases["compare_segments_by_customer_types"].append(
                    _make_strong_candidate_case(
                        family="compare_segments_by_customer_types",
                        source_graph_ids=[company.graph_id],
                        context={"company": company.name, "segment_a": segment_a.name, "segment_b": segment_b.name},
                    )
                )
                cases["compare_segments_by_channels"].append(
                    _make_strong_candidate_case(
                        family="compare_segments_by_channels",
                        source_graph_ids=[company.graph_id],
                        context={"company": company.name, "segment_a": segment_a.name, "segment_b": segment_b.name},
                    )
                )
                cases["compare_segments_by_offering_count"].append(
                    _make_strong_candidate_case(
                        family="compare_segments_by_offering_count",
                        source_graph_ids=[company.graph_id],
                        context={"company": company.name, "segment_a": segment_a.name, "segment_b": segment_b.name},
                    )
                )
            for focus in segment_a.customer_types[:2] + segment_a.channels[:2]:
                cases["why_segment_matches"].append(
                    _make_strong_candidate_case(
                        family="why_segment_matches",
                        source_graph_ids=[company.graph_id],
                        context={"company": company.name, "segment": segment_a.name, "focus": focus},
                    )
                )
        weighted_focuses = []
        if company.segments:
            weighted_focuses.extend(company.segments[0].channels[:2])
            weighted_focuses.extend(company.segments[0].customer_types[:1])
        if not weighted_focuses:
            weighted_focuses.append(company.name)
        for focus in weighted_focuses:
            cases["weighted_ranking_request"].append(
                _make_strong_candidate_case(
                    family="weighted_ranking_request",
                    source_graph_ids=[company.graph_id],
                    context={"company": company.name, "focus": focus},
                )
            )

    if len(companies) >= 2:
        ordered = sorted(companies, key=lambda company: company.name)
        for index, company_a in enumerate(ordered):
            for company_b in ordered[index + 1 :]:
                cases["compare_companies_by_customer_types"].append(
                    _make_strong_candidate_case(
                        family="compare_companies_by_customer_types",
                        source_graph_ids=[company_a.graph_id, company_b.graph_id],
                        context={"company_a": company_a.name, "company_b": company_b.name},
                    )
                )
                cases["compare_companies_by_channels"].append(
                    _make_strong_candidate_case(
                        family="compare_companies_by_channels",
                        source_graph_ids=[company_a.graph_id, company_b.graph_id],
                        context={"company_a": company_a.name, "company_b": company_b.name},
                    )
                )

    return {family: _dedupe_case_ids(materialized) for family, materialized in cases.items()}


def _canonical_refusal_cases(companies: tuple[SyntheticCompany, ...]) -> dict[str, list[CanonicalCase]]:
    cases: dict[str, list[CanonicalCase]] = defaultdict(list)
    for company in companies:
        first_segment = company.segments[0]
        first_offering = first_segment.offerings[0]
        refusal_templates = {
            "unsupported_schema": (
                (
                    f"Which suppliers does {company.name} use?",
                    f"Who supplies {company.name}?",
                    f"List the suppliers for {company.name}.",
                    f"Show the supplier base of {company.name}.",
                ),
                (
                    f"Which employees work on {first_segment.name} at {company.name}?",
                    f"List the employees assigned to {first_segment.name} at {company.name}.",
                    f"Who works on the {first_segment.name} segment at {company.name}?",
                    f"Show the staff attached to {first_segment.name} at {company.name}.",
                ),
                (
                    f"What raw materials feed {first_offering.name} at {company.name}?",
                    f"List the raw-material inputs behind {first_offering.name} at {company.name}.",
                    f"Which upstream inputs support {first_offering.name} at {company.name}?",
                    f"Show the material inputs for {first_offering.name} at {company.name}.",
                ),
            ),
            "unsupported_metric": (
                (
                    f"What is {company.name}'s annual revenue?",
                    f"Show the annual revenue of {company.name}.",
                    f"How much revenue does {company.name} make?",
                    f"List the revenue of {company.name}.",
                ),
                (
                    f"What gross margin does {first_segment.name} have at {company.name}?",
                    f"Show the gross margin for {first_segment.name} at {company.name}.",
                    f"How profitable is {first_segment.name} at {company.name}?",
                    f"List the margin of {first_segment.name} at {company.name}.",
                ),
                (
                    f"What ARR does {first_offering.name} have at {company.name}?",
                    f"Show the ARR for {first_offering.name} at {company.name}.",
                    f"How much recurring revenue does {first_offering.name} make at {company.name}?",
                    f"List the ARR of {first_offering.name} at {company.name}.",
                ),
            ),
            "unsupported_time": (
                (
                    f"What was {company.name}'s revenue in 2024?",
                    f"Show {company.name}'s 2024 revenue.",
                    f"How much revenue did {company.name} report in 2024?",
                    f"List the 2024 revenue of {company.name}.",
                ),
                (
                    f"How did {first_segment.name} change last year at {company.name}?",
                    f"Show how {first_segment.name} changed in 2024 at {company.name}.",
                    f"What changed in {first_segment.name} over the last year at {company.name}?",
                    f"List the year-over-year changes for {first_segment.name} at {company.name}.",
                ),
                (
                    f"Which channel grew fastest last year for {company.name}?",
                    f"Show the fastest-growing channel in 2024 for {company.name}.",
                    f"What channel grew the most last year at {company.name}?",
                    f"List the highest-growth channel for {company.name} last year.",
                ),
            ),
            "ambiguous_closed_label": (
                (
                    f"Which companies sell through affiliates like {company.name}?",
                    f"Show the companies that use affiliates as a channel like {company.name}.",
                    f"List companies going to market through affiliates such as {company.name}.",
                    f"Which companies rely on affiliates the way {company.name} does?",
                    f"What companies use affiliate-style channels like {company.name}?",
                    f"Identify companies whose route to market sounds like affiliates, similar to {company.name}.",
                    f"Which companies appear to sell through affiliate-style partners like {company.name}?",
                ),
                (
                    f"Which segments serve regulated institutions at {company.name}?",
                    f"List the segments at {company.name} that serve regulated institutions.",
                    f"Show me the {company.name} segments for regulated institutions.",
                    f"Which segments at {company.name} target regulated institutions?",
                    f"What segments at {company.name} focus on regulated institutions?",
                    f"Identify the {company.name} segments aimed at regulated institutions.",
                    f"Which {company.name} segments seem built for regulated institutions?",
                ),
            ),
            "ambiguous_request": (
                (
                    f"Which segment is the best one at {company.name}?",
                    f"Show me the best segment at {company.name}.",
                    f"What is the top segment at {company.name}?",
                    f"Identify the best business segment at {company.name}.",
                ),
                (
                    f"What are the main business lines of {company.name}?",
                    f"List the main business lines at {company.name}.",
                    f"Show the main lines of business for {company.name}.",
                    f"What are the core business lines of {company.name}?",
                ),
                (
                    f"Which are the strongest channels at {company.name}?",
                    f"Show the strongest channels for {company.name}.",
                    f"What are the top channels at {company.name}?",
                    f"Identify the strongest go-to-market channels at {company.name}.",
                ),
            ),
            "write_request": (
                (
                    f"Add a new segment to {company.name}.",
                    f"Create a new business segment for {company.name}.",
                    f"Insert a new segment for {company.name}.",
                    f"Add another segment under {company.name}.",
                ),
                (
                    f"Delete {first_offering.name} from {company.name}.",
                    f"Remove the offering {first_offering.name} from {company.name}.",
                    f"Drop {first_offering.name} from the graph for {company.name}.",
                    f"Erase {first_offering.name} from {company.name}.",
                ),
                (
                    f"Rename {first_segment.name} at {company.name}.",
                    f"Change the name of {first_segment.name} for {company.name}.",
                    f"Update the segment name {first_segment.name} at {company.name}.",
                    f"Modify the segment label {first_segment.name} at {company.name}.",
                ),
            ),
            "beyond_local_coverage": (
                (
                    f"Which companies serve developers but not retailers like {company.name}?",
                    f"List companies that serve developers and exclude retailers, similar to {company.name}.",
                    f"Show the companies that serve developers except retailers, using {company.name} as a reference.",
                    f"Which companies match developers but not retailers in a way comparable to {company.name}?",
                ),
                (
                    f"Rank {company.name}'s segments using a custom weighted score.",
                    f"Show a weighted ranking of {company.name}'s segments with a custom formula.",
                    f"Compute a custom score to rank the segments of {company.name}.",
                    f"Order the segments of {company.name} by a weighted custom score.",
                ),
                (
                    f"Explain why {company.name} should rank ahead of peers with a weighted explanation.",
                    f"Give a weighted rationale for why {company.name} should outrank comparable companies.",
                    f"Show the weighted reasoning that would place {company.name} ahead of peers.",
                    f"Why should {company.name} rank first under a custom weighted rubric?",
                ),
            ),
        }
        for reason, template_sets in refusal_templates.items():
            for template_group in template_sets:
                cases[reason].append(
                    _make_refusal_case(
                        family="refuse",
                        bucket="refuse",
                        reason=reason,
                        source_graph_ids=[company.graph_id],
                        context={"templates": template_group},
                    )
                )
    return {reason: _dedupe_case_ids(materialized) for reason, materialized in cases.items()}


def _local_safe_family_targets(split: SplitName, total: int) -> dict[str, int]:
    return _family_targets_for_split(LOCAL_SAFE_FAMILY_TARGETS_TRAIN, total)


def _strong_candidate_family_targets(family_cases: dict[str, list[CanonicalCase]], total: int) -> dict[str, int]:
    available_weights = {
        family: weight
        for family, weight in STRONG_MODEL_CANDIDATE_FAMILY_WEIGHTS.items()
        if family_cases.get(family)
    }
    return _scale_targets(available_weights, total)


def _refusal_reason_targets(total: int) -> dict[str, int]:
    return _scale_targets(REFUSAL_REASON_WEIGHTS, total)


def _build_split_examples(
    companies: tuple[SyntheticCompany, ...],
    *,
    split: SplitName,
    size: int,
    seed: int,
) -> list[DatasetExample]:
    route_targets = _scale_targets(DEFAULT_ROUTE_TARGETS_BY_SPLIT[split], size)
    local_safe_cases = _canonical_local_safe_cases(companies)
    strong_cases = _canonical_strong_candidate_cases(companies)
    refusal_cases = _canonical_refusal_cases(companies)

    seen_questions: set[str] = set()
    seen_pairs: set[str] = set()
    examples: list[DatasetExample] = []

    examples.extend(
        _materialize_family_examples(
            companies=companies,
            family_cases=local_safe_cases,
            family_targets=_local_safe_family_targets(split, route_targets["local_safe"]),
            split=split,
            seed=seed + 11,
            seen_questions=seen_questions,
            seen_pairs=seen_pairs,
        )
    )
    examples.extend(
        _materialize_family_examples(
            companies=companies,
            family_cases=strong_cases,
            family_targets=_strong_candidate_family_targets(strong_cases, route_targets["strong_model_candidate"]),
            split=split,
            seed=seed + 23,
            seen_questions=seen_questions,
            seen_pairs=seen_pairs,
        )
    )
    refusal_targets = _refusal_reason_targets(route_targets["refuse"])
    examples.extend(
        _materialize_family_examples(
            companies=companies,
            family_cases=refusal_cases,
            family_targets=refusal_targets,
            split=split,
            seed=seed + 37,
            seen_questions=seen_questions,
            seen_pairs=seen_pairs,
        )
    )
    if len(examples) != size:
        raise ValueError(f"Split {split} generated {len(examples)} examples instead of {size}.")
    return sorted(examples, key=lambda example: (example.case_id, example.template_id, example.variant_id))


def build_dataset_splits(
    *,
    train_size: int = 8000,
    validation_size: int = 1200,
    release_eval_size: int = 1800,
    seed: int = 7,
) -> dict[str, list[DatasetExample]]:
    companies = build_synthetic_company_graphs()
    company_map = {company.graph_id: company for company in companies}
    graphs_by_split: dict[SplitName, tuple[SyntheticCompany, ...]] = {
        split: tuple(company_map[graph_id] for graph_id in graph_ids)
        for split, graph_ids in GRAPH_IDS_BY_SPLIT.items()
    }
    return {
        "train": _build_split_examples(
            graphs_by_split["train"],
            split="train",
            size=train_size,
            seed=seed + SPLIT_SEED_OFFSETS["train"],
        ),
        "validation": _build_split_examples(
            graphs_by_split["validation"],
            split="validation",
            size=validation_size,
            seed=seed + SPLIT_SEED_OFFSETS["validation"],
        ),
        "release_eval": _build_split_examples(
            graphs_by_split["release_eval"],
            split="release_eval",
            size=release_eval_size,
            seed=seed + SPLIT_SEED_OFFSETS["release_eval"],
        ),
    }


def _summarize_split(examples: list[DatasetExample]) -> dict[str, Any]:
    route_counts = Counter(example.route_label for example in examples)
    family_counts = Counter(example.family for example in examples)
    refusal_reason_counts = Counter(example.metadata.get("refusal_reason") for example in examples if example.metadata.get("refusal_reason"))
    source_graph_distribution: Counter[str] = Counter()
    for example in examples:
        for graph_id in example.metadata.get("source_graph_ids", []):
            source_graph_distribution[graph_id] += 1
    target_keys = [json.dumps(example.target, sort_keys=True, separators=(",", ":")) for example in examples]
    question_keys = [example.question.strip().casefold() for example in examples]
    pair_keys = [f"{question}|{target}" for question, target in zip(question_keys, target_keys)]
    return {
        "count": len(examples),
        "route_counts": dict(sorted(route_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "refusal_reason_counts": dict(sorted(refusal_reason_counts.items())),
        "source_graph_distribution": dict(sorted(source_graph_distribution.items())),
        "unique_question_count": len(set(question_keys)),
        "unique_target_count": len(set(target_keys)),
        "unique_question_target_count": len(set(pair_keys)),
        "duplicate_question_count": len(question_keys) - len(set(question_keys)),
        "duplicate_target_count": len(target_keys) - len(set(target_keys)),
        "duplicate_question_target_count": len(pair_keys) - len(set(pair_keys)),
    }


def build_dataset_manifest(
    *,
    train_size: int = 8000,
    validation_size: int = 1200,
    release_eval_size: int = 1800,
    seed: int = 7,
) -> dict[str, Any]:
    splits = build_dataset_splits(
        train_size=train_size,
        validation_size=validation_size,
        release_eval_size=release_eval_size,
        seed=seed,
    )
    route_targets = {
        split: _scale_targets(DEFAULT_ROUTE_TARGETS_BY_SPLIT[split], size)
        for split, size in {
            "train": train_size,
            "validation": validation_size,
            "release_eval": release_eval_size,
        }.items()
    }
    local_safe_targets = {
        split: _local_safe_family_targets(split, route_targets[split]["local_safe"])
        for split in ("train", "validation", "release_eval")
    }
    strong_targets = {
        split: _strong_candidate_family_targets(
            _canonical_strong_candidate_cases(
                tuple(company for company in build_synthetic_company_graphs() if company.graph_id in GRAPH_IDS_BY_SPLIT[split])  # type: ignore[index]
            ),
            route_targets[split]["strong_model_candidate"],
        )
        for split in ("train", "validation", "release_eval")
    }
    refusal_targets = {
        split: _refusal_reason_targets(route_targets[split]["refuse"])
        for split in ("train", "validation", "release_eval")
    }
    return {
        "seed": seed,
        "split_sizes": {
            "train": train_size,
            "validation": validation_size,
            "release_eval": release_eval_size,
        },
        "route_targets": route_targets,
        "local_safe_bucket_targets": {
            split: _scale_targets(LOCAL_SAFE_BUCKET_TARGETS_TRAIN, route_targets[split]["local_safe"])
            for split in ("train", "validation", "release_eval")
        },
        "local_safe_family_targets": local_safe_targets,
        "strong_model_candidate_targets": strong_targets,
        "refusal_reason_targets": refusal_targets,
        "graph_assignments": {split: list(graph_ids) for split, graph_ids in GRAPH_IDS_BY_SPLIT.items()},
        "split_stats": {split: _summarize_split(examples) for split, examples in splits.items()},
    }


def write_dataset_splits(
    output_dir: Path,
    *,
    train_size: int = 8000,
    validation_size: int = 1200,
    release_eval_size: int = 1800,
    seed: int = 7,
) -> dict[str, Path]:
    splits = build_dataset_splits(
        train_size=train_size,
        validation_size=validation_size,
        release_eval_size=release_eval_size,
        seed=seed,
    )
    manifest = build_dataset_manifest(
        train_size=train_size,
        validation_size=validation_size,
        release_eval_size=release_eval_size,
        seed=seed,
    )
    synthetic_graphs = [asdict(company) for company in build_synthetic_company_graphs()]
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for split_name, examples in splits.items():
        path = output_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
        written[split_name] = path

    graphs_path = output_dir / "synthetic_graphs.json"
    graphs_path.write_text(json.dumps(synthetic_graphs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    written["synthetic_graphs"] = graphs_path

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    written["manifest"] = manifest_path
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic query-planner datasets.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write split JSONL files into.")
    parser.add_argument("--train-size", type=int, default=8000, help="Number of train examples to generate.")
    parser.add_argument("--validation-size", type=int, default=1200, help="Number of validation examples to generate.")
    parser.add_argument("--release-eval-size", type=int, default=1800, help="Number of release-eval examples to generate.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic generation seed.")
    args = parser.parse_args(argv)
    write_dataset_splits(
        args.output_dir,
        train_size=args.train_size,
        validation_size=args.validation_size,
        release_eval_size=args.release_eval_size,
        seed=args.seed,
    )
    return 0


__all__ = [
    "DEFAULT_ROUTE_TARGETS_BY_SPLIT",
    "DatasetExample",
    "GRAPH_IDS_BY_SPLIT",
    "LOCAL_SAFE_BUCKET_TARGETS_TRAIN",
    "LOCAL_SAFE_FAMILY_TARGETS_TRAIN",
    "build_dataset_manifest",
    "build_dataset_splits",
    "main",
    "write_dataset_splits",
]


if __name__ == "__main__":
    raise SystemExit(main())
