from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from ontology.place_hierarchy import classify_place_match, normalize_place_name

from .query_planner import QueryPlanEnvelope, QueryPlanPayload, compile_query_plan

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

DEFAULT_LOCAL_SAFE_FAMILY_TARGETS: dict[str, int] = {
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

GRAPH_IDS_BY_SPLIT: dict[SplitName, tuple[str, ...]] = {
    "train": ("aurora", "redwood", "lattice"),
    "validation": ("nimbus", "vector"),
    "release_eval": ("nimbus", "vector"),
}

SPLIT_SEED_OFFSETS: dict[SplitName, int] = {
    "train": 0,
    "validation": 101,
    "release_eval": 202,
}


@dataclass(frozen=True)
class SyntheticOffering:
    name: str
    revenue_models: tuple[str, ...]
    children: tuple["SyntheticOffering", ...] = ()


@dataclass(frozen=True)
class SyntheticSegment:
    name: str
    customer_types: tuple[str, ...]
    channels: tuple[str, ...]
    offerings: tuple[SyntheticOffering, ...]


@dataclass(frozen=True)
class SyntheticCompany:
    graph_id: str
    name: str
    segments: tuple[SyntheticSegment, ...]
    places: tuple[str, ...]
    partners: tuple[str, ...]


@dataclass(frozen=True)
class DatasetExample:
    question: str
    target: dict[str, Any]
    route_label: RouteLabel
    family: str
    gold_cypher: str | None
    gold_params: dict[str, Any]
    gold_rows: list[dict[str, Any]]
    metadata: dict[str, Any]


def _offering(name: str, revenue_models: tuple[str, ...], *children: SyntheticOffering) -> SyntheticOffering:
    return SyntheticOffering(name=name, revenue_models=revenue_models, children=tuple(children))


def build_synthetic_company_graphs() -> tuple[SyntheticCompany, ...]:
    return (
        SyntheticCompany(
            graph_id="aurora",
            name="Aurora Systems",
            segments=(
                SyntheticSegment(
                    name="Cloud Infrastructure",
                    customer_types=("large enterprises", "developers", "IT professionals"),
                    channels=("direct sales", "system integrators", "managed service providers"),
                    offerings=(
                        _offering(
                            "Cloud Platform",
                            ("subscription", "consumption-based"),
                            _offering("Cloud Platform Edge", ("consumption-based",)),
                            _offering("Cloud Platform Secure", ("subscription",)),
                        ),
                        _offering("Identity Cloud", ("subscription",)),
                        _offering("Developer Toolkit", ("subscription", "service fees")),
                        _offering("Analytics Studio", ("subscription",)),
                        _offering("Automation Suite", ("subscription",)),
                    ),
                ),
                SyntheticSegment(
                    name="Data Products",
                    customer_types=("large enterprises", "mid-market companies", "developers"),
                    channels=("direct sales", "online"),
                    offerings=(
                        _offering(
                            "Analytics Studio",
                            ("subscription",),
                            _offering("Analytics Studio Pro", ("subscription",)),
                            _offering("Analytics Studio Embedded", ("licensing",)),
                        ),
                        _offering("Data Exchange", ("transaction fees",)),
                        _offering("Model Hub", ("subscription",)),
                        _offering("Forecast Engine", ("consumption-based",)),
                        _offering("Insight Stream", ("subscription",)),
                    ),
                ),
                SyntheticSegment(
                    name="Commerce Cloud",
                    customer_types=("retailers", "small businesses", "mid-market companies"),
                    channels=("resellers", "marketplaces", "online"),
                    offerings=(
                        _offering(
                            "Marketplace Hub",
                            ("transaction fees",),
                            _offering("Marketplace Hub Seller", ("transaction fees",)),
                            _offering("Marketplace Hub Buyer", ("transaction fees",)),
                        ),
                        _offering("Commerce Engine", ("subscription",)),
                        _offering("Payment Services", ("transaction fees",)),
                        _offering("Inventory Grid", ("subscription",)),
                        _offering("Fulfillment Orchestrator", ("service fees",)),
                    ),
                ),
                SyntheticSegment(
                    name="Public Sector Solutions",
                    customer_types=("government agencies", "educational institutions", "healthcare organizations"),
                    channels=("direct sales", "resellers"),
                    offerings=(
                        _offering("Compliance Vault", ("subscription",)),
                        _offering("Citizen Services Portal", ("service fees",)),
                        _offering("Secure Records", ("subscription",)),
                        _offering("Procurement Exchange", ("transaction fees",)),
                        _offering("Field Operations Suite", ("subscription",)),
                    ),
                ),
            ),
            places=("United States", "Italy", "Germany", "EMEA"),
            partners=("Dell", "Fujitsu Limited", "RetailNet"),
        ),
        SyntheticCompany(
            graph_id="nimbus",
            name="Nimbus Health",
            segments=(
                SyntheticSegment(
                    name="Provider Cloud",
                    customer_types=("healthcare organizations", "IT professionals", "large enterprises"),
                    channels=("direct sales", "managed service providers"),
                    offerings=(
                        _offering("Care Platform", ("subscription",), _offering("Care Platform Mobile", ("subscription",))),
                        _offering("Identity Cloud", ("subscription",)),
                        _offering("Secure Records", ("subscription",)),
                        _offering("Analytics Studio", ("subscription",)),
                        _offering("Workflow Automation", ("service fees",)),
                    ),
                ),
                SyntheticSegment(
                    name="Payer Intelligence",
                    customer_types=("financial services firms", "healthcare organizations"),
                    channels=("direct sales", "system integrators"),
                    offerings=(
                        _offering("Claims Exchange", ("transaction fees",)),
                        _offering("Forecast Engine", ("consumption-based",)),
                        _offering("Decision Hub", ("subscription",)),
                        _offering("Model Hub", ("subscription",)),
                        _offering("Compliance Vault", ("subscription",)),
                    ),
                ),
                SyntheticSegment(
                    name="Life Sciences",
                    customer_types=("manufacturers", "large enterprises", "developers"),
                    channels=("direct sales", "resellers"),
                    offerings=(
                        _offering("Developer Toolkit", ("subscription",)),
                        _offering("Lab Marketplace", ("transaction fees",)),
                        _offering("Secure Collaboration", ("subscription",)),
                        _offering("Data Exchange", ("licensing",)),
                        _offering("Automation Suite", ("subscription",)),
                    ),
                ),
                SyntheticSegment(
                    name="Consumer Wellness",
                    customer_types=("consumers", "healthcare organizations"),
                    channels=("online", "retail"),
                    offerings=(
                        _offering("Marketplace Hub", ("transaction fees",)),
                        _offering("Wellness Pass", ("subscription",)),
                        _offering("Care Platform Home", ("subscription",)),
                        _offering("Device Sync", ("hardware sales",)),
                        _offering("Support Desk", ("service fees",)),
                    ),
                ),
            ),
            places=("United Kingdom", "France", "Italy", "Europe"),
            partners=("MediSupply", "CloudAtlas", "Dell"),
        ),
        SyntheticCompany(
            graph_id="redwood",
            name="Redwood Retail",
            segments=(
                SyntheticSegment(
                    name="Merchant Services",
                    customer_types=("retailers", "small businesses", "mid-market companies"),
                    channels=("resellers", "marketplaces", "distributors"),
                    offerings=(
                        _offering("Marketplace Hub", ("transaction fees",)),
                        _offering("Commerce Engine", ("subscription",)),
                        _offering("Payment Services", ("transaction fees",)),
                        _offering("Storefront OS", ("subscription",)),
                        _offering("Support Desk", ("service fees",)),
                    ),
                ),
                SyntheticSegment(
                    name="Consumer Commerce",
                    customer_types=("consumers",),
                    channels=("online", "retail"),
                    offerings=(
                        _offering("Loyalty Cloud", ("subscription",)),
                        _offering("Marketplace Hub", ("transaction fees",)),
                        _offering("Device Sync", ("hardware sales",)),
                        _offering("Recommendation Engine", ("consumption-based",)),
                        _offering("Gift Network", ("transaction fees",)),
                    ),
                ),
                SyntheticSegment(
                    name="Analytics",
                    customer_types=("retailers", "large enterprises", "developers"),
                    channels=("direct sales", "system integrators"),
                    offerings=(
                        _offering("Analytics Studio", ("subscription",)),
                        _offering("Forecast Engine", ("consumption-based",)),
                        _offering("Model Hub", ("subscription",)),
                        _offering("Identity Cloud", ("subscription",)),
                        _offering("Inventory Grid", ("subscription",)),
                    ),
                ),
                SyntheticSegment(
                    name="Wholesale Network",
                    customer_types=("manufacturers", "retailers"),
                    channels=("distributors", "resellers"),
                    offerings=(
                        _offering("Supply Exchange", ("transaction fees",)),
                        _offering("Inventory Grid", ("subscription",)),
                        _offering("Compliance Vault", ("subscription",)),
                        _offering("Fulfillment Orchestrator", ("service fees",)),
                        _offering("Commerce Engine", ("subscription",)),
                    ),
                ),
            ),
            places=("United States", "Mexico", "Latin America", "Japan"),
            partners=("RetailNet", "Fujitsu Limited", "PaySphere"),
        ),
        SyntheticCompany(
            graph_id="lattice",
            name="Lattice Finance",
            segments=(
                SyntheticSegment(
                    name="Enterprise Banking",
                    customer_types=("financial services firms", "large enterprises", "government agencies"),
                    channels=("direct sales", "system integrators"),
                    offerings=(
                        _offering("Risk Console", ("subscription",)),
                        _offering("Decision Hub", ("subscription",)),
                        _offering("Forecast Engine", ("consumption-based",)),
                        _offering("Compliance Vault", ("subscription",)),
                        _offering("Identity Cloud", ("subscription",)),
                    ),
                ),
                SyntheticSegment(
                    name="Payments Network",
                    customer_types=("financial services firms", "small businesses", "retailers"),
                    channels=("direct sales", "resellers", "marketplaces"),
                    offerings=(
                        _offering("Payment Services", ("transaction fees",)),
                        _offering("Marketplace Hub", ("transaction fees",)),
                        _offering("Data Exchange", ("licensing",)),
                        _offering("Fraud Grid", ("subscription",)),
                        _offering("Merchant Console", ("subscription",)),
                    ),
                ),
                SyntheticSegment(
                    name="Developer Platform",
                    customer_types=("developers", "IT professionals"),
                    channels=("online", "direct sales"),
                    offerings=(
                        _offering("Developer Toolkit", ("subscription",)),
                        _offering("Cloud Platform", ("subscription", "consumption-based")),
                        _offering("Model Hub", ("subscription",)),
                        _offering("Secure Collaboration", ("subscription",)),
                        _offering("API Gateway", ("consumption-based",)),
                    ),
                ),
                SyntheticSegment(
                    name="Compliance Services",
                    customer_types=("government agencies", "financial services firms", "healthcare organizations"),
                    channels=("direct sales", "managed service providers"),
                    offerings=(
                        _offering("Secure Records", ("subscription",)),
                        _offering("Compliance Vault", ("subscription",)),
                        _offering("Audit Exchange", ("service fees",)),
                        _offering("Support Desk", ("service fees",)),
                        _offering("Workflow Automation", ("service fees",)),
                    ),
                ),
            ),
            places=("United States", "Canada", "United Kingdom", "Europe"),
            partners=("CloudAtlas", "PaySphere", "Dell"),
        ),
        SyntheticCompany(
            graph_id="vector",
            name="Vector Industrial",
            segments=(
                SyntheticSegment(
                    name="Connected Operations",
                    customer_types=("manufacturers", "large enterprises", "IT professionals"),
                    channels=("direct sales", "system integrators", "managed service providers"),
                    offerings=(
                        _offering("Automation Suite", ("subscription",)),
                        _offering("Cloud Platform", ("consumption-based",)),
                        _offering("Field Operations Suite", ("subscription",)),
                        _offering("Device Sync", ("hardware sales",)),
                        _offering("Support Desk", ("service fees",)),
                    ),
                ),
                SyntheticSegment(
                    name="Supply Chain",
                    customer_types=("manufacturers", "retailers", "mid-market companies"),
                    channels=("distributors", "resellers"),
                    offerings=(
                        _offering("Inventory Grid", ("subscription",)),
                        _offering("Supply Exchange", ("transaction fees",)),
                        _offering("Commerce Engine", ("subscription",)),
                        _offering("Fulfillment Orchestrator", ("service fees",)),
                        _offering("Procurement Exchange", ("transaction fees",)),
                    ),
                ),
                SyntheticSegment(
                    name="Developer Tools",
                    customer_types=("developers", "IT professionals", "educational institutions"),
                    channels=("online", "OEMs"),
                    offerings=(
                        _offering("Developer Toolkit", ("subscription",)),
                        _offering("Model Hub", ("subscription",)),
                        _offering("Identity Cloud", ("subscription",)),
                        _offering("Analytics Studio", ("subscription",)),
                        _offering("API Gateway", ("consumption-based",)),
                    ),
                ),
                SyntheticSegment(
                    name="Energy Services",
                    customer_types=("government agencies", "manufacturers", "large enterprises"),
                    channels=("direct sales", "resellers"),
                    offerings=(
                        _offering("Forecast Engine", ("consumption-based",)),
                        _offering("Compliance Vault", ("subscription",)),
                        _offering("Risk Console", ("subscription",)),
                        _offering("Data Exchange", ("licensing",)),
                        _offering("Secure Records", ("subscription",)),
                    ),
                ),
            ),
            places=("Germany", "Japan", "Australia", "APAC"),
            partners=("Fujitsu Limited", "CloudAtlas", "GridWorks"),
        ),
    )


def _scale_targets(targets: dict[str, int], total: int) -> dict[str, int]:
    base_total = sum(targets.values())
    if total <= 0 or base_total <= 0:
        return {label: 0 for label in targets}

    exact = {label: (value * total) / base_total for label, value in targets.items()}
    scaled = {label: int(amount) for label, amount in exact.items()}
    remainder = total - sum(scaled.values())
    if remainder > 0:
        ranked = sorted(
            targets,
            key=lambda label: (exact[label] - scaled[label], targets[label]),
            reverse=True,
        )
        for label in ranked[:remainder]:
            scaled[label] += 1
    return scaled


def _select_question(
    split: SplitName,
    rng: random.Random,
    *,
    train: tuple[str, ...],
    validation: tuple[str, ...] | None = None,
    release_eval: tuple[str, ...] | None = None,
) -> str:
    if split == "validation" and validation:
        choices = validation
    elif split == "release_eval" and release_eval:
        choices = release_eval
    else:
        choices = train
    return rng.choice(list(choices))


def _graph_source_id(companies: Iterable[SyntheticCompany]) -> str:
    return "+".join(sorted(company.graph_id for company in companies))


def _all_offerings(segment: SyntheticSegment) -> list[SyntheticOffering]:
    offerings: list[SyntheticOffering] = []

    def walk(node: SyntheticOffering) -> None:
        offerings.append(node)
        for child in node.children:
            walk(child)

    for root in segment.offerings:
        walk(root)
    return offerings


def _descendant_offering_names(segment: SyntheticSegment, root_name: str) -> set[str]:
    names: set[str] = set()

    def collect(node: SyntheticOffering) -> None:
        names.add(node.name)
        for child in node.children:
            collect(child)

    def walk(node: SyntheticOffering) -> None:
        if node.name == root_name:
            collect(node)
            return
        for child in node.children:
            walk(child)

    for root in segment.offerings:
        walk(root)
    return names


def _segment_revenue_models(segment: SyntheticSegment, *, descendant: bool) -> set[str]:
    offerings = _all_offerings(segment) if descendant else list(segment.offerings)
    revenue_models: set[str] = set()
    for offering in offerings:
        revenue_models.update(offering.revenue_models)
    return revenue_models


def _company_matches_place(company: SyntheticCompany, places: set[str]) -> bool:
    for requested_place in places:
        normalized_requested = normalize_place_name(requested_place)
        for company_place in company.places:
            if classify_place_match(normalized_requested, company_place) is not None:
                return True
    return False


def _segment_matches(company: SyntheticCompany, segment: SyntheticSegment, payload: QueryPlanPayload) -> bool:
    if payload.companies and company.name not in payload.companies:
        return False
    if payload.segments and segment.name not in payload.segments:
        return False
    if payload.customer_types and not set(payload.customer_types).issubset(segment.customer_types):
        return False
    if payload.channels and not set(payload.channels).issubset(segment.channels):
        return False

    hierarchy_mode = payload.hierarchy_mode or ("descendant" if payload.revenue_models else "direct")
    direct_names = {offering.name for offering in segment.offerings}
    descendant_names = {offering.name for offering in _all_offerings(segment)}
    candidate_names = descendant_names if hierarchy_mode == "descendant" else direct_names
    if payload.offerings and not set(payload.offerings).issubset(candidate_names):
        return False

    if payload.revenue_models:
        revenue_models = _segment_revenue_models(segment, descendant=hierarchy_mode == "descendant")
        if not set(payload.revenue_models).issubset(revenue_models):
            return False
    return True


def _company_matches_cross_segment(company: SyntheticCompany, payload: QueryPlanPayload) -> bool:
    if payload.companies and company.name not in payload.companies:
        return False
    atoms = []
    atoms.extend(("customer_type", value) for value in payload.customer_types)
    atoms.extend(("channel", value) for value in payload.channels)
    atoms.extend(("offering", value) for value in payload.offerings)
    atoms.extend(("revenue_model", value) for value in payload.revenue_models)
    if len(atoms) < 2:
        return False

    for kind, value in atoms:
        if kind == "customer_type":
            if not any(_segment_matches(company, segment, QueryPlanPayload(segments=payload.segments, customer_types=[value])) for segment in company.segments):
                return False
        elif kind == "channel":
            if not any(_segment_matches(company, segment, QueryPlanPayload(segments=payload.segments, channels=[value])) for segment in company.segments):
                return False
        elif kind == "offering":
            if not any(_segment_matches(company, segment, QueryPlanPayload(segments=payload.segments, offerings=[value], hierarchy_mode=payload.hierarchy_mode)) for segment in company.segments):
                return False
        else:
            if not any(_segment_matches(company, segment, QueryPlanPayload(segments=payload.segments, revenue_models=[value], hierarchy_mode=payload.hierarchy_mode)) for segment in company.segments):
                return False
    return True


def _company_matches_descendant_revenue(
    company: SyntheticCompany,
    root_names: set[str],
    revenue_models: set[str],
) -> bool:
    matched_revenue_models: set[str] = set()
    for segment in company.segments:
        for root_name in root_names:
            descendant_names = _descendant_offering_names(segment, root_name)
            if not descendant_names:
                continue
            for offering in _all_offerings(segment):
                if offering.name in descendant_names:
                    matched_revenue_models.update(offering.revenue_models)
    return revenue_models.issubset(matched_revenue_models)


def evaluate_query_plan(companies: tuple[SyntheticCompany, ...], plan: QueryPlanEnvelope) -> list[dict[str, Any]]:
    if not plan.answerable:
        return []

    family = plan.family
    payload = plan.payload or QueryPlanPayload()
    company_map = {company.name: company for company in companies}

    if family == "companies_list":
        return [{"company": company.name} for company in sorted(companies, key=lambda item: item.name)]

    if family == "segments_by_company":
        rows: list[dict[str, Any]] = []
        for company_name in payload.companies:
            company = company_map[company_name]
            for segment in company.segments:
                rows.append({"company": company.name, "segment": segment.name})
        return sorted(rows, key=lambda row: (row["company"], row["segment"]))

    if family == "offerings_by_company":
        rows: set[tuple[str, str]] = set()
        for company_name in payload.companies:
            company = company_map[company_name]
            for segment in company.segments:
                for offering in segment.offerings:
                    rows.add((company.name, offering.name))
        return [
            {"company": company_name, "offering": offering_name}
            for company_name, offering_name in sorted(rows, key=lambda row: (row[0], row[1]))
        ]

    if family == "offerings_by_segment":
        rows = []
        company_names = set(payload.companies) if payload.companies else {company.name for company in companies}
        segment_names = set(payload.segments)
        for company in companies:
            if company.name not in company_names:
                continue
            for segment in company.segments:
                if segment.name not in segment_names:
                    continue
                for offering in segment.offerings:
                    rows.append({"company": company.name, "segment": segment.name, "offering": offering.name})
        return sorted(rows, key=lambda row: (row["company"], row["segment"], row["offering"]))

    if family == "companies_by_segment_filters":
        names = sorted(
            company.name
            for company in companies
            if any(_segment_matches(company, segment, payload) for segment in company.segments)
        )
        return [{"company": name} for name in names]

    if family == "segments_by_segment_filters":
        rows = []
        for company in companies:
            for segment in company.segments:
                if _segment_matches(company, segment, payload):
                    rows.append({"company": company.name, "segment": segment.name})
        return sorted(rows, key=lambda row: (row["company"], row["segment"]))

    if family == "companies_by_cross_segment_filters":
        rows = [{"company": company.name} for company in companies if _company_matches_cross_segment(company, payload)]
        return sorted(rows, key=lambda row: row["company"])

    if family == "descendant_offerings_by_root":
        rows = []
        company_names = set(payload.companies) if payload.companies else {company.name for company in companies}
        root_names = set(payload.offerings)
        for company in companies:
            if company.name not in company_names:
                continue
            names: set[str] = set()
            for segment in company.segments:
                for root_name in root_names:
                    names.update(_descendant_offering_names(segment, root_name))
            for offering_name in sorted(names):
                rows.append({"company": company.name, "offering": offering_name})
        return sorted(rows, key=lambda row: (row["company"], row["offering"]))

    if family == "companies_by_descendant_revenue":
        rows = []
        root_names = set(payload.offerings)
        revenue_models = set(payload.revenue_models)
        company_names = set(payload.companies) if payload.companies else {company.name for company in companies}
        places = {normalize_place_name(place) for place in payload.places}
        for company in companies:
            if company.name not in company_names:
                continue
            if places and not _company_matches_place(company, places):
                continue
            if _company_matches_descendant_revenue(company, root_names, revenue_models):
                rows.append({"company": company.name})
        return sorted(rows, key=lambda row: row["company"])

    if family == "companies_by_place":
        places = {normalize_place_name(place) for place in payload.places}
        rows = [{"company": company.name} for company in companies if _company_matches_place(company, places)]
        return sorted(rows, key=lambda row: row["company"])

    if family == "segments_by_place_and_segment_filters":
        places = {normalize_place_name(place) for place in payload.places}
        rows = []
        for company in companies:
            if not _company_matches_place(company, places):
                continue
            for segment in company.segments:
                if _segment_matches(company, segment, payload):
                    rows.append({"company": company.name, "segment": segment.name})
        return sorted(rows, key=lambda row: (row["company"], row["segment"]))

    if family == "companies_by_partner":
        partners = set(payload.partners)
        company_names = set(payload.companies) if payload.companies else {company.name for company in companies}
        rows = [
            {"company": company.name}
            for company in companies
            if company.name in company_names and partners.intersection(company.partners)
        ]
        return sorted(rows, key=lambda row: row["company"])

    if family == "boolean_exists":
        nested_plan = QueryPlanEnvelope(answerable=True, family=payload.base_family, payload=payload)
        return [{"is_match": bool(evaluate_query_plan(companies, nested_plan))}]

    if family == "count_aggregate":
        aggregate_spec = payload.aggregate_spec or {}
        nested_family = payload.base_family or aggregate_spec.get("base_family")
        nested_plan = QueryPlanEnvelope(answerable=True, family=nested_family, payload=payload)
        rows = evaluate_query_plan(companies, nested_plan)
        count_target = aggregate_spec.get("count_target", "company")
        alias = f"{count_target}_count"
        if count_target == "company":
            value = len({row["company"] for row in rows})
        elif count_target == "segment":
            value = len({(row["company"], row["segment"]) for row in rows})
        else:
            value = len({(row.get("company"), row["offering"]) for row in rows})
        return [{alias: value}]

    if family == "ranking_topk":
        metric = (payload.aggregate_spec or {}).get("ranking_metric")
        limit = payload.limit or 5
        normalized_places = {normalize_place_name(place) for place in payload.places}
        if metric == "customer_type_by_company_count":
            counts: dict[str, set[str]] = {}
            for company in companies:
                if normalized_places and not _company_matches_place(company, normalized_places):
                    continue
                if payload.companies and company.name not in payload.companies:
                    continue
                seen: set[str] = set()
                for segment in company.segments:
                    seen.update(segment.customer_types)
                for value in seen:
                    counts.setdefault(value, set()).add(company.name)
            rows = [{"customer_type": name, "company_count": len(company_names)} for name, company_names in counts.items()]
            return sorted(rows, key=lambda row: (-row["company_count"], row["customer_type"]))[:limit]
        if metric == "channel_by_segment_count":
            counts: dict[str, set[tuple[str, str]]] = {}
            for company in companies:
                if normalized_places and not _company_matches_place(company, normalized_places):
                    continue
                if payload.companies and company.name not in payload.companies:
                    continue
                for segment in company.segments:
                    for channel in segment.channels:
                        counts.setdefault(channel, set()).add((company.name, segment.name))
            rows = [{"channel": name, "segment_count": len(segments)} for name, segments in counts.items()]
            return sorted(rows, key=lambda row: (-row["segment_count"], row["channel"]))[:limit]
        if metric == "revenue_model_by_company_count":
            counts: dict[str, set[str]] = {}
            for company in companies:
                if normalized_places and not _company_matches_place(company, normalized_places):
                    continue
                if payload.companies and company.name not in payload.companies:
                    continue
                seen: set[str] = set()
                for segment in company.segments:
                    seen.update(_segment_revenue_models(segment, descendant=True))
                for value in seen:
                    counts.setdefault(value, set()).add(company.name)
            rows = [{"revenue_model": name, "company_count": len(company_names)} for name, company_names in counts.items()]
            return sorted(rows, key=lambda row: (-row["company_count"], row["revenue_model"]))[:limit]
        counts: dict[str, int] = {}
        for company in companies:
            if normalized_places and not _company_matches_place(company, normalized_places):
                continue
            counts[company.name] = sum(1 for segment in company.segments if _segment_matches(company, segment, payload))
        rows = [{"company": name, "segment_count": count} for name, count in counts.items() if count > 0]
        return sorted(rows, key=lambda row: (-row["segment_count"], row["company"]))[:limit]

    return []


def _metadata_for_example(
    *,
    route_label: RouteLabel,
    family: str,
    payload: QueryPlanPayload | None,
    source_graph_id: str,
) -> dict[str, Any]:
    payload = payload or QueryPlanPayload()
    return {
        "route_label": route_label,
        "family": family,
        "filter_count": sum(
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
        ),
        "boolean_depth": max(
            0,
            len(payload.customer_types) + len(payload.channels) + len(payload.offerings) + len(payload.revenue_models) - 1,
        ),
        "has_geography": bool(payload.places),
        "has_descendant_offering": payload.hierarchy_mode == "descendant",
        "has_aggregation": family in {"count_aggregate", "ranking_topk"},
        "source_graph_id": source_graph_id,
    }


def _example_from_plan(
    companies: tuple[SyntheticCompany, ...],
    *,
    question: str,
    plan: QueryPlanEnvelope,
    route_label: RouteLabel,
    source_graph_id: str,
) -> DatasetExample:
    compiled = compile_query_plan(plan)
    if route_label == "local_safe" and not compiled.answerable:
        raise ValueError(f"Local-safe example compiled to refusal for question: {question}")
    if route_label != "local_safe" and compiled.answerable:
        raise ValueError(f"Non-local example compiled to an answer for question: {question}")
    rows = evaluate_query_plan(companies, plan) if compiled.answerable else []
    payload = plan.payload or QueryPlanPayload()
    resolved_family = plan.family or route_label
    return DatasetExample(
        question=question,
        target=plan.model_dump(mode="json", exclude_none=True),
        route_label=route_label,
        family=resolved_family,
        gold_cypher=compiled.cypher if compiled.answerable else None,
        gold_params=compiled.params if compiled.answerable else {},
        gold_rows=rows,
        metadata=_metadata_for_example(
            route_label=route_label,
            family=resolved_family,
            payload=payload,
            source_graph_id=source_graph_id,
        ),
    )


def _pick_company(companies: tuple[SyntheticCompany, ...], rng: random.Random) -> SyntheticCompany:
    return rng.choice(list(companies))


def _pick_two_companies(companies: tuple[SyntheticCompany, ...], rng: random.Random) -> tuple[SyntheticCompany, SyntheticCompany]:
    return tuple(rng.sample(list(companies), 2))  # type: ignore[return-value]


def _pick_hierarchy_segment(company: SyntheticCompany, rng: random.Random) -> SyntheticSegment:
    candidates = [segment for segment in company.segments if any(offering.children for offering in segment.offerings)]
    if not candidates:
        raise ValueError(f"Company {company.name} does not contain hierarchy-ready offerings.")
    return rng.choice(candidates)


def _join_names(values: list[str]) -> str:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _build_inventory_example(companies: tuple[SyntheticCompany, ...], rng: random.Random, *, split: SplitName) -> DatasetExample:
    family = rng.choice(["companies_list", "segments_by_company", "offerings_by_company", "offerings_by_segment"])
    if family == "companies_list":
        plan = QueryPlanEnvelope(answerable=True, family=family, payload=QueryPlanPayload())
        question = _select_question(
            split,
            rng,
            train=("Which companies are in the graph?",),
            validation=("List the companies represented in the graph.",),
            release_eval=("Name every company represented in the knowledge graph.",),
        )
        return _example_from_plan(
            companies,
            question=question,
            plan=plan,
            route_label="local_safe",
            source_graph_id=_graph_source_id(companies),
        )

    company_group = [_pick_company(companies, rng)]
    if family in {"segments_by_company", "offerings_by_company"} and len(companies) >= 2 and rng.random() < 0.35:
        company_group = list(_pick_two_companies(companies, rng))
    company_names = sorted(company.name for company in company_group)
    source_graph_id = _graph_source_id(company_group)

    if family == "segments_by_company":
        plan = QueryPlanEnvelope(answerable=True, family=family, payload=QueryPlanPayload(companies=company_names))
        question = _select_question(
            split,
            rng,
            train=(f"What are the business segments of {_join_names(company_names)}?",),
            validation=(f"List the business segments for {_join_names(company_names)}.",),
            release_eval=(f"Which business segments belong to {_join_names(company_names)}?",),
        )
        return _example_from_plan(companies, question=question, plan=plan, route_label="local_safe", source_graph_id=source_graph_id)

    if family == "offerings_by_company":
        plan = QueryPlanEnvelope(answerable=True, family=family, payload=QueryPlanPayload(companies=company_names))
        question = _select_question(
            split,
            rng,
            train=(f"Which offerings does {_join_names(company_names)} have?",),
            validation=(f"List the offerings for {_join_names(company_names)}.",),
            release_eval=(f"What offerings belong to {_join_names(company_names)}?",),
        )
        return _example_from_plan(companies, question=question, plan=plan, route_label="local_safe", source_graph_id=source_graph_id)

    company = company_group[0]
    segment = rng.choice(list(company.segments))
    plan = QueryPlanEnvelope(
        answerable=True,
        family=family,
        payload=QueryPlanPayload(companies=[company.name], segments=[segment.name]),
    )
    question = _select_question(
        split,
        rng,
        train=(f"What offerings are in the {segment.name} segment at {company.name}?",),
        validation=(f"List the offerings in the {segment.name} segment for {company.name}.",),
        release_eval=(f"Which offerings sit inside {company.name}'s {segment.name} segment?",),
    )
    return _example_from_plan(companies, question=question, plan=plan, route_label="local_safe", source_graph_id=company.graph_id)


def _build_same_segment_example(companies: tuple[SyntheticCompany, ...], rng: random.Random, *, split: SplitName) -> DatasetExample:
    company = _pick_company(companies, rng)
    segment = rng.choice(list(company.segments))
    family = rng.choice(["companies_by_segment_filters", "segments_by_segment_filters"])
    plan = QueryPlanEnvelope(
        answerable=True,
        family=family,
        payload=QueryPlanPayload(
            customer_types=[segment.customer_types[0]],
            channels=[segment.channels[0]],
            binding_scope="same_segment",
        ),
    )
    if family == "companies_by_segment_filters":
        question = _select_question(
            split,
            rng,
            train=(f"Which companies sell to {segment.customer_types[0]} through {segment.channels[0]}?",),
            validation=(f"List the companies with a segment that serves {segment.customer_types[0]} via {segment.channels[0]}.",),
            release_eval=(f"Name the companies that reach {segment.customer_types[0]} through {segment.channels[0]}.",),
        )
    else:
        question = _select_question(
            split,
            rng,
            train=(f"Which segments sell to {segment.customer_types[0]} through {segment.channels[0]}?",),
            validation=(f"List the segments that serve {segment.customer_types[0]} via {segment.channels[0]}.",),
            release_eval=(f"Which business segments reach {segment.customer_types[0]} through {segment.channels[0]}?",),
        )
    return _example_from_plan(companies, question=question, plan=plan, route_label="local_safe", source_graph_id=company.graph_id)


def _build_cross_segment_example(companies: tuple[SyntheticCompany, ...], rng: random.Random, *, split: SplitName) -> DatasetExample:
    company = _pick_company(companies, rng)
    customer_types = sorted({customer_type for segment in company.segments for customer_type in segment.customer_types})
    if len(customer_types) < 2:
        raise ValueError(f"Company {company.name} does not have enough distinct customer types for a cross-segment example.")
    first_customer_type, second_customer_type = rng.sample(customer_types, 2)
    plan = QueryPlanEnvelope(
        answerable=True,
        family="companies_by_cross_segment_filters",
        payload=QueryPlanPayload(
            customer_types=[first_customer_type, second_customer_type],
            binding_scope="across_segments",
        ),
    )
    question = _select_question(
        split,
        rng,
        train=(f"Which companies serve {first_customer_type} and {second_customer_type}?",),
        validation=(f"List the companies that cover both {first_customer_type} and {second_customer_type}, even across segments.",),
        release_eval=(f"Which companies reach both {first_customer_type} and {second_customer_type} across their segment portfolio?",),
    )
    return _example_from_plan(companies, question=question, plan=plan, route_label="local_safe", source_graph_id=company.graph_id)


def _build_hierarchy_example(companies: tuple[SyntheticCompany, ...], rng: random.Random, *, split: SplitName) -> DatasetExample:
    hierarchy_companies = [company for company in companies if any(any(offering.children for offering in segment.offerings) for segment in company.segments)]
    if not hierarchy_companies:
        raise ValueError("The selected split does not contain any hierarchy-ready companies.")
    company = rng.choice(hierarchy_companies)
    segment = _pick_hierarchy_segment(company, rng)
    root = rng.choice([offering for offering in segment.offerings if offering.children])
    if rng.random() < 0.5:
        plan = QueryPlanEnvelope(
            answerable=True,
            family="descendant_offerings_by_root",
            payload=QueryPlanPayload(companies=[company.name], offerings=[root.name], hierarchy_mode="descendant"),
        )
        question = _select_question(
            split,
            rng,
            train=(f"Which offerings descend from {root.name} at {company.name}?",),
            validation=(f"List the descendant offerings under {root.name} for {company.name}.",),
            release_eval=(f"Which offerings sit under the {root.name} family at {company.name}?",),
        )
    else:
        plan = QueryPlanEnvelope(
            answerable=True,
            family="companies_by_descendant_revenue",
            payload=QueryPlanPayload(
                offerings=[root.name],
                revenue_models=[root.revenue_models[0]],
                hierarchy_mode="descendant",
            ),
        )
        question = _select_question(
            split,
            rng,
            train=(f"Which companies monetize descendant offerings of {root.name} via {root.revenue_models[0]}?",),
            validation=(f"List the companies whose descendants of {root.name} monetize through {root.revenue_models[0]}.",),
            release_eval=(f"Which companies monetize the {root.name} offering family through {root.revenue_models[0]}?",),
        )
    return _example_from_plan(companies, question=question, plan=plan, route_label="local_safe", source_graph_id=company.graph_id)


def _build_geography_example(companies: tuple[SyntheticCompany, ...], rng: random.Random, *, split: SplitName) -> DatasetExample:
    company = _pick_company(companies, rng)
    place = rng.choice(list(company.places))
    if rng.random() < 0.5:
        plan = QueryPlanEnvelope(answerable=True, family="companies_by_place", payload=QueryPlanPayload(places=[place]))
        question = _select_question(
            split,
            rng,
            train=(f"Which companies operate in {place}?",),
            validation=(f"List the companies operating in {place}.",),
            release_eval=(f"Which companies have an operating footprint in {place}?",),
        )
    else:
        segment = rng.choice(list(company.segments))
        plan = QueryPlanEnvelope(
            answerable=True,
            family="segments_by_place_and_segment_filters",
            payload=QueryPlanPayload(places=[place], channels=[segment.channels[0]], binding_scope="same_segment"),
        )
        question = _select_question(
            split,
            rng,
            train=(f"Which company segments at companies operating in {place} sell through {segment.channels[0]}?",),
            validation=(f"List the segments at companies in {place} that sell through {segment.channels[0]}.",),
            release_eval=(f"Which segments belong to companies operating in {place} and sell through {segment.channels[0]}?",),
        )
    return _example_from_plan(companies, question=question, plan=plan, route_label="local_safe", source_graph_id=company.graph_id)


def _build_partner_example(companies: tuple[SyntheticCompany, ...], rng: random.Random, *, split: SplitName) -> DatasetExample:
    company = _pick_company(companies, rng)
    partner = rng.choice(list(company.partners))
    plan = QueryPlanEnvelope(answerable=True, family="companies_by_partner", payload=QueryPlanPayload(partners=[partner]))
    question = _select_question(
        split,
        rng,
        train=(f"Which companies partner with {partner}?",),
        validation=(f"List the companies partnered with {partner}.",),
        release_eval=(f"Which companies have a partnership with {partner}?",),
    )
    return _example_from_plan(companies, question=question, plan=plan, route_label="local_safe", source_graph_id=company.graph_id)


def _build_boolean_example(companies: tuple[SyntheticCompany, ...], rng: random.Random, *, split: SplitName) -> DatasetExample:
    company = _pick_company(companies, rng)
    segment = rng.choice(list(company.segments))
    plan = QueryPlanEnvelope(
        answerable=True,
        family="boolean_exists",
        payload=QueryPlanPayload(
            base_family="companies_by_segment_filters",
            companies=[company.name],
            customer_types=[segment.customer_types[0]],
            channels=[segment.channels[0]],
        ),
    )
    question = _select_question(
        split,
        rng,
        train=(f"Does {company.name} sell to {segment.customer_types[0]} through {segment.channels[0]}?",),
        validation=(f"Does {company.name} have a segment serving {segment.customer_types[0]} via {segment.channels[0]}?",),
        release_eval=(f"Is there a {company.name} segment that reaches {segment.customer_types[0]} through {segment.channels[0]}?",),
    )
    return _example_from_plan(companies, question=question, plan=plan, route_label="local_safe", source_graph_id=company.graph_id)


def _build_count_example(companies: tuple[SyntheticCompany, ...], rng: random.Random, *, split: SplitName) -> DatasetExample:
    company = _pick_company(companies, rng)
    segment = rng.choice(list(company.segments))
    plan = QueryPlanEnvelope(
        answerable=True,
        family="count_aggregate",
        payload=QueryPlanPayload(
            customer_types=[segment.customer_types[0]],
            base_family="companies_by_segment_filters",
            aggregate_spec={
                "kind": "count",
                "base_family": "companies_by_segment_filters",
                "count_target": "company",
            },
        ),
    )
    question = _select_question(
        split,
        rng,
        train=(f"How many companies serve {segment.customer_types[0]}?",),
        validation=(f"Count the companies that serve {segment.customer_types[0]}.",),
        release_eval=(f"What is the company count for {segment.customer_types[0]} coverage?",),
    )
    return _example_from_plan(companies, question=question, plan=plan, route_label="local_safe", source_graph_id=company.graph_id)


def _build_ranking_example(companies: tuple[SyntheticCompany, ...], rng: random.Random, *, split: SplitName) -> DatasetExample:
    metric = rng.choice(
        [
            "customer_type_by_company_count",
            "channel_by_segment_count",
            "revenue_model_by_company_count",
            "company_by_matched_segment_count",
        ]
    )
    payload = QueryPlanPayload(
        aggregate_spec={"kind": "ranking", "ranking_metric": metric},
        limit=3,
    )
    if metric == "company_by_matched_segment_count":
        payload = QueryPlanPayload(
            customer_types=["developers"],
            aggregate_spec={"kind": "ranking", "ranking_metric": metric},
            limit=3,
        )
    plan = QueryPlanEnvelope(answerable=True, family="ranking_topk", payload=payload)
    question_map = {
        "customer_type_by_company_count": _select_question(
            split,
            rng,
            train=("What customer type is served by the highest number of companies?",),
            validation=("Which customer type appears across the most companies?",),
            release_eval=("What customer type has the broadest company coverage?",),
        ),
        "channel_by_segment_count": _select_question(
            split,
            rng,
            train=("What channel is used by the highest number of segments?",),
            validation=("Which channel appears across the most segments?",),
            release_eval=("What sales channel has the widest segment usage?",),
        ),
        "revenue_model_by_company_count": _select_question(
            split,
            rng,
            train=("What revenue model appears in the highest number of companies?",),
            validation=("Which revenue model shows up across the most companies?",),
            release_eval=("What revenue model has the broadest company footprint?",),
        ),
        "company_by_matched_segment_count": _select_question(
            split,
            rng,
            train=("Which companies have the highest number of matching segments serving developers?",),
            validation=("Which companies have the most developer-serving matching segments?",),
            release_eval=("Which companies rank highest by matching developer-facing segments?",),
        ),
    }
    return _example_from_plan(
        companies,
        question=question_map[metric],
        plan=plan,
        route_label="local_safe",
        source_graph_id=_graph_source_id(companies),
    )


def _generate_local_safe_examples(
    companies: tuple[SyntheticCompany, ...],
    count: int,
    rng: random.Random,
    *,
    split: SplitName,
) -> list[DatasetExample]:
    quotas = _scale_targets(DEFAULT_LOCAL_SAFE_FAMILY_TARGETS, count)
    builders = {
        "inventory": _build_inventory_example,
        "same_segment": _build_same_segment_example,
        "cross_segment": _build_cross_segment_example,
        "hierarchy": _build_hierarchy_example,
        "geography": _build_geography_example,
        "partner": _build_partner_example,
        "boolean": _build_boolean_example,
        "count": _build_count_example,
        "ranking": _build_ranking_example,
    }
    examples: list[DatasetExample] = []
    for bucket, bucket_count in quotas.items():
        for _ in range(bucket_count):
            examples.append(builders[bucket](companies, rng, split=split))
    return examples


def _generate_strong_model_candidates(
    companies: tuple[SyntheticCompany, ...],
    count: int,
    rng: random.Random,
    *,
    split: SplitName,
) -> list[DatasetExample]:
    examples: list[DatasetExample] = []
    for _ in range(count):
        company_a, company_b = _pick_two_companies(companies, rng)
        questions = [
            _select_question(
                split,
                rng,
                train=(f"Compare {company_a.name} and {company_b.name} by the customer types they serve.",),
                validation=(f"Compare the customer-type coverage of {company_a.name} and {company_b.name}.",),
                release_eval=(f"How do {company_a.name} and {company_b.name} differ in the customer types they cover?",),
            ),
            _select_question(
                split,
                rng,
                train=(f"Which segments does {company_a.name} have that {company_b.name} does not?",),
                validation=(f"List the segments unique to {company_a.name} compared with {company_b.name}.",),
                release_eval=(f"What business segments appear in {company_a.name} but not in {company_b.name}?",),
            ),
            _select_question(
                split,
                rng,
                train=(f"Why does {company_a.name} match this query better than {company_b.name}?",),
                validation=(f"Explain why {company_a.name} is a better match than {company_b.name}.",),
                release_eval=(f"Why should {company_a.name} rank ahead of {company_b.name} for this request?",),
            ),
        ]
        plan = QueryPlanEnvelope(answerable=False, reason="beyond_local_coverage")
        examples.append(
            _example_from_plan(
                companies,
                question=rng.choice(questions),
                plan=plan,
                route_label="strong_model_candidate",
                source_graph_id=_graph_source_id((company_a, company_b)),
            )
        )
    return examples


def _generate_refusal_examples(
    companies: tuple[SyntheticCompany, ...],
    count: int,
    rng: random.Random,
    *,
    split: SplitName,
) -> list[DatasetExample]:
    examples: list[DatasetExample] = []
    reasons = [
        ("unsupported_time", _select_question(split, rng, train=("What was {company}'s revenue in 2024?",), validation=("Show {company}'s revenue for 2024.",), release_eval=("What revenue did {company} report in 2024?",))),
        ("unsupported_metric", _select_question(split, rng, train=("Which company grew fastest year over year?",), validation=("Which company had the strongest year-over-year growth?",), release_eval=("What company posted the fastest annual growth?",))),
        ("unsupported_schema", _select_question(split, rng, train=("Which suppliers does {company} use?",), validation=("List the suppliers for {company}.",), release_eval=("Who supplies {company}?",))),
        ("write_request", _select_question(split, rng, train=("Add a new segment to {company}.",), validation=("Create a new business segment for {company}.",), release_eval=("Insert a new segment for {company}.",))),
        ("ambiguous_closed_label", _select_question(split, rng, train=("Which companies sell through affiliates?",), validation=("List the companies that use affiliates as a channel.",), release_eval=("Which companies go to market through affiliates?",))),
        ("ambiguous_request", _select_question(split, rng, train=("Which segment is the best one?",), validation=("Show me the best segment.",), release_eval=("Which segment should I pick?",))),
    ]
    for _ in range(count):
        company = _pick_company(companies, rng)
        reason, template = rng.choice(reasons)
        examples.append(
            _example_from_plan(
                companies,
                question=template.format(company=company.name),
                plan=QueryPlanEnvelope(answerable=False, reason=reason),
                route_label="refuse",
                source_graph_id=company.graph_id,
            )
        )
    return examples


def _build_split_examples(
    companies: tuple[SyntheticCompany, ...],
    *,
    split: SplitName,
    size: int,
    rng: random.Random,
) -> list[DatasetExample]:
    route_targets = _scale_targets(DEFAULT_ROUTE_TARGETS_BY_SPLIT[split], size)
    examples: list[DatasetExample] = []
    examples.extend(_generate_local_safe_examples(companies, route_targets["local_safe"], rng, split=split))
    examples.extend(_generate_strong_model_candidates(companies, route_targets["strong_model_candidate"], rng, split=split))
    examples.extend(_generate_refusal_examples(companies, route_targets["refuse"], rng, split=split))
    rng.shuffle(examples)
    return examples


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
            rng=random.Random(seed + SPLIT_SEED_OFFSETS["train"]),
        ),
        "validation": _build_split_examples(
            graphs_by_split["validation"],
            split="validation",
            size=validation_size,
            rng=random.Random(seed + SPLIT_SEED_OFFSETS["validation"]),
        ),
        "release_eval": _build_split_examples(
            graphs_by_split["release_eval"],
            split="release_eval",
            size=release_eval_size,
            rng=random.Random(seed + SPLIT_SEED_OFFSETS["release_eval"]),
        ),
    }


def build_dataset_manifest(
    *,
    train_size: int = 8000,
    validation_size: int = 1200,
    release_eval_size: int = 1800,
    seed: int = 7,
) -> dict[str, Any]:
    split_sizes = {
        "train": train_size,
        "validation": validation_size,
        "release_eval": release_eval_size,
    }
    route_targets = {
        split: _scale_targets(DEFAULT_ROUTE_TARGETS_BY_SPLIT[split], size)
        for split, size in split_sizes.items()
    }
    local_safe_targets = {
        split: _scale_targets(DEFAULT_LOCAL_SAFE_FAMILY_TARGETS, route_targets[split]["local_safe"])
        for split in ("train", "validation", "release_eval")
    }
    return {
        "seed": seed,
        "split_sizes": split_sizes,
        "route_targets": route_targets,
        "local_safe_family_targets": local_safe_targets,
        "graph_assignments": {split: list(graph_ids) for split, graph_ids in GRAPH_IDS_BY_SPLIT.items()},
        "route_labels": ["local_safe", "strong_model_candidate", "refuse"],
        "query_families": [
            "companies_list",
            "segments_by_company",
            "offerings_by_company",
            "offerings_by_segment",
            "companies_by_segment_filters",
            "segments_by_segment_filters",
            "companies_by_cross_segment_filters",
            "descendant_offerings_by_root",
            "companies_by_descendant_revenue",
            "companies_by_place",
            "segments_by_place_and_segment_filters",
            "companies_by_partner",
            "boolean_exists",
            "count_aggregate",
            "ranking_topk",
        ],
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
    "DEFAULT_LOCAL_SAFE_FAMILY_TARGETS",
    "DEFAULT_ROUTE_TARGETS_BY_SPLIT",
    "DatasetExample",
    "GRAPH_IDS_BY_SPLIT",
    "SyntheticCompany",
    "build_dataset_manifest",
    "build_dataset_splits",
    "build_synthetic_company_graphs",
    "evaluate_query_plan",
    "main",
    "write_dataset_splits",
]
