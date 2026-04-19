from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ontology.place_hierarchy import classify_place_match, normalize_place_name

from runtime.query_planner import QueryPlanEnvelope, QueryPlanPayload


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
                        _offering(
                            "Care Platform",
                            ("subscription",),
                            _offering("Care Platform Mobile", ("subscription",)),
                            _offering("Care Platform Home", ("subscription",)),
                        ),
                        _offering("Identity Cloud", ("subscription",)),
                        _offering("Secure Records", ("subscription",)),
                        _offering(
                            "Analytics Studio",
                            ("subscription",),
                            _offering("Analytics Studio Clinical", ("subscription",)),
                        ),
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
                        _offering(
                            "Marketplace Hub",
                            ("transaction fees",),
                            _offering("Marketplace Hub Merchant", ("transaction fees",)),
                            _offering("Marketplace Hub Checkout", ("transaction fees",)),
                        ),
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
                        _offering(
                            "Loyalty Cloud",
                            ("subscription",),
                            _offering("Loyalty Cloud Rewards", ("subscription",)),
                        ),
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
                        _offering(
                            "Decision Hub",
                            ("subscription",),
                            _offering("Decision Hub Credit", ("subscription",)),
                            _offering("Decision Hub Treasury", ("subscription",)),
                        ),
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
                        _offering(
                            "Merchant Console",
                            ("subscription",),
                            _offering("Merchant Console Insights", ("subscription",)),
                        ),
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
                        _offering(
                            "Automation Suite",
                            ("subscription",),
                            _offering("Automation Suite Remote", ("subscription",)),
                            _offering("Automation Suite Safety", ("subscription",)),
                        ),
                        _offering(
                            "Cloud Platform",
                            ("consumption-based",),
                            _offering("Cloud Platform Field", ("consumption-based",)),
                        ),
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
                        _offering("Support Desk", ("service fees",)),
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
                        _offering(
                            "Analytics Studio",
                            ("subscription",),
                            _offering("Analytics Studio Edge", ("subscription",)),
                        ),
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


def all_offerings(segment: SyntheticSegment) -> list[SyntheticOffering]:
    offerings: list[SyntheticOffering] = []

    def walk(node: SyntheticOffering) -> None:
        offerings.append(node)
        for child in node.children:
            walk(child)

    for root in segment.offerings:
        walk(root)
    return offerings


def root_offerings_with_children(segment: SyntheticSegment) -> tuple[SyntheticOffering, ...]:
    return tuple(offering for offering in segment.offerings if offering.children)


def descendant_offering_names(segment: SyntheticSegment, root_name: str) -> set[str]:
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


def segment_revenue_models(segment: SyntheticSegment, *, descendant: bool) -> set[str]:
    offerings = all_offerings(segment) if descendant else list(segment.offerings)
    revenue_models: set[str] = set()
    for offering in offerings:
        revenue_models.update(offering.revenue_models)
    return revenue_models


def company_matches_place(company: SyntheticCompany, places: set[str]) -> bool:
    for requested_place in places:
        normalized_requested = normalize_place_name(requested_place)
        for company_place in company.places:
            if classify_place_match(normalized_requested, company_place) is not None:
                return True
    return False


def segment_matches(company: SyntheticCompany, segment: SyntheticSegment, payload: QueryPlanPayload) -> bool:
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
    descendant_names = {offering.name for offering in all_offerings(segment)}
    candidate_names = descendant_names if hierarchy_mode == "descendant" else direct_names
    if payload.offerings and not set(payload.offerings).issubset(candidate_names):
        return False

    if payload.revenue_models:
        revenue_models = segment_revenue_models(segment, descendant=hierarchy_mode == "descendant")
        if not set(payload.revenue_models).issubset(revenue_models):
            return False
    return True


def company_matches_cross_segment(company: SyntheticCompany, payload: QueryPlanPayload) -> bool:
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
            if not any(segment_matches(company, segment, QueryPlanPayload(segments=payload.segments, customer_types=[value])) for segment in company.segments):
                return False
        elif kind == "channel":
            if not any(segment_matches(company, segment, QueryPlanPayload(segments=payload.segments, channels=[value])) for segment in company.segments):
                return False
        elif kind == "offering":
            if not any(segment_matches(company, segment, QueryPlanPayload(segments=payload.segments, offerings=[value], hierarchy_mode=payload.hierarchy_mode)) for segment in company.segments):
                return False
        else:
            if not any(segment_matches(company, segment, QueryPlanPayload(segments=payload.segments, revenue_models=[value], hierarchy_mode=payload.hierarchy_mode)) for segment in company.segments):
                return False
    return True


def company_matches_descendant_revenue(
    company: SyntheticCompany,
    root_names: set[str],
    revenue_models: set[str],
) -> bool:
    for segment in company.segments:
        for root_name in root_names:
            descendant_names = descendant_offering_names(segment, root_name)
            if not descendant_names:
                continue
            for offering in all_offerings(segment):
                if offering.name in descendant_names and revenue_models.issubset(offering.revenue_models):
                    return True
    return False


def matching_graph_ids_for_plan(
    companies: tuple[SyntheticCompany, ...],
    plan: QueryPlanEnvelope,
    rows: list[dict[str, Any]] | None = None,
) -> tuple[str, ...]:
    if not plan.answerable:
        return ()

    payload = plan.payload or QueryPlanPayload()
    company_by_name = {company.name: company for company in companies}
    rows = evaluate_query_plan(companies, plan) if rows is None else rows

    def graph_ids_from_company_names(company_names: set[str]) -> tuple[str, ...]:
        return tuple(sorted({company_by_name[name].graph_id for name in company_names if name in company_by_name}))

    if any("company" in row for row in rows):
        return graph_ids_from_company_names({row["company"] for row in rows if "company" in row})

    if plan.family == "boolean_exists":
        nested_plan = QueryPlanEnvelope(answerable=True, family=payload.base_family, payload=payload)
        return matching_graph_ids_for_plan(companies, nested_plan)

    if plan.family == "count_aggregate":
        aggregate_spec = payload.aggregate_spec or {}
        nested_family = payload.base_family or aggregate_spec.get("base_family")
        nested_plan = QueryPlanEnvelope(answerable=True, family=nested_family, payload=payload)
        return matching_graph_ids_for_plan(companies, nested_plan)

    if plan.family == "ranking_topk":
        metric = (payload.aggregate_spec or {}).get("ranking_metric")
        normalized_places = {normalize_place_name(place) for place in payload.places}
        if metric == "customer_type_by_company_count":
            contributors: dict[str, set[str]] = {}
            for company in companies:
                if normalized_places and not company_matches_place(company, normalized_places):
                    continue
                if payload.companies and company.name not in payload.companies:
                    continue
                seen: set[str] = set()
                for segment in company.segments:
                    seen.update(segment.customer_types)
                for customer_type in seen:
                    contributors.setdefault(customer_type, set()).add(company.name)
            return graph_ids_from_company_names(
                {company_name for row in rows for company_name in contributors.get(row["customer_type"], set())}
            )
        if metric == "channel_by_segment_count":
            contributors: dict[str, set[str]] = {}
            for company in companies:
                if normalized_places and not company_matches_place(company, normalized_places):
                    continue
                if payload.companies and company.name not in payload.companies:
                    continue
                for segment in company.segments:
                    for channel in segment.channels:
                        contributors.setdefault(channel, set()).add(company.name)
            return graph_ids_from_company_names(
                {company_name for row in rows for company_name in contributors.get(row["channel"], set())}
            )
        if metric == "revenue_model_by_company_count":
            contributors: dict[str, set[str]] = {}
            for company in companies:
                if normalized_places and not company_matches_place(company, normalized_places):
                    continue
                if payload.companies and company.name not in payload.companies:
                    continue
                seen: set[str] = set()
                for segment in company.segments:
                    seen.update(segment_revenue_models(segment, descendant=True))
                for revenue_model in seen:
                    contributors.setdefault(revenue_model, set()).add(company.name)
            return graph_ids_from_company_names(
                {company_name for row in rows for company_name in contributors.get(row["revenue_model"], set())}
            )
        return graph_ids_from_company_names({row["company"] for row in rows if "company" in row})

    return ()


def evaluate_query_plan(companies: tuple[SyntheticCompany, ...], plan: QueryPlanEnvelope) -> list[dict[str, Any]]:
    if not plan.answerable:
        return []

    family = plan.family
    payload = plan.payload or QueryPlanPayload()
    company_map = {company.name: company for company in companies}

    if family == "companies_list":
        rows = [{"company": company.name} for company in sorted(companies, key=lambda item: item.name)]
        return rows[: payload.limit] if payload.limit else rows

    if family == "segments_by_company":
        rows: list[dict[str, Any]] = []
        for company_name in payload.companies:
            company = company_map[company_name]
            for segment in company.segments:
                rows.append({"company": company.name, "segment": segment.name})
        rows = sorted(rows, key=lambda row: (row["company"], row["segment"]))
        return rows[: payload.limit] if payload.limit else rows

    if family == "offerings_by_company":
        rows: set[tuple[str, str]] = set()
        for company_name in payload.companies:
            company = company_map[company_name]
            for segment in company.segments:
                for offering in segment.offerings:
                    rows.add((company.name, offering.name))
        materialized = [
            {"company": company_name, "offering": offering_name}
            for company_name, offering_name in sorted(rows, key=lambda row: (row[0], row[1]))
        ]
        return materialized[: payload.limit] if payload.limit else materialized

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
        rows = sorted(rows, key=lambda row: (row["company"], row["segment"], row["offering"]))
        return rows[: payload.limit] if payload.limit else rows

    if family == "companies_by_segment_filters":
        rows = [
            {"company": company.name}
            for company in sorted(companies, key=lambda item: item.name)
            if any(segment_matches(company, segment, payload) for segment in company.segments)
        ]
        return rows[: payload.limit] if payload.limit else rows

    if family == "segments_by_segment_filters":
        rows = []
        for company in companies:
            for segment in company.segments:
                if segment_matches(company, segment, payload):
                    rows.append({"company": company.name, "segment": segment.name})
        rows = sorted(rows, key=lambda row: (row["company"], row["segment"]))
        return rows[: payload.limit] if payload.limit else rows

    if family == "companies_by_cross_segment_filters":
        rows = [{"company": company.name} for company in companies if company_matches_cross_segment(company, payload)]
        rows = sorted(rows, key=lambda row: row["company"])
        return rows[: payload.limit] if payload.limit else rows

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
                    names.update(descendant_offering_names(segment, root_name))
            for offering_name in sorted(names):
                rows.append({"company": company.name, "offering": offering_name})
        rows = sorted(rows, key=lambda row: (row["company"], row["offering"]))
        return rows[: payload.limit] if payload.limit else rows

    if family == "companies_by_descendant_revenue":
        rows = []
        root_names = set(payload.offerings)
        revenue_models = set(payload.revenue_models)
        company_names = set(payload.companies) if payload.companies else {company.name for company in companies}
        places = {normalize_place_name(place) for place in payload.places}
        for company in companies:
            if company.name not in company_names:
                continue
            if places and not company_matches_place(company, places):
                continue
            if company_matches_descendant_revenue(company, root_names, revenue_models):
                rows.append({"company": company.name})
        rows = sorted(rows, key=lambda row: row["company"])
        return rows[: payload.limit] if payload.limit else rows

    if family == "companies_by_place":
        places = {normalize_place_name(place) for place in payload.places}
        rows = [{"company": company.name} for company in companies if company_matches_place(company, places)]
        rows = sorted(rows, key=lambda row: row["company"])
        return rows[: payload.limit] if payload.limit else rows

    if family == "segments_by_place_and_segment_filters":
        places = {normalize_place_name(place) for place in payload.places}
        rows = []
        for company in companies:
            if not company_matches_place(company, places):
                continue
            for segment in company.segments:
                if segment_matches(company, segment, payload):
                    rows.append({"company": company.name, "segment": segment.name})
        rows = sorted(rows, key=lambda row: (row["company"], row["segment"]))
        return rows[: payload.limit] if payload.limit else rows

    if family == "companies_by_partner":
        partners = set(payload.partners)
        company_names = set(payload.companies) if payload.companies else {company.name for company in companies}
        rows = [
            {"company": company.name}
            for company in companies
            if company.name in company_names and partners.intersection(company.partners)
        ]
        rows = sorted(rows, key=lambda row: row["company"])
        return rows[: payload.limit] if payload.limit else rows

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
            value = len({(row["company"], row["segment"]) for row in rows if "segment" in row})
        else:
            value = len({(row.get("company"), row["offering"]) for row in rows if "offering" in row})
        return [{alias: value}]

    if family == "ranking_topk":
        metric = (payload.aggregate_spec or {}).get("ranking_metric")
        limit = payload.limit or 5
        normalized_places = {normalize_place_name(place) for place in payload.places}
        if metric == "customer_type_by_company_count":
            counts: dict[str, set[str]] = {}
            for company in companies:
                if normalized_places and not company_matches_place(company, normalized_places):
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
                if normalized_places and not company_matches_place(company, normalized_places):
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
                if normalized_places and not company_matches_place(company, normalized_places):
                    continue
                if payload.companies and company.name not in payload.companies:
                    continue
                seen: set[str] = set()
                for segment in company.segments:
                    seen.update(segment_revenue_models(segment, descendant=True))
                for value in seen:
                    counts.setdefault(value, set()).add(company.name)
            rows = [{"revenue_model": name, "company_count": len(company_names)} for name, company_names in counts.items()]
            return sorted(rows, key=lambda row: (-row["company_count"], row["revenue_model"]))[:limit]
        counts: dict[str, int] = {}
        for company in companies:
            if normalized_places and not company_matches_place(company, normalized_places):
                continue
            counts[company.name] = sum(1 for segment in company.segments if segment_matches(company, segment, payload))
        rows = [{"company": name, "segment_count": count} for name, count in counts.items() if count > 0]
        return sorted(rows, key=lambda row: (-row["segment_count"], row["company"]))[:limit]

    return []


__all__ = [
    "SyntheticCompany",
    "SyntheticOffering",
    "SyntheticSegment",
    "all_offerings",
    "build_synthetic_company_graphs",
    "company_matches_place",
    "descendant_offering_names",
    "evaluate_query_plan",
    "matching_graph_ids_for_plan",
    "root_offerings_with_children",
    "segment_matches",
    "segment_revenue_models",
]
