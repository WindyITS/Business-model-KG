from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ontology.config import canonical_labels
from ontology.place_hierarchy import normalize_place_name
from ontology.validator import clean_entity_name
from pydantic import BaseModel, Field, model_validator

from .cypher_validation import validate_params_match, validate_read_only_cypher

RefusalReason = Literal[
    "unsupported_schema",
    "unsupported_metric",
    "unsupported_time",
    "ambiguous_closed_label",
    "ambiguous_request",
    "write_request",
    "beyond_local_coverage",
]

LookupFamily = Literal[
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
]

QueryFamily = Literal[
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
]

BindingScope = Literal["same_segment", "across_segments"]
HierarchyMode = Literal["direct", "descendant"]
CountTarget = Literal["company", "segment", "offering"]
RankingMetric = Literal[
    "customer_type_by_company_count",
    "channel_by_segment_count",
    "revenue_model_by_company_count",
    "company_by_matched_segment_count",
]

SUPPORTED_QUERY_FAMILIES: tuple[QueryFamily, ...] = (
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
)

LOOKUP_FAMILIES: tuple[LookupFamily, ...] = (
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
)

SUPPORTED_REFUSAL_REASONS: tuple[RefusalReason, ...] = (
    "unsupported_schema",
    "unsupported_metric",
    "unsupported_time",
    "ambiguous_closed_label",
    "ambiguous_request",
    "write_request",
    "beyond_local_coverage",
)

SAME_SEGMENT_FAMILIES = {
    "companies_by_segment_filters",
    "segments_by_segment_filters",
    "segments_by_place_and_segment_filters",
}
WRAPPER_BASE_FAMILIES = {
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
}

COUNT_TARGET_COMPATIBILITY: dict[LookupFamily, set[CountTarget]] = {
    "companies_list": {"company"},
    "segments_by_company": {"company", "segment"},
    "offerings_by_company": {"company", "offering"},
    "offerings_by_segment": {"company", "offering"},
    "companies_by_segment_filters": {"company"},
    "segments_by_segment_filters": {"company", "segment"},
    "companies_by_cross_segment_filters": {"company"},
    "descendant_offerings_by_root": {"company", "offering"},
    "companies_by_descendant_revenue": {"company"},
    "companies_by_place": {"company"},
    "segments_by_place_and_segment_filters": {"company", "segment"},
    "companies_by_partner": {"company"},
}

_CUSTOMER_TYPES = tuple(canonical_labels("CustomerType"))
_CHANNELS = tuple(canonical_labels("Channel"))
_REVENUE_MODELS = tuple(canonical_labels("RevenueModel"))


def _canonical_key(value: str) -> str:
    return clean_entity_name(value).casefold()


def _alias_map(label_type: str) -> dict[str, str]:
    canonical = {
        _canonical_key(label): label
        for label in canonical_labels(label_type)
    }
    if label_type == "CustomerType":
        canonical.update(
            {
                "government": "government agencies",
                "public sector": "government agencies",
                "agencies": "government agencies",
                "agency": "government agencies",
                "healthcare": "healthcare organizations",
                "healthcare firms": "healthcare organizations",
                "providers": "healthcare organizations",
                "health systems": "healthcare organizations",
                "hospitals": "healthcare organizations",
                "enterprise": "large enterprises",
                "enterprise customers": "large enterprises",
                "enterprises": "large enterprises",
                "end users": "consumers",
                "individuals": "consumers",
                "smb": "small businesses",
                "small business": "small businesses",
                "mid market": "mid-market companies",
                "midsize companies": "mid-market companies",
                "midsized companies": "mid-market companies",
            }
        )
    elif label_type == "Channel":
        canonical.update(
            {
                "direct": "direct sales",
                "direct selling": "direct sales",
                "website": "online",
                "web": "online",
                "self service": "online",
                "marketplace": "marketplaces",
                "marketplaces": "marketplaces",
                "oem": "OEMs",
                "oems": "OEMs",
                "systems integrators": "system integrators",
                "integrators": "system integrators",
                "msp": "managed service providers",
                "msps": "managed service providers",
            }
        )
    elif label_type == "RevenueModel":
        canonical.update(
            {
                "recurring": "subscription",
                "subscriptions": "subscription",
                "ads": "advertising",
                "advertisements": "advertising",
                "usage based": "consumption-based",
                "usage-based": "consumption-based",
                "pay as you go": "consumption-based",
                "pay-as-you-go": "consumption-based",
                "hardware": "hardware sales",
                "services": "service fees",
                "professional services": "service fees",
                "transactions": "transaction fees",
                "transaction": "transaction fees",
            }
        )
    return canonical


_CUSTOMER_TYPE_ALIASES = _alias_map("CustomerType")
_CHANNEL_ALIASES = _alias_map("Channel")
_REVENUE_MODEL_ALIASES = _alias_map("RevenueModel")


class QueryPlanPayload(BaseModel):
    companies: list[str] = Field(default_factory=list)
    segments: list[str] = Field(default_factory=list)
    offerings: list[str] = Field(default_factory=list)
    customer_types: list[str] = Field(default_factory=list)
    channels: list[str] = Field(default_factory=list)
    revenue_models: list[str] = Field(default_factory=list)
    places: list[str] = Field(default_factory=list)
    partners: list[str] = Field(default_factory=list)
    binding_scope: BindingScope | None = None
    hierarchy_mode: HierarchyMode | None = None
    aggregate_spec: dict[str, Any] | None = None
    base_family: LookupFamily | None = None
    limit: int | None = None


class QueryPlanEnvelope(BaseModel):
    answerable: Literal[True]
    family: QueryFamily
    payload: QueryPlanPayload


class QueryResult(BaseModel):
    answerable: bool
    cypher: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    reason: RefusalReason | None = None

    @model_validator(mode="after")
    def _validate_contract(self) -> "QueryResult":
        if self.answerable:
            if not self.cypher or not self.cypher.strip():
                raise ValueError("Answerable responses must include a non-empty cypher string.")
            if self.reason is not None:
                raise ValueError("Answerable responses must not include a refusal reason.")
            return self

        if self.reason is None:
            raise ValueError("Refusal responses must include a refusal reason.")
        if self.cypher is not None:
            raise ValueError("Refusal responses must not include cypher.")
        if self.params:
            raise ValueError("Refusal responses must not include params.")
        return self


@dataclass(frozen=True)
class QueryParts:
    clauses: tuple[str, ...]
    params: dict[str, Any]


class QueryPlanRefusal(Exception):
    def __init__(self, reason: RefusalReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


def refusal_result(reason: RefusalReason) -> QueryResult:
    return QueryResult(answerable=False, reason=reason)


def validate_compiled_query(result: QueryResult) -> list[str]:
    if not result.answerable:
        return []
    failures = list(validate_read_only_cypher(result.cypher or ""))
    failures.extend(validate_params_match(result.cypher or "", result.params))
    return failures


def compile_query_plan(plan: QueryPlanEnvelope) -> QueryResult:
    try:
        payload = _normalize_payload(plan.payload)
        family = plan.family
        if family == "companies_list":
            result = _compile_companies_list(payload)
        elif family == "segments_by_company":
            result = _compile_segments_by_company(payload)
        elif family == "offerings_by_company":
            result = _compile_offerings_by_company(payload)
        elif family == "offerings_by_segment":
            result = _compile_offerings_by_segment(payload)
        elif family == "companies_by_segment_filters":
            result = _compile_companies_by_segment_filters(payload)
        elif family == "segments_by_segment_filters":
            result = _compile_segments_by_segment_filters(payload)
        elif family == "companies_by_cross_segment_filters":
            result = _compile_companies_by_cross_segment_filters(payload)
        elif family == "descendant_offerings_by_root":
            result = _compile_descendant_offerings_by_root(payload)
        elif family == "companies_by_descendant_revenue":
            result = _compile_companies_by_descendant_revenue(payload)
        elif family == "companies_by_place":
            result = _compile_companies_by_place(payload)
        elif family == "segments_by_place_and_segment_filters":
            result = _compile_segments_by_place_and_segment_filters(payload)
        elif family == "companies_by_partner":
            result = _compile_companies_by_partner(payload)
        elif family == "boolean_exists":
            result = _compile_boolean_exists(payload)
        elif family == "count_aggregate":
            result = _compile_count_aggregate(payload)
        elif family == "ranking_topk":
            result = _compile_ranking_topk(payload)
        else:
            raise QueryPlanRefusal("beyond_local_coverage", f"Unsupported family {family!r}.")
    except QueryPlanRefusal as exc:
        return refusal_result(exc.reason)

    failures = validate_compiled_query(result)
    if failures:
        return refusal_result("beyond_local_coverage")
    return result


def _normalize_payload(payload: QueryPlanPayload) -> QueryPlanPayload:
    return QueryPlanPayload(
        companies=_normalize_entities(payload.companies),
        segments=_normalize_entities(payload.segments),
        offerings=_normalize_entities(payload.offerings),
        customer_types=_normalize_closed_labels(payload.customer_types, "CustomerType"),
        channels=_normalize_closed_labels(payload.channels, "Channel"),
        revenue_models=_normalize_closed_labels(payload.revenue_models, "RevenueModel"),
        places=_normalize_places(payload.places),
        partners=_normalize_entities(payload.partners),
        binding_scope=payload.binding_scope,
        hierarchy_mode=payload.hierarchy_mode,
        aggregate_spec=payload.aggregate_spec,
        base_family=payload.base_family,
        limit=payload.limit,
    )


def _normalize_entities(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = clean_entity_name(value)
        if not cleaned:
            continue
        key = _canonical_key(cleaned)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    return normalized


def _normalize_places(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = clean_entity_name(value)
        if not cleaned:
            continue
        place = normalize_place_name(cleaned)
        key = _canonical_key(place)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(place)
    return normalized


def _normalize_closed_labels(values: list[str], label_type: str) -> list[str]:
    if label_type == "CustomerType":
        aliases = _CUSTOMER_TYPE_ALIASES
    elif label_type == "Channel":
        aliases = _CHANNEL_ALIASES
    else:
        aliases = _REVENUE_MODEL_ALIASES

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = clean_entity_name(value)
        if not cleaned:
            continue
        key = _canonical_key(cleaned)
        canonical = aliases.get(key)
        if canonical is None:
            raise QueryPlanRefusal(
                "ambiguous_closed_label",
                f"Value {value!r} does not map cleanly to an approved {label_type} label.",
            )
        if canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(canonical)
    return normalized


def _make_result(cypher: str, params: dict[str, Any] | None = None) -> QueryResult:
    return QueryResult(answerable=True, cypher=cypher.strip(), params=params or {})


def _require(condition: bool, reason: RefusalReason, message: str) -> None:
    if not condition:
        raise QueryPlanRefusal(reason, message)


def _single_or_indexed(prefix: str, values: list[str]) -> tuple[dict[str, Any], list[str]]:
    params: dict[str, Any] = {}
    names: list[str] = []
    for index, value in enumerate(values, start=1):
        name = prefix if len(values) == 1 else f"{prefix}_{index}"
        params[name] = value
        names.append(name)
    return params, names


def _place_where_clause(place_var: str = "place", param_name: str = "places") -> str:
    return (
        f"any(requested_place IN ${param_name} WHERE "
        f"requested_place = {place_var}.name OR "
        f"requested_place IN coalesce({place_var}.includes_places, []) OR "
        f"requested_place IN coalesce({place_var}.within_places, []))"
    )


def _limit_clause(limit: int | None) -> str:
    if limit is None:
        return ""
    _require(limit > 0, "ambiguous_request", "Limit must be positive.")
    return f" LIMIT {limit}"


def _ensure_no_extra_filters(payload: QueryPlanPayload, *, allowed: set[str]) -> None:
    slots = {
        "companies": bool(payload.companies),
        "segments": bool(payload.segments),
        "offerings": bool(payload.offerings),
        "customer_types": bool(payload.customer_types),
        "channels": bool(payload.channels),
        "revenue_models": bool(payload.revenue_models),
        "places": bool(payload.places),
        "partners": bool(payload.partners),
        "binding_scope": payload.binding_scope is not None,
        "hierarchy_mode": payload.hierarchy_mode is not None,
        "aggregate_spec": payload.aggregate_spec is not None,
        "base_family": payload.base_family is not None,
    }
    extras = sorted(name for name, present in slots.items() if present and name not in allowed)
    _require(not extras, "beyond_local_coverage", f"Unsupported payload fields: {extras}.")


def _filter_atoms(payload: QueryPlanPayload) -> list[tuple[str, str]]:
    atoms: list[tuple[str, str]] = []
    atoms.extend(("customer_type", value) for value in payload.customer_types)
    atoms.extend(("channel", value) for value in payload.channels)
    atoms.extend(("offering", value) for value in payload.offerings)
    atoms.extend(("revenue_model", value) for value in payload.revenue_models)
    return atoms


def _company_and_segment_seed(
    payload: QueryPlanPayload,
    *,
    company_var: str = "c",
    segment_var: str = "s",
) -> QueryParts:
    clauses = [
        f"MATCH ({company_var}:Company)-[:HAS_SEGMENT]->({segment_var}:BusinessSegment)",
        f"WHERE {segment_var}.company_name = {company_var}.name",
    ]
    params: dict[str, Any] = {}
    if payload.companies:
        params["companies"] = payload.companies
        clauses[-1] += f" AND {company_var}.name IN $companies"
    if payload.segments:
        params["segments"] = payload.segments
        clauses[-1] += f" AND {segment_var}.name IN $segments"
    return QueryParts(tuple(clauses), params)


def _append_filter_matches(
    parts: QueryParts,
    payload: QueryPlanPayload,
    *,
    company_var: str = "c",
    segment_var: str = "s",
    scope: BindingScope,
) -> QueryParts:
    clauses = list(parts.clauses)
    params = dict(parts.params)
    hierarchy_mode = payload.hierarchy_mode or ("descendant" if payload.revenue_models else "direct")

    atoms = _filter_atoms(payload)
    _require(atoms, "ambiguous_request", "At least one semantic filter is required.")

    if scope == "same_segment":
        for atom_index, (kind, value) in enumerate(atoms, start=1):
            if kind == "customer_type":
                param_name = f"customer_type_{atom_index}" if len(payload.customer_types) > 1 else "customer_type"
                params[param_name] = value
                clauses.append(
                    f"MATCH ({segment_var})-[:SERVES]->(:CustomerType {{name: ${param_name}}})"
                )
            elif kind == "channel":
                param_name = f"channel_{atom_index}" if len(payload.channels) > 1 else "channel"
                params[param_name] = value
                clauses.append(
                    f"MATCH ({segment_var})-[:SELLS_THROUGH]->(:Channel {{name: ${param_name}}})"
                )
            elif kind == "offering":
                param_name = f"offering_{atom_index}" if len(payload.offerings) > 1 else "offering"
                params[param_name] = value
                if hierarchy_mode == "descendant":
                    root_var = f"root_offering_{atom_index}"
                    offer_var = f"offering_match_{atom_index}"
                    clauses.append(
                        f"MATCH ({segment_var})-[:OFFERS]->({root_var}:Offering)"
                    )
                    clauses.append(
                        f"WHERE {root_var}.company_name = {company_var}.name"
                    )
                    clauses.append(
                        f"MATCH ({root_var})-[:OFFERS*0..]->({offer_var}:Offering)"
                    )
                    clauses.append(
                        f"WHERE {offer_var}.company_name = {company_var}.name AND {offer_var}.name = ${param_name}"
                    )
                else:
                    offer_var = f"offering_match_{atom_index}"
                    clauses.append(
                        f"MATCH ({segment_var})-[:OFFERS]->({offer_var}:Offering)"
                    )
                    clauses.append(
                        f"WHERE {offer_var}.company_name = {company_var}.name AND {offer_var}.name = ${param_name}"
                    )
            else:
                param_name = (
                    f"revenue_model_{atom_index}" if len(payload.revenue_models) > 1 else "revenue_model"
                )
                params[param_name] = value
                if hierarchy_mode == "descendant":
                    root_var = f"root_revenue_{atom_index}"
                    offer_var = f"revenue_offering_{atom_index}"
                    clauses.append(
                        f"MATCH ({segment_var})-[:OFFERS]->({root_var}:Offering)"
                    )
                    clauses.append(
                        f"WHERE {root_var}.company_name = {company_var}.name"
                    )
                    clauses.append(
                        f"MATCH ({root_var})-[:OFFERS*0..]->({offer_var}:Offering)-[:MONETIZES_VIA]->(:RevenueModel {{name: ${param_name}}})"
                    )
                    clauses.append(f"WHERE {offer_var}.company_name = {company_var}.name")
                else:
                    offer_var = f"revenue_offering_{atom_index}"
                    clauses.append(
                        f"MATCH ({segment_var})-[:OFFERS]->({offer_var}:Offering)-[:MONETIZES_VIA]->(:RevenueModel {{name: ${param_name}}})"
                    )
                    clauses.append(f"WHERE {offer_var}.company_name = {company_var}.name")
    else:
        for atom_index, (kind, value) in enumerate(atoms, start=1):
            segment_alias = f"s{atom_index}"
            seed = _company_and_segment_seed(payload, company_var=company_var, segment_var=segment_alias)
            for clause in seed.clauses:
                clauses.append(clause)
            params.update(seed.params)
            if kind == "customer_type":
                param_name = f"customer_type_{atom_index}"
                params[param_name] = value
                clauses.append(
                    f"MATCH ({segment_alias})-[:SERVES]->(:CustomerType {{name: ${param_name}}})"
                )
            elif kind == "channel":
                param_name = f"channel_{atom_index}"
                params[param_name] = value
                clauses.append(
                    f"MATCH ({segment_alias})-[:SELLS_THROUGH]->(:Channel {{name: ${param_name}}})"
                )
            elif kind == "offering":
                param_name = f"offering_{atom_index}"
                params[param_name] = value
                if hierarchy_mode == "descendant":
                    root_var = f"root_offering_{atom_index}"
                    offer_var = f"offering_match_{atom_index}"
                    clauses.append(f"MATCH ({segment_alias})-[:OFFERS]->({root_var}:Offering)")
                    clauses.append(f"WHERE {root_var}.company_name = {company_var}.name")
                    clauses.append(f"MATCH ({root_var})-[:OFFERS*0..]->({offer_var}:Offering)")
                    clauses.append(
                        f"WHERE {offer_var}.company_name = {company_var}.name AND {offer_var}.name = ${param_name}"
                    )
                else:
                    offer_var = f"offering_match_{atom_index}"
                    clauses.append(f"MATCH ({segment_alias})-[:OFFERS]->({offer_var}:Offering)")
                    clauses.append(
                        f"WHERE {offer_var}.company_name = {company_var}.name AND {offer_var}.name = ${param_name}"
                    )
            else:
                param_name = f"revenue_model_{atom_index}"
                params[param_name] = value
                if hierarchy_mode == "descendant":
                    root_var = f"root_revenue_{atom_index}"
                    offer_var = f"revenue_offering_{atom_index}"
                    clauses.append(f"MATCH ({segment_alias})-[:OFFERS]->({root_var}:Offering)")
                    clauses.append(f"WHERE {root_var}.company_name = {company_var}.name")
                    clauses.append(
                        f"MATCH ({root_var})-[:OFFERS*0..]->({offer_var}:Offering)-[:MONETIZES_VIA]->(:RevenueModel {{name: ${param_name}}})"
                    )
                    clauses.append(f"WHERE {offer_var}.company_name = {company_var}.name")
                else:
                    offer_var = f"revenue_offering_{atom_index}"
                    clauses.append(
                        f"MATCH ({segment_alias})-[:OFFERS]->({offer_var}:Offering)-[:MONETIZES_VIA]->(:RevenueModel {{name: ${param_name}}})"
                    )
                    clauses.append(f"WHERE {offer_var}.company_name = {company_var}.name")

    return QueryParts(tuple(clauses), params)


def _compile_companies_list(payload: QueryPlanPayload) -> QueryResult:
    _ensure_no_extra_filters(payload, allowed={"limit"})
    cypher = f"MATCH (company:Company) RETURN DISTINCT company.name AS company ORDER BY company{_limit_clause(payload.limit)}"
    return _make_result(cypher)


def _compile_segments_by_company(payload: QueryPlanPayload) -> QueryResult:
    _ensure_no_extra_filters(payload, allowed={"companies", "limit"})
    _require(payload.companies, "ambiguous_request", "segments_by_company requires at least one company.")
    cypher = "\n".join(
        [
            "MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment)",
            "WHERE s.company_name = c.name AND c.name IN $companies",
            f"RETURN DISTINCT c.name AS company, s.name AS segment ORDER BY company, segment{_limit_clause(payload.limit)}",
        ]
    )
    return _make_result(cypher, {"companies": payload.companies})


def _compile_offerings_by_company(payload: QueryPlanPayload) -> QueryResult:
    _ensure_no_extra_filters(payload, allowed={"companies", "limit"})
    _require(payload.companies, "ambiguous_request", "offerings_by_company requires at least one company.")
    cypher = "\n".join(
        [
            "MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment)-[:OFFERS]->(o:Offering)",
            "WHERE s.company_name = c.name AND o.company_name = c.name AND c.name IN $companies",
            f"RETURN DISTINCT c.name AS company, o.name AS offering ORDER BY company, offering{_limit_clause(payload.limit)}",
        ]
    )
    return _make_result(cypher, {"companies": payload.companies})


def _compile_offerings_by_segment(payload: QueryPlanPayload) -> QueryResult:
    _ensure_no_extra_filters(payload, allowed={"companies", "segments", "limit"})
    _require(payload.segments, "ambiguous_request", "offerings_by_segment requires at least one segment.")
    params: dict[str, Any] = {"segments": payload.segments}
    where = ["s.company_name = c.name", "o.company_name = c.name", "s.name IN $segments"]
    if payload.companies:
        params["companies"] = payload.companies
        where.append("c.name IN $companies")
    cypher = "\n".join(
        [
            "MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment)-[:OFFERS]->(o:Offering)",
            f"WHERE {' AND '.join(where)}",
            f"RETURN DISTINCT c.name AS company, s.name AS segment, o.name AS offering ORDER BY company, segment, offering{_limit_clause(payload.limit)}",
        ]
    )
    return _make_result(cypher, params)


def _compile_companies_by_segment_filters(payload: QueryPlanPayload) -> QueryResult:
    _ensure_no_extra_filters(
        payload,
        allowed={"companies", "segments", "offerings", "customer_types", "channels", "revenue_models", "binding_scope", "hierarchy_mode", "limit"},
    )
    _require(
        payload.binding_scope in {None, "same_segment"},
        "ambiguous_request",
        "companies_by_segment_filters only supports same_segment binding.",
    )
    seed = _company_and_segment_seed(payload)
    parts = _append_filter_matches(seed, payload, scope="same_segment")
    cypher = "\n".join(
        [
            *parts.clauses,
            f"RETURN DISTINCT c.name AS company ORDER BY company{_limit_clause(payload.limit)}",
        ]
    )
    return _make_result(cypher, parts.params)


def _compile_segments_by_segment_filters(payload: QueryPlanPayload) -> QueryResult:
    _ensure_no_extra_filters(
        payload,
        allowed={"companies", "segments", "offerings", "customer_types", "channels", "revenue_models", "binding_scope", "hierarchy_mode", "limit"},
    )
    _require(
        payload.binding_scope in {None, "same_segment"},
        "ambiguous_request",
        "segments_by_segment_filters only supports same_segment binding.",
    )
    seed = _company_and_segment_seed(payload)
    parts = _append_filter_matches(seed, payload, scope="same_segment")
    cypher = "\n".join(
        [
            *parts.clauses,
            f"RETURN DISTINCT c.name AS company, s.name AS segment ORDER BY company, segment{_limit_clause(payload.limit)}",
        ]
    )
    return _make_result(cypher, parts.params)


def _compile_companies_by_cross_segment_filters(payload: QueryPlanPayload) -> QueryResult:
    _ensure_no_extra_filters(
        payload,
        allowed={"companies", "segments", "offerings", "customer_types", "channels", "revenue_models", "binding_scope", "hierarchy_mode", "limit"},
    )
    _require(
        payload.binding_scope in {None, "across_segments"},
        "ambiguous_request",
        "companies_by_cross_segment_filters only supports across_segments binding.",
    )
    _require(
        len(_filter_atoms(payload)) >= 2,
        "ambiguous_request",
        "companies_by_cross_segment_filters requires at least two semantic filters.",
    )
    seed = QueryParts(("MATCH (c:Company)",), {"companies": payload.companies} if payload.companies else {})
    clauses = list(seed.clauses)
    if payload.companies:
        clauses.append("WHERE c.name IN $companies")
    parts = QueryParts(tuple(clauses), seed.params)
    parts = _append_filter_matches(parts, payload, company_var="c", scope="across_segments")
    cypher = "\n".join(
        [
            *parts.clauses,
            f"RETURN DISTINCT c.name AS company ORDER BY company{_limit_clause(payload.limit)}",
        ]
    )
    return _make_result(cypher, parts.params)


def _compile_descendant_offerings_by_root(payload: QueryPlanPayload) -> QueryResult:
    _ensure_no_extra_filters(payload, allowed={"companies", "offerings", "hierarchy_mode", "limit"})
    _require(payload.offerings, "ambiguous_request", "descendant_offerings_by_root requires at least one root offering.")
    params: dict[str, Any] = {"offerings": payload.offerings}
    where = ["root.name IN $offerings", "root.company_name = c.name", "o.company_name = c.name"]
    if payload.companies:
        params["companies"] = payload.companies
        where.append("c.name IN $companies")
    cypher = "\n".join(
        [
            "MATCH (c:Company)-[:HAS_SEGMENT]->(:BusinessSegment)-[:OFFERS]->(root:Offering)",
            "MATCH (root)-[:OFFERS*0..]->(o:Offering)",
            f"WHERE {' AND '.join(where)}",
            f"RETURN DISTINCT c.name AS company, o.name AS offering ORDER BY company, offering{_limit_clause(payload.limit)}",
        ]
    )
    return _make_result(cypher, params)


def _compile_companies_by_descendant_revenue(payload: QueryPlanPayload) -> QueryResult:
    _ensure_no_extra_filters(
        payload,
        allowed={"companies", "offerings", "revenue_models", "places", "hierarchy_mode", "limit"},
    )
    _require(payload.offerings, "ambiguous_request", "companies_by_descendant_revenue requires at least one root offering.")
    _require(payload.revenue_models, "ambiguous_request", "companies_by_descendant_revenue requires at least one revenue model.")
    params: dict[str, Any] = {"offerings": payload.offerings}
    clauses = []
    if payload.places:
        params["places"] = payload.places
        clauses.extend(
            [
                "MATCH (company:Company)-[:OPERATES_IN]->(place:Place)",
                f"WHERE {_place_where_clause()}",
                "WITH DISTINCT company",
            ]
        )
        company_var = "company"
    else:
        clauses.append("MATCH (company:Company)")
        company_var = "company"
    clauses.append(f"MATCH ({company_var})-[:HAS_SEGMENT]->(:BusinessSegment)-[:OFFERS]->(root:Offering)")
    clauses.append("MATCH (root)-[:OFFERS*0..]->(o:Offering)")
    clauses.append(
        f"WHERE root.name IN $offerings AND root.company_name = {company_var}.name AND o.company_name = {company_var}.name"
    )
    if payload.companies:
        params["companies"] = payload.companies
        clauses[-1] += f" AND {company_var}.name IN $companies"
    revenue_params, names = _single_or_indexed("revenue_model", payload.revenue_models)
    params.update(revenue_params)
    for name in names:
        clauses.append(
            f"MATCH (o)-[:MONETIZES_VIA]->(:RevenueModel {{name: ${name}}})"
        )
    cypher = "\n".join(
        [
            *clauses,
            f"RETURN DISTINCT {company_var}.name AS company ORDER BY company{_limit_clause(payload.limit)}",
        ]
    )
    return _make_result(cypher, params)


def _compile_companies_by_place(payload: QueryPlanPayload) -> QueryResult:
    _ensure_no_extra_filters(payload, allowed={"places", "limit"})
    _require(payload.places, "ambiguous_request", "companies_by_place requires at least one place.")
    cypher = "\n".join(
        [
            "MATCH (company:Company)-[:OPERATES_IN]->(place:Place)",
            f"WHERE {_place_where_clause()}",
            f"RETURN DISTINCT company.name AS company ORDER BY company{_limit_clause(payload.limit)}",
        ]
    )
    return _make_result(cypher, {"places": payload.places})


def _compile_segments_by_place_and_segment_filters(payload: QueryPlanPayload) -> QueryResult:
    _ensure_no_extra_filters(
        payload,
        allowed={"companies", "segments", "offerings", "customer_types", "channels", "revenue_models", "places", "binding_scope", "hierarchy_mode", "limit"},
    )
    _require(
        payload.binding_scope in {None, "same_segment"},
        "ambiguous_request",
        "segments_by_place_and_segment_filters only supports same_segment binding.",
    )
    _require(payload.places, "ambiguous_request", "segments_by_place_and_segment_filters requires at least one place.")
    params: dict[str, Any] = {"places": payload.places}
    clauses = [
        "MATCH (company:Company)-[:OPERATES_IN]->(place:Place)",
        f"WHERE {_place_where_clause()}",
        "WITH DISTINCT company",
    ]
    seed = _company_and_segment_seed(payload, company_var="company", segment_var="s")
    clauses.extend(seed.clauses)
    params.update(seed.params)
    parts = _append_filter_matches(QueryParts(tuple(clauses), params), payload, company_var="company", segment_var="s", scope="same_segment")
    cypher = "\n".join(
        [
            *parts.clauses,
            f"RETURN DISTINCT company.name AS company, s.name AS segment ORDER BY company, segment{_limit_clause(payload.limit)}",
        ]
    )
    return _make_result(cypher, parts.params)


def _compile_companies_by_partner(payload: QueryPlanPayload) -> QueryResult:
    _ensure_no_extra_filters(payload, allowed={"companies", "partners", "limit"})
    _require(payload.partners, "ambiguous_request", "companies_by_partner requires at least one partner.")
    params: dict[str, Any] = {"partners": payload.partners}
    where = ["partner.name IN $partners"]
    if payload.companies:
        params["companies"] = payload.companies
        where.append("company.name IN $companies")
    cypher = "\n".join(
        [
            "MATCH (company:Company)-[:PARTNERS_WITH]->(partner:Company)",
            f"WHERE {' AND '.join(where)}",
            f"RETURN DISTINCT company.name AS company ORDER BY company{_limit_clause(payload.limit)}",
        ]
    )
    return _make_result(cypher, params)


def _compile_boolean_exists(payload: QueryPlanPayload) -> QueryResult:
    _require(payload.base_family in WRAPPER_BASE_FAMILIES, "beyond_local_coverage", "boolean_exists requires a supported base_family.")
    base_parts, expression = _compile_base_parts(payload.base_family, payload)
    cypher = "\n".join([*base_parts.clauses, f"RETURN COUNT(DISTINCT {expression}) > 0 AS is_match"])
    return _make_result(cypher, base_parts.params)


def _compile_count_aggregate(payload: QueryPlanPayload) -> QueryResult:
    spec = payload.aggregate_spec or {}
    _require(spec.get("kind") == "count", "beyond_local_coverage", "count_aggregate requires a count aggregate_spec.")
    base_family = spec.get("base_family")
    count_target = spec.get("count_target")
    _require(base_family in WRAPPER_BASE_FAMILIES, "beyond_local_coverage", "Unsupported count base_family.")
    _require(count_target in {"company", "segment", "offering"}, "beyond_local_coverage", "Unsupported count target.")
    _require(
        count_target in COUNT_TARGET_COMPATIBILITY[base_family],
        "beyond_local_coverage",
        f"{base_family} does not support {count_target} counts.",
    )
    base_parts, expression = _compile_base_parts(base_family, payload, count_target=count_target)
    alias = f"{count_target}_count"
    cypher = "\n".join([*base_parts.clauses, f"RETURN COUNT(DISTINCT {expression}) AS {alias}"])
    return _make_result(cypher, base_parts.params)


def _compile_ranking_topk(payload: QueryPlanPayload) -> QueryResult:
    spec = payload.aggregate_spec or {}
    _require(spec.get("kind") == "ranking", "beyond_local_coverage", "ranking_topk requires a ranking aggregate_spec.")
    metric = spec.get("ranking_metric")
    limit = payload.limit if payload.limit is not None else spec.get("limit", 5)
    _require(isinstance(limit, int) and limit > 0, "ambiguous_request", "ranking_topk requires a positive limit.")
    if metric == "customer_type_by_company_count":
        _ensure_no_extra_filters(payload, allowed={"companies", "places", "aggregate_spec", "limit"})
        params: dict[str, Any] = {"limit": limit}
        clauses = []
        if payload.places:
            params["places"] = payload.places
            clauses.extend(
                [
                    "MATCH (company:Company)-[:OPERATES_IN]->(place:Place)",
                    f"WHERE {_place_where_clause()}",
                    "WITH DISTINCT company",
                ]
            )
            company_var = "company"
        else:
            clauses.append("MATCH (company:Company)")
            company_var = "company"
        if payload.companies:
            params["companies"] = payload.companies
            clauses.append(f"WHERE {company_var}.name IN $companies")
        clauses.append(f"MATCH ({company_var})-[:HAS_SEGMENT]->(s:BusinessSegment)-[:SERVES]->(customer_type:CustomerType)")
        clauses.append(f"WHERE s.company_name = {company_var}.name")
        cypher = "\n".join(
            [
                *clauses,
                "RETURN customer_type.name AS customer_type, COUNT(DISTINCT company.name) AS company_count",
                "ORDER BY company_count DESC, customer_type",
                "LIMIT $limit",
            ]
        )
        return _make_result(cypher, params)
    if metric == "channel_by_segment_count":
        _ensure_no_extra_filters(payload, allowed={"companies", "places", "aggregate_spec", "limit"})
        params = {"limit": limit}
        clauses = []
        if payload.places:
            params["places"] = payload.places
            clauses.extend(
                [
                    "MATCH (company:Company)-[:OPERATES_IN]->(place:Place)",
                    f"WHERE {_place_where_clause()}",
                    "WITH DISTINCT company",
                ]
            )
            company_var = "company"
        else:
            clauses.append("MATCH (company:Company)")
            company_var = "company"
        if payload.companies:
            params["companies"] = payload.companies
            clauses.append(f"WHERE {company_var}.name IN $companies")
        clauses.append(f"MATCH ({company_var})-[:HAS_SEGMENT]->(s:BusinessSegment)-[:SELLS_THROUGH]->(channel:Channel)")
        clauses.append(f"WHERE s.company_name = {company_var}.name")
        cypher = "\n".join(
            [
                *clauses,
                "RETURN channel.name AS channel, COUNT(DISTINCT [company.name, s.name]) AS segment_count",
                "ORDER BY segment_count DESC, channel",
                "LIMIT $limit",
            ]
        )
        return _make_result(cypher, params)
    if metric == "revenue_model_by_company_count":
        _ensure_no_extra_filters(payload, allowed={"companies", "places", "aggregate_spec", "limit"})
        params = {"limit": limit}
        clauses = []
        if payload.places:
            params["places"] = payload.places
            clauses.extend(
                [
                    "MATCH (company:Company)-[:OPERATES_IN]->(place:Place)",
                    f"WHERE {_place_where_clause()}",
                    "WITH DISTINCT company",
                ]
            )
            company_var = "company"
        else:
            clauses.append("MATCH (company:Company)")
            company_var = "company"
        if payload.companies:
            params["companies"] = payload.companies
            clauses.append(f"WHERE {company_var}.name IN $companies")
        clauses.append(f"MATCH ({company_var})-[:HAS_SEGMENT]->(:BusinessSegment)-[:OFFERS]->(root:Offering)")
        clauses.append(f"WHERE root.company_name = {company_var}.name")
        clauses.append("MATCH (root)-[:OFFERS*0..]->(o:Offering)-[:MONETIZES_VIA]->(revenue_model:RevenueModel)")
        clauses.append(f"WHERE o.company_name = {company_var}.name")
        cypher = "\n".join(
            [
                *clauses,
                "RETURN revenue_model.name AS revenue_model, COUNT(DISTINCT company.name) AS company_count",
                "ORDER BY company_count DESC, revenue_model",
                "LIMIT $limit",
            ]
        )
        return _make_result(cypher, params)
    if metric == "company_by_matched_segment_count":
        _ensure_no_extra_filters(
            payload,
            allowed={"companies", "segments", "offerings", "customer_types", "channels", "revenue_models", "places", "binding_scope", "hierarchy_mode", "aggregate_spec", "limit"},
        )
        clauses = []
        params: dict[str, Any] = {"limit": limit}
        if payload.places:
            params["places"] = payload.places
            clauses.extend(
                [
                    "MATCH (company:Company)-[:OPERATES_IN]->(place:Place)",
                    f"WHERE {_place_where_clause()}",
                    "WITH DISTINCT company",
                ]
            )
            seed = _company_and_segment_seed(payload, company_var="company", segment_var="s")
            clauses.extend(seed.clauses)
            params.update(seed.params)
            parts = _append_filter_matches(QueryParts(tuple(clauses), params), payload, company_var="company", segment_var="s", scope="same_segment")
        else:
            seed = _company_and_segment_seed(payload, company_var="company", segment_var="s")
            parts = _append_filter_matches(seed, payload, company_var="company", segment_var="s", scope="same_segment")
        cypher = "\n".join(
            [
                *parts.clauses,
                "RETURN company.name AS company, COUNT(DISTINCT s.name) AS segment_count",
                "ORDER BY segment_count DESC, company",
                "LIMIT $limit",
            ]
        )
        return _make_result(cypher, parts.params | {"limit": limit})
    raise QueryPlanRefusal("beyond_local_coverage", f"Unsupported ranking metric {metric!r}.")


def _compile_base_parts(
    family: LookupFamily,
    payload: QueryPlanPayload,
    *,
    count_target: CountTarget | None = None,
) -> tuple[QueryParts, str]:
    if family == "companies_list":
        _require(
            count_target in {None, "company"},
            "beyond_local_coverage",
            "companies_list only supports company counts.",
        )
        return QueryParts(("MATCH (company:Company)",), {}), "company.name"
    if family == "segments_by_company":
        _ensure_no_extra_filters(payload, allowed={"companies", "base_family", "aggregate_spec", "limit"})
        _require(payload.companies, "ambiguous_request", "segments_by_company requires at least one company.")
        return (
            QueryParts(
                (
                    "MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment)",
                    "WHERE s.company_name = c.name AND c.name IN $companies",
                ),
                {"companies": payload.companies},
            ),
            "c.name" if count_target == "company" else ("[c.name, s.name]" if count_target == "segment" else "s.name"),
        )
    if family == "offerings_by_company":
        _ensure_no_extra_filters(payload, allowed={"companies", "base_family", "aggregate_spec", "limit"})
        _require(payload.companies, "ambiguous_request", "offerings_by_company requires at least one company.")
        return (
            QueryParts(
                (
                    "MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment)-[:OFFERS]->(o:Offering)",
                    "WHERE s.company_name = c.name AND o.company_name = c.name AND c.name IN $companies",
                ),
                {"companies": payload.companies},
            ),
            "c.name" if count_target == "company" else ("[c.name, o.name]" if count_target == "offering" else "o.name"),
        )
    if family == "offerings_by_segment":
        _ensure_no_extra_filters(payload, allowed={"companies", "segments", "base_family", "aggregate_spec", "limit"})
        _require(payload.segments, "ambiguous_request", "offerings_by_segment requires at least one segment.")
        params: dict[str, Any] = {"segments": payload.segments}
        where = ["s.company_name = c.name", "o.company_name = c.name", "s.name IN $segments"]
        if payload.companies:
            params["companies"] = payload.companies
            where.append("c.name IN $companies")
        return (
            QueryParts(
                (
                    "MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment)-[:OFFERS]->(o:Offering)",
                    f"WHERE {' AND '.join(where)}",
                ),
                params,
            ),
            "c.name" if count_target == "company" else ("[c.name, o.name]" if count_target == "offering" else "o.name"),
        )
    if family == "companies_by_segment_filters":
        _require(
            count_target in {None, "company"},
            "beyond_local_coverage",
            "companies_by_segment_filters only supports company counts.",
        )
        seed = _company_and_segment_seed(payload)
        parts = _append_filter_matches(seed, payload, scope="same_segment")
        return parts, "c.name"
    if family == "segments_by_segment_filters":
        seed = _company_and_segment_seed(payload)
        parts = _append_filter_matches(seed, payload, scope="same_segment")
        return parts, "c.name" if count_target == "company" else ("[c.name, s.name]" if count_target == "segment" else "s.name")
    if family == "companies_by_cross_segment_filters":
        _require(
            count_target in {None, "company"},
            "beyond_local_coverage",
            "companies_by_cross_segment_filters only supports company counts.",
        )
        _require(
            len(_filter_atoms(payload)) >= 2,
            "ambiguous_request",
            "companies_by_cross_segment_filters requires at least two semantic filters.",
        )
        clauses = ["MATCH (c:Company)"]
        params = {}
        if payload.companies:
            params["companies"] = payload.companies
            clauses.append("WHERE c.name IN $companies")
        parts = _append_filter_matches(QueryParts(tuple(clauses), params), payload, company_var="c", scope="across_segments")
        return parts, "c.name"
    if family == "descendant_offerings_by_root":
        _ensure_no_extra_filters(payload, allowed={"companies", "offerings", "base_family", "aggregate_spec", "limit"})
        _require(payload.offerings, "ambiguous_request", "descendant_offerings_by_root requires at least one root offering.")
        params = {"offerings": payload.offerings}
        where = ["root.name IN $offerings", "root.company_name = c.name", "o.company_name = c.name"]
        if payload.companies:
            params["companies"] = payload.companies
            where.append("c.name IN $companies")
        return (
            QueryParts(
                (
                    "MATCH (c:Company)-[:HAS_SEGMENT]->(:BusinessSegment)-[:OFFERS]->(root:Offering)",
                    "MATCH (root)-[:OFFERS*0..]->(o:Offering)",
                    f"WHERE {' AND '.join(where)}",
                ),
                params,
            ),
            "c.name" if count_target == "company" else ("[c.name, o.name]" if count_target == "offering" else "o.name"),
        )
    if family == "companies_by_descendant_revenue":
        _require(
            count_target in {None, "company"},
            "beyond_local_coverage",
            "companies_by_descendant_revenue only supports company counts.",
        )
        result = _compile_companies_by_descendant_revenue(payload)
        return QueryParts(tuple(line for line in (result.cypher or "").splitlines()[:-1]), result.params), "company.name"
    if family == "companies_by_place":
        _require(
            count_target in {None, "company"},
            "beyond_local_coverage",
            "companies_by_place only supports company counts.",
        )
        _ensure_no_extra_filters(payload, allowed={"places", "base_family", "aggregate_spec", "limit"})
        _require(payload.places, "ambiguous_request", "companies_by_place requires at least one place.")
        return (
            QueryParts(
                (
                    "MATCH (company:Company)-[:OPERATES_IN]->(place:Place)",
                    f"WHERE {_place_where_clause()}",
                ),
                {"places": payload.places},
            ),
            "company.name",
        )
    if family == "segments_by_place_and_segment_filters":
        _require(payload.places, "ambiguous_request", "segments_by_place_and_segment_filters requires at least one place.")
        params: dict[str, Any] = {"places": payload.places}
        clauses = [
            "MATCH (company:Company)-[:OPERATES_IN]->(place:Place)",
            f"WHERE {_place_where_clause()}",
            "WITH DISTINCT company",
        ]
        seed = _company_and_segment_seed(payload, company_var="company", segment_var="s")
        clauses.extend(seed.clauses)
        params.update(seed.params)
        parts = _append_filter_matches(QueryParts(tuple(clauses), params), payload, company_var="company", segment_var="s", scope="same_segment")
        return parts, "company.name" if count_target == "company" else ("[company.name, s.name]" if count_target == "segment" else "s.name")
    _require(
        count_target in {None, "company"},
        "beyond_local_coverage",
        "companies_by_partner only supports company counts.",
    )
    _ensure_no_extra_filters(payload, allowed={"companies", "partners", "base_family", "aggregate_spec", "limit"})
    _require(payload.partners, "ambiguous_request", "companies_by_partner requires at least one partner.")
    params = {"partners": payload.partners}
    where = ["partner.name IN $partners"]
    if payload.companies:
        params["companies"] = payload.companies
        where.append("company.name IN $companies")
    return (
        QueryParts(
            (
                "MATCH (company:Company)-[:PARTNERS_WITH]->(partner:Company)",
                f"WHERE {' AND '.join(where)}",
            ),
            params,
        ),
        "company.name",
    )


__all__ = [
    "LOOKUP_FAMILIES",
    "QueryFamily",
    "QueryPlanEnvelope",
    "QueryPlanPayload",
    "QueryResult",
    "SUPPORTED_QUERY_FAMILIES",
    "SUPPORTED_REFUSAL_REASONS",
    "compile_query_plan",
    "refusal_result",
    "validate_compiled_query",
]
