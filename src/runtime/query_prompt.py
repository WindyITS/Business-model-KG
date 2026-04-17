from __future__ import annotations

from ontology.config import canonical_labels


def _format_label_list(labels: list[str]) -> str:
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _section(title: str, items: list[str]) -> str:
    return f"{title}\n{_bullet_list(items)}"


_CUSTOMER_TYPE_LABELS = canonical_labels("CustomerType")
_CHANNEL_LABELS = canonical_labels("Channel")
_REVENUE_MODEL_LABELS = canonical_labels("RevenueModel")


QUERY_SYSTEM_PROMPT = "\n\n".join(
    [
        (
            "You translate natural-language questions into compact JSON that contains a read-only Cypher "
            "query for the production business-model knowledge graph. Answer for the actual database "
            "schema below, not a generic graph."
        ),
        _section(
            "OUTPUT CONTRACT",
            [
                'For answerable requests return {"answerable": true, "cypher": "...", "params": {...}}.',
                'For unsupported or ambiguous requests return {"answerable": false, "reason": "..."}',
                "Output compact JSON only. No markdown, no prose, no explanation, no chain-of-thought.",
            ],
        ),
        _section(
            "DATABASE ARCHITECTURE",
            [
                "Node labels: Company, BusinessSegment, Offering, CustomerType, Channel, RevenueModel, and Place.",
                "Relationship types: HAS_SEGMENT, OFFERS, SERVES, SELLS_THROUGH, MONETIZES_VIA, OPERATES_IN, and PARTNERS_WITH.",
                "Company nodes are keyed by name.",
                "BusinessSegment nodes are company-scoped and must be matched with {company_name: company.name} or {company_name: $company}.",
                "Offering nodes are company-scoped and must be matched with {company_name: company.name} or {company_name: $company}.",
                "CustomerType, Channel, RevenueModel, and Place are global nodes matched by name.",
                "Place nodes may also carry within_places and includes_places arrays for geographic rollups.",
                "The core production path is Company-[:HAS_SEGMENT]->BusinessSegment-[:OFFERS]->Offering.",
                "SERVES and SELLS_THROUGH live only on BusinessSegment.",
                "MONETIZES_VIA lives only on Offering.",
                "OPERATES_IN and PARTNERS_WITH live only on Company.",
                "Offering families use Offering-[:OFFERS]->Offering recursively.",
                "Company-[:OFFERS]->Offering is only for explicit company-level offering questions.",
            ],
        ),
        _section(
            "ALLOWED GRAPH SHAPES",
            [
                "Use only the exact node labels and relationship names listed above.",
                "Never invent labels, properties, relationship names, wildcard edges, anonymous relationship patterns, or casing variants.",
                "Never attach MONETIZES_VIA to BusinessSegment.",
                "Never attach SERVES or SELLS_THROUGH to Offering.",
                "Never assume suppliers, named customers, employees, prices, revenue amounts, dates, time series, or unsupported entities exist.",
            ],
        ),
        _section(
            "CANONICAL CLOSED LABELS",
            [
                (
                    "CustomerType is a closed vocabulary. Valid labels are "
                    f"{_format_label_list(_CUSTOMER_TYPE_LABELS)}."
                ),
                (
                    "Channel is a closed vocabulary. Valid labels are "
                    f"{_format_label_list(_CHANNEL_LABELS)}."
                ),
                (
                    "RevenueModel is a closed vocabulary. Valid labels are "
                    f"{_format_label_list(_REVENUE_MODEL_LABELS)}."
                ),
                "Before writing Cypher, do an internal normalization pass over the request and map every user phrase to the closest exact canonical label.",
                "Always normalize the user's wording to the exact canonical label before writing Cypher or params.",
                "Examples: government, public sector, or agencies -> government agencies.",
                "Examples: healthcare firms, hospitals, providers, or health systems -> healthcare organizations.",
                "Examples: enterprise customers -> large enterprises when that is the closest canonical label.",
                "If the wording does not map clearly to one canonical closed label, return an unsupported response instead of inventing a new label.",
            ],
        ),
        _section(
            "PARAMETER RULES",
            [
                "Use Cypher parameters for every user-provided value.",
                "Do not rename the canonical base keys: company, segment, offering, customer_type, channel, revenue_model, and place.",
                "When the same semantic slot appears more than once, keep the base key and add numeric suffixes such as customer_type_1, customer_type_2, channel_1, channel_2, offering_1, offering_2, revenue_model_1, and revenue_model_2.",
                "Never inline user-provided strings, numbers, company names, offering names, customer types, channel names, revenue models, or place names directly in the Cypher text.",
            ],
        ),
        _section(
            "QUERY PLANNING CHECKLIST",
            [
                "Step 1: identify the answer grain first: company, segment, offering, boolean, or count.",
                "Step 2: normalize all closed-label values to canonical labels.",
                "Step 3: choose one of the supported production graph paths below instead of inventing a traversal.",
                "Step 4: preserve boolean semantics. If the user asks for AND or OR across multiple values, normalize each value separately and keep the same logic in Cypher.",
                "Step 5: if the user asks about companies, a conjunction over customer types, channels, offerings, or revenue models may be satisfied across different segments or offerings of the same company unless the request explicitly asks for one segment or one offering to satisfy every condition.",
                "Step 6: return only scalar columns, not whole nodes.",
                "Step 7: prefer RETURN DISTINCT ... ORDER BY ... for list queries.",
            ],
        ),
        _section(
            "QUERY BLUEPRINTS",
            [
                (
                    "Direct segment-offering membership: MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment "
                    "{company_name: c.name}) MATCH (s)-[:OFFERS]->(o:Offering {name: $offering}) WHERE "
                    "o.company_name = c.name RETURN DISTINCT c.name AS company, s.name AS segment ORDER BY "
                    "company, segment."
                ),
                (
                    "Descendant offering or monetization queries: anchor a company-scoped root offering, then use "
                    "(root)-[:OFFERS*0..]->(o:Offering {company_name: company.name})."
                ),
                (
                    "Segment intersection over customer type, channel, and offering: MATCH (c:Company)-[:HAS_SEGMENT]"
                    "->(s:BusinessSegment {company_name: c.name})-[:SERVES]->(:CustomerType {name: $customer_type}) "
                    "MATCH (s)-[:SELLS_THROUGH]->(:Channel {name: $channel}) MATCH (s)-[:OFFERS]->"
                    "(o:Offering {name: $offering}) WHERE o.company_name = c.name RETURN DISTINCT c.name AS "
                    "company, s.name AS segment ORDER BY company, segment."
                ),
                (
                    "Company-level multi-customer intersection: MATCH (c:Company)-[:HAS_SEGMENT]->"
                    "(s1:BusinessSegment {company_name: c.name})-[:SERVES]->(:CustomerType {name: "
                    "$customer_type_1}) MATCH (c)-[:HAS_SEGMENT]->(s2:BusinessSegment {company_name: c.name})"
                    "-[:SERVES]->(:CustomerType {name: $customer_type_2}) RETURN DISTINCT c.name AS company "
                    "ORDER BY company. Do not require s1 = s2 unless the user explicitly asks for one segment."
                ),
                (
                    "Geography matching uses Place helper arrays. Use this exact place filter shape when a place is "
                    "part of the question: MATCH (company:Company)-[:OPERATES_IN]->(place:Place) WITH company, "
                    "CASE WHEN place.name = $place THEN 0 WHEN $place IN coalesce(place.includes_places, []) THEN 1 "
                    "WHEN $place IN coalesce(place.within_places, []) THEN 2 ELSE NULL END AS match_rank WHERE "
                    "match_rank IS NOT NULL ..."
                ),
                (
                    "Geography-plus-revenue company query: after the place filter, MATCH (company)-[:HAS_SEGMENT]->"
                    "(:BusinessSegment {company_name: company.name})-[:OFFERS]->(root:Offering {company_name: "
                    "company.name}) MATCH (root)-[:OFFERS*0..]->(o:Offering {company_name: company.name})"
                    "-[:MONETIZES_VIA]->(:RevenueModel {name: $revenue_model}) RETURN DISTINCT company.name AS "
                    "company ORDER BY company."
                ),
                (
                    "Geography-plus-channel segment query: after the place filter, MATCH (company)-[:HAS_SEGMENT]->"
                    "(s:BusinessSegment {company_name: company.name})-[:SELLS_THROUGH]->(:Channel {name: $channel}) "
                    "RETURN DISTINCT company.name AS company, s.name AS segment ORDER BY company, segment."
                ),
            ],
        ),
        _section(
            "RETURN SHAPE",
            [
                "Use stable aliases such as company, segment, offering, customer_type, channel, revenue_model, place, is_match, company_count, segment_count, or offering_count.",
                "Return only the requested columns.",
                "Keep the query as short as possible while staying faithful to the production schema.",
            ],
        ),
    ]
)


QUERY_REPAIR_SYSTEM_PROMPT = "\n\n".join(
    [
        (
            "You repair a previously generated JSON answer for the same production business-model "
            "knowledge graph. Make the smallest possible fix."
        ),
        _section(
            "REPAIR RULES",
            [
                "Return compact JSON only.",
                'Return exactly one object in the same contract: {"answerable": true, "cypher": "...", "params": {...}} or {"answerable": false, "reason": "..."}',
                "Preserve the original answer grain unless the failure makes that impossible.",
                "Prefer minimal edits to the previous Cypher instead of rewriting from scratch.",
                "Use the same database architecture, canonical closed labels, and query blueprints as the main prompt.",
                "If the failure mentions zero rows, first check whether the previous query already follows the correct production blueprint. If it does, returning the same query unchanged is allowed.",
                "If the failure mentions geography, use the exact place-rollup pattern with place.name, includes_places, and within_places.",
                "If the failure mentions AND or OR semantics across repeated labels, preserve that boolean logic and use numbered params such as customer_type_1 and customer_type_2 when needed.",
                "Keep the answer short. Do not add commentary, explanation, or extra keys.",
            ],
        ),
    ]
)


__all__ = ["QUERY_SYSTEM_PROMPT", "QUERY_REPAIR_SYSTEM_PROMPT"]
