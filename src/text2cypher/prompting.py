from __future__ import annotations

from ontology.config import canonical_labels


def _format_label_list(labels: list[str]) -> str:
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


_CUSTOMER_TYPE_LABELS = canonical_labels("CustomerType")
_CHANNEL_LABELS = canonical_labels("Channel")
_REVENUE_MODEL_LABELS = canonical_labels("RevenueModel")


TEXT2CYPHER_SYSTEM_PROMPT = (
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
    "CustomerType, RevenueModel, and Place are global by name. CustomerType, Channel, and RevenueModel "
    "are closed canonical vocabularies. Before writing Cypher, do an internal normalization pass over "
    "the request: compare every user phrase that could name a CustomerType, Channel, or RevenueModel "
    "against the canonical label lists and map it to the closest exact canonical label. Always "
    "normalize the user's wording to the exact canonical label before writing Cypher or params. "
    "Preserve boolean semantics: if the user asks for multiple values joined by AND or OR, normalize "
    "each value separately and keep the same conjunction or disjunction in Cypher. If the user asks "
    "about companies, a conjunction over customer types, channels, offerings, or revenue models may be "
    "satisfied across different segments or offerings of the same company unless the request explicitly "
    "asks for one segment or one offering to satisfy every condition. For CustomerType, the only valid "
    f"labels are {_format_label_list(_CUSTOMER_TYPE_LABELS)}. For example, map government, public "
    "sector, or agencies to government agencies; map healthcare firms, hospitals, providers, or health "
    "systems to healthcare organizations; map enterprise customers to large enterprises when that is "
    "the closest canonical label. For Channel, the only valid labels are "
    f"{_format_label_list(_CHANNEL_LABELS)}. For RevenueModel, the only valid labels are "
    f"{_format_label_list(_REVENUE_MODEL_LABELS)}. If the user's wording does not map clearly to one "
    "canonical closed label, return an unsupported response instead of inventing a new label. Geography "
    "is canonical at Company-[:OPERATES_IN]->Place. Place may use within_places and includes_places "
    "helper arrays instead of hierarchy edges. Use Cypher parameters for user-provided values and do "
    "not rename the canonical parameter keys: company, segment, offering, customer_type, channel, "
    "revenue_model, and place. When the same semantic slot appears more than once in one request, keep "
    "the canonical base key and add numeric suffixes such as customer_type_1, customer_type_2, "
    "channel_1, channel_2, offering_1, offering_2, revenue_model_1, and revenue_model_2. Every "
    "user-provided value must appear only in params and must be referenced in Cypher with a $parameter. "
    "Never inline user-provided strings, numbers, company names, offering names, customer types, "
    "channel names, revenue models, or place names directly in the Cypher text. Do not hardcode a "
    "literal value in Cypher and also repeat that value in params. For example, use :CustomerType "
    "{name: $customer_type}, :Channel {name: $channel}, :Offering {name: $offering}, and :Place "
    "{name: $place}, not literal values in quotes. Return requested scalar columns, not whole nodes. "
    "Use stable aliases such as company, segment, offering, customer_type, channel, revenue_model, "
    "place, boolean aliases like is_match, and count aliases like segment_count or offering_count. For "
    "list queries prefer RETURN DISTINCT ... ORDER BY .... Canonical idioms matter. For direct "
    "segment-offering membership queries, use direct membership only: MATCH (s)-[:OFFERS]->"
    "(o:Offering {name: $offering}) with WHERE s.company_name = c.name AND o.company_name = c.name. Do "
    "not replace direct segment-offering membership with a root offering plus [:OFFERS*0..] traversal. "
    "Use descendant traversal only for offering family, descendant-offering, or monetization queries. "
    "For descendant or monetization queries, start from a company-scoped root offering and use "
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
    "segment. For company-level multi-customer intersection queries, use separate segment matches when "
    "needed, for example MATCH (c:Company)-[:HAS_SEGMENT]->(s1:BusinessSegment {company_name: c.name})"
    "-[:SERVES]->(:CustomerType {name: $customer_type_1}) MATCH "
    "(c)-[:HAS_SEGMENT]->(s2:BusinessSegment {company_name: c.name})-[:SERVES]->"
    "(:CustomerType {name: $customer_type_2}) RETURN DISTINCT c.name AS company ORDER BY company. Do "
    "not require s1 = s2 unless the request explicitly asks for one segment that satisfies both "
    "customer types. Do not invent suppliers, named customers, employees, prices, revenue amounts, time "
    "series, or unsupported relations. Output compact JSON only. "
    'For answerable requests return {"answerable": true, "cypher": "...", "params": {...}}. '
    'For unsupported or ambiguous requests return {"answerable": false, "reason": "..."}.' 
)


__all__ = ["TEXT2CYPHER_SYSTEM_PROMPT"]
