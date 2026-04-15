from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .models import (
    DatasetSpec,
    FixtureEdgeSpec,
    FixtureNodeSpec,
    FixtureSpec,
    ResultColumnSpec,
    SourceExampleSpec,
)


SPLIT_ROTATION = ("train", "train", "train", "train", "train", "dev", "test")


def _node(node_id: str, label: str, name: str, **properties: Any) -> FixtureNodeSpec:
    return FixtureNodeSpec(node_id=node_id, label=label, name=name, properties=properties)


def _company(node_id: str, name: str) -> FixtureNodeSpec:
    return _node(node_id, "Company", name)


def _segment(node_id: str, company_name: str, name: str) -> FixtureNodeSpec:
    return _node(node_id, "BusinessSegment", name, company_name=company_name)


def _offering(node_id: str, company_name: str, name: str) -> FixtureNodeSpec:
    return _node(node_id, "Offering", name, company_name=company_name)


def _customer_type(node_id: str, name: str) -> FixtureNodeSpec:
    return _node(node_id, "CustomerType", name)


def _channel(node_id: str, name: str) -> FixtureNodeSpec:
    return _node(node_id, "Channel", name)


def _revenue_model(node_id: str, name: str) -> FixtureNodeSpec:
    return _node(node_id, "RevenueModel", name)


def _place(
    node_id: str,
    name: str,
    *,
    within_places: Sequence[str] = (),
    includes_places: Sequence[str] = (),
) -> FixtureNodeSpec:
    properties: dict[str, Any] = {}
    if within_places:
        properties["within_places"] = list(within_places)
    if includes_places:
        properties["includes_places"] = list(includes_places)
    return _node(node_id, "Place", name, **properties)


def _edge(source: str, relation: str, target: str) -> FixtureEdgeSpec:
    return FixtureEdgeSpec(source=source, type=relation, target=target)


def _fixture(
    fixture_id: str,
    graph_purpose: str,
    covered_families: Sequence[str],
    nodes: Sequence[FixtureNodeSpec],
    edges: Sequence[FixtureEdgeSpec],
    invariants_satisfied: Sequence[str],
    authoring_notes: Sequence[str],
) -> FixtureSpec:
    return FixtureSpec(
        fixture_id=fixture_id,
        graph_id=fixture_id,
        graph_purpose=graph_purpose,
        covered_families=list(covered_families),
        nodes=list(nodes),
        edges=list(edges),
        invariants_satisfied=list(invariants_satisfied),
        authoring_notes=list(authoring_notes),
    )


def _col(column: str, type_: str, description: str | None = None) -> ResultColumnSpec:
    return ResultColumnSpec(column=column, type=type_, description=description)


QUESTION_PARAPHRASE_BOOSTS: dict[str, tuple[str, str]] = {
    "What business segments does {company} have?": (
        "Show me the business segments for {company}.",
        "List all business segments owned by {company}.",
    ),
    "How many business segments does {company} have?": (
        "What is the total business segment count for {company}?",
        "Give me the number of business segments for {company}.",
    ),
    "Does {company} have {segment}?": (
        "Can you confirm that {company} has the {segment} segment?",
        "Is {segment} part of {company}'s business segment set?",
    ),
    "What offerings does {company} have at the company level?": (
        "Show me {company}'s company-level offerings.",
        "List all of {company}'s company-level offerings.",
    ),
    "How many company-level offerings does {company} have?": (
        "What is the total count of company-level offerings for {company}?",
        "Give me the number of company-level offerings for {company}.",
    ),
    "Does {company} directly offer {offering}?": (
        "Can you confirm that {company} directly offers {offering}?",
        "Is {offering} part of {company}'s direct offering inventory?",
    ),
    "What direct offerings does {company}'s {segment} segment have?": (
        "Show the direct offerings in {company}'s {segment} segment.",
        "List all offerings owned by {company}'s {segment} segment.",
    ),
    "How many direct offerings does {company}'s {segment} segment have?": (
        "What is the total count of direct offerings in {company}'s {segment} segment?",
        "Give me the number of offerings in {company}'s {segment} segment.",
    ),
    "Does {company}'s {segment} segment offer {offering}?": (
        "Can you confirm that {company}'s {segment} segment offers {offering}?",
        "Is {offering} part of {company}'s {segment} segment portfolio?",
    ),
    "What is inside {company}'s {offering}?": (
        "Show the immediate children of {company}'s {offering}.",
        "List the direct child offerings under {company}'s {offering}.",
    ),
    "How many child offerings does {company}'s {offering} have?": (
        "What is the number of immediate children for {company}'s {offering}?",
        "Give me the count of direct children inside {company}'s {offering}.",
    ),
    "Is {child_offering} inside {company}'s {offering}?": (
        "Can you confirm that {child_offering} is a direct child of {company}'s {offering}?",
        "Is {child_offering} listed directly beneath {company}'s {offering}?",
    ),
    "What offerings sit under {company}'s {offering}?": (
        "Show all descendant offerings under {company}'s {offering}.",
        "List every descendant of {company}'s {offering}.",
    ),
    "How many offerings sit under {company}'s {offering}?": (
        "What is the total descendant count for {company}'s {offering}?",
        "Give me the number of descendant offerings below {company}'s {offering}.",
    ),
    "Which leaf offerings are under {company}'s {offering}?": (
        "Show only the leaf descendants of {company}'s {offering}.",
        "List the terminal offerings under {company}'s {offering}.",
    ),
    "Where does {company} operate?": (
        "Show me the places where {company} operates.",
        "List all operating geographies for {company}.",
    ),
    "How many places does {company} operate in?": (
        "What is the total number of places where {company} operates?",
        "Give me the number of operating geographies for {company}.",
    ),
    "Does {company} partner with {partner}?": (
        "Can you confirm that {company} partners with {partner}?",
        "Is {partner} one of {company}'s partners?",
    ),
    "Who does {company} partner with?": (
        "Show me {company}'s partners.",
        "List all companies that partner with {company}.",
    ),
    "Which customer types does {company}'s {segment} segment serve?": (
        "Show the customer types served by {company}'s {segment} segment.",
        "List all customer types for {company}'s {segment} segment.",
    ),
    "How many customer types does {company}'s {segment} segment serve?": (
        "What is the total number of customer types served by {company}'s {segment} segment?",
        "Give me the customer type count for {company}'s {segment} segment.",
    ),
    "Does {company}'s {segment} segment serve {customer_type}?": (
        "Can you confirm that {company}'s {segment} segment serves {customer_type}?",
        "Is {customer_type} part of the target audience for {company}'s {segment} segment?",
    ),
    "Which channels does {company}'s {segment} segment sell through?": (
        "Show the channels used by {company}'s {segment} segment.",
        "List all sales channels for {company}'s {segment} segment.",
    ),
    "How many channels does {company}'s {segment} segment sell through?": (
        "What is the total number of channels used by {company}'s {segment} segment?",
        "Give me the channel count for {company}'s {segment} segment.",
    ),
    "Does {company}'s {segment} segment sell through {channel}?": (
        "Can you confirm that {company}'s {segment} segment sells through {channel}?",
        "Is {channel} a sales channel for {company}'s {segment} segment?",
    ),
    "How is {company}'s {offering} sold?": (
        "Show the channels that sell {company}'s {offering}.",
        "List all sales channels for {company}'s {offering}.",
    ),
    "How many channels does {company}'s {offering} use?": (
        "What is the total number of channels used by {company}'s {offering}?",
        "Give me the channel count for {company}'s {offering}.",
    ),
    "Does {company}'s {offering} sell through {channel}?": (
        "Can you confirm that {company}'s {offering} sells through {channel}?",
        "Is {channel} a sales channel for {company}'s {offering}?",
    ),
    "How does {company}'s {offering} make money?": (
        "Show the revenue models used by {company}'s {offering}.",
        "List all revenue models for {company}'s {offering}.",
    ),
    "How many revenue models does {company}'s {offering} use?": (
        "What is the total number of revenue models used by {company}'s {offering}?",
        "Give me the revenue model count for {company}'s {offering}.",
    ),
    "Does {company}'s {offering} monetize via {revenue_model}?": (
        "Can you confirm that {company}'s {offering} monetizes via {revenue_model}?",
        "Is {revenue_model} a revenue model for {company}'s {offering}?",
    ),
    "Which segments sell through {channel}?": (
        "Show me the segments that use {channel}.",
        "List all segments that sell through {channel}.",
    ),
    "How many segments sell through {channel}?": (
        "What is the total number of segments that use {channel}?",
        "Give me the number of segments that sell through {channel}.",
    ),
    "Which segments serve {customer_type}?": (
        "Show me the segments that serve {customer_type}.",
        "List all segments that target {customer_type}.",
    ),
    "How many segments serve {customer_type}?": (
        "What is the total number of segments that serve {customer_type}?",
        "Give me the number of segments that serve {customer_type}.",
    ),
    "Which offerings at {company} sit under segments that serve {customer_type}?": (
        "Show the offerings at {company} under segments serving {customer_type}.",
        "List all {company} offerings beneath segments serving {customer_type}.",
    ),
    "How many offerings at {company} sit under segments that serve {customer_type}?": (
        "What is the total number of {company} offerings under segments serving {customer_type}?",
        "Give me the number of {company} offerings under segments serving {customer_type}.",
    ),
    "List {company}'s offerings alphabetically.": (
        "Show me {company}'s offerings in alphabetical order.",
        "List {company}'s offerings sorted alphabetically.",
    ),
    "List the first {limit} offerings for {company} alphabetically.": (
        "Show the first {limit} offerings for {company} alphabetically.",
        "Return the first {limit} offerings for {company} in sorted order.",
    ),
    "List {company}'s {segment} segment offerings alphabetically.": (
        "Show the offerings in {company}'s {segment} segment sorted alphabetically.",
        "List the offerings in {company}'s {segment} segment in alphabetical order.",
    ),
    "List the offerings under {company}'s {offering} in sorted order.": (
        "Show all offerings under {company}'s {offering} in sorted order.",
        "List the descendants of {company}'s {offering} alphabetically.",
    ),
    "List the places where {company} operates alphabetically.": (
        "Show me the places where {company} operates sorted alphabetically.",
        "List all places where {company} operates in alphabetical order.",
    ),
    "Is {partner} explicitly recorded as a partner of {company}?": (
        "Can you confirm that the graph records {partner} as a partner of {company}?",
        "Does the KG explicitly list {partner} as a partner for {company}?",
    ),
    "Is {customer_type} explicitly recorded as a customer type for {company}'s {segment} segment?": (
        "Can you confirm that the graph records {customer_type} for {company}'s {segment} segment?",
        "Does the KG explicitly list {customer_type} as a customer type for {company}'s {segment} segment?",
    ),
    "Is {revenue_model} explicitly recorded as a revenue model for {company}'s {offering}?": (
        "Can you confirm that the graph records {revenue_model} for {company}'s {offering}?",
        "Does the KG explicitly list {revenue_model} as a revenue model for {company}'s {offering}?",
    ),
    "Is {place} explicitly recorded as an operating place for {company}?": (
        "Can you confirm that the graph records {place} as an operating place for {company}?",
        "Does the KG explicitly list {place} in {company}'s operating footprint?",
    ),
    "Which companies have a segment called {segment_name}?": (
        "Show every company with a segment called {segment_name}.",
        "List the companies that expose a segment named {segment_name}.",
    ),
    "How many companies have a segment called {segment_name}?": (
        "What is the number of companies with a segment called {segment_name}?",
        "Give me the company count for segment name {segment_name}.",
    ),
    "Which companies have an offering called {offering_name}?": (
        "Show every company with an offering called {offering_name}.",
        "List the companies that expose an offering named {offering_name}.",
    ),
    "How many companies have an offering called {offering_name}?": (
        "What is the number of companies with an offering called {offering_name}?",
        "Give me the company count for offering name {offering_name}.",
    ),
    "Does {company} have an offering called {offering_name}?": (
        "Can you confirm that {company} has an offering called {offering_name}?",
        "Is {offering_name} available as a company-scoped offering for {company}?",
    ),
    "Does {company} have a segment called {segment_name}?": (
        "Can you confirm that {company} has a segment called {segment_name}?",
        "Is {segment_name} available as a company-scoped segment for {company}?",
    ),
    "What does {surface_name} offer?": (
        "Show me what {surface_name} offers.",
        "List the companies that expose {surface_name} as an offering.",
    ),
    "How does {surface_name} make money?": (
        "Show me how {surface_name} makes money.",
        "List the revenue models associated with {surface_name}.",
    ),
    "What does the surface name {surface_name} offer?": (
        "Show me what the surface name {surface_name} offers.",
        "What offerings belong to the surface form {surface_name}?",
    ),
    "How does the surface name {surface_name} make money?": (
        "Show me how the surface name {surface_name} makes money.",
        "What revenue models attach to the surface form {surface_name}?",
    ),
    "Which segment is {company}'s {offering} under?": (
        "Show me the segment that contains {company}'s {offering}.",
        "What segment anchors {company}'s {offering}?",
    ),
    "What parent offering anchors {company}'s {offering}?": (
        "Show me the root offering for {company}'s {offering}.",
        "Which top-level offering anchors {company}'s {offering}?",
    ),
    "Show the segment-to-offering path for {company}'s {offering}.": (
        "Show the full breadcrumb from the segment to {company}'s {offering}.",
        "List the breadcrumb trail for {company}'s {offering}.",
    ),
    "What parent offering is {company}'s {offering} under?": (
        "Which offering directly contains {company}'s {offering}?",
        "Show me the direct parent offering for {company}'s {offering}.",
    ),
    "Which offerings are above {company}'s {offering}?": (
        "Show all ancestors of {company}'s {offering}.",
        "List the parent chain for {company}'s {offering}.",
    ),
    "What is the root offering for {company}'s {offering}?": (
        "Show me the top-level offering for {company}'s {offering}.",
        "Which root offering anchors {company}'s {offering}?",
    ),
    "What is the topmost offering above {company}'s {offering}?": (
        "Show me the root ancestor offering for {company}'s {offering}.",
        "Which topmost ancestor sits above {company}'s {offering}?",
    ),
    "Is {company}'s {segment} segment one of the segments that sell through {channel}?": (
        "Can you confirm that the reverse channel lookup for {channel} includes {company}'s {segment} segment?",
        "Does the segment set for {channel} include {company}'s {segment} segment?",
    ),
    "Which companies operate in {place}?": (
        "Show all companies that operate in {place}.",
        "List the companies that match {place} geographically.",
    ),
    "How many companies operate in {place}?": (
        "What is the number of companies that operate in {place}?",
        "Give me the company count for {place}.",
    ),
    "Does {company} operate in {place}?": (
        "Can you confirm that {company} operates in {place}?",
        "Is {place} part of {company}'s operating footprint?",
    ),
    "How does {company} match {place}?": (
        "What geography match class applies to {company} for {place}?",
        "Show the match label for {company} and {place}.",
    ),
    "Does {company}'s operating footprint match {place} anywhere in the geography hierarchy?": (
        "Can you confirm that {company} matches {place} through an exact, broader, or narrower geography?",
        "Is {place} matched anywhere in {company}'s geography hierarchy footprint?",
    ),
}


def _lowercase_first(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:]


def _generic_messy_paraphrases(question_template: str) -> tuple[str, ...]:
    lowered = _lowercase_first(question_template)
    return (
        f"Need the graph answer here: {lowered}",
        f"Can you just pull this from Neo4j: {lowered}",
        f"Trying to sanity-check this one: {lowered}",
    )


def _merge_paraphrases(question_template: str, paraphrase_templates: Sequence[str]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for template in (
        *paraphrase_templates,
        *QUESTION_PARAPHRASE_BOOSTS.get(question_template, ()),
        *_generic_messy_paraphrases(question_template),
    ):
        cleaned = template.strip()
        if not cleaned or cleaned == question_template or cleaned in seen:
            continue
        merged.append(cleaned)
        seen.add(cleaned)
    return tuple(merged)


def _intent(
    *,
    intent_id: str,
    family_id: str,
    question_template: str,
    cypher: str | None,
    result_shape: Sequence[ResultColumnSpec] | None,
    bindings: Sequence[dict[str, Any]],
    difficulty: str,
    paraphrase_templates: Sequence[str] = (),
    answerable: bool = True,
    refusal_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "intent_id": intent_id,
        "family_id": family_id,
        "question_template": question_template,
        "cypher": cypher,
        "result_shape": tuple(result_shape) if result_shape is not None else None,
        "bindings": list(bindings),
        "difficulty": difficulty,
        "paraphrase_templates": list(_merge_paraphrases(question_template, paraphrase_templates)),
        "answerable": answerable,
        "refusal_reason": refusal_reason,
    }


def _expand_intents(intent_specs: Sequence[dict[str, Any]]) -> list[SourceExampleSpec]:
    source_examples: list[SourceExampleSpec] = []
    for index, spec in enumerate(intent_specs):
        split = SPLIT_ROTATION[index % len(SPLIT_ROTATION)]
        for binding in spec["bindings"]:
            params = dict(binding["params"])
            question_canonical = spec["question_template"].format(**params)
            paraphrases = tuple(template.format(**params) for template in spec["paraphrase_templates"])
            source_examples.append(
                SourceExampleSpec(
                    example_id=f"{spec['intent_id']}__{binding['binding_id']}",
                    intent_id=spec["intent_id"],
                    family_id=spec["family_id"],
                    fixture_id=binding["fixture_id"],
                    graph_id=binding["graph_id"],
                    binding_id=binding["binding_id"],
                    question_canonical=question_canonical,
                    gold_cypher=spec["cypher"],
                    params=params,
                    answerable=spec["answerable"],
                    refusal_reason=spec["refusal_reason"],
                    result_shape=spec["result_shape"],
                    difficulty=spec["difficulty"],
                    split=split,
                    paraphrases=paraphrases,
                )
            )
    return source_examples


COMPANY_LIST_SHAPE = (_col("company", "string"),)
SEGMENT_LIST_SHAPE = (_col("segment", "string"),)
COMPANY_SEGMENT_SHAPE = (
    _col("company", "string"),
    _col("segment", "string"),
)
OFFERING_LIST_SHAPE = (_col("offering", "string"),)
CHILD_OFFERING_LIST_SHAPE = (_col("child_offering", "string"),)
DESCENDANT_OFFERING_LIST_SHAPE = (_col("descendant_offering", "string"),)
LEAF_OFFERING_LIST_SHAPE = (_col("leaf_offering", "string"),)
PLACE_LIST_SHAPE = (_col("place", "string", "Canonical place name"),)
PARTNER_LIST_SHAPE = (_col("partner", "string"),)
CUSTOMER_TYPE_LIST_SHAPE = (_col("customer_type", "string"),)
CHANNEL_LIST_SHAPE = (_col("channel", "string"),)
REVENUE_MODEL_LIST_SHAPE = (_col("revenue_model", "string"),)

COUNT_SHAPE = lambda column: (_col(column, "integer"),)
BOOL_SHAPE = lambda description: (_col("is_match", "boolean", description),)

GEOGRAPHY_MATCH_SHAPE = (
    _col("company", "string"),
    _col("matched_place", "string"),
    _col("geography_match", "string"),
)

COLLISION_SHAPE = (
    _col("company_name", "string"),
    _col("entity_name", "string"),
)


fixtures = [
    _fixture(
        "fx01_company_segment_core_northstar_v1",
        "Northstar Systems segment core for company-to-segment listing, counting, and membership queries.",
        ("QF01", "QF21", "QF22"),
        [
            _company("company_northstar_systems", "Northstar Systems"),
            _segment("segment_industrial_ai", "Northstar Systems", "Industrial AI"),
            _segment("segment_public_sector", "Northstar Systems", "Public Sector"),
            _segment("segment_developer_ecosystem", "Northstar Systems", "Developer Ecosystem"),
            _segment("segment_applied_security", "Northstar Systems", "Applied Security"),
        ],
        [
            _edge("company_northstar_systems", "HAS_SEGMENT", "segment_industrial_ai"),
            _edge("company_northstar_systems", "HAS_SEGMENT", "segment_public_sector"),
            _edge("company_northstar_systems", "HAS_SEGMENT", "segment_developer_ecosystem"),
            _edge("company_northstar_systems", "HAS_SEGMENT", "segment_applied_security"),
        ],
        (
            "BusinessSegment nodes carry company_name so the query contract matches production scoping.",
            "The fixture stays segment-only, with no offerings or geography facts mixed in.",
        ),
        (
            "Company-scoped segment lookup should remain deterministic under the composite identity model.",
            "This fixture gives us a clean anchor for direct list, count, and membership queries.",
        ),
    ),
    _fixture(
        "fx01_company_segment_core_harborline_v1",
        "Alternate company-to-segment core fixture with a second company context and denser segment inventory.",
        ("QF01", "QF21", "QF22"),
        [
            _company("company_harborline_systems", "Harborline Systems"),
            _segment("segment_industrial_security", "Harborline Systems", "Industrial Security"),
            _segment("segment_cloud_operations", "Harborline Systems", "Cloud Operations"),
            _segment("segment_risk_intelligence", "Harborline Systems", "Risk Intelligence"),
            _segment("segment_compliance_analytics", "Harborline Systems", "Compliance Analytics"),
        ],
        [
            _edge("company_harborline_systems", "HAS_SEGMENT", "segment_industrial_security"),
            _edge("company_harborline_systems", "HAS_SEGMENT", "segment_cloud_operations"),
            _edge("company_harborline_systems", "HAS_SEGMENT", "segment_risk_intelligence"),
            _edge("company_harborline_systems", "HAS_SEGMENT", "segment_compliance_analytics"),
        ],
        (
            "BusinessSegment nodes carry company_name so same-named segments can coexist across companies.",
            "The fixture gives an alternate company context for direct segment queries and ordering variants.",
        ),
        (
            "This is intentionally plain so the model learns the base company-to-segment shape before harder joins.",
        ),
    ),
    _fixture(
        "fx02_company_direct_offerings_meridian_v1",
        "Company-scoped direct-offering fixture for company-level lookup, counting, and ordering.",
        ("QF02", "QF21", "QF22"),
        [
            _company("company_meridian_nexus", "Meridian Nexus"),
            _offering("offering_meridian_atlas", "Meridian Nexus", "Meridian Atlas"),
            _offering("offering_meridian_studio", "Meridian Nexus", "Meridian Studio"),
            _offering("offering_meridian_vault", "Meridian Nexus", "Meridian Vault"),
            _offering("offering_meridian_horizon", "Meridian Nexus", "Meridian Horizon"),
        ],
        [
            _edge("company_meridian_nexus", "OFFERS", "offering_meridian_atlas"),
            _edge("company_meridian_nexus", "OFFERS", "offering_meridian_studio"),
            _edge("company_meridian_nexus", "OFFERS", "offering_meridian_vault"),
            _edge("company_meridian_nexus", "OFFERS", "offering_meridian_horizon"),
        ],
        (
            "Offering nodes carry company_name to preserve company-level identity during query execution.",
            "The offerings are direct company inventory, not segment-owned inventory.",
        ),
        (
            "This fixture supports the fallback company-offers family plus ordering and existence checks.",
        ),
    ),
    _fixture(
        "fx02_company_direct_offerings_latticeforge_v1",
        "Alternate company-scoped direct-offering fixture with a second company context and same-shape inventory.",
        ("QF02", "QF21", "QF22"),
        [
            _company("company_lattice_forge", "Lattice Forge"),
            _offering("offering_lattice_atlas", "Lattice Forge", "Lattice Atlas"),
            _offering("offering_lattice_studio", "Lattice Forge", "Lattice Studio"),
            _offering("offering_lattice_vault", "Lattice Forge", "Lattice Vault"),
            _offering("offering_lattice_horizon", "Lattice Forge", "Lattice Horizon"),
        ],
        [
            _edge("company_lattice_forge", "OFFERS", "offering_lattice_atlas"),
            _edge("company_lattice_forge", "OFFERS", "offering_lattice_studio"),
            _edge("company_lattice_forge", "OFFERS", "offering_lattice_vault"),
            _edge("company_lattice_forge", "OFFERS", "offering_lattice_horizon"),
        ],
        (
            "Offering nodes carry company_name so same-named offerings remain distinct across companies.",
            "The inventory is intentionally symmetrical with the Meridian fixture to make ordering and membership easy to validate.",
        ),
        (
            "This second company context helps the model learn that company-scoped offerings are not globally unique.",
        ),
    ),
    _fixture(
        "fx03_segment_direct_offerings_asteron_v1",
        "Canonical company -> segment -> offering fixture for direct segment-owned offerings.",
        ("QF03", "QF21", "QF22"),
        [
            _company("company_asteron_analytics", "Asteron Analytics"),
            _segment("segment_industrial_ai", "Asteron Analytics", "Industrial AI"),
            _offering("offering_asteron_control_suite", "Asteron Analytics", "Asteron Control Suite"),
            _offering("offering_asteron_insight_cloud", "Asteron Analytics", "Asteron Insight Cloud"),
            _offering("offering_asteron_model_forge", "Asteron Analytics", "Asteron Model Forge"),
            _offering("offering_asteron_signal_hub", "Asteron Analytics", "Asteron Signal Hub"),
        ],
        [
            _edge("company_asteron_analytics", "HAS_SEGMENT", "segment_industrial_ai"),
            _edge("segment_industrial_ai", "OFFERS", "offering_asteron_control_suite"),
            _edge("segment_industrial_ai", "OFFERS", "offering_asteron_insight_cloud"),
            _edge("segment_industrial_ai", "OFFERS", "offering_asteron_model_forge"),
            _edge("segment_industrial_ai", "OFFERS", "offering_asteron_signal_hub"),
        ],
        (
            "BusinessSegment and Offering both carry company_name so the direct ownership chain mirrors production.",
            "The segment owns four offerings so list, count, and membership examples are all non-trivial.",
        ),
        (
            "This is the canonical direct segment-owned offering shape used by the core dataset.",
        ),
    ),
    _fixture(
        "fx03_segment_direct_offerings_aurum_v1",
        "Alternate company -> segment -> offering fixture for a second company and segment context.",
        ("QF03", "QF21", "QF22"),
        [
            _company("company_aurum_metrics", "Aurum Metrics"),
            _segment("segment_decision_intelligence", "Aurum Metrics", "Decision Intelligence"),
            _offering("offering_aurum_control_suite", "Aurum Metrics", "Aurum Control Suite"),
            _offering("offering_aurum_insight_cloud", "Aurum Metrics", "Aurum Insight Cloud"),
            _offering("offering_aurum_model_forge", "Aurum Metrics", "Aurum Model Forge"),
            _offering("offering_aurum_signal_hub", "Aurum Metrics", "Aurum Signal Hub"),
        ],
        [
            _edge("company_aurum_metrics", "HAS_SEGMENT", "segment_decision_intelligence"),
            _edge("segment_decision_intelligence", "OFFERS", "offering_aurum_control_suite"),
            _edge("segment_decision_intelligence", "OFFERS", "offering_aurum_insight_cloud"),
            _edge("segment_decision_intelligence", "OFFERS", "offering_aurum_model_forge"),
            _edge("segment_decision_intelligence", "OFFERS", "offering_aurum_signal_hub"),
        ],
        (
            "BusinessSegment and Offering both carry company_name so the segment-owned inventory remains company-scoped.",
            "The second company context helps the model generalize beyond a single corporate shell.",
        ),
        (
            "This fixture mirrors the Asteron shape with different synthetic names to avoid memorization.",
        ),
    ),
    _fixture(
        "fx04_offering_hierarchy_tree_northstar_v1",
        "Segment-anchored offering hierarchy for immediate-child, descendant, and ancestor-path queries.",
        ("QF04", "QF05", "QF21", "QF23", "QF24"),
        [
            _company("company_northstar_systems", "Northstar Systems"),
            _segment("segment_industrial_ai", "Northstar Systems", "Industrial AI"),
            _offering("offering_northstar_platform", "Northstar Systems", "Northstar Platform"),
            _offering("offering_vision_grid", "Northstar Systems", "Vision Grid"),
            _offering("offering_predict_forge", "Northstar Systems", "Predict Forge"),
            _offering("offering_civic_graph", "Northstar Systems", "Civic Graph"),
            _offering("offering_vision_grid_edge", "Northstar Systems", "Vision Grid Edge"),
            _offering("offering_vision_grid_mobile", "Northstar Systems", "Vision Grid Mobile"),
            _offering("offering_predict_forge_edge", "Northstar Systems", "Predict Forge Edge"),
        ],
        [
            _edge("company_northstar_systems", "HAS_SEGMENT", "segment_industrial_ai"),
            _edge("segment_industrial_ai", "OFFERS", "offering_northstar_platform"),
            _edge("offering_northstar_platform", "OFFERS", "offering_vision_grid"),
            _edge("offering_northstar_platform", "OFFERS", "offering_predict_forge"),
            _edge("offering_northstar_platform", "OFFERS", "offering_civic_graph"),
            _edge("offering_vision_grid", "OFFERS", "offering_vision_grid_edge"),
            _edge("offering_vision_grid", "OFFERS", "offering_vision_grid_mobile"),
            _edge("offering_predict_forge", "OFFERS", "offering_predict_forge_edge"),
        ],
        (
            "Offering nodes carry company_name, and the tree is single-parent so reverse traversal is deterministic.",
            "The hierarchy has multiple children and at least one deeper descendant for recursive path supervision.",
        ),
        (
            "This is the core hierarchy fixture for immediate children, descendants, and anchor-path queries.",
        ),
    ),
    _fixture(
        "fx04_offering_hierarchy_tree_aurelia_v1",
        "Second offering hierarchy with a deeper branch and distinct synthetic names for ancestor recovery.",
        ("QF04", "QF05", "QF21", "QF23", "QF24"),
        [
            _company("company_aurelia_systems", "Aurelia Systems"),
            _segment("segment_data_platforms", "Aurelia Systems", "Data Platforms"),
            _offering("offering_aurelia_nexus", "Aurelia Systems", "Aurelia Nexus"),
            _offering("offering_nexus_insight", "Aurelia Systems", "Nexus Insight"),
            _offering("offering_nexus_relay", "Aurelia Systems", "Nexus Relay"),
            _offering("offering_nexus_guard", "Aurelia Systems", "Nexus Guard"),
            _offering("offering_nexus_insight_edge", "Aurelia Systems", "Nexus Insight Edge"),
            _offering("offering_nexus_relay_mobile", "Aurelia Systems", "Nexus Relay Mobile"),
            _offering("offering_nexus_guard_connect", "Aurelia Systems", "Nexus Guard Connect"),
        ],
        [
            _edge("company_aurelia_systems", "HAS_SEGMENT", "segment_data_platforms"),
            _edge("segment_data_platforms", "OFFERS", "offering_aurelia_nexus"),
            _edge("offering_aurelia_nexus", "OFFERS", "offering_nexus_insight"),
            _edge("offering_aurelia_nexus", "OFFERS", "offering_nexus_relay"),
            _edge("offering_aurelia_nexus", "OFFERS", "offering_nexus_guard"),
            _edge("offering_nexus_insight", "OFFERS", "offering_nexus_insight_edge"),
            _edge("offering_nexus_relay", "OFFERS", "offering_nexus_relay_mobile"),
            _edge("offering_nexus_guard", "OFFERS", "offering_nexus_guard_connect"),
        ],
        (
            "Offering nodes carry company_name so ancestor and path queries stay company-scoped.",
            "The hierarchy has a clear root plus multiple branches, including a deeper descendant under one branch.",
        ),
        (
            "This second hierarchy fixture broadens the training signal for path anchoring and parent lookup.",
        ),
    ),
    _fixture(
        "fx04_offering_hierarchy_tree_polaris_v1",
        "Third hierarchy fixture with a deeper branch and an alternate root for recursive child and ancestor queries.",
        ("QF04", "QF05", "QF21", "QF23", "QF24"),
        [
            _company("company_polaris_core", "Polaris Core"),
            _segment("segment_strategic_platforms", "Polaris Core", "Strategic Platforms"),
            _offering("offering_polaris_core", "Polaris Core", "Polaris Core"),
            _offering("offering_polaris_edge", "Polaris Core", "Polaris Edge"),
            _offering("offering_polaris_guard", "Polaris Core", "Polaris Guard"),
            _offering("offering_polaris_edge_drift", "Polaris Core", "Polaris Edge Drift"),
            _offering("offering_polaris_guard_signal", "Polaris Core", "Polaris Guard Signal"),
        ],
        [
            _edge("company_polaris_core", "HAS_SEGMENT", "segment_strategic_platforms"),
            _edge("segment_strategic_platforms", "OFFERS", "offering_polaris_core"),
            _edge("offering_polaris_core", "OFFERS", "offering_polaris_edge"),
            _edge("offering_polaris_core", "OFFERS", "offering_polaris_guard"),
            _edge("offering_polaris_edge", "OFFERS", "offering_polaris_edge_drift"),
            _edge("offering_polaris_guard", "OFFERS", "offering_polaris_guard_signal"),
        ],
        (
            "Offering nodes carry company_name so deeper ancestor and breadcrumb queries stay company-scoped.",
            "The fixture introduces another multi-branch tree with one deeper descendant branch for recursion coverage.",
        ),
        (
            "This third hierarchy fixture adds one more high-value deep-query anchor to the core spec.",
        ),
    ),
    _fixture(
        "fx05_geo_partner_profile_europe_v1",
        "Europe-centric geography and partner fixture with helper arrays for broader and narrower place matches.",
        ("QF06", "QF07", "QF22", "QF29", "QF21"),
        [
            _place(
                "place_europe",
                "Europe",
                includes_places=("Western Europe", "Eastern Europe", "Italy", "Germany"),
            ),
            _place(
                "place_western_europe",
                "Western Europe",
                within_places=("Europe", "EMEA", "European Union"),
                includes_places=("Italy", "Germany"),
            ),
            _place(
                "place_italy",
                "Italy",
                within_places=("Western Europe", "Europe", "EMEA", "European Union"),
            ),
            _place(
                "place_germany",
                "Germany",
                within_places=("Western Europe", "Europe", "EMEA", "European Union"),
            ),
            _company("company_helioforge", "Helioforge"),
            _company("company_panatlantic_software", "PanAtlantic Software"),
            _company("company_euroscope_systems", "EuroScope Systems"),
            _company("company_cobalt_ridge", "Cobalt Ridge"),
            _company("company_lumen_grid", "Lumen Grid"),
            _company("company_verity_works", "Verity Works"),
            _company("company_meridian_nexus", "Meridian Nexus"),
            _company("company_harborline_systems", "Harborline Systems"),
        ],
        [
            _edge("company_helioforge", "OPERATES_IN", "place_italy"),
            _edge("company_helioforge", "OPERATES_IN", "place_germany"),
            _edge("company_panatlantic_software", "OPERATES_IN", "place_western_europe"),
            _edge("company_euroscope_systems", "OPERATES_IN", "place_europe"),
            _edge("company_helioforge", "PARTNERS_WITH", "company_cobalt_ridge"),
            _edge("company_helioforge", "PARTNERS_WITH", "company_lumen_grid"),
            _edge("company_helioforge", "PARTNERS_WITH", "company_verity_works"),
            _edge("company_panatlantic_software", "PARTNERS_WITH", "company_meridian_nexus"),
            _edge("company_euroscope_systems", "PARTNERS_WITH", "company_harborline_systems"),
        ],
        (
            "Place helper arrays are attached to the Place nodes so downstream queries can recover broader and narrower matches.",
            "The fixture includes exact, broader, and narrower geography relationships in one compact graph.",
        ),
        (
            "Helioforge is the main subject company, while the other companies give QF29 broader and narrower comparison points.",
        ),
    ),
    _fixture(
        "fx05_geo_partner_profile_apac_v1",
        "APAC-centric geography and partner fixture with helper arrays for broader and narrower place matches.",
        ("QF06", "QF07", "QF22", "QF29", "QF21"),
        [
            _place(
                "place_apac",
                "APAC",
                includes_places=("Asia Pacific", "Southeast Asia", "Japan", "Singapore"),
            ),
            _place(
                "place_asia_pacific",
                "Asia Pacific",
                within_places=("APAC",),
                includes_places=("Southeast Asia", "Japan", "Singapore"),
            ),
            _place(
                "place_southeast_asia",
                "Southeast Asia",
                within_places=("Asia Pacific", "APAC"),
                includes_places=("Singapore",),
            ),
            _place(
                "place_singapore",
                "Singapore",
                within_places=("Southeast Asia", "Asia Pacific", "APAC"),
            ),
            _place(
                "place_japan",
                "Japan",
                within_places=("Asia Pacific", "APAC"),
            ),
            _company("company_moonbridge_systems", "Moonbridge Systems"),
            _company("company_apex_digital", "Apex Digital"),
            _company("company_harborline_asia", "Harborline Asia"),
            _company("company_cobalt_ridge", "Cobalt Ridge"),
            _company("company_meridian_nexus", "Meridian Nexus"),
            _company("company_lattice_forge", "Lattice Forge"),
        ],
        [
            _edge("company_moonbridge_systems", "OPERATES_IN", "place_singapore"),
            _edge("company_apex_digital", "OPERATES_IN", "place_apac"),
            _edge("company_harborline_asia", "OPERATES_IN", "place_asia_pacific"),
            _edge("company_moonbridge_systems", "PARTNERS_WITH", "company_cobalt_ridge"),
            _edge("company_apex_digital", "PARTNERS_WITH", "company_meridian_nexus"),
            _edge("company_harborline_asia", "PARTNERS_WITH", "company_lattice_forge"),
        ],
        (
            "Place helper arrays are attached to the Place nodes so downstream queries can recover broader and narrower matches.",
            "The fixture includes exact, broader, and narrower geography relationships for APAC-style coverage.",
        ),
        (
            "Moonbridge is the main subject company, while the other companies make the hierarchy-aware matching examples more varied.",
        ),
    ),
    _fixture(
        "fx05_geo_partner_profile_americas_v1",
        "Americas-centric geography fixture with deeper hierarchy levels for broader, narrower, and exact geography matches.",
        ("QF29",),
        [
            _place(
                "place_americas",
                "Americas",
                includes_places=("North America", "South America", "United States", "Canada"),
            ),
            _place(
                "place_north_america",
                "North America",
                within_places=("Americas",),
                includes_places=("United States", "Canada"),
            ),
            _place(
                "place_united_states",
                "United States",
                within_places=("North America", "Americas"),
            ),
            _place(
                "place_canada",
                "Canada",
                within_places=("North America", "Americas"),
            ),
            _company("company_northlight_systems", "Northlight Systems"),
            _company("company_northlight_labs", "Northlight Labs"),
            _company("company_northlight_canada", "Northlight Canada"),
        ],
        [
            _edge("company_northlight_systems", "OPERATES_IN", "place_north_america"),
            _edge("company_northlight_labs", "OPERATES_IN", "place_united_states"),
            _edge("company_northlight_canada", "OPERATES_IN", "place_canada"),
        ],
        (
            "Place helper arrays are attached to the Place nodes so broader and narrower matches can both be exercised.",
            "The Americas chain adds a deeper geography path than the Europe and APAC fixtures.",
        ),
        (
            "This fixture gives QF29 another hierarchy shape with exact and inherited geography matches.",
        ),
    ),
    _fixture(
        "fx06_segment_customer_channel_profile_cobalt_ridge_v1",
        "Segment customer-type and channel fixture with a company-scoped segment structure and contrastive predicates.",
        ("QF08", "QF09", "QF17", "QF18", "QF22"),
        [
            _company("company_cobalt_ridge", "Cobalt Ridge"),
            _segment("segment_field_intelligence", "Cobalt Ridge", "Field Intelligence"),
            _segment("segment_logistics_cloud", "Cobalt Ridge", "Logistics Cloud"),
            _offering("offering_field_lens", "Cobalt Ridge", "Field Lens"),
            _offering("offering_scout_mesh", "Cobalt Ridge", "Scout Mesh"),
            _offering("offering_route_sphere", "Cobalt Ridge", "Route Sphere"),
            _offering("offering_cargo_pulse", "Cobalt Ridge", "Cargo Pulse"),
            _customer_type("customer_type_manufacturers", "manufacturers"),
            _customer_type("customer_type_large_enterprises", "large enterprises"),
            _customer_type("customer_type_government_agencies", "government agencies"),
            _customer_type("customer_type_it_professionals", "IT professionals"),
            _channel("channel_direct_sales", "direct sales"),
            _channel("channel_resellers", "resellers"),
            _channel("channel_system_integrators", "system integrators"),
            _channel("channel_online", "online"),
        ],
        [
            _edge("company_cobalt_ridge", "HAS_SEGMENT", "segment_field_intelligence"),
            _edge("company_cobalt_ridge", "HAS_SEGMENT", "segment_logistics_cloud"),
            _edge("segment_field_intelligence", "OFFERS", "offering_field_lens"),
            _edge("segment_field_intelligence", "OFFERS", "offering_scout_mesh"),
            _edge("segment_logistics_cloud", "OFFERS", "offering_route_sphere"),
            _edge("segment_logistics_cloud", "OFFERS", "offering_cargo_pulse"),
            _edge("segment_field_intelligence", "SERVES", "customer_type_manufacturers"),
            _edge("segment_field_intelligence", "SERVES", "customer_type_large_enterprises"),
            _edge("segment_field_intelligence", "SELLS_THROUGH", "channel_direct_sales"),
            _edge("segment_field_intelligence", "SELLS_THROUGH", "channel_resellers"),
            _edge("segment_logistics_cloud", "SERVES", "customer_type_government_agencies"),
            _edge("segment_logistics_cloud", "SERVES", "customer_type_it_professionals"),
            _edge("segment_logistics_cloud", "SELLS_THROUGH", "channel_system_integrators"),
            _edge("segment_logistics_cloud", "SELLS_THROUGH", "channel_online"),
        ],
        (
            "BusinessSegment nodes carry company_name and the closed-label vocabularies remain canonical.",
            "The two segments create a clean contrast for customer-type and channel lookups.",
        ),
        (
            "This fixture is the core segment/customer/channel training graph.",
        ),
    ),
    _fixture(
        "fx06_segment_customer_channel_profile_orion_v1",
        "Alternate segment customer-type and channel fixture with broader contrast across segments.",
        ("QF08", "QF09", "QF17", "QF18", "QF22"),
        [
            _company("company_orion_fabric", "Orion Fabric"),
            _segment("segment_industrial_platform", "Orion Fabric", "Industrial Platform"),
            _segment("segment_public_services", "Orion Fabric", "Public Services"),
            _offering("offering_forge_control", "Orion Fabric", "Forge Control"),
            _offering("offering_signal_orchestrator", "Orion Fabric", "Signal Orchestrator"),
            _offering("offering_civic_stream", "Orion Fabric", "Civic Stream"),
            _offering("offering_service_ledger", "Orion Fabric", "Service Ledger"),
            _customer_type("customer_type_retailers", "retailers"),
            _customer_type("customer_type_developers", "developers"),
            _customer_type("customer_type_educational_institutions", "educational institutions"),
            _customer_type("customer_type_healthcare_organizations", "healthcare organizations"),
            _channel("channel_direct_sales", "direct sales"),
            _channel("channel_oems", "OEMs"),
            _channel("channel_distributors", "distributors"),
            _channel("channel_online", "online"),
        ],
        [
            _edge("company_orion_fabric", "HAS_SEGMENT", "segment_industrial_platform"),
            _edge("company_orion_fabric", "HAS_SEGMENT", "segment_public_services"),
            _edge("segment_industrial_platform", "OFFERS", "offering_forge_control"),
            _edge("segment_industrial_platform", "OFFERS", "offering_signal_orchestrator"),
            _edge("segment_public_services", "OFFERS", "offering_civic_stream"),
            _edge("segment_public_services", "OFFERS", "offering_service_ledger"),
            _edge("segment_industrial_platform", "SERVES", "customer_type_retailers"),
            _edge("segment_industrial_platform", "SERVES", "customer_type_developers"),
            _edge("segment_industrial_platform", "SELLS_THROUGH", "channel_direct_sales"),
            _edge("segment_industrial_platform", "SELLS_THROUGH", "channel_oems"),
            _edge("segment_public_services", "SERVES", "customer_type_educational_institutions"),
            _edge("segment_public_services", "SERVES", "customer_type_healthcare_organizations"),
            _edge("segment_public_services", "SELLS_THROUGH", "channel_distributors"),
            _edge("segment_public_services", "SELLS_THROUGH", "channel_online"),
        ],
        (
            "BusinessSegment nodes carry company_name and the closed-label vocabularies remain canonical.",
            "This fixture gives the core dataset a second segment/customer/channel context with different label combinations.",
        ),
        (
            "The overlap between the two segments is deliberate so filtering and existence checks stay meaningful.",
        ),
    ),
    _fixture(
        "fx07_offering_fallback_channel_harborline_v1",
        "Fallback offering channel fixture with no segment anchor so offering-scoped SELLS_THROUGH queries remain necessary.",
        ("QF10", "QF21", "QF22"),
        [
            _company("company_harborline_works", "Harborline Works"),
            _offering("offering_relay_studio", "Harborline Works", "Relay Studio"),
            _channel("channel_direct_sales", "direct sales"),
            _channel("channel_online", "online"),
            _channel("channel_resellers", "resellers"),
        ],
        [
            _edge("company_harborline_works", "OFFERS", "offering_relay_studio"),
            _edge("offering_relay_studio", "SELLS_THROUGH", "channel_direct_sales"),
            _edge("offering_relay_studio", "SELLS_THROUGH", "channel_online"),
            _edge("offering_relay_studio", "SELLS_THROUGH", "channel_resellers"),
        ],
        (
            "Offering nodes carry company_name, but the fixture intentionally omits any segment anchor.",
            "This keeps the fallback offering channel path separate from the segment-first channel path.",
        ),
        (
            "The offering must be queried directly because there is no BusinessSegment parent anywhere in this graph.",
        ),
    ),
    _fixture(
        "fx07_offering_fallback_channel_kestrel_v1",
        "Second fallback offering fixture with a different company context and the same no-segment-anchor rule.",
        ("QF10", "QF21", "QF22"),
        [
            _company("company_kestrel_foundry", "Kestrel Foundry"),
            _offering("offering_studio_edge", "Kestrel Foundry", "Studio Edge"),
            _channel("channel_online", "online"),
            _channel("channel_direct_sales", "direct sales"),
            _channel("channel_oems", "OEMs"),
        ],
        [
            _edge("company_kestrel_foundry", "OFFERS", "offering_studio_edge"),
            _edge("offering_studio_edge", "SELLS_THROUGH", "channel_online"),
            _edge("offering_studio_edge", "SELLS_THROUGH", "channel_direct_sales"),
            _edge("offering_studio_edge", "SELLS_THROUGH", "channel_oems"),
        ],
        (
            "Offering nodes carry company_name, but the graph deliberately has no segment anchor for the offering.",
            "This fixture broadens the fallback channel supervision with a second company context.",
        ),
        (
            "The model should learn that not every offering can be queried through a segment path.",
        ),
    ),
    _fixture(
        "fx08_offering_revenue_profile_helioforge_v1",
        "Direct offering-to-revenue-model fixture for monetization listing, counting, and membership queries.",
        ("QF11", "QF21", "QF22"),
        [
            _company("company_helioforge", "Helioforge"),
            _offering("offering_helioforge_studio", "Helioforge", "Helioforge Studio"),
            _revenue_model("revenue_model_subscription", "subscription"),
            _revenue_model("revenue_model_service_fees", "service fees"),
            _revenue_model("revenue_model_licensing", "licensing"),
        ],
        [
            _edge("company_helioforge", "OFFERS", "offering_helioforge_studio"),
            _edge("offering_helioforge_studio", "MONETIZES_VIA", "revenue_model_subscription"),
            _edge("offering_helioforge_studio", "MONETIZES_VIA", "revenue_model_service_fees"),
            _edge("offering_helioforge_studio", "MONETIZES_VIA", "revenue_model_licensing"),
        ],
        (
            "Offering nodes carry company_name, but the monetization relation remains strictly offering-scoped.",
            "The fixture uses three canonical revenue models to make counting and membership non-trivial.",
        ),
        (
            "This is the core direct monetization fixture for the answerable revenue family.",
        ),
    ),
    _fixture(
        "fx08_offering_revenue_profile_orion_v1",
        "Alternate direct offering-to-revenue-model fixture with a different company and revenue mix.",
        ("QF11", "QF21", "QF22"),
        [
            _company("company_orion", "Orion"),
            _offering("offering_orion_control_cloud", "Orion", "Orion Control Cloud"),
            _revenue_model("revenue_model_subscription", "subscription"),
            _revenue_model("revenue_model_licensing", "licensing"),
            _revenue_model("revenue_model_consumption_based", "consumption-based"),
        ],
        [
            _edge("company_orion", "OFFERS", "offering_orion_control_cloud"),
            _edge("offering_orion_control_cloud", "MONETIZES_VIA", "revenue_model_subscription"),
            _edge("offering_orion_control_cloud", "MONETIZES_VIA", "revenue_model_licensing"),
            _edge("offering_orion_control_cloud", "MONETIZES_VIA", "revenue_model_consumption_based"),
        ],
        (
            "Offering nodes carry company_name, but the monetization relation remains strictly offering-scoped.",
            "The alternate mix broadens the revenue-model vocabulary in the core dataset.",
        ),
        (
            "This second monetization fixture gives the model a different revenue-model combination to memorize against.",
        ),
    ),
    _fixture(
        "fx30_inventory_collision_atlas_v1",
        "Company-scoped collision fixture where the same surface form appears as a company, segment, and offering across companies.",
        ("QF30",),
        [
            _company("company_atlas", "Atlas"),
            _company("company_atlas_dynamics", "Atlas Dynamics"),
            _company("company_atlas_fabric", "Atlas Fabric"),
            _company("company_beacon", "Beacon"),
            _company("company_beacon_labs", "Beacon Labs"),
            _segment("segment_atlas_atlas", "Atlas", "Atlas"),
            _segment("segment_atlas_dynamics", "Atlas Dynamics", "Atlas"),
            _segment("segment_atlas_fabric", "Atlas Fabric", "Atlas"),
            _segment("segment_beacon_beacon", "Beacon", "Beacon"),
            _segment("segment_beacon_labs", "Beacon Labs", "Beacon"),
            _offering("offering_atlas_atlas", "Atlas", "Atlas"),
            _offering("offering_atlas_dynamics", "Atlas Dynamics", "Atlas"),
            _offering("offering_atlas_fabric", "Atlas Fabric", "Atlas"),
            _offering("offering_atlas_cloud", "Atlas Dynamics", "Atlas Cloud"),
            _offering("offering_beacon_beacon", "Beacon", "Beacon"),
            _offering("offering_beacon_labs", "Beacon Labs", "Beacon"),
            _offering("offering_beacon_signal", "Beacon Labs", "Beacon Signal"),
        ],
        [
            _edge("company_atlas", "HAS_SEGMENT", "segment_atlas_atlas"),
            _edge("company_atlas_dynamics", "HAS_SEGMENT", "segment_atlas_dynamics"),
            _edge("company_atlas_fabric", "HAS_SEGMENT", "segment_atlas_fabric"),
            _edge("company_beacon", "HAS_SEGMENT", "segment_beacon_beacon"),
            _edge("company_beacon_labs", "HAS_SEGMENT", "segment_beacon_labs"),
            _edge("company_atlas", "OFFERS", "offering_atlas_atlas"),
            _edge("company_atlas_dynamics", "OFFERS", "offering_atlas_dynamics"),
            _edge("company_atlas_dynamics", "OFFERS", "offering_atlas_cloud"),
            _edge("company_atlas_fabric", "OFFERS", "offering_atlas_fabric"),
            _edge("company_beacon", "OFFERS", "offering_beacon_beacon"),
            _edge("company_beacon_labs", "OFFERS", "offering_beacon_labs"),
            _edge("company_beacon_labs", "OFFERS", "offering_beacon_signal"),
        ],
        (
            "BusinessSegment and Offering carry company_name so same-surface nodes stay separated by company.",
            "The graph intentionally repeats Atlas and Beacon across company, segment, and offering scopes.",
        ),
        (
            "This is the primary collision fixture for company-scoped inventory lookup and ambiguity supervision.",
        ),
    ),
    _fixture(
        "fx30_inventory_collision_orbit_v1",
        "Second collision fixture with another same-surface inventory cluster for the harder company-scoped lookup family.",
        ("QF30",),
        [
            _company("company_orbit", "Orbit"),
            _company("company_orbit_labs", "Orbit Labs"),
            _company("company_orbit_works", "Orbit Works"),
            _segment("segment_orbit_orbit", "Orbit", "Orbit"),
            _segment("segment_orbit_labs", "Orbit Labs", "Orbit"),
            _segment("segment_orbit_works", "Orbit Works", "Orbit"),
            _offering("offering_orbit_orbit", "Orbit", "Orbit"),
            _offering("offering_orbit_labs", "Orbit Labs", "Orbit"),
            _offering("offering_orbit_labs_cloud", "Orbit Labs", "Orbit Cloud"),
            _offering("offering_orbit_works", "Orbit Works", "Orbit"),
            _offering("offering_orbit_works_signal", "Orbit Works", "Orbit Signal"),
        ],
        [
            _edge("company_orbit", "HAS_SEGMENT", "segment_orbit_orbit"),
            _edge("company_orbit_labs", "HAS_SEGMENT", "segment_orbit_labs"),
            _edge("company_orbit_works", "HAS_SEGMENT", "segment_orbit_works"),
            _edge("company_orbit", "OFFERS", "offering_orbit_orbit"),
            _edge("company_orbit_labs", "OFFERS", "offering_orbit_labs"),
            _edge("company_orbit_labs", "OFFERS", "offering_orbit_labs_cloud"),
            _edge("company_orbit_works", "OFFERS", "offering_orbit_works"),
            _edge("company_orbit_works", "OFFERS", "offering_orbit_works_signal"),
        ],
        (
            "BusinessSegment and Offering carry company_name so same-surface inventory stays separated by company.",
            "The repeated Orbit surface gives the model another disambiguation cluster beyond Atlas and Beacon.",
        ),
        (
            "This second collision fixture broadens the high-value company-scoped lookup supervision.",
        ),
    ),
]


COMPANY_SEGMENT_BINDINGS = [
    {"binding_id": "northstar", "fixture_id": "fx01_company_segment_core_northstar_v1", "graph_id": "fx01_company_segment_core_northstar_v1", "params": {"company": "Northstar Systems", "segment": "Industrial AI"}},
    {"binding_id": "harborline", "fixture_id": "fx01_company_segment_core_harborline_v1", "graph_id": "fx01_company_segment_core_harborline_v1", "params": {"company": "Harborline Systems", "segment": "Industrial Security"}},
]

COMPANY_DIRECT_OFFERING_BINDINGS = [
    {"binding_id": "meridian", "fixture_id": "fx02_company_direct_offerings_meridian_v1", "graph_id": "fx02_company_direct_offerings_meridian_v1", "params": {"company": "Meridian Nexus", "offering": "Meridian Atlas"}},
    {"binding_id": "lattice", "fixture_id": "fx02_company_direct_offerings_latticeforge_v1", "graph_id": "fx02_company_direct_offerings_latticeforge_v1", "params": {"company": "Lattice Forge", "offering": "Lattice Atlas"}},
]

SEGMENT_DIRECT_OFFERING_BINDINGS = [
    {"binding_id": "asteron", "fixture_id": "fx03_segment_direct_offerings_asteron_v1", "graph_id": "fx03_segment_direct_offerings_asteron_v1", "params": {"company": "Asteron Analytics", "segment": "Industrial AI", "offering": "Asteron Control Suite"}},
    {"binding_id": "aurum", "fixture_id": "fx03_segment_direct_offerings_aurum_v1", "graph_id": "fx03_segment_direct_offerings_aurum_v1", "params": {"company": "Aurum Metrics", "segment": "Decision Intelligence", "offering": "Aurum Control Suite"}},
]

HIERARCHY_BINDINGS = [
    {
        "binding_id": "northstar",
        "fixture_id": "fx04_offering_hierarchy_tree_northstar_v1",
        "graph_id": "fx04_offering_hierarchy_tree_northstar_v1",
        "params": {
            "company": "Northstar Systems",
            "segment": "Industrial AI",
            "offering": "Northstar Platform",
            "child_offering": "Vision Grid",
            "descendant_offering": "Vision Grid Edge",
        },
    },
    {
        "binding_id": "aurelia",
        "fixture_id": "fx04_offering_hierarchy_tree_aurelia_v1",
        "graph_id": "fx04_offering_hierarchy_tree_aurelia_v1",
        "params": {
            "company": "Aurelia Systems",
            "segment": "Data Platforms",
            "offering": "Aurelia Nexus",
            "child_offering": "Nexus Insight",
            "descendant_offering": "Nexus Insight Edge",
        },
    },
    {
        "binding_id": "polaris",
        "fixture_id": "fx04_offering_hierarchy_tree_polaris_v1",
        "graph_id": "fx04_offering_hierarchy_tree_polaris_v1",
        "params": {
            "company": "Polaris Core",
            "segment": "Strategic Platforms",
            "offering": "Polaris Core",
            "child_offering": "Polaris Edge",
            "descendant_offering": "Polaris Edge Drift",
        },
    },
]

REVERSE_HIERARCHY_BINDINGS = [
    {
        "binding_id": binding["binding_id"],
        "fixture_id": binding["fixture_id"],
        "graph_id": binding["graph_id"],
        "params": {
            "company": binding["params"]["company"],
            "offering": binding["params"]["descendant_offering"],
        },
    }
    for binding in HIERARCHY_BINDINGS
]

GEO_PARTNER_EU_BINDINGS = [
    {"binding_id": "helioforge", "fixture_id": "fx05_geo_partner_profile_europe_v1", "graph_id": "fx05_geo_partner_profile_europe_v1", "params": {"company": "Helioforge", "place": "Germany", "partner": "Cobalt Ridge"}},
    {"binding_id": "panatlantic", "fixture_id": "fx05_geo_partner_profile_europe_v1", "graph_id": "fx05_geo_partner_profile_europe_v1", "params": {"company": "PanAtlantic Software", "place": "Western Europe", "partner": "Meridian Nexus"}},
    {"binding_id": "euroscope", "fixture_id": "fx05_geo_partner_profile_europe_v1", "graph_id": "fx05_geo_partner_profile_europe_v1", "params": {"company": "EuroScope Systems", "place": "Europe", "partner": "Harborline Systems"}},
]

GEO_PARTNER_APAC_BINDINGS = [
    {"binding_id": "moonbridge", "fixture_id": "fx05_geo_partner_profile_apac_v1", "graph_id": "fx05_geo_partner_profile_apac_v1", "params": {"company": "Moonbridge Systems", "place": "Singapore", "partner": "Cobalt Ridge"}},
    {"binding_id": "apex", "fixture_id": "fx05_geo_partner_profile_apac_v1", "graph_id": "fx05_geo_partner_profile_apac_v1", "params": {"company": "Apex Digital", "place": "APAC", "partner": "Meridian Nexus"}},
    {"binding_id": "harborline_asia", "fixture_id": "fx05_geo_partner_profile_apac_v1", "graph_id": "fx05_geo_partner_profile_apac_v1", "params": {"company": "Harborline Asia", "place": "Asia Pacific", "partner": "Lattice Forge"}},
]

SEGMENT_CUSTOMER_CHANNEL_BINDINGS = [
    {
        "binding_id": "cobalt",
        "fixture_id": "fx06_segment_customer_channel_profile_cobalt_ridge_v1",
        "graph_id": "fx06_segment_customer_channel_profile_cobalt_ridge_v1",
        "params": {
            "company": "Cobalt Ridge",
            "segment": "Field Intelligence",
            "customer_type": "manufacturers",
            "channel": "direct sales",
        },
    },
    {
        "binding_id": "orion",
        "fixture_id": "fx06_segment_customer_channel_profile_orion_v1",
        "graph_id": "fx06_segment_customer_channel_profile_orion_v1",
        "params": {
            "company": "Orion Fabric",
            "segment": "Industrial Platform",
            "customer_type": "developers",
            "channel": "OEMs",
        },
    },
]

FALLBACK_CHANNEL_BINDINGS = [
    {
        "binding_id": "harborline",
        "fixture_id": "fx07_offering_fallback_channel_harborline_v1",
        "graph_id": "fx07_offering_fallback_channel_harborline_v1",
        "params": {
            "company": "Harborline Works",
            "offering": "Relay Studio",
            "channel": "online",
        },
    },
    {
        "binding_id": "kestrel",
        "fixture_id": "fx07_offering_fallback_channel_kestrel_v1",
        "graph_id": "fx07_offering_fallback_channel_kestrel_v1",
        "params": {
            "company": "Kestrel Foundry",
            "offering": "Studio Edge",
            "channel": "OEMs",
        },
    },
]

REVENUE_BINDINGS = [
    {
        "binding_id": "helioforge",
        "fixture_id": "fx08_offering_revenue_profile_helioforge_v1",
        "graph_id": "fx08_offering_revenue_profile_helioforge_v1",
        "params": {
            "company": "Helioforge",
            "offering": "Helioforge Studio",
            "revenue_model": "subscription",
        },
    },
    {
        "binding_id": "orion",
        "fixture_id": "fx08_offering_revenue_profile_orion_v1",
        "graph_id": "fx08_offering_revenue_profile_orion_v1",
        "params": {
            "company": "Orion",
            "offering": "Orion Control Cloud",
            "revenue_model": "licensing",
        },
    },
]

QF29_BINDINGS = [
    {
        "binding_id": "helioforge",
        "fixture_id": "fx05_geo_partner_profile_europe_v1",
        "graph_id": "fx05_geo_partner_profile_europe_v1",
        "params": {"company": "Helioforge", "place": "Europe"},
    },
    {
        "binding_id": "panatlantic",
        "fixture_id": "fx05_geo_partner_profile_europe_v1",
        "graph_id": "fx05_geo_partner_profile_europe_v1",
        "params": {"company": "PanAtlantic Software", "place": "Italy"},
    },
    {
        "binding_id": "euroscope",
        "fixture_id": "fx05_geo_partner_profile_europe_v1",
        "graph_id": "fx05_geo_partner_profile_europe_v1",
        "params": {"company": "EuroScope Systems", "place": "Western Europe"},
    },
    {
        "binding_id": "moonbridge",
        "fixture_id": "fx05_geo_partner_profile_apac_v1",
        "graph_id": "fx05_geo_partner_profile_apac_v1",
        "params": {"company": "Moonbridge Systems", "place": "APAC"},
    },
    {
        "binding_id": "apex",
        "fixture_id": "fx05_geo_partner_profile_apac_v1",
        "graph_id": "fx05_geo_partner_profile_apac_v1",
        "params": {"company": "Apex Digital", "place": "Singapore"},
    },
    {
        "binding_id": "harborline_asia",
        "fixture_id": "fx05_geo_partner_profile_apac_v1",
        "graph_id": "fx05_geo_partner_profile_apac_v1",
        "params": {"company": "Harborline Asia", "place": "Asia Pacific"},
    },
    {
        "binding_id": "northlight_systems",
        "fixture_id": "fx05_geo_partner_profile_americas_v1",
        "graph_id": "fx05_geo_partner_profile_americas_v1",
        "params": {"company": "Northlight Systems", "place": "North America"},
    },
    {
        "binding_id": "northlight_labs",
        "fixture_id": "fx05_geo_partner_profile_americas_v1",
        "graph_id": "fx05_geo_partner_profile_americas_v1",
        "params": {"company": "Northlight Labs", "place": "United States"},
    },
    {
        "binding_id": "northlight_canada",
        "fixture_id": "fx05_geo_partner_profile_americas_v1",
        "graph_id": "fx05_geo_partner_profile_americas_v1",
        "params": {"company": "Northlight Canada", "place": "Canada"},
    },
]

QF30_OFFERING_BINDINGS = [
    {
        "binding_id": "atlas",
        "fixture_id": "fx30_inventory_collision_atlas_v1",
        "graph_id": "fx30_inventory_collision_atlas_v1",
        "params": {"offering_name": "Atlas"},
    },
    {
        "binding_id": "beacon",
        "fixture_id": "fx30_inventory_collision_atlas_v1",
        "graph_id": "fx30_inventory_collision_atlas_v1",
        "params": {"offering_name": "Beacon"},
    },
    {
        "binding_id": "orbit",
        "fixture_id": "fx30_inventory_collision_orbit_v1",
        "graph_id": "fx30_inventory_collision_orbit_v1",
        "params": {"offering_name": "Orbit"},
    },
]

QF30_SEGMENT_BINDINGS = [
    {
        "binding_id": "atlas",
        "fixture_id": "fx30_inventory_collision_atlas_v1",
        "graph_id": "fx30_inventory_collision_atlas_v1",
        "params": {"segment_name": "Atlas"},
    },
    {
        "binding_id": "beacon",
        "fixture_id": "fx30_inventory_collision_atlas_v1",
        "graph_id": "fx30_inventory_collision_atlas_v1",
        "params": {"segment_name": "Beacon"},
    },
    {
        "binding_id": "orbit",
        "fixture_id": "fx30_inventory_collision_orbit_v1",
        "graph_id": "fx30_inventory_collision_orbit_v1",
        "params": {"segment_name": "Orbit"},
    },
]

QF30_SCOPED_BINDINGS = [
    {
        "binding_id": "atlas_company",
        "fixture_id": "fx30_inventory_collision_atlas_v1",
        "graph_id": "fx30_inventory_collision_atlas_v1",
        "params": {"company": "Atlas", "offering_name": "Atlas", "segment_name": "Atlas"},
    },
    {
        "binding_id": "beacon_company",
        "fixture_id": "fx30_inventory_collision_atlas_v1",
        "graph_id": "fx30_inventory_collision_atlas_v1",
        "params": {"company": "Beacon", "offering_name": "Beacon", "segment_name": "Beacon"},
    },
    {
        "binding_id": "orbit_company",
        "fixture_id": "fx30_inventory_collision_orbit_v1",
        "graph_id": "fx30_inventory_collision_orbit_v1",
        "params": {"company": "Orbit", "offering_name": "Orbit", "segment_name": "Orbit"},
    },
    {
        "binding_id": "orbit_labs_company",
        "fixture_id": "fx30_inventory_collision_orbit_v1",
        "graph_id": "fx30_inventory_collision_orbit_v1",
        "params": {"company": "Orbit Labs", "offering_name": "Orbit", "segment_name": "Orbit"},
    },
]

QF30_REFUSAL_BINDINGS = [
    {
        "binding_id": "atlas",
        "fixture_id": "fx30_inventory_collision_atlas_v1",
        "graph_id": "fx30_inventory_collision_atlas_v1",
        "params": {"surface_name": "Atlas"},
    },
    {
        "binding_id": "beacon",
        "fixture_id": "fx30_inventory_collision_atlas_v1",
        "graph_id": "fx30_inventory_collision_atlas_v1",
        "params": {"surface_name": "Beacon"},
    },
    {
        "binding_id": "orbit",
        "fixture_id": "fx30_inventory_collision_orbit_v1",
        "graph_id": "fx30_inventory_collision_orbit_v1",
        "params": {"surface_name": "Orbit"},
    },
]


intent_specs = [
    _intent(
        intent_id="qf01_company_segments_list",
        family_id="QF01",
        question_template="What business segments does {company} have?",
        paraphrase_templates=("List the business segments for {company}.",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(s:BusinessSegment {company_name: $company}) RETURN DISTINCT s.name AS segment ORDER BY segment",
        result_shape=SEGMENT_LIST_SHAPE,
        bindings=COMPANY_SEGMENT_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf01_company_segments_count",
        family_id="QF01",
        question_template="How many business segments does {company} have?",
        paraphrase_templates=("Count the business segments for {company}.",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(s:BusinessSegment {company_name: $company}) RETURN COUNT(DISTINCT s) AS segment_count",
        result_shape=COUNT_SHAPE("segment_count"),
        bindings=COMPANY_SEGMENT_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf01_company_segments_membership",
        family_id="QF01",
        question_template="Does {company} have {segment}?",
        paraphrase_templates=("Is {segment} one of {company}'s business segments?",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(s:BusinessSegment {company_name: $company, name: $segment}) RETURN COUNT(DISTINCT s) > 0 AS is_match",
        result_shape=BOOL_SHAPE("Whether the company has the given segment"),
        bindings=COMPANY_SEGMENT_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf02_company_offerings_list",
        family_id="QF02",
        question_template="What offerings does {company} have at the company level?",
        paraphrase_templates=("List {company}'s company-level offerings.",),
        cypher="MATCH (:Company {name: $company})-[:OFFERS]->(o:Offering {company_name: $company}) RETURN DISTINCT o.name AS offering ORDER BY offering",
        result_shape=OFFERING_LIST_SHAPE,
        bindings=COMPANY_DIRECT_OFFERING_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf02_company_offerings_count",
        family_id="QF02",
        question_template="How many company-level offerings does {company} have?",
        paraphrase_templates=("Count {company}'s company-level offerings.",),
        cypher="MATCH (:Company {name: $company})-[:OFFERS]->(o:Offering {company_name: $company}) RETURN COUNT(DISTINCT o) AS offering_count",
        result_shape=COUNT_SHAPE("offering_count"),
        bindings=COMPANY_DIRECT_OFFERING_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf02_company_offerings_membership",
        family_id="QF02",
        question_template="Does {company} directly offer {offering}?",
        paraphrase_templates=("Is {offering} a direct company-level offering for {company}?",),
        cypher="MATCH (:Company {name: $company})-[:OFFERS]->(o:Offering {company_name: $company, name: $offering}) RETURN COUNT(DISTINCT o) > 0 AS is_match",
        result_shape=BOOL_SHAPE("Whether the company directly offers the given offering"),
        bindings=COMPANY_DIRECT_OFFERING_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf03_segment_direct_offerings_list",
        family_id="QF03",
        question_template="What direct offerings does {company}'s {segment} segment have?",
        paraphrase_templates=("List the direct offerings owned by {company}'s {segment} segment.",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company, name: $segment})-[:OFFERS]->(o:Offering {company_name: $company}) RETURN DISTINCT o.name AS offering ORDER BY offering",
        result_shape=OFFERING_LIST_SHAPE,
        bindings=SEGMENT_DIRECT_OFFERING_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf03_segment_direct_offerings_count",
        family_id="QF03",
        question_template="How many direct offerings does {company}'s {segment} segment have?",
        paraphrase_templates=("Count the offerings in {company}'s {segment} segment.",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company, name: $segment})-[:OFFERS]->(o:Offering {company_name: $company}) RETURN COUNT(DISTINCT o) AS offering_count",
        result_shape=COUNT_SHAPE("offering_count"),
        bindings=SEGMENT_DIRECT_OFFERING_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf03_segment_direct_offerings_membership",
        family_id="QF03",
        question_template="Does {company}'s {segment} segment offer {offering}?",
        paraphrase_templates=("Is {offering} directly under {company}'s {segment} segment?",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company, name: $segment})-[:OFFERS]->(o:Offering {company_name: $company, name: $offering}) RETURN COUNT(DISTINCT o) > 0 AS is_match",
        result_shape=BOOL_SHAPE("Whether the segment directly offers the given offering"),
        bindings=SEGMENT_DIRECT_OFFERING_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf04_offering_children_list",
        family_id="QF04",
        question_template="What is inside {company}'s {offering}?",
        paraphrase_templates=("What are the immediate children of {company}'s {offering}?",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company, name: $segment})-[:OFFERS]->(root:Offering {company_name: $company, name: $offering})-[:OFFERS]->(child:Offering {company_name: $company}) RETURN DISTINCT child.name AS child_offering ORDER BY child_offering",
        result_shape=CHILD_OFFERING_LIST_SHAPE,
        bindings=HIERARCHY_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf04_offering_children_count",
        family_id="QF04",
        question_template="How many child offerings does {company}'s {offering} have?",
        paraphrase_templates=("Count the immediate children inside {company}'s {offering}.",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company, name: $segment})-[:OFFERS]->(root:Offering {company_name: $company, name: $offering})-[:OFFERS]->(child:Offering {company_name: $company}) RETURN COUNT(DISTINCT child) AS child_offering_count",
        result_shape=COUNT_SHAPE("child_offering_count"),
        bindings=HIERARCHY_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf04_offering_children_membership",
        family_id="QF04",
        question_template="Is {child_offering} inside {company}'s {offering}?",
        paraphrase_templates=("Does {company}'s {offering} contain {child_offering}?",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company, name: $segment})-[:OFFERS]->(:Offering {company_name: $company, name: $offering})-[:OFFERS]->(child:Offering {company_name: $company, name: $child_offering}) RETURN COUNT(DISTINCT child) > 0 AS is_match",
        result_shape=BOOL_SHAPE("Whether the child offering is directly inside the given parent offering"),
        bindings=HIERARCHY_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf05_offering_descendants_list",
        family_id="QF05",
        question_template="What offerings sit under {company}'s {offering}?",
        paraphrase_templates=("List all descendant offerings under {company}'s {offering}.",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company, name: $segment})-[:OFFERS]->(:Offering {company_name: $company, name: $offering})-[:OFFERS*1..]->(descendant:Offering {company_name: $company}) RETURN DISTINCT descendant.name AS descendant_offering ORDER BY descendant_offering",
        result_shape=DESCENDANT_OFFERING_LIST_SHAPE,
        bindings=HIERARCHY_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf05_offering_descendants_count",
        family_id="QF05",
        question_template="How many offerings sit under {company}'s {offering}?",
        paraphrase_templates=("Count the descendants of {company}'s {offering}.",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company, name: $segment})-[:OFFERS]->(:Offering {company_name: $company, name: $offering})-[:OFFERS*1..]->(descendant:Offering {company_name: $company}) RETURN COUNT(DISTINCT descendant) AS descendant_count",
        result_shape=COUNT_SHAPE("descendant_count"),
        bindings=HIERARCHY_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf05_offering_leaf_descendants_list",
        family_id="QF05",
        question_template="Which leaf offerings are under {company}'s {offering}?",
        paraphrase_templates=("List only the leaf descendants of {company}'s {offering}.",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company, name: $segment})-[:OFFERS]->(:Offering {company_name: $company, name: $offering})-[:OFFERS*1..]->(leaf:Offering {company_name: $company}) WHERE NOT (leaf)-[:OFFERS]->(:Offering {company_name: $company}) RETURN DISTINCT leaf.name AS leaf_offering ORDER BY leaf_offering",
        result_shape=LEAF_OFFERING_LIST_SHAPE,
        bindings=HIERARCHY_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf06_company_places_list",
        family_id="QF06",
        question_template="Where does {company} operate?",
        paraphrase_templates=("What places does {company} operate in?",),
        cypher="MATCH (:Company {name: $company})-[:OPERATES_IN]->(place:Place) RETURN DISTINCT place.name AS place ORDER BY place",
        result_shape=PLACE_LIST_SHAPE,
        bindings=GEO_PARTNER_EU_BINDINGS[:2] + GEO_PARTNER_APAC_BINDINGS[:2],
        difficulty="low",
    ),
    _intent(
        intent_id="qf06_company_places_count",
        family_id="QF06",
        question_template="How many places does {company} operate in?",
        paraphrase_templates=("Count the places where {company} operates.",),
        cypher="MATCH (:Company {name: $company})-[:OPERATES_IN]->(place:Place) RETURN COUNT(DISTINCT place) AS place_count",
        result_shape=COUNT_SHAPE("place_count"),
        bindings=GEO_PARTNER_EU_BINDINGS[:2] + GEO_PARTNER_APAC_BINDINGS[:2],
        difficulty="low",
    ),
    _intent(
        intent_id="qf06_company_places_membership",
        family_id="QF06",
        question_template="Does {company} operate in {place}?",
        paraphrase_templates=("Is {place} one of {company}'s operating geographies?",),
        cypher="MATCH (:Company {name: $company})-[:OPERATES_IN]->(place_node:Place {name: $place}) RETURN COUNT(DISTINCT place_node) > 0 AS is_match",
        result_shape=BOOL_SHAPE("Whether the company operates in the given place"),
        bindings=GEO_PARTNER_EU_BINDINGS + GEO_PARTNER_APAC_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf07_company_partners_list",
        family_id="QF07",
        question_template="Who does {company} partner with?",
        paraphrase_templates=("List {company}'s partners.",),
        cypher="MATCH (:Company {name: $company})-[:PARTNERS_WITH]->(partner:Company) RETURN DISTINCT partner.name AS partner ORDER BY partner",
        result_shape=PARTNER_LIST_SHAPE,
        bindings=GEO_PARTNER_EU_BINDINGS + GEO_PARTNER_APAC_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf07_company_partners_count",
        family_id="QF07",
        question_template="How many partners does {company} have?",
        paraphrase_templates=("Count {company}'s partners.",),
        cypher="MATCH (:Company {name: $company})-[:PARTNERS_WITH]->(partner:Company) RETURN COUNT(DISTINCT partner) AS partner_count",
        result_shape=COUNT_SHAPE("partner_count"),
        bindings=GEO_PARTNER_EU_BINDINGS + GEO_PARTNER_APAC_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf07_company_partners_membership",
        family_id="QF07",
        question_template="Does {company} partner with {partner}?",
        paraphrase_templates=("Is {partner} a partner of {company}?",),
        cypher="MATCH (:Company {name: $company})-[:PARTNERS_WITH]->(partner:Company {name: $partner}) RETURN COUNT(DISTINCT partner) > 0 AS is_match",
        result_shape=BOOL_SHAPE("Whether the company partners with the given company"),
        bindings=GEO_PARTNER_EU_BINDINGS + GEO_PARTNER_APAC_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf08_segment_customer_types_list",
        family_id="QF08",
        question_template="Which customer types does {company}'s {segment} segment serve?",
        paraphrase_templates=("List the customer types served by {company}'s {segment} segment.",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company, name: $segment})-[:SERVES]->(c:CustomerType) RETURN DISTINCT c.name AS customer_type ORDER BY customer_type",
        result_shape=CUSTOMER_TYPE_LIST_SHAPE,
        bindings=SEGMENT_CUSTOMER_CHANNEL_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf08_segment_customer_types_count",
        family_id="QF08",
        question_template="How many customer types does {company}'s {segment} segment serve?",
        paraphrase_templates=("Count the customer types served by {company}'s {segment} segment.",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company, name: $segment})-[:SERVES]->(c:CustomerType) RETURN COUNT(DISTINCT c) AS customer_type_count",
        result_shape=COUNT_SHAPE("customer_type_count"),
        bindings=SEGMENT_CUSTOMER_CHANNEL_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf08_segment_customer_types_membership",
        family_id="QF08",
        question_template="Does {company}'s {segment} segment serve {customer_type}?",
        paraphrase_templates=("Is {customer_type} a target customer type for {company}'s {segment} segment?",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company, name: $segment})-[:SERVES]->(c:CustomerType {name: $customer_type}) RETURN COUNT(DISTINCT c) > 0 AS is_match",
        result_shape=BOOL_SHAPE("Whether the segment serves the given customer type"),
        bindings=SEGMENT_CUSTOMER_CHANNEL_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf09_segment_channels_list",
        family_id="QF09",
        question_template="Which channels does {company}'s {segment} segment sell through?",
        paraphrase_templates=("List the channels used by {company}'s {segment} segment.",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company, name: $segment})-[:SELLS_THROUGH]->(ch:Channel) RETURN DISTINCT ch.name AS channel ORDER BY channel",
        result_shape=CHANNEL_LIST_SHAPE,
        bindings=SEGMENT_CUSTOMER_CHANNEL_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf09_segment_channels_count",
        family_id="QF09",
        question_template="How many channels does {company}'s {segment} segment sell through?",
        paraphrase_templates=("Count the channels used by {company}'s {segment} segment.",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company, name: $segment})-[:SELLS_THROUGH]->(ch:Channel) RETURN COUNT(DISTINCT ch) AS channel_count",
        result_shape=COUNT_SHAPE("channel_count"),
        bindings=SEGMENT_CUSTOMER_CHANNEL_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf09_segment_channels_membership",
        family_id="QF09",
        question_template="Does {company}'s {segment} segment sell through {channel}?",
        paraphrase_templates=("Is {channel} a channel for {company}'s {segment} segment?",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company, name: $segment})-[:SELLS_THROUGH]->(ch:Channel {name: $channel}) RETURN COUNT(DISTINCT ch) > 0 AS is_match",
        result_shape=BOOL_SHAPE("Whether the segment sells through the given channel"),
        bindings=SEGMENT_CUSTOMER_CHANNEL_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf10_offering_channels_list",
        family_id="QF10",
        question_template="How is {company}'s {offering} sold?",
        paraphrase_templates=("Which channels sell {company}'s {offering}?",),
        cypher="MATCH (:Company {name: $company})-[:OFFERS]->(o:Offering {company_name: $company, name: $offering})-[:SELLS_THROUGH]->(ch:Channel) RETURN DISTINCT ch.name AS channel ORDER BY channel",
        result_shape=CHANNEL_LIST_SHAPE,
        bindings=FALLBACK_CHANNEL_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf10_offering_channels_count",
        family_id="QF10",
        question_template="How many channels does {company}'s {offering} use?",
        paraphrase_templates=("Count the channels for {company}'s {offering}.",),
        cypher="MATCH (:Company {name: $company})-[:OFFERS]->(o:Offering {company_name: $company, name: $offering})-[:SELLS_THROUGH]->(ch:Channel) RETURN COUNT(DISTINCT ch) AS channel_count",
        result_shape=COUNT_SHAPE("channel_count"),
        bindings=FALLBACK_CHANNEL_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf10_offering_channels_membership",
        family_id="QF10",
        question_template="Does {company}'s {offering} sell through {channel}?",
        paraphrase_templates=("Is {channel} a sales channel for {company}'s {offering}?",),
        cypher="MATCH (:Company {name: $company})-[:OFFERS]->(o:Offering {company_name: $company, name: $offering})-[:SELLS_THROUGH]->(ch:Channel {name: $channel}) RETURN COUNT(DISTINCT ch) > 0 AS is_match",
        result_shape=BOOL_SHAPE("Whether the offering sells through the given channel"),
        bindings=FALLBACK_CHANNEL_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf11_offering_revenue_models_list",
        family_id="QF11",
        question_template="How does {company}'s {offering} make money?",
        paraphrase_templates=("Which revenue models does {company}'s {offering} use?",),
        cypher="MATCH (:Company {name: $company})-[:OFFERS]->(o:Offering {company_name: $company, name: $offering})-[:MONETIZES_VIA]->(r:RevenueModel) RETURN DISTINCT r.name AS revenue_model ORDER BY revenue_model",
        result_shape=REVENUE_MODEL_LIST_SHAPE,
        bindings=REVENUE_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf11_offering_revenue_models_count",
        family_id="QF11",
        question_template="How many revenue models does {company}'s {offering} use?",
        paraphrase_templates=("Count the revenue models for {company}'s {offering}.",),
        cypher="MATCH (:Company {name: $company})-[:OFFERS]->(o:Offering {company_name: $company, name: $offering})-[:MONETIZES_VIA]->(r:RevenueModel) RETURN COUNT(DISTINCT r) AS revenue_model_count",
        result_shape=COUNT_SHAPE("revenue_model_count"),
        bindings=REVENUE_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf11_offering_revenue_model_membership",
        family_id="QF11",
        question_template="Does {company}'s {offering} monetize via {revenue_model}?",
        paraphrase_templates=("Is {revenue_model} one of {company}'s {offering} revenue models?",),
        cypher="MATCH (:Company {name: $company})-[:OFFERS]->(o:Offering {company_name: $company, name: $offering})-[:MONETIZES_VIA]->(r:RevenueModel {name: $revenue_model}) RETURN COUNT(DISTINCT r) > 0 AS is_match",
        result_shape=BOOL_SHAPE("Whether the offering monetizes via the given revenue model"),
        bindings=REVENUE_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf17_channel_segment_reverse_list",
        family_id="QF17",
        question_template="Which segments sell through {channel}?",
        paraphrase_templates=("Show the segments that use {channel}.",),
        cypher="MATCH (s:BusinessSegment)-[:SELLS_THROUGH]->(:Channel {name: $channel}) RETURN DISTINCT s.company_name AS company, s.name AS segment ORDER BY company, segment",
        result_shape=COMPANY_SEGMENT_SHAPE,
        bindings=SEGMENT_CUSTOMER_CHANNEL_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf17_channel_segment_reverse_count",
        family_id="QF17",
        question_template="How many segments sell through {channel}?",
        paraphrase_templates=("Count the segments that use {channel}.",),
        cypher="MATCH (s:BusinessSegment)-[:SELLS_THROUGH]->(:Channel {name: $channel}) RETURN COUNT(DISTINCT s) AS segment_count",
        result_shape=COUNT_SHAPE("segment_count"),
        bindings=SEGMENT_CUSTOMER_CHANNEL_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf17_channel_segment_membership",
        family_id="QF17",
        question_template="Is {company}'s {segment} segment one of the segments that sell through {channel}?",
        paraphrase_templates=("Is {company}'s {segment} segment included in the segment set for {channel}?",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(s:BusinessSegment {company_name: $company, name: $segment})-[:SELLS_THROUGH]->(:Channel {name: $channel}) RETURN COUNT(DISTINCT s) > 0 AS is_match",
        result_shape=BOOL_SHAPE("Whether the segment uses the given channel"),
        bindings=SEGMENT_CUSTOMER_CHANNEL_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf17_company_channel_segment_list",
        family_id="QF17",
        question_template="Which segments at {company} sell through {channel}?",
        paraphrase_templates=("List the {company} segments that use {channel}.",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(s:BusinessSegment {company_name: $company})-[:SELLS_THROUGH]->(:Channel {name: $channel}) RETURN DISTINCT s.name AS segment ORDER BY segment",
        result_shape=SEGMENT_LIST_SHAPE,
        bindings=SEGMENT_CUSTOMER_CHANNEL_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf18_segment_customer_segment_list",
        family_id="QF18",
        question_template="Which segments serve {customer_type}?",
        paraphrase_templates=("Show the segments that serve {customer_type}.",),
        cypher="MATCH (s:BusinessSegment)-[:SERVES]->(:CustomerType {name: $customer_type}) RETURN DISTINCT s.company_name AS company, s.name AS segment ORDER BY company, segment",
        result_shape=COMPANY_SEGMENT_SHAPE,
        bindings=SEGMENT_CUSTOMER_CHANNEL_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf18_segment_customer_segment_count",
        family_id="QF18",
        question_template="How many segments serve {customer_type}?",
        paraphrase_templates=("Count the segments that serve {customer_type}.",),
        cypher="MATCH (s:BusinessSegment)-[:SERVES]->(:CustomerType {name: $customer_type}) RETURN COUNT(DISTINCT s) AS segment_count",
        result_shape=COUNT_SHAPE("segment_count"),
        bindings=SEGMENT_CUSTOMER_CHANNEL_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf18_segment_customer_offering_list",
        family_id="QF18",
        question_template="Which offerings at {company} sit under segments that serve {customer_type}?",
        paraphrase_templates=("List the {company} offerings under segments serving {customer_type}.",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(s:BusinessSegment {company_name: $company})-[:SERVES]->(:CustomerType {name: $customer_type}) MATCH (s)-[:OFFERS]->(o:Offering {company_name: $company}) RETURN DISTINCT o.name AS offering ORDER BY offering",
        result_shape=OFFERING_LIST_SHAPE,
        bindings=SEGMENT_CUSTOMER_CHANNEL_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf18_segment_customer_offering_count",
        family_id="QF18",
        question_template="How many offerings at {company} sit under segments that serve {customer_type}?",
        paraphrase_templates=("Count the {company} offerings under segments serving {customer_type}.",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(s:BusinessSegment {company_name: $company})-[:SERVES]->(:CustomerType {name: $customer_type}) MATCH (s)-[:OFFERS]->(o:Offering {company_name: $company}) RETURN COUNT(DISTINCT o) AS offering_count",
        result_shape=COUNT_SHAPE("offering_count"),
        bindings=SEGMENT_CUSTOMER_CHANNEL_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf21_company_offerings_alpha_list",
        family_id="QF21",
        question_template="List {company}'s offerings alphabetically.",
        paraphrase_templates=("Show {company}'s offerings in alphabetical order.",),
        cypher="MATCH (:Company {name: $company})-[:OFFERS]->(o:Offering {company_name: $company}) RETURN DISTINCT o.name AS offering ORDER BY offering",
        result_shape=OFFERING_LIST_SHAPE,
        bindings=COMPANY_DIRECT_OFFERING_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf21_company_offerings_alpha_limited_list",
        family_id="QF21",
        question_template="List the first {limit} offerings for {company} alphabetically.",
        paraphrase_templates=("Show the first {limit} of {company}'s offerings in alphabetical order.",),
        cypher="MATCH (:Company {name: $company})-[:OFFERS]->(o:Offering {company_name: $company}) RETURN DISTINCT o.name AS offering ORDER BY offering LIMIT $limit",
        result_shape=OFFERING_LIST_SHAPE,
        bindings=[
            {**binding, "binding_id": f"{binding['binding_id']}_limit2", "params": {**binding["params"], "limit": 2}}
            for binding in COMPANY_DIRECT_OFFERING_BINDINGS
        ],
        difficulty="low",
    ),
    _intent(
        intent_id="qf21_segment_offerings_alpha_list",
        family_id="QF21",
        question_template="List {company}'s {segment} segment offerings alphabetically.",
        paraphrase_templates=("Show the offerings in {company}'s {segment} segment in alphabetical order.",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company, name: $segment})-[:OFFERS]->(o:Offering {company_name: $company}) RETURN DISTINCT o.name AS offering ORDER BY offering",
        result_shape=OFFERING_LIST_SHAPE,
        bindings=SEGMENT_DIRECT_OFFERING_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf21_descendant_offerings_ordered_list",
        family_id="QF21",
        question_template="List the offerings under {company}'s {offering} in sorted order.",
        paraphrase_templates=("Show the descendants of {company}'s {offering} in alphabetical order.",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company, name: $segment})-[:OFFERS]->(:Offering {company_name: $company, name: $offering})-[:OFFERS*1..]->(descendant:Offering {company_name: $company}) RETURN DISTINCT descendant.name AS descendant_offering ORDER BY descendant_offering",
        result_shape=DESCENDANT_OFFERING_LIST_SHAPE,
        bindings=HIERARCHY_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf21_places_alpha_list",
        family_id="QF21",
        question_template="List the places where {company} operates alphabetically.",
        paraphrase_templates=("What places does {company} operate in, in alphabetical order?",),
        cypher="MATCH (:Company {name: $company})-[:OPERATES_IN]->(place:Place) RETURN DISTINCT place.name AS place ORDER BY place",
        result_shape=PLACE_LIST_SHAPE,
        bindings=GEO_PARTNER_EU_BINDINGS + GEO_PARTNER_APAC_BINDINGS,
        difficulty="low",
    ),
    _intent(
        intent_id="qf22_partner_existence",
        family_id="QF22",
        question_template="Is {partner} explicitly recorded as a partner of {company}?",
        paraphrase_templates=("Is {partner} listed in the KG as a partner of {company}?",),
        cypher="MATCH (:Company {name: $company})-[:PARTNERS_WITH]->(partner:Company {name: $partner}) RETURN COUNT(DISTINCT partner) > 0 AS is_match",
        result_shape=BOOL_SHAPE("Whether the company partners with the given company"),
        bindings=GEO_PARTNER_EU_BINDINGS + GEO_PARTNER_APAC_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf22_segment_customer_existence",
        family_id="QF22",
        question_template="Is {customer_type} explicitly recorded as a customer type for {company}'s {segment} segment?",
        paraphrase_templates=("Is {customer_type} listed in the KG for {company}'s {segment} segment?",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company, name: $segment})-[:SERVES]->(c:CustomerType {name: $customer_type}) RETURN COUNT(DISTINCT c) > 0 AS is_match",
        result_shape=BOOL_SHAPE("Whether the segment serves the given customer type"),
        bindings=SEGMENT_CUSTOMER_CHANNEL_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf22_offering_revenue_existence",
        family_id="QF22",
        question_template="Is {revenue_model} explicitly recorded as a revenue model for {company}'s {offering}?",
        paraphrase_templates=("Is {revenue_model} listed in the KG for {company}'s {offering}?",),
        cypher="MATCH (:Company {name: $company})-[:OFFERS]->(o:Offering {company_name: $company, name: $offering})-[:MONETIZES_VIA]->(r:RevenueModel {name: $revenue_model}) RETURN COUNT(DISTINCT r) > 0 AS is_match",
        result_shape=BOOL_SHAPE("Whether the offering monetizes via the given revenue model"),
        bindings=REVENUE_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf22_company_place_existence",
        family_id="QF22",
        question_template="Is {place} explicitly recorded as an operating place for {company}?",
        paraphrase_templates=("Is {place} listed in the KG as an operating place for {company}?",),
        cypher="MATCH (:Company {name: $company})-[:OPERATES_IN]->(place_node:Place {name: $place}) RETURN COUNT(DISTINCT place_node) > 0 AS is_match",
        result_shape=BOOL_SHAPE("Whether the company operates in the given place"),
        bindings=GEO_PARTNER_EU_BINDINGS + GEO_PARTNER_APAC_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf23_offering_segment_anchor",
        family_id="QF23",
        question_template="Which segment is {company}'s {offering} under?",
        paraphrase_templates=("What segment anchors {company}'s {offering}?",),
        cypher="MATCH (s:BusinessSegment {company_name: $company})-[:OFFERS*1..]->(o:Offering {company_name: $company, name: $offering}) RETURN DISTINCT s.name AS segment",
        result_shape=SEGMENT_LIST_SHAPE,
        bindings=REVERSE_HIERARCHY_BINDINGS,
        difficulty="high",
    ),
    _intent(
        intent_id="qf23_offering_root_anchor",
        family_id="QF23",
        question_template="What parent offering anchors {company}'s {offering}?",
        paraphrase_templates=("Which root offering anchors {company}'s {offering}?",),
        cypher="MATCH p=(root:Offering {company_name: $company})-[:OFFERS*1..]->(o:Offering {company_name: $company, name: $offering}) WITH root, length(p) AS depth ORDER BY depth DESC LIMIT 1 RETURN root.name AS root_offering",
        result_shape=(_col("root_offering", "string"),),
        bindings=REVERSE_HIERARCHY_BINDINGS,
        difficulty="high",
    ),
    _intent(
        intent_id="qf23_offering_anchor_path",
        family_id="QF23",
        question_template="Show the segment-to-offering path for {company}'s {offering}.",
        paraphrase_templates=("Show the full breadcrumb from the segment to {company}'s {offering}.",),
        cypher="MATCH p=(s:BusinessSegment {company_name: $company})-[:OFFERS]->(root:Offering {company_name: $company})-[:OFFERS*0..]->(o:Offering {company_name: $company, name: $offering}) WITH s.name AS segment, nodes(p)[1..] AS offerings UNWIND range(0, size(offerings) - 1) AS depth RETURN segment, offerings[depth].name AS offering, depth ORDER BY depth",
        result_shape=(
            _col("segment", "string"),
            _col("offering", "string"),
            _col("depth", "integer"),
        ),
        bindings=REVERSE_HIERARCHY_BINDINGS,
        difficulty="high",
    ),
    _intent(
        intent_id="qf24_offering_parent_list",
        family_id="QF24",
        question_template="What parent offering is {company}'s {offering} under?",
        paraphrase_templates=("Which offering directly contains {company}'s {offering}?",),
        cypher="MATCH (:Offering {company_name: $company, name: $offering})<-[:OFFERS]-(parent:Offering {company_name: $company}) RETURN DISTINCT parent.name AS parent_offering",
        result_shape=(_col("parent_offering", "string"),),
        bindings=REVERSE_HIERARCHY_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf24_offering_ancestor_list",
        family_id="QF24",
        question_template="Which offerings are above {company}'s {offering}?",
        paraphrase_templates=("List the ancestors of {company}'s {offering}.",),
        cypher="MATCH p=(ancestor:Offering {company_name: $company})-[:OFFERS*1..]->(o:Offering {company_name: $company, name: $offering}) WITH ancestor.name AS ancestor_offering, length(p) AS depth RETURN ancestor_offering ORDER BY depth",
        result_shape=(_col("ancestor_offering", "string"),),
        bindings=REVERSE_HIERARCHY_BINDINGS,
        difficulty="high",
    ),
    _intent(
        intent_id="qf24_offering_root_ancestor",
        family_id="QF24",
        question_template="What is the topmost offering above {company}'s {offering}?",
        paraphrase_templates=("What is the root ancestor offering for {company}'s {offering}?",),
        cypher="MATCH p=(root:Offering {company_name: $company})-[:OFFERS*1..]->(o:Offering {company_name: $company, name: $offering}) WITH root.name AS root_offering, length(p) AS depth ORDER BY depth DESC LIMIT 1 RETURN root_offering",
        result_shape=(_col("root_offering", "string"),),
        bindings=REVERSE_HIERARCHY_BINDINGS,
        difficulty="high",
    ),
    _intent(
        intent_id="qf29_company_geography_match_list",
        family_id="QF29",
        question_template="Which companies operate in {place}?",
        paraphrase_templates=("Show the companies that match {place} in the geography hierarchy.",),
        cypher=(
            "MATCH (company:Company)-[:OPERATES_IN]->(place:Place) "
            "WITH company, place.name AS matched_place, "
            "CASE "
            "WHEN place.name = $place THEN 0 "
            "WHEN $place IN coalesce(place.includes_places, []) THEN 1 "
            "WHEN $place IN coalesce(place.within_places, []) THEN 2 "
            "ELSE NULL "
            "END AS match_rank "
            "WHERE match_rank IS NOT NULL "
            "WITH company, matched_place, match_rank, "
            "CASE match_rank WHEN 0 THEN 'exact' WHEN 1 THEN 'narrower_place' ELSE 'broader_region' END AS geography_match "
            "RETURN company.name AS company, matched_place, geography_match "
            "ORDER BY match_rank, company, matched_place"
        ),
        result_shape=GEOGRAPHY_MATCH_SHAPE,
        bindings=QF29_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf29_company_geography_match_count",
        family_id="QF29",
        question_template="How many companies operate in {place}?",
        paraphrase_templates=("Count the companies that match {place} geographically.",),
        cypher=(
            "MATCH (company:Company)-[:OPERATES_IN]->(place:Place) "
            "WITH company, place.name AS matched_place, "
            "CASE "
            "WHEN place.name = $place THEN 0 "
            "WHEN $place IN coalesce(place.includes_places, []) THEN 1 "
            "WHEN $place IN coalesce(place.within_places, []) THEN 2 "
            "ELSE NULL "
            "END AS match_rank "
            "WHERE match_rank IS NOT NULL "
            "RETURN COUNT(DISTINCT company) AS company_count"
        ),
        result_shape=COUNT_SHAPE("company_count"),
        bindings=QF29_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf29_company_geography_match_boolean",
        family_id="QF29",
        question_template="Does {company}'s operating footprint match {place} anywhere in the geography hierarchy?",
        paraphrase_templates=("Is {place} matched anywhere in {company}'s geography hierarchy footprint?",),
        cypher=(
            "MATCH (:Company {name: $company})-[:OPERATES_IN]->(place:Place) "
            "WITH place.name AS matched_place, "
            "CASE "
            "WHEN place.name = $place THEN 0 "
            "WHEN $place IN coalesce(place.includes_places, []) THEN 1 "
            "WHEN $place IN coalesce(place.within_places, []) THEN 2 "
            "ELSE NULL "
            "END AS match_rank "
            "WHERE match_rank IS NOT NULL "
            "RETURN COUNT(*) > 0 AS is_match"
        ),
        result_shape=BOOL_SHAPE("Whether the company matches the geography hierarchy for the given place"),
        bindings=QF29_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf29_company_geography_match_class",
        family_id="QF29",
        question_template="How does {company} match {place}?",
        paraphrase_templates=("What geography match class does {company} have for {place}?",),
        cypher=(
            "MATCH (:Company {name: $company})-[:OPERATES_IN]->(place:Place) "
            "WITH place.name AS matched_place, "
            "CASE "
            "WHEN place.name = $place THEN 0 "
            "WHEN $place IN coalesce(place.includes_places, []) THEN 1 "
            "WHEN $place IN coalesce(place.within_places, []) THEN 2 "
            "ELSE NULL "
            "END AS match_rank "
            "WHERE match_rank IS NOT NULL "
            "WITH $company AS company, matched_place, match_rank "
            "RETURN company, matched_place, "
            "CASE match_rank WHEN 0 THEN 'exact' WHEN 1 THEN 'narrower_place' ELSE 'broader_region' END AS geography_match "
            "ORDER BY match_rank, matched_place"
        ),
        result_shape=(
            _col("company", "string"),
            _col("matched_place", "string"),
            _col("geography_match", "string"),
        ),
        bindings=QF29_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf30_offering_collision_list",
        family_id="QF30",
        question_template="Which companies have an offering called {offering_name}?",
        paraphrase_templates=("List the companies that expose an offering named {offering_name}.",),
        cypher="MATCH (o:Offering {name: $offering_name}) RETURN DISTINCT o.company_name AS company_name, o.name AS entity_name ORDER BY company_name",
        result_shape=COLLISION_SHAPE,
        bindings=QF30_OFFERING_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf30_offering_collision_count",
        family_id="QF30",
        question_template="How many companies have an offering called {offering_name}?",
        paraphrase_templates=("Count the companies that expose an offering named {offering_name}.",),
        cypher="MATCH (o:Offering {name: $offering_name}) RETURN COUNT(DISTINCT o.company_name) AS company_count",
        result_shape=COUNT_SHAPE("company_count"),
        bindings=QF30_OFFERING_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf30_segment_collision_list",
        family_id="QF30",
        question_template="Which companies have a segment called {segment_name}?",
        paraphrase_templates=("List the companies that expose a segment named {segment_name}.",),
        cypher="MATCH (s:BusinessSegment {name: $segment_name}) RETURN DISTINCT s.company_name AS company_name, s.name AS entity_name ORDER BY company_name",
        result_shape=COLLISION_SHAPE,
        bindings=QF30_SEGMENT_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf30_segment_collision_count",
        family_id="QF30",
        question_template="How many companies have a segment called {segment_name}?",
        paraphrase_templates=("Count the companies that expose a segment named {segment_name}.",),
        cypher="MATCH (s:BusinessSegment {name: $segment_name}) RETURN COUNT(DISTINCT s.company_name) AS company_count",
        result_shape=COUNT_SHAPE("company_count"),
        bindings=QF30_SEGMENT_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf30_company_scoped_offering_membership",
        family_id="QF30",
        question_template="Does {company} have an offering called {offering_name}?",
        paraphrase_templates=("Is {offering_name} a company-scoped offering for {company}?",),
        cypher="MATCH (:Company {name: $company})-[:OFFERS]->(o:Offering {company_name: $company, name: $offering_name}) RETURN COUNT(DISTINCT o) > 0 AS is_match",
        result_shape=BOOL_SHAPE("Whether the company has the given company-scoped offering"),
        bindings=QF30_SCOPED_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf30_company_scoped_segment_membership",
        family_id="QF30",
        question_template="Does {company} have a segment called {segment_name}?",
        paraphrase_templates=("Is {segment_name} a company-scoped segment for {company}?",),
        cypher="MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(s:BusinessSegment {company_name: $company, name: $segment_name}) RETURN COUNT(DISTINCT s) > 0 AS is_match",
        result_shape=BOOL_SHAPE("Whether the company has the given company-scoped segment"),
        bindings=QF30_SCOPED_BINDINGS,
        difficulty="medium",
    ),
    _intent(
        intent_id="qf30_ambiguous_surface_request",
        family_id="QF30",
        question_template="What does the surface name {surface_name} offer?",
        paraphrase_templates=("How does the surface name {surface_name} make money?",),
        cypher=None,
        result_shape=None,
        bindings=QF30_REFUSAL_BINDINGS,
        difficulty="medium",
        answerable=False,
        refusal_reason="ambiguous_scope",
    ),
]


def build_spec() -> DatasetSpec:
    return DatasetSpec(fixtures=fixtures, source_examples=_expand_intents(intent_specs))
