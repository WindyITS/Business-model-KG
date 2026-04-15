from __future__ import annotations

from collections import defaultdict

from .models import (
    DatasetSpec,
    FixtureEdgeSpec,
    FixtureNodeSpec,
    FixtureSpec,
    ResultColumnSpec,
    SourceExampleSpec,
)

INTENT_SPLIT_OVERRIDES = {
    "qf31_place_company_revenue_model_list": "train",
    "qf31_place_company_revenue_model_count": "dev",
    "qf31_place_segment_channel_list": "test",
}


def _split_for_intent(intent_id: str) -> str:
    if intent_id in INTENT_SPLIT_OVERRIDES:
        return INTENT_SPLIT_OVERRIDES[intent_id]
    bucket = sum(ord(char) for char in intent_id) % 10
    if bucket in {0, 1}:
        return "test"
    if bucket in {2, 3}:
        return "dev"
    return "train"


def _lowercase_first(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:]


def _paraphrases(question: str) -> tuple[str, ...]:
    lowered = _lowercase_first(question)
    return (
        f"Need the graph answer here: {lowered}",
        f"Can you just pull this from Neo4j: {lowered}",
        f"Trying to sanity-check this one: {lowered}",
        f"Quick gut check from the KG: {lowered}",
        f"Need this out of the graph asap: {lowered}",
        f"For a deck I'm building: {lowered}",
        f"Can you pull the exact graph answer for this: {lowered}",
        f"I'm trying to untangle this ask: {lowered}",
        f"One more thing I need from the graph: {lowered}",
        f"Need the Cypher-ready answer for this: {lowered}",
        f"Not looking for prose, just the graph result: {lowered}",
        f"Can you sanity-check this against the KG: {lowered}",
        f"I need a clean graph lookup for this: {lowered}",
        f"What does the graph say here: {lowered}",
        f"Need the exact answer from the KG, not a guess: {lowered}",
    )


def _col(column: str, type_name: str, description: str | None = None) -> ResultColumnSpec:
    return ResultColumnSpec(column=column, type=type_name, description=description)


def _node(node_id: str, label: str, name: str, **properties: object) -> FixtureNodeSpec:
    return FixtureNodeSpec(node_id=node_id, label=label, name=name, properties=properties)


def _place(
    node_id: str,
    name: str,
    *,
    within_places: tuple[str, ...] = (),
    includes_places: tuple[str, ...] = (),
) -> FixtureNodeSpec:
    properties: dict[str, object] = {}
    if within_places:
        properties["within_places"] = list(within_places)
    if includes_places:
        properties["includes_places"] = list(includes_places)
    return _node(node_id, "Place", name, **properties)


def _edge(source: str, rel_type: str, target: str) -> FixtureEdgeSpec:
    return FixtureEdgeSpec(source=source, type=rel_type, target=target)


def _example(
    *,
    example_id: str,
    intent_id: str,
    family_id: str,
    fixture_id: str,
    graph_id: str,
    binding_id: str,
    question: str,
    gold_cypher: str,
    params: dict[str, object],
    result_shape: list[ResultColumnSpec],
    difficulty: str,
) -> SourceExampleSpec:
    return SourceExampleSpec(
        example_id=example_id,
        intent_id=intent_id,
        family_id=family_id,
        fixture_id=fixture_id,
        graph_id=graph_id,
        binding_id=binding_id,
        question_canonical=question,
        gold_cypher=gold_cypher,
        params=params,
        answerable=True,
        refusal_reason=None,
        result_shape=result_shape,
        difficulty=difficulty,
        split=_split_for_intent(intent_id),
        paraphrases=_paraphrases(question),
    )


def _build_fx09_segment_revenue_rollup_tree() -> tuple[FixtureSpec, list[SourceExampleSpec]]:
    fixture_id = "FX09_segment_revenue_rollup_tree"
    graph_id = "fx09_segment_revenue_rollup_tree_v2"
    segment_specs = {
        ("Northstar Systems", "Industrial AI"): {
            "root": ("Atlas Platform", ("subscription", "consumption-based")),
            "children": {
                "Atlas Vision": ("Atlas Vision Edge", "Atlas Vision Signals"),
                "Atlas Predict": ("Atlas Predict Live",),
                "Atlas Civic": (),
            },
            "direct_offerings": (("Atlas Services", ("service fees",)),),
        },
        ("Silverline Harbor", "Operations Cloud"): {
            "root": ("Atlas Platform", ("service fees",)),
            "children": {
                "Atlas Guard": ("Atlas Guard Mobile", "Atlas Guard Air"),
                "Atlas Flow": ("Atlas Flow Edge",),
                "Atlas Ledger": (),
            },
            "direct_offerings": (("Harbor Control", ("licensing",)), ("Harbor Sync", ("platform fees",))),
        },
        ("Rivergrid Systems", "Developer Infrastructure"): {
            "root": ("Beacon Suite", ("subscription",)),
            "children": {
                "Beacon Runtime": ("Beacon Runtime Edge",),
                "Beacon Deploy": (),
                "Beacon Trace": (),
            },
            "direct_offerings": (("Beacon Meter", ("consumption-based",)),),
        },
        ("Asteron Analytics", "Industrial AI"): {
            "root": ("Atlas Platform", ("usage fees",)),
            "children": {
                "Atlas Vision": ("Atlas Vision Cloud",),
                "Atlas Forecast": ("Atlas Forecast Live",),
                "Atlas Ops": (),
            },
            "direct_offerings": (("Atlas Insights", ("usage fees",)), ("Atlas Signals", ("subscription",))),
        },
        ("Harborline Systems", "Operations Cloud"): {
            "root": ("Beacon Suite", ("service fees",)),
            "children": {
                "Beacon Guard": ("Beacon Guard Mobile",),
                "Beacon Flow": ("Beacon Flow Studio",),
                "Beacon Audit": (),
            },
            "direct_offerings": (("Harbor Control", ("licensing",)), ("Harbor Ledger", ("transaction fees",))),
        },
    }

    nodes: list[FixtureNodeSpec] = []
    edges: list[FixtureEdgeSpec] = []
    revenue_ids: dict[str, str] = {}

    for company_index, ((company, segment), spec) in enumerate(segment_specs.items(), start=1):
        company_id = f"company_{company_index}"
        segment_id = f"{company_id}_segment"
        root_name, root_revenue = spec["root"]
        root_id = f"{company_id}_root"
        nodes.extend(
            [
                _node(company_id, "Company", company),
                _node(segment_id, "BusinessSegment", segment, company_name=company),
                _node(root_id, "Offering", root_name, company_name=company),
            ]
        )
        edges.extend(
            [
                _edge(company_id, "HAS_SEGMENT", segment_id),
                _edge(segment_id, "OFFERS", root_id),
            ]
        )
        for revenue_model in root_revenue:
            revenue_id = revenue_ids.setdefault(revenue_model, f"revenue_{len(revenue_ids) + 1}")
            if not any(node.node_id == revenue_id for node in nodes):
                nodes.append(_node(revenue_id, "RevenueModel", revenue_model))
            edges.append(_edge(root_id, "MONETIZES_VIA", revenue_id))

        child_ids: dict[str, str] = {}
        for child_index, (child, grandchildren) in enumerate(spec["children"].items(), start=1):
            child_id = f"{company_id}_child_{child_index}"
            child_ids[child] = child_id
            nodes.append(_node(child_id, "Offering", child, company_name=company))
            edges.append(_edge(root_id, "OFFERS", child_id))
            for grandchild_index, grandchild in enumerate(grandchildren, start=1):
                grandchild_id = f"{child_id}_grandchild_{grandchild_index}"
                nodes.append(_node(grandchild_id, "Offering", grandchild, company_name=company))
                edges.append(_edge(child_id, "OFFERS", grandchild_id))

        for direct_index, (direct_name, direct_revenue) in enumerate(spec["direct_offerings"], start=1):
            direct_id = f"{company_id}_direct_{direct_index}"
            nodes.append(_node(direct_id, "Offering", direct_name, company_name=company))
            edges.append(_edge(segment_id, "OFFERS", direct_id))
            for revenue_model in direct_revenue:
                revenue_id = revenue_ids.setdefault(revenue_model, f"revenue_{len(revenue_ids) + 1}")
                if not any(node.node_id == revenue_id for node in nodes):
                    nodes.append(_node(revenue_id, "RevenueModel", revenue_model))
                edges.append(_edge(direct_id, "MONETIZES_VIA", revenue_id))

    fixture = FixtureSpec(
        fixture_id=fixture_id,
        graph_id=graph_id,
        graph_purpose="Segment-level monetization rollup tree with repeated root names across companies and direct monetized siblings.",
        covered_families=("QF12", "QF16", "QF20"),
        nodes=nodes,
        edges=edges,
        invariants_satisfied=(
            "MONETIZES_VIA attaches only to root offerings or direct segment-owned offerings.",
            "Repeated root names such as Atlas Platform are disambiguated by company_name.",
            "Every segment has at least one monetized path and one non-monetized descendant for contrast.",
        ),
        authoring_notes=(
            "This is the core segment-level rollup fixture for revenue traversal and filtered monetization queries.",
            "Northstar has only a monetized root, while Silverline and Rivergrid also have direct monetized siblings.",
        ),
    )

    rollup_list_cypher = (
        "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->"
        "(s:BusinessSegment {name: $segment, company_name: $company})-[:OFFERS]->"
        "(root:Offering {company_name: $company}) "
        "MATCH (root)-[:OFFERS*0..]->(o:Offering {company_name: $company}) "
        "MATCH (o)-[:MONETIZES_VIA]->(r:RevenueModel) "
        "RETURN DISTINCT r.name AS revenue_model ORDER BY revenue_model"
    )
    rollup_count_cypher = (
        "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->"
        "(s:BusinessSegment {name: $segment, company_name: $company})-[:OFFERS]->"
        "(root:Offering {company_name: $company}) "
        "MATCH (root)-[:OFFERS*0..]->(o:Offering {company_name: $company}) "
        "MATCH (o)-[:MONETIZES_VIA]->(r:RevenueModel) "
        "RETURN COUNT(DISTINCT r) AS revenue_model_count"
    )
    rollup_membership_cypher = (
        "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->"
        "(s:BusinessSegment {name: $segment, company_name: $company})-[:OFFERS]->"
        "(root:Offering {company_name: $company}) "
        "MATCH (root)-[:OFFERS*0..]->(o:Offering {company_name: $company}) "
        "MATCH (o)-[:MONETIZES_VIA]->(r:RevenueModel {name: $revenue_model}) "
        "RETURN COUNT(DISTINCT r) > 0 AS is_match"
    )
    contributor_cypher = (
        "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->"
        "(s:BusinessSegment {name: $segment, company_name: $company})-[:OFFERS]->"
        "(root:Offering {company_name: $company}) "
        "MATCH (root)-[:OFFERS*0..]->(o:Offering {company_name: $company}) "
        "MATCH (o)-[:MONETIZES_VIA]->(r:RevenueModel {name: $revenue_model}) "
        "RETURN DISTINCT o.name AS offering, r.name AS revenue_model ORDER BY offering, revenue_model"
    )
    monetized_offerings_list_cypher = (
        "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->"
        "(s:BusinessSegment {name: $segment, company_name: $company})-[:OFFERS]->"
        "(root:Offering {company_name: $company}) "
        "MATCH (root)-[:OFFERS*0..]->(o:Offering {company_name: $company}) "
        "MATCH (o)-[:MONETIZES_VIA]->(r:RevenueModel {name: $revenue_model}) "
        "RETURN DISTINCT o.name AS offering ORDER BY offering"
    )
    monetized_offerings_count_cypher = (
        "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->"
        "(s:BusinessSegment {name: $segment, company_name: $company})-[:OFFERS]->"
        "(root:Offering {company_name: $company}) "
        "MATCH (root)-[:OFFERS*0..]->(o:Offering {company_name: $company}) "
        "MATCH (o)-[:MONETIZES_VIA]->(r:RevenueModel {name: $revenue_model}) "
        "RETURN COUNT(DISTINCT o) AS offering_count"
    )
    monetized_pairs_cypher = (
        "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->"
        "(s:BusinessSegment {name: $segment, company_name: $company})-[:OFFERS]->"
        "(root:Offering {company_name: $company}) "
        "MATCH (root)-[:OFFERS*0..]->(o:Offering {company_name: $company}) "
        "MATCH (o)-[:MONETIZES_VIA]->(r:RevenueModel) "
        "RETURN DISTINCT o.name AS offering, r.name AS revenue_model ORDER BY offering, revenue_model"
    )
    monetized_membership_cypher = (
        "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->"
        "(s:BusinessSegment {name: $segment, company_name: $company})-[:OFFERS]->"
        "(root:Offering {company_name: $company}) "
        "MATCH (root)-[:OFFERS*0..]->(o:Offering {name: $offering, company_name: $company}) "
        "MATCH (o)-[:MONETIZES_VIA]->(r:RevenueModel {name: $revenue_model}) "
        "RETURN COUNT(DISTINCT o) > 0 AS is_match"
    )
    descendant_count_cypher = (
        "MATCH (:Offering {name: $offering, company_name: $company})-[:OFFERS*1..]->"
        "(descendant:Offering {company_name: $company}) "
        "RETURN COUNT(DISTINCT descendant) AS descendant_count"
    )

    examples: list[SourceExampleSpec] = []
    qf12_bindings = (
        ("Northstar Systems", "Industrial AI", "subscription"),
        ("Silverline Harbor", "Operations Cloud", "licensing"),
        ("Rivergrid Systems", "Developer Infrastructure", "consumption-based"),
        ("Asteron Analytics", "Industrial AI", "usage fees"),
        ("Harborline Systems", "Operations Cloud", "service fees"),
    )
    for company, segment, revenue_model in qf12_bindings:
        slug = company.lower().replace(" ", "_")
        segment_slug = segment.lower().replace(" ", "_")
        examples.extend(
            [
                _example(
                    example_id=f"qf12_segment_revenue_rollup_list__{slug}_{segment_slug}",
                    intent_id="qf12_segment_revenue_rollup_list",
                    family_id="QF12",
                    fixture_id=fixture_id,
                    graph_id=graph_id,
                    binding_id=f"{slug}_{segment_slug}_revenue_list",
                    question=f"How does {company}'s {segment} segment make money?",
                    gold_cypher=rollup_list_cypher,
                    params={"company": company, "segment": segment},
                    result_shape=[_col("revenue_model", "string")],
                    difficulty="high",
                ),
                _example(
                    example_id=f"qf12_segment_revenue_rollup_count__{slug}_{segment_slug}",
                    intent_id="qf12_segment_revenue_rollup_count",
                    family_id="QF12",
                    fixture_id=fixture_id,
                    graph_id=graph_id,
                    binding_id=f"{slug}_{segment_slug}_revenue_count",
                    question=f"How many distinct revenue models does {company}'s {segment} segment use?",
                    gold_cypher=rollup_count_cypher,
                    params={"company": company, "segment": segment},
                    result_shape=[_col("revenue_model_count", "integer")],
                    difficulty="high",
                ),
                _example(
                    example_id=f"qf12_segment_revenue_rollup_membership__{slug}_{segment_slug}_{revenue_model.replace(' ', '_')}",
                    intent_id="qf12_segment_revenue_rollup_membership",
                    family_id="QF12",
                    fixture_id=fixture_id,
                    graph_id=graph_id,
                    binding_id=f"{slug}_{segment_slug}_{revenue_model.replace(' ', '_')}",
                    question=f"Does {company}'s {segment} segment monetize via {revenue_model}?",
                    gold_cypher=rollup_membership_cypher,
                    params={"company": company, "segment": segment, "revenue_model": revenue_model},
                    result_shape=[_col("is_match", "boolean")],
                    difficulty="high",
                ),
                _example(
                    example_id=f"qf12_segment_revenue_contributors_list__{slug}_{segment_slug}_{revenue_model.replace(' ', '_')}",
                    intent_id="qf12_segment_revenue_contributors_list",
                    family_id="QF12",
                    fixture_id=fixture_id,
                    graph_id=graph_id,
                    binding_id=f"{slug}_{segment_slug}_{revenue_model.replace(' ', '_')}_contributors",
                    question=f"Which offerings under {company}'s {segment} segment monetize via {revenue_model}?",
                    gold_cypher=contributor_cypher,
                    params={"company": company, "segment": segment, "revenue_model": revenue_model},
                    result_shape=[_col("offering", "string"), _col("revenue_model", "string")],
                    difficulty="high",
                ),
            ]
        )

    qf16_bindings = (
        ("Northstar Systems", "Industrial AI", "Atlas Platform", "subscription"),
        ("Silverline Harbor", "Operations Cloud", "Harbor Control", "licensing"),
        ("Rivergrid Systems", "Developer Infrastructure", "Beacon Meter", "consumption-based"),
        ("Asteron Analytics", "Industrial AI", "Atlas Insights", "usage fees"),
        ("Harborline Systems", "Operations Cloud", "Beacon Suite", "service fees"),
    )
    for company, segment, offering, revenue_model in qf16_bindings:
        slug = company.lower().replace(" ", "_")
        examples.extend(
            [
                _example(
                    example_id=f"qf16_segment_offering_monetization_list__{slug}_{offering.lower().replace(' ', '_')}",
                    intent_id="qf16_segment_offering_monetization_list",
                    family_id="QF16",
                    fixture_id=fixture_id,
                    graph_id=graph_id,
                    binding_id=f"{slug}_{offering.lower().replace(' ', '_')}_list",
                    question=f"Which offerings in {company}'s {segment} segment monetize via {revenue_model}?",
                    gold_cypher=monetized_offerings_list_cypher,
                    params={"company": company, "segment": segment, "revenue_model": revenue_model},
                    result_shape=[_col("offering", "string")],
                    difficulty="high",
                ),
                _example(
                    example_id=f"qf16_segment_offering_monetization_count__{slug}_{offering.lower().replace(' ', '_')}",
                    intent_id="qf16_segment_offering_monetization_count",
                    family_id="QF16",
                    fixture_id=fixture_id,
                    graph_id=graph_id,
                    binding_id=f"{slug}_{offering.lower().replace(' ', '_')}_count",
                    question=f"How many offerings in {company}'s {segment} segment monetize via {revenue_model}?",
                    gold_cypher=monetized_offerings_count_cypher,
                    params={"company": company, "segment": segment, "revenue_model": revenue_model},
                    result_shape=[_col("offering_count", "integer")],
                    difficulty="high",
                ),
                _example(
                    example_id=f"qf16_segment_offering_monetization_membership__{slug}_{offering.lower().replace(' ', '_')}",
                    intent_id="qf16_segment_offering_monetization_membership",
                    family_id="QF16",
                    fixture_id=fixture_id,
                    graph_id=graph_id,
                    binding_id=f"{slug}_{offering.lower().replace(' ', '_')}_membership",
                    question=f"Does {company}'s {segment} segment include {offering} monetizing via {revenue_model}?",
                    gold_cypher=monetized_membership_cypher,
                    params={
                        "company": company,
                        "segment": segment,
                        "offering": offering,
                        "revenue_model": revenue_model,
                    },
                    result_shape=[_col("is_match", "boolean")],
                    difficulty="high",
                ),
            ]
        )
    for company, segment in (
        ("Northstar Systems", "Industrial AI"),
        ("Silverline Harbor", "Operations Cloud"),
        ("Rivergrid Systems", "Developer Infrastructure"),
        ("Asteron Analytics", "Industrial AI"),
        ("Harborline Systems", "Operations Cloud"),
    ):
        slug = company.lower().replace(" ", "_")
        examples.append(
            _example(
                example_id=f"qf16_segment_offering_monetization_pairs__{slug}_{segment.lower().replace(' ', '_')}",
                intent_id="qf16_segment_offering_monetization_pairs",
                family_id="QF16",
                fixture_id=fixture_id,
                graph_id=graph_id,
                binding_id=f"{slug}_{segment.lower().replace(' ', '_')}_pairs",
                question=f"Which offerings under {company}'s {segment} segment use which revenue models?",
                gold_cypher=monetized_pairs_cypher,
                params={"company": company, "segment": segment},
                result_shape=[_col("offering", "string"), _col("revenue_model", "string")],
                difficulty="high",
            )
        )

    for company, offering in (
        ("Northstar Systems", "Atlas Platform"),
        ("Northstar Systems", "Atlas Vision"),
        ("Silverline Harbor", "Atlas Platform"),
        ("Silverline Harbor", "Atlas Guard"),
        ("Rivergrid Systems", "Beacon Suite"),
        ("Asteron Analytics", "Atlas Platform"),
        ("Harborline Systems", "Beacon Suite"),
    ):
        slug = company.lower().replace(" ", "_")
        examples.append(
            _example(
                example_id=f"qf20_rollup_descendant_count__{slug}_{offering.lower().replace(' ', '_')}",
                intent_id="qf20_rollup_descendant_count",
                family_id="QF20",
                fixture_id=fixture_id,
                graph_id=graph_id,
                binding_id=f"{slug}_{offering.lower().replace(' ', '_')}_descendants",
                question=f"How many descendant offerings sit under {company}'s {offering} family?",
                gold_cypher=descendant_count_cypher,
                params={"company": company, "offering": offering},
                result_shape=[_col("descendant_count", "integer")],
                difficulty="medium",
            )
        )
    for company, segment, revenue_model in (
        ("Northstar Systems", "Industrial AI", "subscription"),
        ("Silverline Harbor", "Operations Cloud", "licensing"),
        ("Rivergrid Systems", "Developer Infrastructure", "consumption-based"),
        ("Asteron Analytics", "Industrial AI", "usage fees"),
        ("Harborline Systems", "Operations Cloud", "service fees"),
    ):
        slug = company.lower().replace(" ", "_")
        examples.append(
            _example(
                example_id=f"qf20_filtered_match_count__{slug}_{revenue_model.replace(' ', '_')}",
                intent_id="qf20_filtered_match_count",
                family_id="QF20",
                fixture_id=fixture_id,
                graph_id=graph_id,
                binding_id=f"{slug}_{revenue_model.replace(' ', '_')}_filtered_count",
                question=f"Count the offerings in {company}'s {segment} segment that monetize via {revenue_model}.",
                gold_cypher=monetized_offerings_count_cypher,
                params={"company": company, "segment": segment, "revenue_model": revenue_model},
                result_shape=[_col("offering_count", "integer")],
                difficulty="medium",
            )
        )

    return fixture, examples


def _build_fx10_company_rollup_profile() -> tuple[FixtureSpec, list[SourceExampleSpec]]:
    fixture_id = "FX10_company_rollup_profile"
    graph_id = "fx10_company_rollup_profile_v2"
    place_specs = {
        "North America": {"includes_places": ("United States", "Canada", "Mexico")},
        "United States": {"within_places": ("North America",), "includes_places": ("California", "New York", "Texas")},
        "Canada": {"within_places": ("North America",), "includes_places": ("Ontario", "Quebec")},
        "Mexico": {"within_places": ("North America",)},
        "Europe": {"includes_places": ("Germany", "Italy", "France", "Spain")},
        "Germany": {"within_places": ("Europe",), "includes_places": ("Bavaria", "Berlin", "Hesse")},
        "Italy": {"within_places": ("Europe",), "includes_places": ("Lombardy", "Lazio", "Piedmont")},
        "France": {"within_places": ("Europe",), "includes_places": ("Ile-de-France", "Occitanie")},
        "Spain": {"within_places": ("Europe",)},
        "APAC": {"includes_places": ("Japan", "Australia", "Singapore", "South Korea")},
        "Japan": {"within_places": ("APAC",), "includes_places": ("Kanto", "Kansai")},
        "Australia": {"within_places": ("APAC",)},
        "Singapore": {"within_places": ("APAC",)},
    }
    company_specs = {
        "Meridian Nexus": {
            "places": ("United States", "Europe"),
            "segments": {
                "Public Sector": {
                    "customers": ("government agencies", "IT professionals"),
                    "channels": ("direct sales", "system integrators"),
                    "offerings": {
                        "Civic Relay": ("licensing",),
                    },
                },
                "Developer Cloud": {
                    "customers": ("developers", "IT professionals"),
                    "channels": ("marketplaces", "direct sales"),
                    "offerings": {
                        "Builder Studio": ("subscription", "service fees"),
                    },
                },
                "Commerce Systems": {
                    "customers": ("large enterprises", "financial services firms"),
                    "channels": ("marketplaces", "direct sales"),
                    "offerings": {
                        "Merchant Grid": ("transaction fees",),
                    },
                },
                "Platform Services": {
                    "customers": ("developers", "IT professionals"),
                    "channels": ("direct sales", "resellers"),
                    "offerings": {
                        "Signal Portal": ("subscription",),
                    },
                },
            },
        },
        "Orion Fabric": {
            "places": ("Germany", "APAC"),
            "segments": {
                "Industrial Platform": {
                    "customers": ("large enterprises", "manufacturers"),
                    "channels": ("direct sales", "system integrators"),
                    "offerings": {
                        "Control Tower": ("subscription", "service fees"),
                    },
                },
                "Public Sector": {
                    "customers": ("government agencies",),
                    "channels": ("system integrators",),
                    "offerings": {
                        "Civic Beacon": ("licensing",),
                    },
                },
                "Commerce Systems": {
                    "customers": ("large enterprises", "financial services firms"),
                    "channels": ("marketplaces", "resellers"),
                    "offerings": {
                        "Merchant Grid": ("service fees",),
                    },
                },
                "Edge Intelligence": {
                    "customers": ("developers", "IT professionals"),
                    "channels": ("direct sales", "marketplaces"),
                    "offerings": {
                        "Beacon Loop": ("subscription",),
                    },
                },
            },
        },
        "Velora Matrix": {
            "places": ("Italy", "North America"),
            "segments": {
                "Developer Cloud": {
                    "customers": ("developers", "IT professionals"),
                    "channels": ("marketplaces", "direct sales"),
                    "offerings": {
                        "Runtime Hub": ("consumption-based", "subscription"),
                    },
                },
                "Enterprise Ops": {
                    "customers": ("IT professionals", "large enterprises"),
                    "channels": ("direct sales", "managed service providers"),
                    "offerings": {
                        "Ops Ledger": ("subscription", "licensing"),
                    },
                },
                "Data Systems": {
                    "customers": ("large enterprises", "developers"),
                    "channels": ("marketplaces", "direct sales"),
                    "offerings": {
                        "Insight Rail": ("service fees",),
                    },
                },
            },
        },
        "Asteron Analytics": {
            "places": ("Canada", "Germany"),
            "segments": {
                "Industrial Platform": {
                    "customers": ("large enterprises", "manufacturers"),
                    "channels": ("direct sales", "system integrators"),
                    "offerings": {
                        "Signal Grid": ("usage-based", "subscription"),
                    },
                },
                "Data Cloud": {
                    "customers": ("developers", "IT professionals"),
                    "channels": ("marketplaces", "direct sales"),
                    "offerings": {
                        "Predict Stack": ("service fees",),
                    },
                },
                "Commerce Systems": {
                    "customers": ("large enterprises", "financial services firms"),
                    "channels": ("marketplaces", "resellers"),
                    "offerings": {
                        "Merchant Atlas": ("transaction fees",),
                    },
                },
            },
        },
        "Harborline Systems": {
            "places": ("France", "Japan"),
            "segments": {
                "Public Sector": {
                    "customers": ("government agencies", "IT professionals"),
                    "channels": ("system integrators",),
                    "offerings": {
                        "Civic Beacon": ("licensing",),
                    },
                },
                "Developer Cloud": {
                    "customers": ("developers", "IT professionals"),
                    "channels": ("marketplaces", "direct sales"),
                    "offerings": {
                        "Runtime Dock": ("subscription",),
                    },
                },
                "Enterprise Ops": {
                    "customers": ("IT professionals", "large enterprises"),
                    "channels": ("direct sales", "managed service providers"),
                    "offerings": {
                        "Ops Ledger": ("service fees", "subscription"),
                    },
                },
                "Commerce Systems": {
                    "customers": ("large enterprises", "financial services firms"),
                    "channels": ("marketplaces", "resellers"),
                    "offerings": {
                        "Merchant Grid": ("transaction fees",),
                    },
                },
            },
        },
        "Solstice Works": {
            "places": ("Spain", "Australia"),
            "segments": {
                "Industrial Platform": {
                    "customers": ("large enterprises", "manufacturers"),
                    "channels": ("direct sales", "system integrators"),
                    "offerings": {
                        "Control Tower": ("service fees",),
                    },
                },
                "Developer Cloud": {
                    "customers": ("developers", "IT professionals"),
                    "channels": ("marketplaces", "direct sales"),
                    "offerings": {
                        "Runtime Hub": ("subscription",),
                    },
                },
            },
        },
    }

    nodes: list[FixtureNodeSpec] = []
    edges: list[FixtureEdgeSpec] = []
    place_ids: dict[str, str] = {}
    customer_ids: dict[str, str] = {}
    channel_ids: dict[str, str] = {}
    revenue_ids: dict[str, str] = {}

    for company_index, (company, company_spec) in enumerate(company_specs.items(), start=1):
        company_id = f"company_{company_index}"
        nodes.append(_node(company_id, "Company", company))
        for place in company_spec["places"]:
            place_id = place_ids.setdefault(place, f"place_{len(place_ids) + 1}")
            if not any(node.node_id == place_id for node in nodes):
                place_spec = place_specs.get(place, {})
                nodes.append(
                    _place(
                        place_id,
                        place,
                        within_places=tuple(place_spec.get("within_places", ())),
                        includes_places=tuple(place_spec.get("includes_places", ())),
                    )
                )
            edges.append(_edge(company_id, "OPERATES_IN", place_id))
        for segment_index, (segment, segment_spec) in enumerate(company_spec["segments"].items(), start=1):
            segment_id = f"{company_id}_segment_{segment_index}"
            nodes.append(_node(segment_id, "BusinessSegment", segment, company_name=company))
            edges.append(_edge(company_id, "HAS_SEGMENT", segment_id))
            for customer in segment_spec["customers"]:
                customer_id = customer_ids.setdefault(customer, f"customer_{len(customer_ids) + 1}")
                if not any(node.node_id == customer_id for node in nodes):
                    nodes.append(_node(customer_id, "CustomerType", customer))
                edges.append(_edge(segment_id, "SERVES", customer_id))
            for channel in segment_spec["channels"]:
                channel_id = channel_ids.setdefault(channel, f"channel_{len(channel_ids) + 1}")
                if not any(node.node_id == channel_id for node in nodes):
                    nodes.append(_node(channel_id, "Channel", channel))
                edges.append(_edge(segment_id, "SELLS_THROUGH", channel_id))
            for offering_index, (offering, revenue_models) in enumerate(segment_spec["offerings"].items(), start=1):
                offering_id = f"{segment_id}_offering_{offering_index}"
                nodes.append(_node(offering_id, "Offering", offering, company_name=company))
                edges.append(_edge(segment_id, "OFFERS", offering_id))
                for revenue_model in revenue_models:
                    revenue_id = revenue_ids.setdefault(revenue_model, f"revenue_{len(revenue_ids) + 1}")
                    if not any(node.node_id == revenue_id for node in nodes):
                        nodes.append(_node(revenue_id, "RevenueModel", revenue_model))
                    edges.append(_edge(offering_id, "MONETIZES_VIA", revenue_id))

    fixture = FixtureSpec(
        fixture_id=fixture_id,
        graph_id=graph_id,
        graph_purpose="Company-level rollup profile with revenue, channel, customer, and geography coverage.",
        covered_families=("QF13", "QF14", "QF15", "QF20", "QF31"),
        nodes=nodes,
        edges=edges,
        invariants_satisfied=(
            "Company rollups derive from segment and offering facts rather than synthetic company-level convenience edges.",
            "The fixture includes direct place tags so geography+inventory composite queries can be trained.",
            "Segment names and offering names overlap across companies to keep rollups company-aware.",
        ),
        authoring_notes=(
            "Meridian, Orion, and Velora intentionally share some customer and channel labels while differing on revenue mix.",
            "Place tags were added here specifically for the optional geography+inventory composite family.",
        ),
    )

    company_revenue_list_cypher = (
        "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company})-[:OFFERS]->"
        "(root:Offering {company_name: $company}) "
        "MATCH (root)-[:OFFERS*0..]->(o:Offering {company_name: $company}) "
        "MATCH (o)-[:MONETIZES_VIA]->(r:RevenueModel) "
        "RETURN DISTINCT r.name AS revenue_model ORDER BY revenue_model"
    )
    company_revenue_count_cypher = (
        "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company})-[:OFFERS]->"
        "(root:Offering {company_name: $company}) "
        "MATCH (root)-[:OFFERS*0..]->(o:Offering {company_name: $company}) "
        "MATCH (o)-[:MONETIZES_VIA]->(r:RevenueModel) "
        "RETURN COUNT(DISTINCT r) AS revenue_model_count"
    )
    company_revenue_membership_cypher = (
        "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company})-[:OFFERS]->"
        "(root:Offering {company_name: $company}) "
        "MATCH (root)-[:OFFERS*0..]->(o:Offering {company_name: $company}) "
        "MATCH (o)-[:MONETIZES_VIA]->(r:RevenueModel {name: $revenue_model}) "
        "RETURN COUNT(DISTINCT r) > 0 AS is_match"
    )
    company_revenue_sources_cypher = (
        "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(s:BusinessSegment {company_name: $company})-[:OFFERS]->"
        "(root:Offering {company_name: $company}) "
        "MATCH (root)-[:OFFERS*0..]->(o:Offering {company_name: $company}) "
        "MATCH (o)-[:MONETIZES_VIA]->(r:RevenueModel {name: $revenue_model}) "
        "RETURN DISTINCT s.name AS segment, o.name AS offering, r.name AS revenue_model "
        "ORDER BY segment, offering, revenue_model"
    )
    company_channels_list_cypher = (
        "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company})-[:SELLS_THROUGH]->"
        "(ch:Channel) RETURN DISTINCT ch.name AS channel ORDER BY channel"
    )
    company_channels_count_cypher = (
        "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company})-[:SELLS_THROUGH]->"
        "(ch:Channel) RETURN COUNT(DISTINCT ch) AS channel_count"
    )
    company_channels_membership_cypher = (
        "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company})-[:SELLS_THROUGH]->"
        "(ch:Channel {name: $channel}) RETURN COUNT(DISTINCT ch) > 0 AS is_match"
    )
    company_channel_sources_cypher = (
        "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(s:BusinessSegment {company_name: $company})-[:SELLS_THROUGH]->"
        "(ch:Channel {name: $channel}) RETURN DISTINCT s.name AS segment, ch.name AS channel ORDER BY segment, channel"
    )
    company_customer_list_cypher = (
        "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company})-[:SERVES]->"
        "(ct:CustomerType) RETURN DISTINCT ct.name AS customer_type ORDER BY customer_type"
    )
    company_customer_count_cypher = (
        "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company})-[:SERVES]->"
        "(ct:CustomerType) RETURN COUNT(DISTINCT ct) AS customer_type_count"
    )
    company_customer_membership_cypher = (
        "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company})-[:SERVES]->"
        "(ct:CustomerType {name: $customer_type}) RETURN COUNT(DISTINCT ct) > 0 AS is_match"
    )
    company_customer_sources_cypher = (
        "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(s:BusinessSegment {company_name: $company})-[:SERVES]->"
        "(ct:CustomerType {name: $customer_type}) "
        "RETURN DISTINCT s.name AS segment, ct.name AS customer_type ORDER BY segment, customer_type"
    )
    company_descendant_count_cypher = (
        "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(:BusinessSegment {company_name: $company})-[:OFFERS]->"
        "(o:Offering {company_name: $company}) "
        "MATCH (o)-[:OFFERS*0..]->(d:Offering {company_name: $company}) "
        "RETURN COUNT(DISTINCT d) AS offering_count"
    )
    geography_revenue_company_list_cypher = (
        "MATCH (company:Company)-[:OPERATES_IN]->(place:Place) "
        "WITH company, place.name AS matched_place, "
        "CASE "
        "WHEN place.name = $place THEN 0 "
        "WHEN $place IN coalesce(place.includes_places, []) THEN 1 "
        "WHEN $place IN coalesce(place.within_places, []) THEN 2 "
        "ELSE NULL END AS match_rank "
        "WHERE match_rank IS NOT NULL "
        "MATCH (company)-[:HAS_SEGMENT]->(:BusinessSegment {company_name: company.name})-[:OFFERS]->"
        "(root:Offering {company_name: company.name}) "
        "MATCH (root)-[:OFFERS*0..]->(o:Offering {company_name: company.name})-[:MONETIZES_VIA]->"
        "(r:RevenueModel {name: $revenue_model}) "
        "RETURN DISTINCT company.name AS company ORDER BY company"
    )
    geography_channel_segments_cypher = (
        "MATCH (company:Company)-[:OPERATES_IN]->(place:Place) "
        "WITH company, place.name AS matched_place, "
        "CASE "
        "WHEN place.name = $place THEN 0 "
        "WHEN $place IN coalesce(place.includes_places, []) THEN 1 "
        "WHEN $place IN coalesce(place.within_places, []) THEN 2 "
        "ELSE NULL END AS match_rank "
        "WHERE match_rank IS NOT NULL "
        "MATCH (company)-[:HAS_SEGMENT]->(s:BusinessSegment {company_name: company.name})-[:SELLS_THROUGH]->"
        "(ch:Channel {name: $channel}) "
        "RETURN DISTINCT company.name AS company, s.name AS segment ORDER BY company, segment"
    )
    geography_revenue_company_count_cypher = (
        "MATCH (company:Company)-[:OPERATES_IN]->(place:Place) "
        "WITH company, place.name AS matched_place, "
        "CASE "
        "WHEN place.name = $place THEN 0 "
        "WHEN $place IN coalesce(place.includes_places, []) THEN 1 "
        "WHEN $place IN coalesce(place.within_places, []) THEN 2 "
        "ELSE NULL END AS match_rank "
        "WHERE match_rank IS NOT NULL "
        "MATCH (company)-[:HAS_SEGMENT]->(:BusinessSegment {company_name: company.name})-[:OFFERS]->"
        "(root:Offering {company_name: company.name}) "
        "MATCH (root)-[:OFFERS*0..]->(o:Offering {company_name: company.name})-[:MONETIZES_VIA]->"
        "(r:RevenueModel {name: $revenue_model}) "
        "RETURN COUNT(DISTINCT company) AS company_count"
    )

    examples: list[SourceExampleSpec] = []
    companies = tuple(company_specs.keys())
    for company in companies:
        slug = company.lower().replace(" ", "_")
        examples.extend(
            [
                _example(
                    example_id=f"qf13_company_revenue_rollup_list__{slug}",
                    intent_id="qf13_company_revenue_rollup_list",
                    family_id="QF13",
                    fixture_id=fixture_id,
                    graph_id=graph_id,
                    binding_id=f"{slug}_company_revenue_list",
                    question=f"How does {company} monetize its business?",
                    gold_cypher=company_revenue_list_cypher,
                    params={"company": company},
                    result_shape=[_col("revenue_model", "string")],
                    difficulty="high",
                ),
                _example(
                    example_id=f"qf13_company_revenue_rollup_count__{slug}",
                    intent_id="qf13_company_revenue_rollup_count",
                    family_id="QF13",
                    fixture_id=fixture_id,
                    graph_id=graph_id,
                    binding_id=f"{slug}_company_revenue_count",
                    question=f"How many revenue models does {company} use?",
                    gold_cypher=company_revenue_count_cypher,
                    params={"company": company},
                    result_shape=[_col("revenue_model_count", "integer")],
                    difficulty="high",
                ),
                _example(
                    example_id=f"qf14_company_channels_list__{slug}",
                    intent_id="qf14_company_channels_list",
                    family_id="QF14",
                    fixture_id=fixture_id,
                    graph_id=graph_id,
                    binding_id=f"{slug}_channel_list",
                    question=f"What channels does {company} use?",
                    gold_cypher=company_channels_list_cypher,
                    params={"company": company},
                    result_shape=[_col("channel", "string")],
                    difficulty="medium",
                ),
                _example(
                    example_id=f"qf14_company_channels_count__{slug}",
                    intent_id="qf14_company_channels_count",
                    family_id="QF14",
                    fixture_id=fixture_id,
                    graph_id=graph_id,
                    binding_id=f"{slug}_channel_count",
                    question=f"How many channels does {company} use?",
                    gold_cypher=company_channels_count_cypher,
                    params={"company": company},
                    result_shape=[_col("channel_count", "integer")],
                    difficulty="medium",
                ),
                _example(
                    example_id=f"qf15_company_customer_types_list__{slug}",
                    intent_id="qf15_company_customer_types_list",
                    family_id="QF15",
                    fixture_id=fixture_id,
                    graph_id=graph_id,
                    binding_id=f"{slug}_customer_list",
                    question=f"What customer types does {company} serve?",
                    gold_cypher=company_customer_list_cypher,
                    params={"company": company},
                    result_shape=[_col("customer_type", "string")],
                    difficulty="medium",
                ),
                _example(
                    example_id=f"qf15_company_customer_types_count__{slug}",
                    intent_id="qf15_company_customer_types_count",
                    family_id="QF15",
                    fixture_id=fixture_id,
                    graph_id=graph_id,
                    binding_id=f"{slug}_customer_count",
                    question=f"How many customer types does {company} serve?",
                    gold_cypher=company_customer_count_cypher,
                    params={"company": company},
                    result_shape=[_col("customer_type_count", "integer")],
                    difficulty="medium",
                ),
                _example(
                    example_id=f"qf20_company_descendant_offerings_count__{slug}",
                    intent_id="qf20_company_descendant_offerings_count",
                    family_id="QF20",
                    fixture_id=fixture_id,
                    graph_id=graph_id,
                    binding_id=f"{slug}_descendant_offerings",
                    question=f"How many offerings does {company} have across all segments?",
                    gold_cypher=company_descendant_count_cypher,
                    params={"company": company},
                    result_shape=[_col("offering_count", "integer")],
                    difficulty="medium",
                ),
            ]
        )

    for company, revenue_model in (
        ("Meridian Nexus", "licensing"),
        ("Orion Fabric", "service fees"),
        ("Velora Matrix", "consumption-based"),
        ("Asteron Analytics", "usage-based"),
        ("Harborline Systems", "subscription"),
        ("Solstice Works", "service fees"),
    ):
        slug = company.lower().replace(" ", "_")
        examples.extend(
            [
                _example(
                    example_id=f"qf13_company_revenue_rollup_membership__{slug}_{revenue_model.replace(' ', '_')}",
                    intent_id="qf13_company_revenue_rollup_membership",
                    family_id="QF13",
                    fixture_id=fixture_id,
                    graph_id=graph_id,
                    binding_id=f"{slug}_{revenue_model.replace(' ', '_')}_company_revenue_membership",
                    question=f"Does {company} monetize via {revenue_model}?",
                    gold_cypher=company_revenue_membership_cypher,
                    params={"company": company, "revenue_model": revenue_model},
                    result_shape=[_col("is_match", "boolean")],
                    difficulty="high",
                ),
                _example(
                    example_id=f"qf13_company_revenue_sources_list__{slug}_{revenue_model.replace(' ', '_')}",
                    intent_id="qf13_company_revenue_sources_list",
                    family_id="QF13",
                    fixture_id=fixture_id,
                    graph_id=graph_id,
                    binding_id=f"{slug}_{revenue_model.replace(' ', '_')}_company_revenue_sources",
                    question=f"Which segments or offerings make money for {company} via {revenue_model}?",
                    gold_cypher=company_revenue_sources_cypher,
                    params={"company": company, "revenue_model": revenue_model},
                    result_shape=[_col("segment", "string"), _col("offering", "string"), _col("revenue_model", "string")],
                    difficulty="high",
                ),
            ]
        )

    for company, channel in (
        ("Meridian Nexus", "marketplaces"),
        ("Orion Fabric", "system integrators"),
        ("Velora Matrix", "managed service providers"),
        ("Asteron Analytics", "direct sales"),
        ("Harborline Systems", "resellers"),
        ("Solstice Works", "marketplaces"),
    ):
        slug = company.lower().replace(" ", "_")
        examples.extend(
            [
                _example(
                    example_id=f"qf14_company_channels_membership__{slug}_{channel.replace(' ', '_')}",
                    intent_id="qf14_company_channels_membership",
                    family_id="QF14",
                    fixture_id=fixture_id,
                    graph_id=graph_id,
                    binding_id=f"{slug}_{channel.replace(' ', '_')}_company_channel_membership",
                    question=f"Does {company} use {channel}?",
                    gold_cypher=company_channels_membership_cypher,
                    params={"company": company, "channel": channel},
                    result_shape=[_col("is_match", "boolean")],
                    difficulty="medium",
                ),
                _example(
                    example_id=f"qf14_company_channel_sources_list__{slug}_{channel.replace(' ', '_')}",
                    intent_id="qf14_company_channel_sources_list",
                    family_id="QF14",
                    fixture_id=fixture_id,
                    graph_id=graph_id,
                    binding_id=f"{slug}_{channel.replace(' ', '_')}_company_channel_sources",
                    question=f"Which segments use {channel} at {company}?",
                    gold_cypher=company_channel_sources_cypher,
                    params={"company": company, "channel": channel},
                    result_shape=[_col("segment", "string"), _col("channel", "string")],
                    difficulty="medium",
                ),
            ]
        )

    for company, customer_type in (
        ("Meridian Nexus", "government agencies"),
        ("Orion Fabric", "manufacturers"),
        ("Velora Matrix", "developers"),
        ("Asteron Analytics", "financial services firms"),
        ("Harborline Systems", "IT professionals"),
        ("Solstice Works", "large enterprises"),
    ):
        slug = company.lower().replace(" ", "_")
        examples.extend(
            [
                _example(
                    example_id=f"qf15_company_customer_types_membership__{slug}_{customer_type.replace(' ', '_')}",
                    intent_id="qf15_company_customer_types_membership",
                    family_id="QF15",
                    fixture_id=fixture_id,
                    graph_id=graph_id,
                    binding_id=f"{slug}_{customer_type.replace(' ', '_')}_company_customer_membership",
                    question=f"Does {company} serve {customer_type}?",
                    gold_cypher=company_customer_membership_cypher,
                    params={"company": company, "customer_type": customer_type},
                    result_shape=[_col("is_match", "boolean")],
                    difficulty="medium",
                ),
                _example(
                    example_id=f"qf15_company_customer_sources_list__{slug}_{customer_type.replace(' ', '_')}",
                    intent_id="qf15_company_customer_sources_list",
                    family_id="QF15",
                    fixture_id=fixture_id,
                    graph_id=graph_id,
                    binding_id=f"{slug}_{customer_type.replace(' ', '_')}_company_customer_sources",
                    question=f"Which segments serve {customer_type} at {company}?",
                    gold_cypher=company_customer_sources_cypher,
                    params={"company": company, "customer_type": customer_type},
                    result_shape=[_col("segment", "string"), _col("customer_type", "string")],
                    difficulty="medium",
                ),
            ]
        )

    for place, revenue_model in (
        ("Europe", "licensing"),
        ("Europe", "subscription"),
        ("Germany", "usage-based"),
        ("APAC", "subscription"),
        ("North America", "transaction fees"),
        ("Italy", "consumption-based"),
        ("France", "service fees"),
        ("Spain", "service fees"),
        ("Canada", "usage-based"),
    ):
        examples.append(
            _example(
                example_id=f"qf31_place_company_revenue_model_list__{place.lower().replace(' ', '_')}_{revenue_model.replace(' ', '_')}",
                intent_id="qf31_place_company_revenue_model_list",
                family_id="QF31",
                fixture_id=fixture_id,
                graph_id=graph_id,
                binding_id=f"{place.lower().replace(' ', '_')}_{revenue_model.replace(' ', '_')}_company_revenue",
                question=f"Which companies operating in {place} monetize via {revenue_model}?",
                gold_cypher=geography_revenue_company_list_cypher,
                params={"place": place, "revenue_model": revenue_model},
                result_shape=[_col("company", "string")],
                difficulty="high",
            )
        )
    for place, channel in (
        ("Europe", "marketplaces"),
        ("Europe", "direct sales"),
        ("Germany", "system integrators"),
        ("APAC", "system integrators"),
        ("North America", "managed service providers"),
        ("Italy", "direct sales"),
        ("France", "resellers"),
        ("Spain", "marketplaces"),
    ):
        examples.append(
            _example(
                example_id=f"qf31_place_segment_channel_list__{place.lower().replace(' ', '_')}_{channel.replace(' ', '_')}",
                intent_id="qf31_place_segment_channel_list",
                family_id="QF31",
                fixture_id=fixture_id,
                graph_id=graph_id,
                binding_id=f"{place.lower().replace(' ', '_')}_{channel.replace(' ', '_')}_segment_channel",
                question=f"Which company segments at companies operating in {place} sell through {channel}?",
                gold_cypher=geography_channel_segments_cypher,
                params={"place": place, "channel": channel},
                result_shape=[_col("company", "string"), _col("segment", "string")],
                difficulty="high",
            )
        )
    for place, revenue_model in (
        ("Europe", "licensing"),
        ("Germany", "usage-based"),
        ("APAC", "subscription"),
        ("North America", "transaction fees"),
        ("Italy", "consumption-based"),
        ("France", "service fees"),
        ("Spain", "service fees"),
        ("Canada", "usage-based"),
    ):
        examples.append(
            _example(
                example_id=f"qf31_place_company_revenue_model_count__{place.lower().replace(' ', '_')}_{revenue_model.replace(' ', '_')}",
                intent_id="qf31_place_company_revenue_model_count",
                family_id="QF31",
                fixture_id=fixture_id,
                graph_id=graph_id,
                binding_id=f"{place.lower().replace(' ', '_')}_{revenue_model.replace(' ', '_')}_company_revenue_count",
                question=f"How many companies operating in {place} monetize via {revenue_model}?",
                gold_cypher=geography_revenue_company_count_cypher,
                params={"place": place, "revenue_model": revenue_model},
                result_shape=[_col("company_count", "integer")],
                difficulty="high",
            )
        )

    return fixture, examples


def _build_fx11_intersection_and_filtering_mesh() -> tuple[FixtureSpec, list[SourceExampleSpec]]:
    fixture_id = "FX11_intersection_and_filtering_mesh"
    graph_id = "fx11_intersection_and_filtering_mesh_v2"
    segment_specs = {
        ("Asteron Analytics", "Industrial AI"): {
            "customers": ("developers", "large enterprises"),
            "channels": ("direct sales", "system integrators"),
            "offerings": ("Control Suite", "Insight Cloud", "Model Forge"),
        },
        ("Orion Fabric", "Public Sector"): {
            "customers": ("government agencies", "IT professionals"),
            "channels": ("direct sales", "system integrators"),
            "offerings": ("Control Suite", "Civic Graph"),
        },
        ("Velora Matrix", "Developer Cloud"): {
            "customers": ("developers", "IT professionals"),
            "channels": ("direct sales", "marketplaces"),
            "offerings": ("Runtime Hub", "Deploy Suite"),
        },
        ("Harborline Systems", "Commerce Systems"): {
            "customers": ("large enterprises", "financial services firms"),
            "channels": ("marketplaces", "resellers"),
            "offerings": ("Merchant Grid", "Control Suite"),
        },
        ("Meridian Nexus", "Enterprise Ops"): {
            "customers": ("IT professionals", "large enterprises"),
            "channels": ("direct sales", "managed service providers"),
            "offerings": ("Ops Ledger", "Control Suite"),
        },
        ("Asteron Analytics", "Developer Cloud"): {
            "customers": ("developers", "IT professionals"),
            "channels": ("direct sales", "marketplaces"),
            "offerings": ("Control Suite", "Signal Hub"),
        },
        ("Harborline Systems", "Public Sector"): {
            "customers": ("government agencies", "IT professionals"),
            "channels": ("direct sales", "system integrators"),
            "offerings": ("Civic Graph", "Control Suite"),
        },
        ("Solstice Works", "Commerce Systems"): {
            "customers": ("large enterprises", "financial services firms"),
            "channels": ("marketplaces", "resellers"),
            "offerings": ("Merchant Grid", "Control Suite"),
        },
    }
    nodes: list[FixtureNodeSpec] = []
    edges: list[FixtureEdgeSpec] = []
    company_ids: dict[str, str] = {}
    customer_ids: dict[str, str] = {}
    channel_ids: dict[str, str] = {}
    for index, ((company, segment), spec) in enumerate(segment_specs.items(), start=1):
        company_id = company_ids.setdefault(company, f"company_{len(company_ids) + 1}")
        if not any(node.node_id == company_id for node in nodes):
            nodes.append(_node(company_id, "Company", company))
        segment_id = f"{company_id}_segment_{index}"
        nodes.append(_node(segment_id, "BusinessSegment", segment, company_name=company))
        edges.append(_edge(company_id, "HAS_SEGMENT", segment_id))
        for customer in spec["customers"]:
            customer_id = customer_ids.setdefault(customer, f"customer_{len(customer_ids) + 1}")
            if not any(node.node_id == customer_id for node in nodes):
                nodes.append(_node(customer_id, "CustomerType", customer))
            edges.append(_edge(segment_id, "SERVES", customer_id))
        for channel in spec["channels"]:
            channel_id = channel_ids.setdefault(channel, f"channel_{len(channel_ids) + 1}")
            if not any(node.node_id == channel_id for node in nodes):
                nodes.append(_node(channel_id, "Channel", channel))
            edges.append(_edge(segment_id, "SELLS_THROUGH", channel_id))
        for offering_index, offering in enumerate(spec["offerings"], start=1):
            offering_id = f"{segment_id}_offering_{offering_index}"
            nodes.append(_node(offering_id, "Offering", offering, company_name=company))
            edges.append(_edge(segment_id, "OFFERS", offering_id))

    fixture = FixtureSpec(
        fixture_id=fixture_id,
        graph_id=graph_id,
        graph_purpose="Overlap-heavy segment mesh for global intersections and company+segment output rows.",
        covered_families=("QF19",),
        nodes=nodes,
        edges=edges,
        invariants_satisfied=(
            "The fixture intentionally shares channels, customers, and offering names across companies.",
            "Every positive intersection has at least one near-miss elsewhere in the graph.",
            "Offerings remain direct children of segments so intersection logic stays inspectable.",
        ),
        authoring_notes=(
            "This is the main cross-company intersection fixture for the rebuilt corpus.",
            "Control Suite appears under multiple companies, but only some segments satisfy the full predicate set used by the queries.",
        ),
    )

    customer_channel_segments_cypher = (
        "MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment)-[:SERVES]->(ct:CustomerType {name: $customer_type}) "
        "MATCH (s)-[:SELLS_THROUGH]->(ch:Channel {name: $channel}) "
        "WHERE s.company_name = c.name "
        "RETURN DISTINCT c.name AS company, s.name AS segment ORDER BY company, segment"
    )
    customer_offering_segments_cypher = (
        "MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment)-[:SERVES]->(ct:CustomerType {name: $customer_type}) "
        "MATCH (s)-[:OFFERS]->(o:Offering {name: $offering}) "
        "WHERE s.company_name = c.name AND o.company_name = c.name "
        "RETURN DISTINCT c.name AS company, s.name AS segment ORDER BY company, segment"
    )
    customer_channel_offering_segments_cypher = (
        "MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment)-[:SERVES]->(ct:CustomerType {name: $customer_type}) "
        "MATCH (s)-[:SELLS_THROUGH]->(ch:Channel {name: $channel}) "
        "MATCH (s)-[:OFFERS]->(o:Offering {name: $offering}) "
        "WHERE s.company_name = c.name AND o.company_name = c.name "
        "RETURN DISTINCT c.name AS company, s.name AS segment ORDER BY company, segment"
    )
    intersection_count_cypher = (
        "MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment)-[:SERVES]->(ct:CustomerType {name: $customer_type}) "
        "MATCH (s)-[:SELLS_THROUGH]->(ch:Channel {name: $channel}) "
        "MATCH (s)-[:OFFERS]->(o:Offering {name: $offering}) "
        "WHERE s.company_name = c.name AND o.company_name = c.name "
        "RETURN COUNT(DISTINCT c.name + '|' + s.name) AS segment_count"
    )

    examples: list[SourceExampleSpec] = []
    for customer_type, channel in (
        ("developers", "direct sales"),
        ("government agencies", "system integrators"),
        ("large enterprises", "marketplaces"),
        ("IT professionals", "managed service providers"),
        ("financial services firms", "resellers"),
        ("developers", "marketplaces"),
    ):
        slug = f"{customer_type.replace(' ', '_')}_{channel.replace(' ', '_')}"
        examples.append(
            _example(
                example_id=f"qf19_segment_customer_channel_list__{slug}",
                intent_id="qf19_segment_customer_channel_list",
                family_id="QF19",
                fixture_id=fixture_id,
                graph_id=graph_id,
                binding_id=f"customer_channel_{slug}",
                question=f"Which company segments serve {customer_type} and sell through {channel}?",
                gold_cypher=customer_channel_segments_cypher,
                params={"customer_type": customer_type, "channel": channel},
                result_shape=[_col("company", "string"), _col("segment", "string")],
                difficulty="high",
            )
        )
    for customer_type, offering in (
        ("developers", "Control Suite"),
        ("large enterprises", "Merchant Grid"),
        ("government agencies", "Civic Graph"),
        ("IT professionals", "Ops Ledger"),
        ("developers", "Signal Hub"),
        ("financial services firms", "Merchant Grid"),
    ):
        slug = f"{customer_type.replace(' ', '_')}_{offering.lower().replace(' ', '_')}"
        examples.append(
            _example(
                example_id=f"qf19_segment_customer_offering_list__{slug}",
                intent_id="qf19_segment_customer_offering_list",
                family_id="QF19",
                fixture_id=fixture_id,
                graph_id=graph_id,
                binding_id=f"customer_offering_{slug}",
                question=f"Which company segments serve {customer_type} and offer {offering}?",
                gold_cypher=customer_offering_segments_cypher,
                params={"customer_type": customer_type, "offering": offering},
                result_shape=[_col("company", "string"), _col("segment", "string")],
                difficulty="high",
            )
        )
    for customer_type, channel, offering in (
        ("developers", "direct sales", "Control Suite"),
        ("government agencies", "system integrators", "Civic Graph"),
        ("large enterprises", "marketplaces", "Merchant Grid"),
        ("IT professionals", "managed service providers", "Ops Ledger"),
        ("developers", "marketplaces", "Signal Hub"),
        ("financial services firms", "resellers", "Merchant Grid"),
    ):
        slug = f"{customer_type.replace(' ', '_')}_{channel.replace(' ', '_')}_{offering.lower().replace(' ', '_')}"
        examples.append(
            _example(
                example_id=f"qf19_segment_customer_channel_offering_list__{slug}",
                intent_id="qf19_segment_customer_channel_offering_list",
                family_id="QF19",
                fixture_id=fixture_id,
                graph_id=graph_id,
                binding_id=f"customer_channel_offering_{slug}",
                question=f"Which company segments serve {customer_type}, sell through {channel}, and offer {offering}?",
                gold_cypher=customer_channel_offering_segments_cypher,
                params={"customer_type": customer_type, "channel": channel, "offering": offering},
                result_shape=[_col("company", "string"), _col("segment", "string")],
                difficulty="high",
            )
        )
    for customer_type, channel, offering in (
        ("developers", "direct sales", "Control Suite"),
        ("government agencies", "system integrators", "Civic Graph"),
        ("large enterprises", "marketplaces", "Merchant Grid"),
        ("IT professionals", "managed service providers", "Ops Ledger"),
        ("developers", "marketplaces", "Signal Hub"),
        ("financial services firms", "resellers", "Merchant Grid"),
    ):
        slug = f"{customer_type.replace(' ', '_')}_{channel.replace(' ', '_')}_{offering.lower().replace(' ', '_')}"
        examples.append(
            _example(
                example_id=f"qf19_segment_intersection_count__{slug}",
                intent_id="qf19_segment_intersection_count",
                family_id="QF19",
                fixture_id=fixture_id,
                graph_id=graph_id,
                binding_id=f"intersection_count_{slug}",
                question=f"How many company segments serve {customer_type}, sell through {channel}, and offer {offering}?",
                gold_cypher=intersection_count_cypher,
                params={"customer_type": customer_type, "channel": channel, "offering": offering},
                result_shape=[_col("segment_count", "integer")],
                difficulty="high",
            )
        )

    return fixture, examples


def build_spec() -> DatasetSpec:
    fixtures: list[FixtureSpec] = []
    source_examples: list[SourceExampleSpec] = []
    for builder in (
        _build_fx09_segment_revenue_rollup_tree,
        _build_fx10_company_rollup_profile,
        _build_fx11_intersection_and_filtering_mesh,
    ):
        fixture, examples = builder()
        fixtures.append(fixture)
        source_examples.extend(examples)

    by_intent: dict[str, int] = defaultdict(int)
    for example in source_examples:
        by_intent[example.intent_id] += 1
    if not by_intent:
        raise ValueError("spec_rollups did not generate any examples")

    return DatasetSpec(fixtures=fixtures, source_examples=source_examples)
