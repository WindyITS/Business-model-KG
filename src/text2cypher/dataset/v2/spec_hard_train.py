from __future__ import annotations

from collections.abc import Sequence

from .models import (
    DatasetSpec,
    FixtureEdgeSpec,
    FixtureNodeSpec,
    FixtureSpec,
    ResultColumnSpec,
    SourceExampleSpec,
)


def _slug(text: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in text)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _node(node_id: str, label: str, name: str, **properties: object) -> FixtureNodeSpec:
    return FixtureNodeSpec(node_id=node_id, label=label, name=name, properties=properties)


def _company(node_id: str, name: str) -> FixtureNodeSpec:
    return _node(node_id, "Company", name)


def _segment(node_id: str, company_name: str, name: str) -> FixtureNodeSpec:
    return _node(node_id, "BusinessSegment", name, company_name=company_name)


def _offering(node_id: str, company_name: str, name: str) -> FixtureNodeSpec:
    return _node(node_id, "Offering", name, company_name=company_name)


def _place(node_id: str, name: str, *, within_places: Sequence[str] = (), includes_places: Sequence[str] = ()) -> FixtureNodeSpec:
    properties: dict[str, object] = {}
    if within_places:
        properties["within_places"] = list(within_places)
    if includes_places:
        properties["includes_places"] = list(includes_places)
    return _node(node_id, "Place", name, **properties)


def _customer_type(node_id: str, name: str) -> FixtureNodeSpec:
    return _node(node_id, "CustomerType", name)


def _channel(node_id: str, name: str) -> FixtureNodeSpec:
    return _node(node_id, "Channel", name)


def _revenue_model(node_id: str, name: str) -> FixtureNodeSpec:
    return _node(node_id, "RevenueModel", name)


def _edge(source: str, relation: str, target: str) -> FixtureEdgeSpec:
    return FixtureEdgeSpec(source=source, type=relation, target=target)


def _col(column: str, type_name: str, description: str | None = None) -> ResultColumnSpec:
    return ResultColumnSpec(column=column, type=type_name, description=description)


def _hard_train_paraphrases(question: str) -> tuple[str, ...]:
    lowered = question[0].lower() + question[1:] if question else question
    return (
        f"Need the exact graph answer here: {lowered}",
        f"Can you pull this straight from Neo4j: {lowered}",
        f"Please resolve this using only the KG: {lowered}",
        f"Quick graph check on this request: {lowered}",
        f"Need the Cypher-ready answer for this ask: {lowered}",
        f"I need the exact lookup from the graph: {lowered}",
        f"Can you sanity-check this against the knowledge graph: {lowered}",
        f"Please give me the read-only graph result: {lowered}",
        f"Trying to untangle this graph question: {lowered}",
        f"What does the KG return for this request: {lowered}",
        f"Need the exact business-model graph answer here: {lowered}",
        f"Can you give me the structured graph result: {lowered}",
        f"Need the precise ontology-backed answer for this: {lowered}",
        f"I only want the graph lookup for this: {lowered}",
        f"One more hard query for the KG: {lowered}",
    )


COMPANY_SPECS: tuple[dict[str, object], ...] = (
    {
        "company": "Mariner Vale",
        "places": (
            {"name": "Iberia", "includes_places": ("Spain", "Portugal")},
            {"name": "Benelux", "includes_places": ("Belgium", "Netherlands", "Luxembourg")},
        ),
        "segments": (
            {
                "segment": "Signal Grid",
                "customers": ("developers", "large enterprises"),
                "channels": ("direct sales", "marketplaces"),
                "offerings": (
                    ("Harbor Console", ("subscription", "usage-based")),
                    ("Harbor Stream", ("subscription",)),
                ),
            },
            {
                "segment": "Route Engine",
                "customers": ("government agencies", "IT professionals"),
                "channels": ("system integrators", "resellers"),
                "offerings": (
                    ("Harbor Route", ("usage-based",)),
                    ("Harbor Beacon", ("subscription", "usage-based")),
                ),
            },
        ),
    },
    {
        "company": "Northfield Loom",
        "places": (
            {"name": "Nordics", "includes_places": ("Sweden", "Finland", "Norway")},
            {"name": "DACH", "includes_places": ("Germany", "Austria", "Switzerland")},
        ),
        "segments": (
            {
                "segment": "Civic Mesh",
                "customers": ("government agencies", "large enterprises"),
                "channels": ("system integrators", "managed service providers"),
                "offerings": (
                    ("Loom Civic", ("service fees", "subscription")),
                    ("Loom Registry", ("service fees",)),
                ),
            },
            {
                "segment": "Runtime Fabric",
                "customers": ("developers", "IT professionals"),
                "channels": ("direct sales", "online"),
                "offerings": (
                    ("Loom Runtime", ("licensing", "consumption-based")),
                    ("Loom Relay", ("licensing",)),
                ),
            },
        ),
    },
    {
        "company": "Peregrine Stack",
        "places": (
            {"name": "Southeast Asia", "includes_places": ("Singapore", "Thailand", "Malaysia")},
            {"name": "ANZ", "includes_places": ("Australia", "New Zealand")},
        ),
        "segments": (
            {
                "segment": "Planning Studio",
                "customers": ("financial services firms", "large enterprises"),
                "channels": ("direct sales", "resellers"),
                "offerings": (
                    ("Peregrine Atlas", ("transaction fees", "subscription")),
                    ("Peregrine Ledger", ("transaction fees",)),
                ),
            },
            {
                "segment": "Field Exchange",
                "customers": ("government agencies", "developers"),
                "channels": ("marketplaces", "system integrators"),
                "offerings": (
                    ("Peregrine Field", ("subscription",)),
                    ("Peregrine Arc", ("transaction fees", "subscription")),
                ),
            },
        ),
    },
    {
        "company": "Quarry Harbor",
        "places": (
            {"name": "East Africa", "includes_places": ("Kenya", "Tanzania", "Uganda")},
            {"name": "Gulf", "includes_places": ("UAE", "Saudi Arabia", "Qatar")},
        ),
        "segments": (
            {
                "segment": "Exchange Core",
                "customers": ("IT professionals", "developers"),
                "channels": ("direct sales", "marketplaces"),
                "offerings": (
                    ("Quarry Core", ("consumption-based", "service fees")),
                    ("Quarry Prism", ("consumption-based",)),
                ),
            },
            {
                "segment": "Risk Ledger",
                "customers": ("financial services firms", "large enterprises"),
                "channels": ("managed service providers", "resellers"),
                "offerings": (
                    ("Quarry Shield", ("service fees",)),
                    ("Quarry Ledger", ("consumption-based", "service fees")),
                ),
            },
        ),
    },
)


def _build_fixture() -> tuple[FixtureSpec, list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    fixture_id = "FX12_hard_train_mesh_v1"
    graph_id = "fx12_hard_train_mesh_v1"

    nodes: list[FixtureNodeSpec] = []
    edges: list[FixtureEdgeSpec] = []

    qf19_bindings: list[dict[str, str]] = []
    qf31_revenue_bindings: list[dict[str, str]] = []
    qf31_channel_bindings: list[dict[str, str]] = []

    place_ids: dict[str, str] = {}
    customer_ids: dict[str, str] = {}
    channel_ids: dict[str, str] = {}
    revenue_ids: dict[str, str] = {}

    for company_index, company_spec in enumerate(COMPANY_SPECS, start=1):
        company_name = str(company_spec["company"])
        company_id = f"company_{company_index}"
        nodes.append(_company(company_id, company_name))

        for place_spec in company_spec["places"]:  # type: ignore[index]
            place_name = str(place_spec["name"])
            place_id = place_ids.setdefault(place_name, f"place_{len(place_ids) + 1}")
            if not any(node.node_id == place_id for node in nodes):
                nodes.append(
                    _place(
                        place_id,
                        place_name,
                        within_places=tuple(place_spec.get("within_places", ())),
                        includes_places=tuple(place_spec.get("includes_places", ())),
                    )
                )
            edges.append(_edge(company_id, "OPERATES_IN", place_id))

        company_revenue_models: set[str] = set()
        segment_specs = tuple(company_spec["segments"])  # type: ignore[index]
        for segment_index, segment_spec in enumerate(segment_specs, start=1):
            segment_name = str(segment_spec["segment"])
            segment_id = f"{company_id}_segment_{segment_index}"
            nodes.append(_segment(segment_id, company_name, segment_name))
            edges.append(_edge(company_id, "HAS_SEGMENT", segment_id))

            customers = tuple(segment_spec["customers"])  # type: ignore[index]
            channels = tuple(segment_spec["channels"])  # type: ignore[index]
            offerings = tuple(segment_spec["offerings"])  # type: ignore[index]

            for customer in customers:
                customer_id = customer_ids.setdefault(customer, f"customer_{len(customer_ids) + 1}")
                if not any(node.node_id == customer_id for node in nodes):
                    nodes.append(_customer_type(customer_id, customer))
                edges.append(_edge(segment_id, "SERVES", customer_id))

            for channel in channels:
                channel_id = channel_ids.setdefault(channel, f"channel_{len(channel_ids) + 1}")
                if not any(node.node_id == channel_id for node in nodes):
                    nodes.append(_channel(channel_id, channel))
                edges.append(_edge(segment_id, "SELLS_THROUGH", channel_id))

            for offering_index, (offering_name, offering_revenue_models) in enumerate(offerings, start=1):
                offering_id = f"{segment_id}_offering_{offering_index}"
                nodes.append(_offering(offering_id, company_name, offering_name))
                edges.append(_edge(segment_id, "OFFERS", offering_id))
                for revenue_model in offering_revenue_models:
                    company_revenue_models.add(revenue_model)
                    revenue_id = revenue_ids.setdefault(revenue_model, f"revenue_{len(revenue_ids) + 1}")
                    if not any(node.node_id == revenue_id for node in nodes):
                        nodes.append(_revenue_model(revenue_id, revenue_model))
                    edges.append(_edge(offering_id, "MONETIZES_VIA", revenue_id))

            qf19_bindings.extend(
                [
                    {
                        "customer_type": customers[0],
                        "channel": channels[0],
                        "offering": offerings[0][0],
                        "slug": _slug(
                            f"{company_name}_{segment_name}_{customers[0]}_{channels[0]}_{offerings[0][0]}"
                        ),
                    },
                    {
                        "customer_type": customers[1],
                        "channel": channels[1],
                        "offering": offerings[1][0],
                        "slug": _slug(
                            f"{company_name}_{segment_name}_{customers[1]}_{channels[1]}_{offerings[1][0]}"
                        ),
                    },
                ]
            )

            if segment_index == 1:
                for place_spec in company_spec["places"]:  # type: ignore[index]
                    qf31_channel_bindings.append(
                        {
                            "place": str(place_spec["name"]),
                            "channel": channels[0],
                            "slug": _slug(f"{company_name}_{place_spec['name']}_{channels[0]}"),
                        }
                    )

        for place_spec in company_spec["places"]:  # type: ignore[index]
            primary_revenue_model = sorted(company_revenue_models)[0]
            qf31_revenue_bindings.append(
                {
                    "place": str(place_spec["name"]),
                    "revenue_model": primary_revenue_model,
                    "slug": _slug(f"{company_name}_{place_spec['name']}_{primary_revenue_model}"),
                }
            )

    if len(qf19_bindings) != 16:
        raise ValueError(f"Expected 16 hard-train QF19 bindings, found {len(qf19_bindings)}")
    if len(qf31_revenue_bindings) != 8:
        raise ValueError(f"Expected 8 hard-train QF31 revenue bindings, found {len(qf31_revenue_bindings)}")
    if len(qf31_channel_bindings) != 8:
        raise ValueError(f"Expected 8 hard-train QF31 channel bindings, found {len(qf31_channel_bindings)}")

    fixture = FixtureSpec(
        fixture_id=fixture_id,
        graph_id=graph_id,
        graph_purpose=(
            "High-difficulty training mesh for multi-constraint customer/channel/offering intersections "
            "and geography-constrained revenue/channel queries."
        ),
        covered_families=("QF19", "QF31"),
        nodes=nodes,
        edges=edges,
        invariants_satisfied=(
            "BusinessSegment and Offering nodes remain company-scoped through company_name properties.",
            "Each segment supports at least two hard QF19 intersection bindings with near-miss alternatives elsewhere in the graph.",
            "Each company operates in two places and exposes revenue/channel signals suitable for geography-filtered rollup queries.",
        ),
        authoring_notes=(
            "This fixture is intentionally overlap-heavy so the model sees harder multi-constraint training cases.",
            "Place nodes use includes_places arrays to preserve hierarchy-aware geography behavior.",
            "The names are distinct from the held-out fixture so the evaluation set can stay fresh.",
        ),
    )

    return fixture, qf19_bindings, qf31_revenue_bindings, qf31_channel_bindings


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
    result_shape: Sequence[ResultColumnSpec],
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
        difficulty="high",
        split="train",
        paraphrases=_hard_train_paraphrases(question),
    )


def build_spec() -> DatasetSpec:
    fixture, qf19_bindings, qf31_revenue_bindings, qf31_channel_bindings = _build_fixture()

    qf19_list_cypher = (
        "MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment)-[:SERVES]->"
        "(ct:CustomerType {name: $customer_type}) "
        "MATCH (s)-[:SELLS_THROUGH]->(ch:Channel {name: $channel}) "
        "MATCH (s)-[:OFFERS]->(o:Offering {name: $offering}) "
        "WHERE s.company_name = c.name AND o.company_name = c.name "
        "RETURN DISTINCT c.name AS company, s.name AS segment ORDER BY company, segment"
    )
    qf19_count_cypher = (
        "MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment)-[:SERVES]->"
        "(ct:CustomerType {name: $customer_type}) "
        "MATCH (s)-[:SELLS_THROUGH]->(ch:Channel {name: $channel}) "
        "MATCH (s)-[:OFFERS]->(o:Offering {name: $offering}) "
        "WHERE s.company_name = c.name AND o.company_name = c.name "
        "RETURN COUNT(DISTINCT c.name + '|' + s.name) AS segment_count"
    )
    qf31_revenue_cypher = (
        "MATCH (company:Company)-[:OPERATES_IN]->(place:Place {name: $place}) "
        "MATCH (company)-[:HAS_SEGMENT]->(:BusinessSegment {company_name: company.name})-[:OFFERS]->"
        "(root:Offering {company_name: company.name}) "
        "MATCH (root)-[:OFFERS*0..]->(o:Offering {company_name: company.name})-[:MONETIZES_VIA]->"
        "(r:RevenueModel {name: $revenue_model}) "
        "RETURN DISTINCT company.name AS company ORDER BY company"
    )
    qf31_channel_cypher = (
        "MATCH (company:Company)-[:OPERATES_IN]->(place:Place) "
        "WITH company, "
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

    source_examples: list[SourceExampleSpec] = []

    for binding in qf19_bindings[:8]:
        question = (
            "Which company segments serve {customer_type}, sell through {channel}, and offer {offering}?"
        ).format(**binding)
        source_examples.append(
            _example(
                example_id=f"hard_qf19_segment_customer_channel_offering_list__{binding['slug']}",
                intent_id=f"hard_qf19_segment_customer_channel_offering_list__{binding['slug']}",
                family_id="QF19",
                fixture_id=fixture.fixture_id,
                graph_id=fixture.graph_id,
                binding_id=f"qf19_list_{binding['slug']}",
                question=question,
                gold_cypher=qf19_list_cypher,
                params={
                    "customer_type": binding["customer_type"],
                    "channel": binding["channel"],
                    "offering": binding["offering"],
                },
                result_shape=[_col("company", "string"), _col("segment", "string")],
            )
        )

    for binding in qf19_bindings[8:]:
        question = (
            "How many company segments serve {customer_type}, sell through {channel}, and offer {offering}?"
        ).format(**binding)
        source_examples.append(
            _example(
                example_id=f"hard_qf19_segment_intersection_count__{binding['slug']}",
                intent_id=f"hard_qf19_segment_intersection_count__{binding['slug']}",
                family_id="QF19",
                fixture_id=fixture.fixture_id,
                graph_id=fixture.graph_id,
                binding_id=f"qf19_count_{binding['slug']}",
                question=question,
                gold_cypher=qf19_count_cypher,
                params={
                    "customer_type": binding["customer_type"],
                    "channel": binding["channel"],
                    "offering": binding["offering"],
                },
                result_shape=[_col("segment_count", "integer")],
            )
        )

    for binding in qf31_revenue_bindings:
        question = "Which companies operating in {place} monetize via {revenue_model}?".format(**binding)
        source_examples.append(
            _example(
                example_id=f"hard_qf31_place_company_revenue_model_list__{binding['slug']}",
                intent_id=f"hard_qf31_place_company_revenue_model_list__{binding['slug']}",
                family_id="QF31",
                fixture_id=fixture.fixture_id,
                graph_id=fixture.graph_id,
                binding_id=f"qf31_revenue_{binding['slug']}",
                question=question,
                gold_cypher=qf31_revenue_cypher,
                params={
                    "place": binding["place"],
                    "revenue_model": binding["revenue_model"],
                },
                result_shape=[_col("company", "string")],
            )
        )

    for binding in qf31_channel_bindings:
        question = "Which company segments at companies operating in {place} sell through {channel}?".format(**binding)
        source_examples.append(
            _example(
                example_id=f"hard_qf31_place_segment_channel_list__{binding['slug']}",
                intent_id=f"hard_qf31_place_segment_channel_list__{binding['slug']}",
                family_id="QF31",
                fixture_id=fixture.fixture_id,
                graph_id=fixture.graph_id,
                binding_id=f"qf31_channel_{binding['slug']}",
                question=question,
                gold_cypher=qf31_channel_cypher,
                params={
                    "place": binding["place"],
                    "channel": binding["channel"],
                },
                result_shape=[_col("company", "string"), _col("segment", "string")],
            )
        )

    if len(source_examples) != 32:
        raise ValueError(f"Expected 32 hard-train source examples, found {len(source_examples)}")
    for example in source_examples:
        if len(example.paraphrases) != 15:
            raise ValueError(f"Example {example.example_id} must have exactly 15 paraphrases")

    return DatasetSpec(fixtures=[fixture], source_examples=source_examples)
