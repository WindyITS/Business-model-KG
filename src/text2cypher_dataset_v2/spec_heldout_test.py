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


def _place(node_id: str, name: str) -> FixtureNodeSpec:
    return _node(node_id, "Place", name)


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


def _heldout_paraphrases(question: str) -> tuple[str, ...]:
    lowered = question[0].lower() + question[1:] if question else question
    return (
        f"Need the exact graph answer for this: {lowered}",
        f"Can you pull the KG-backed result for this: {lowered}",
        f"Quick graph lookup for this one: {lowered}",
        f"I need the Cypher-ready answer here: {lowered}",
        f"Please answer from the knowledge graph: {lowered}",
        f"Can you sanity-check this against the KG: {lowered}",
        f"What does the graph return for this request: {lowered}",
        f"Need a clean read from the graph here: {lowered}",
        f"Help me resolve this Neo4j query: {lowered}",
        f"I am after the exact graph match for this: {lowered}",
        f"Please give me the structured graph answer: {lowered}",
        f"Need the read-only graph result for this: {lowered}",
        f"Can you get the precise KG lookup for this: {lowered}",
        f"I'm trying to pin down this graph ask: {lowered}",
        f"One more graph check for this: {lowered}",
    )


COMPANY_SPECS: tuple[dict[str, object], ...] = (
    {
        "company": "Alder Quanta",
        "places": ("Larkport", "Mosspoint"),
        "revenue_models": ("subscription", "usage-based"),
        "segments": (
            {
                "segment": "Signal Planning",
                "customers": ("developers", "large enterprises"),
                "channels": ("direct sales", "marketplaces"),
                "offerings": (
                    ("Aquila Console", ("subscription", "usage-based")),
                    ("Aquila Studio", ("subscription",)),
                ),
            },
            {
                "segment": "Grid Navigation",
                "customers": ("government agencies", "IT professionals"),
                "channels": ("system integrators", "resellers"),
                "offerings": (
                    ("Aquila Route", ("usage-based",)),
                    ("Aquila Radar", ("subscription", "usage-based")),
                ),
            },
        ),
    },
    {
        "company": "Beryl Harbor",
        "places": ("Northmere", "Stone Wharf"),
        "revenue_models": ("licensing", "consumption-based"),
        "segments": (
            {
                "segment": "Demand Studio",
                "customers": ("developers", "IT professionals"),
                "channels": ("direct sales", "online"),
                "offerings": (
                    ("Beryl Forecast", ("licensing", "consumption-based")),
                    ("Beryl Pulse", ("licensing",)),
                ),
            },
            {
                "segment": "Operations Ledger",
                "customers": ("large enterprises", "government agencies"),
                "channels": ("marketplaces", "system integrators"),
                "offerings": (
                    ("Beryl Vault", ("consumption-based",)),
                    ("Beryl Ledger", ("licensing", "consumption-based")),
                ),
            },
        ),
    },
    {
        "company": "Crescent Quarry",
        "places": ("Juniper Bay", "Slate Valley"),
        "revenue_models": ("service fees", "subscription"),
        "segments": (
            {
                "segment": "Risk Workshop",
                "customers": ("financial services firms", "large enterprises"),
                "channels": ("direct sales", "resellers"),
                "offerings": (
                    ("Crescent Shield", ("service fees", "subscription")),
                    ("Crescent Scan", ("service fees",)),
                ),
            },
            {
                "segment": "Experience Cloud",
                "customers": ("developers", "government agencies"),
                "channels": ("online", "marketplaces"),
                "offerings": (
                    ("Crescent Atlas", ("subscription",)),
                    ("Crescent Spark", ("service fees", "subscription")),
                ),
            },
        ),
    },
    {
        "company": "Dovetail Current",
        "places": ("Amber Bluff", "Thorn Port"),
        "revenue_models": ("consumption-based", "service fees"),
        "segments": (
            {
                "segment": "Planning Engine",
                "customers": ("IT professionals", "developers"),
                "channels": ("direct sales", "marketplaces"),
                "offerings": (
                    ("Dovetail Core", ("consumption-based", "service fees")),
                    ("Dovetail Prism", ("consumption-based",)),
                ),
            },
            {
                "segment": "Civic Exchange",
                "customers": ("government agencies", "large enterprises"),
                "channels": ("system integrators", "online"),
                "offerings": (
                    ("Dovetail Civic", ("service fees",)),
                    ("Dovetail Relay", ("consumption-based", "service fees")),
                ),
            },
        ),
    },
)


def _build_fixture() -> tuple[FixtureSpec, list[dict[str, str]], list[dict[str, str]]]:
    fixture_id = "FX32_heldout_evaluation_mesh_v1"
    graph_id = "fx32_heldout_evaluation_mesh_v1"

    nodes: list[FixtureNodeSpec] = []
    edges: list[FixtureEdgeSpec] = []
    qf19_bindings: list[dict[str, str]] = []
    qf31_bindings: list[dict[str, str]] = []

    place_ids: dict[str, str] = {}
    customer_ids: dict[str, str] = {}
    channel_ids: dict[str, str] = {}
    revenue_ids: dict[str, str] = {}

    for company_index, company_spec in enumerate(COMPANY_SPECS, start=1):
        company_name = str(company_spec["company"])
        company_id = f"company_{company_index}"
        nodes.append(_company(company_id, company_name))

        for place_name in company_spec["places"]:  # type: ignore[index]
            place_id = place_ids.setdefault(place_name, f"place_{len(place_ids) + 1}")
            if not any(node.node_id == place_id for node in nodes):
                nodes.append(_place(place_id, place_name))
            edges.append(_edge(company_id, "OPERATES_IN", place_id))

        for revenue_model in company_spec["revenue_models"]:  # type: ignore[index]
            revenue_id = revenue_ids.setdefault(revenue_model, f"revenue_{len(revenue_ids) + 1}")
            if not any(node.node_id == revenue_id for node in nodes):
                nodes.append(_revenue_model(revenue_id, revenue_model))

        for segment_index, segment_spec in enumerate(company_spec["segments"], start=1):  # type: ignore[index]
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
                    revenue_id = revenue_ids[revenue_model]
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

        for place_name in company_spec["places"]:  # type: ignore[index]
            for revenue_model in company_spec["revenue_models"]:  # type: ignore[index]
                qf31_bindings.append(
                    {
                        "company": company_name,
                        "place": place_name,
                        "revenue_model": revenue_model,
                        "slug": _slug(f"{company_name}_{place_name}_{revenue_model}"),
                    }
                )

    if len(qf19_bindings) != 16:
        raise ValueError(f"Expected 16 held-out QF19 bindings, found {len(qf19_bindings)}")
    if len(qf31_bindings) != 16:
        raise ValueError(f"Expected 16 held-out QF31 bindings, found {len(qf31_bindings)}")

    fixture = FixtureSpec(
        fixture_id=fixture_id,
        graph_id=graph_id,
        graph_purpose=(
            "Held-out evaluation mesh with fresh companies, segments, places, and company-scoped "
            "monetization facts for QF19 and QF31 style queries."
        ),
        covered_families=("QF19", "QF31"),
        nodes=nodes,
        edges=edges,
        invariants_satisfied=(
            "BusinessSegment and Offering nodes carry company_name so company-scoped joins stay explicit.",
            "Each segment exposes two answerable QF19-style customer, channel, and offering combinations.",
            "Each company has two unique places and two monetization models for QF31-style geography + revenue lookups.",
        ),
        authoring_notes=(
            "This fixture is dedicated to held-out evaluation only.",
            "The names are intentionally fresh so none of the train fixtures need to be reused here.",
            "The graph keeps the same ontology and Cypher idioms as the training corpus while avoiding train leakage.",
        ),
    )

    return fixture, qf19_bindings, qf31_bindings


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
        split="heldout_test",
        paraphrases=_heldout_paraphrases(question),
    )


def build_spec() -> DatasetSpec:
    fixture, qf19_bindings, qf31_bindings = _build_fixture()

    qf19_cypher = (
        "MATCH (c:Company)-[:HAS_SEGMENT]->(s:BusinessSegment)-[:SERVES]->"
        "(ct:CustomerType {name: $customer_type}) "
        "MATCH (s)-[:SELLS_THROUGH]->(ch:Channel {name: $channel}) "
        "MATCH (s)-[:OFFERS]->(o:Offering {name: $offering}) "
        "WHERE s.company_name = c.name AND o.company_name = c.name "
        "RETURN DISTINCT c.name AS company, s.name AS segment ORDER BY company, segment"
    )
    qf31_cypher = (
        "MATCH (company:Company)-[:OPERATES_IN]->(place:Place {name: $place}) "
        "MATCH (company)-[:HAS_SEGMENT]->(:BusinessSegment {company_name: company.name})-[:OFFERS]->"
        "(root:Offering {company_name: company.name}) "
        "MATCH (root)-[:OFFERS*0..]->(o:Offering {company_name: company.name})-[:MONETIZES_VIA]->"
        "(r:RevenueModel {name: $revenue_model}) "
        "RETURN DISTINCT company.name AS company ORDER BY company"
    )

    source_examples: list[SourceExampleSpec] = []

    for binding in qf19_bindings:
        question = (
            "Which company segments serve {customer_type}, sell through {channel}, and offer {offering}?"
        ).format(**binding)
        source_examples.append(
            _example(
                example_id=f"heldout_qf19_segment_customer_channel_offering_list__{binding['slug']}",
                intent_id=f"heldout_qf19_segment_customer_channel_offering_list__{binding['slug']}",
                family_id="QF19",
                fixture_id=fixture.fixture_id,
                graph_id=fixture.graph_id,
                binding_id=f"qf19_{binding['slug']}",
                question=question,
                gold_cypher=qf19_cypher,
                params={
                    "customer_type": binding["customer_type"],
                    "channel": binding["channel"],
                    "offering": binding["offering"],
                },
                result_shape=[_col("company", "string"), _col("segment", "string")],
            )
        )

    for binding in qf31_bindings:
        question = "Which companies operate in {place} and monetize via {revenue_model}?".format(**binding)
        source_examples.append(
            _example(
                example_id=f"heldout_qf31_company_place_revenue_list__{binding['slug']}",
                intent_id=f"heldout_qf31_company_place_revenue_list__{binding['slug']}",
                family_id="QF31",
                fixture_id=fixture.fixture_id,
                graph_id=fixture.graph_id,
                binding_id=f"qf31_{binding['slug']}",
                question=question,
                gold_cypher=qf31_cypher,
                params={
                    "place": binding["place"],
                    "revenue_model": binding["revenue_model"],
                },
                result_shape=[_col("company", "string")],
            )
        )

    if len(source_examples) != 32:
        raise ValueError(f"Expected 32 held-out source examples, found {len(source_examples)}")
    for example in source_examples:
        if len(example.paraphrases) != 15:
            raise ValueError(f"Example {example.example_id} must have exactly 15 paraphrases")

    return DatasetSpec(fixtures=[fixture], source_examples=source_examples)
