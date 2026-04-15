from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

JsonValue = Any
SplitName = str


@dataclass(frozen=True, slots=True)
class ResultColumnSpec:
    column: str
    type: str
    description: str | None = None


@dataclass(frozen=True, slots=True)
class FixtureNodeSpec:
    node_id: str
    label: str
    name: str
    properties: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FixtureEdgeSpec:
    source: str
    type: str
    target: str


@dataclass(frozen=True, slots=True)
class FixtureSpec:
    fixture_id: str
    graph_id: str
    graph_purpose: str
    covered_families: Sequence[str]
    nodes: Sequence[FixtureNodeSpec]
    edges: Sequence[FixtureEdgeSpec]
    invariants_satisfied: Sequence[str]
    authoring_notes: Sequence[str]


@dataclass(frozen=True, slots=True)
class SourceExampleSpec:
    example_id: str
    intent_id: str
    family_id: str
    fixture_id: str | None
    graph_id: str | None
    binding_id: str | None
    question_canonical: str
    gold_cypher: str | None
    params: Mapping[str, JsonValue]
    answerable: bool
    refusal_reason: str | None
    result_shape: Sequence[ResultColumnSpec] | None
    difficulty: str
    split: SplitName
    paraphrases: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    fixtures: Sequence[FixtureSpec]
    source_examples: Sequence[SourceExampleSpec]

