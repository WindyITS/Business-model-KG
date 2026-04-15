from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


NodeType = Literal["Company", "BusinessSegment", "Offering", "CustomerType", "Channel", "Place", "RevenueModel"]
RelationType = Literal[
    "HAS_SEGMENT",
    "OFFERS",
    "SERVES",
    "OPERATES_IN",
    "SELLS_THROUGH",
    "PARTNERS_WITH",
    "MONETIZES_VIA",
]


class Triple(BaseModel):
    subject: str
    subject_type: NodeType
    relation: RelationType
    object: str
    object_type: NodeType


class KnowledgeGraphExtraction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    extraction_notes: str = Field(
        default="",
        validation_alias=AliasChoices("extraction_notes", "chain_of_thought_reasoning"),
        serialization_alias="extraction_notes",
    )
    triples: list[Triple] = Field(default_factory=list)


class CanonicalPipelineResult(BaseModel):
    success: bool
    skeleton_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    pass2_channels_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    pass2_revenue_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    pass2_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    pass3_serves_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    pass4_corporate_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    pre_reflection_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    rule_reflection_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    final_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    skeleton_audit: dict[str, Any] = Field(default_factory=dict)
    pass2_channels_audit: dict[str, Any] = Field(default_factory=dict)
    pass2_revenue_audit: dict[str, Any] = Field(default_factory=dict)
    pass2_audit: dict[str, Any] = Field(default_factory=dict)
    pass3_serves_audit: dict[str, Any] = Field(default_factory=dict)
    pass4_corporate_audit: dict[str, Any] = Field(default_factory=dict)
    pre_reflection_audit: dict[str, Any] = Field(default_factory=dict)
    rule_reflection_audit: dict[str, Any] = Field(default_factory=dict)
    final_reflection_audit: dict[str, Any] = Field(default_factory=dict)
    raw_skeleton_response: str | None = None
    raw_pass2_channels_response: str | None = None
    raw_pass2_revenue_response: str | None = None
    raw_pass2_response: str | None = None
    raw_pass3_serves_response: str | None = None
    raw_pass4_corporate_response: str | None = None
    raw_rule_reflection_response: str | None = None
    raw_final_reflection_response: str | None = None
    skeleton_attempts_used: int = 0
    pass2_channels_attempts_used: int = 0
    pass2_revenue_attempts_used: int = 0
    pass2_attempts_used: int = 0
    pass3_serves_attempts_used: int = 0
    pass4_corporate_attempts_used: int = 0
    rule_reflection_attempts_used: int = 0
    final_reflection_attempts_used: int = 0
    error: str | None = None


class ExtractionError(RuntimeError):
    pass
