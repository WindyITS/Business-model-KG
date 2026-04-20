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


class ExtractionPipelineResult(BaseModel):
    success: bool
    final_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    error: str | None = None


class AnalystEvidence(BaseModel):
    explicit_support: list[str] = Field(default_factory=list)
    analyst_inference: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


class AnalystNamedClaim(BaseModel):
    name: str
    rationale: str = ""
    support: AnalystEvidence = Field(default_factory=AnalystEvidence)


class AnalystCanonicalLabelClaim(BaseModel):
    label: str
    rationale: str = ""
    support: AnalystEvidence = Field(default_factory=AnalystEvidence)


class AnalystSegment(BaseModel):
    name: str
    role_in_business_model: str = ""
    support: AnalystEvidence = Field(default_factory=AnalystEvidence)
    customer_types: list[AnalystCanonicalLabelClaim] = Field(default_factory=list)
    channels: list[AnalystCanonicalLabelClaim] = Field(default_factory=list)


class AnalystOffering(BaseModel):
    name: str
    role_in_business_model: str = ""
    support: AnalystEvidence = Field(default_factory=AnalystEvidence)
    segment_anchors: list[AnalystNamedClaim] = Field(default_factory=list)
    parent_offering: AnalystNamedClaim | None = None
    channels: list[AnalystCanonicalLabelClaim] = Field(default_factory=list)
    revenue_models: list[AnalystCanonicalLabelClaim] = Field(default_factory=list)


class AnalystCorporateScope(BaseModel):
    operating_geographies: list[AnalystNamedClaim] = Field(default_factory=list)
    key_partners: list[AnalystNamedClaim] = Field(default_factory=list)


class AnalystBusinessModelMemo(BaseModel):
    content: str = ""


class CanonicalPipelineResult(ExtractionPipelineResult):
    skeleton_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    pass2_channels_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    pass2_revenue_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    pass2_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    pass3_serves_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    pass4_corporate_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    pre_reflection_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    rule_reflection_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
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


class AnalystPipelineResult(ExtractionPipelineResult):
    foundation_memo: AnalystBusinessModelMemo = Field(default_factory=AnalystBusinessModelMemo)
    augmented_memo: AnalystBusinessModelMemo = Field(default_factory=AnalystBusinessModelMemo)
    compiled_graph_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    foundation_memo_audit: dict[str, Any] = Field(default_factory=dict)
    augmented_memo_audit: dict[str, Any] = Field(default_factory=dict)
    compiled_graph_audit: dict[str, Any] = Field(default_factory=dict)
    critique_audit: dict[str, Any] = Field(default_factory=dict)
    raw_foundation_memo_response: str | None = None
    raw_augmented_memo_response: str | None = None
    raw_compiled_graph_response: str | None = None
    raw_critique_response: str | None = None
    foundation_memo_attempts_used: int = 0
    augmented_memo_attempts_used: int = 0
    compiled_graph_attempts_used: int = 0
    critique_attempts_used: int = 0


class ZeroShotPipelineResult(ExtractionPipelineResult):
    zero_shot_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    zero_shot_audit: dict[str, Any] = Field(default_factory=dict)
    raw_zero_shot_response: str | None = None
    zero_shot_attempts_used: int = 0


class ExtractionError(RuntimeError):
    pass
