from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llm_extraction.audit import aggregate_extraction_audits, audit_knowledge_graph_payload
from llm_extraction.models import CanonicalPipelineResult, ExtractionError, KnowledgeGraphExtraction
from llm_extraction.pipelines.canonical.prompts import (
    canonical_final_reflection_prompt,
    canonical_pass1_prompt,
    canonical_pass2_channels_prompt,
    canonical_pass2_revenue_prompt,
    canonical_pass3_serves_prompt,
    canonical_pass4_corporate_prompt,
    canonical_pipeline_system_prompt,
    canonical_reflection_system_prompt,
    canonical_rule_reflection_prompt,
    canonical_rule_reflection_system_prompt,
)

if TYPE_CHECKING:
    from llm_extractor import LLMExtractor


@dataclass
class CanonicalPipelineState:
    skeleton_extraction: KnowledgeGraphExtraction = field(default_factory=KnowledgeGraphExtraction)
    pass2_channels_extraction: KnowledgeGraphExtraction = field(default_factory=KnowledgeGraphExtraction)
    pass2_revenue_extraction: KnowledgeGraphExtraction = field(default_factory=KnowledgeGraphExtraction)
    pass2_extraction: KnowledgeGraphExtraction = field(default_factory=KnowledgeGraphExtraction)
    pass3_serves_extraction: KnowledgeGraphExtraction = field(default_factory=KnowledgeGraphExtraction)
    pass4_corporate_extraction: KnowledgeGraphExtraction = field(default_factory=KnowledgeGraphExtraction)
    pre_reflection_extraction: KnowledgeGraphExtraction = field(default_factory=KnowledgeGraphExtraction)
    rule_reflection_extraction: KnowledgeGraphExtraction = field(default_factory=KnowledgeGraphExtraction)
    final_extraction: KnowledgeGraphExtraction = field(default_factory=KnowledgeGraphExtraction)
    skeleton_audit: dict[str, Any] = field(default_factory=dict)
    pass2_channels_audit: dict[str, Any] = field(default_factory=dict)
    pass2_revenue_audit: dict[str, Any] = field(default_factory=dict)
    pass2_audit: dict[str, Any] = field(default_factory=dict)
    pass3_serves_audit: dict[str, Any] = field(default_factory=dict)
    pass4_corporate_audit: dict[str, Any] = field(default_factory=dict)
    pre_reflection_audit: dict[str, Any] = field(default_factory=dict)
    rule_reflection_audit: dict[str, Any] = field(default_factory=dict)
    final_reflection_audit: dict[str, Any] = field(default_factory=dict)
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
    pass3_serves_attempts_used: int = 0
    pass4_corporate_attempts_used: int = 0
    rule_reflection_attempts_used: int = 0
    final_reflection_attempts_used: int = 0

    def to_result(self, *, success: bool, error: str | None = None) -> CanonicalPipelineResult:
        return CanonicalPipelineResult(
            success=success,
            skeleton_extraction=self.skeleton_extraction,
            pass2_channels_extraction=self.pass2_channels_extraction,
            pass2_revenue_extraction=self.pass2_revenue_extraction,
            pass2_extraction=self.pass2_extraction,
            pass3_serves_extraction=self.pass3_serves_extraction,
            pass4_corporate_extraction=self.pass4_corporate_extraction,
            pre_reflection_extraction=self.pre_reflection_extraction,
            rule_reflection_extraction=self.rule_reflection_extraction,
            final_extraction=self.final_extraction,
            skeleton_audit=self.skeleton_audit,
            pass2_channels_audit=self.pass2_channels_audit,
            pass2_revenue_audit=self.pass2_revenue_audit,
            pass2_audit=self.pass2_audit,
            pass3_serves_audit=self.pass3_serves_audit,
            pass4_corporate_audit=self.pass4_corporate_audit,
            pre_reflection_audit=self.pre_reflection_audit,
            rule_reflection_audit=self.rule_reflection_audit,
            final_reflection_audit=self.final_reflection_audit,
            raw_skeleton_response=self.raw_skeleton_response,
            raw_pass2_channels_response=self.raw_pass2_channels_response,
            raw_pass2_revenue_response=self.raw_pass2_revenue_response,
            raw_pass2_response=self.raw_pass2_response,
            raw_pass3_serves_response=self.raw_pass3_serves_response,
            raw_pass4_corporate_response=self.raw_pass4_corporate_response,
            raw_rule_reflection_response=self.raw_rule_reflection_response,
            raw_final_reflection_response=self.raw_final_reflection_response,
            skeleton_attempts_used=self.skeleton_attempts_used,
            pass2_channels_attempts_used=self.pass2_channels_attempts_used,
            pass2_revenue_attempts_used=self.pass2_revenue_attempts_used,
            pass2_attempts_used=self.pass2_channels_attempts_used + self.pass2_revenue_attempts_used,
            pass3_serves_attempts_used=self.pass3_serves_attempts_used,
            pass4_corporate_attempts_used=self.pass4_corporate_attempts_used,
            rule_reflection_attempts_used=self.rule_reflection_attempts_used,
            final_reflection_attempts_used=self.final_reflection_attempts_used,
            error=error,
        )


class CanonicalPipelineRunner:
    def __init__(self, extractor: "LLMExtractor"):
        self.extractor = extractor

    def _run_structured_stage(
        self,
        *,
        index: int,
        title: str,
        user_prompt: str,
        system_prompt: str,
        fallback_payload: str,
        max_retries: int,
        extracts: str | None = None,
    ) -> tuple[KnowledgeGraphExtraction, str | None, int, dict[str, Any]]:
        self.extractor._emit_progress(
            "stage_start",
            index=index,
            title=title,
            extracts=extracts,
        )
        extraction, raw_response, attempts_used, audit = self.extractor._call_structured_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            schema_name="KnowledgeGraphExtraction",
            schema_model=KnowledgeGraphExtraction,
            fallback_payload=fallback_payload,
            max_retries=max_retries,
            ontology_version="canonical",
        )
        self.extractor._emit_progress("stage_complete", details=[("result", f"{len(extraction.triples)} triples")])
        return extraction, raw_response, attempts_used, audit

    def run(
        self,
        *,
        full_text: str,
        company_name: str | None = None,
        max_retries: int = 2,
        stop_after_pass1: bool = False,
    ) -> CanonicalPipelineResult:
        state = CanonicalPipelineState()
        pipeline_system_prompt = canonical_pipeline_system_prompt(full_text)

        try:
            (
                state.skeleton_extraction,
                state.raw_skeleton_response,
                state.skeleton_attempts_used,
                state.skeleton_audit,
            ) = self._run_structured_stage(
                index=2,
                title="Pass 1 - Structural skeleton",
                extracts="HAS_SEGMENT, OFFERS",
                user_prompt=canonical_pass1_prompt(company_name),
                system_prompt=pipeline_system_prompt,
                fallback_payload='{"extraction_notes":"Truncated skeleton extraction.","triples":[]}',
                max_retries=max_retries,
            )
        except ExtractionError as exc:
            self.extractor._emit_progress("stage_failed", error=str(exc))
            return state.to_result(success=False, error=str(exc))

        if stop_after_pass1:
            return state.to_result(success=True)

        skeleton_graph = self.extractor._compact_json(state.skeleton_extraction.model_dump())

        try:
            (
                state.pass2_channels_extraction,
                state.raw_pass2_channels_response,
                state.pass2_channels_attempts_used,
                state.pass2_channels_audit,
            ) = self._run_structured_stage(
                index=3,
                title="Pass 2A - Channels",
                extracts="SELLS_THROUGH",
                user_prompt=canonical_pass2_channels_prompt(skeleton_graph),
                system_prompt=pipeline_system_prompt,
                fallback_payload='{"extraction_notes":"Truncated pass-2 channels extraction.","triples":[]}',
                max_retries=max_retries,
            )
        except ExtractionError as exc:
            self.extractor._emit_progress("stage_failed", error=str(exc))
            return state.to_result(success=False, error=str(exc))

        state.pass2_extraction = self.extractor._merge_relation_subset_into_base(
            state.skeleton_extraction,
            state.pass2_channels_extraction,
            allowed_relations={"SELLS_THROUGH"},
        )
        state.pass2_audit = aggregate_extraction_audits([state.pass2_channels_audit])
        state.raw_pass2_response = state.raw_pass2_channels_response

        try:
            (
                state.pass2_revenue_extraction,
                state.raw_pass2_revenue_response,
                state.pass2_revenue_attempts_used,
                state.pass2_revenue_audit,
            ) = self._run_structured_stage(
                index=4,
                title="Pass 2B - Revenue models",
                extracts="MONETIZES_VIA",
                user_prompt=canonical_pass2_revenue_prompt(skeleton_graph),
                system_prompt=pipeline_system_prompt,
                fallback_payload='{"extraction_notes":"Truncated pass-2 revenue extraction.","triples":[]}',
                max_retries=max_retries,
            )
        except ExtractionError as exc:
            self.extractor._emit_progress("stage_failed", error=str(exc))
            return state.to_result(success=False, error=str(exc))

        pass2_effective_extraction = self.extractor._merge_relation_subset_into_base(
            state.pass2_extraction,
            state.pass2_revenue_extraction,
            allowed_relations={"MONETIZES_VIA"},
        )
        state.pass2_extraction = KnowledgeGraphExtraction(
            extraction_notes="Merged channel and revenue-model passes.",
            triples=pass2_effective_extraction.triples,
        )
        state.pass2_audit = aggregate_extraction_audits([state.pass2_channels_audit, state.pass2_revenue_audit])
        state.raw_pass2_response = state.raw_pass2_revenue_response

        try:
            (
                state.pass3_serves_extraction,
                state.raw_pass3_serves_response,
                state.pass3_serves_attempts_used,
                state.pass3_serves_audit,
            ) = self._run_structured_stage(
                index=5,
                title="Pass 3 - Customer types",
                extracts="SERVES",
                user_prompt=canonical_pass3_serves_prompt(company_name, skeleton_graph),
                system_prompt=pipeline_system_prompt,
                fallback_payload='{"extraction_notes":"Truncated serves extraction.","triples":[]}',
                max_retries=max_retries,
            )
        except ExtractionError as exc:
            self.extractor._emit_progress("stage_failed", error=str(exc))
            return state.to_result(success=False, error=str(exc))

        pass3_effective_extraction = self.extractor._merge_serves_into_base(
            pass2_effective_extraction,
            state.pass3_serves_extraction,
        )
        state.pre_reflection_extraction = KnowledgeGraphExtraction(
            extraction_notes="Merged structure, channels, revenue models, and customer types before corporate-shell extraction.",
            triples=pass3_effective_extraction.triples,
        )
        state.pre_reflection_audit = state.pass3_serves_audit

        try:
            (
                state.pass4_corporate_extraction,
                state.raw_pass4_corporate_response,
                state.pass4_corporate_attempts_used,
                state.pass4_corporate_audit,
            ) = self._run_structured_stage(
                index=6,
                title="Pass 4 - Corporate shell facts",
                extracts="OPERATES_IN, PARTNERS_WITH",
                user_prompt=canonical_pass4_corporate_prompt(),
                system_prompt=pipeline_system_prompt,
                fallback_payload='{"extraction_notes":"Truncated corporate-shell extraction.","triples":[]}',
                max_retries=max_retries,
            )
        except ExtractionError as exc:
            self.extractor._emit_progress("stage_failed", error=str(exc))
            return state.to_result(success=False, error=str(exc))

        pass4_effective_extraction = self.extractor._merge_relation_subset_into_base(
            pass3_effective_extraction,
            state.pass4_corporate_extraction,
            allowed_relations={"OPERATES_IN", "PARTNERS_WITH"},
        )
        state.pre_reflection_extraction = KnowledgeGraphExtraction(
            extraction_notes="Merged structure, channels, revenue models, customer types, and corporate-shell facts before final reflection.",
            triples=pass4_effective_extraction.triples,
        )
        _, state.pre_reflection_audit = audit_knowledge_graph_payload(
            state.pre_reflection_extraction.model_dump(mode="json"),
            ontology_version="canonical",
        )

        self.extractor._emit_progress(
            "stage_start",
            index=7,
            title="Reflection 1 - Ontology compliance",
            details=[("triples in", self.extractor._triple_count(state.pre_reflection_extraction))],
        )
        (
            state.rule_reflection_extraction,
            state.raw_rule_reflection_response,
            state.rule_reflection_attempts_used,
            state.rule_reflection_audit,
        ) = self.extractor.reflect_extraction(
            full_text=full_text,
            current_extraction=state.pre_reflection_extraction,
            company_name=company_name,
            max_retries=max_retries,
            strict=False,
            system_prompt=canonical_rule_reflection_system_prompt(),
            user_prompt=canonical_rule_reflection_prompt(
                company_name,
                self.extractor._compact_json(state.pre_reflection_extraction.model_dump()),
            ),
            stage_label="Rule reflection",
            ontology_version="canonical",
        )
        self.extractor._emit_progress(
            "stage_complete",
            details=self.extractor._triple_delta_details(
                state.pre_reflection_extraction,
                state.rule_reflection_extraction,
            ),
        )

        self.extractor._emit_progress(
            "stage_start",
            index=8,
            title="Reflection 2 - Filing reconciliation",
            details=[("triples in", self.extractor._triple_count(state.rule_reflection_extraction))],
        )
        (
            state.final_extraction,
            state.raw_final_reflection_response,
            state.final_reflection_attempts_used,
            state.final_reflection_audit,
        ) = self.extractor.reflect_extraction(
            full_text=full_text,
            current_extraction=state.rule_reflection_extraction,
            company_name=company_name,
            max_retries=max_retries,
            strict=False,
            system_prompt=canonical_reflection_system_prompt(full_text),
            user_prompt=canonical_final_reflection_prompt(
                company_name,
                self.extractor._compact_json(state.rule_reflection_extraction.model_dump()),
            ),
            stage_label="Filing reflection",
            ontology_version="canonical",
        )
        self.extractor._emit_progress(
            "stage_complete",
            details=self.extractor._triple_delta_details(
                state.rule_reflection_extraction,
                state.final_extraction,
            ),
        )

        return state.to_result(success=True)
