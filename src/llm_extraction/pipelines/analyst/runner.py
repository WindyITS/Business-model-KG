from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llm_extraction.models import (
    AnalystBusinessModelMemo,
    AnalystPipelineResult,
    ExtractionError,
    KnowledgeGraphExtraction,
)
from llm_extraction.pipelines.analyst.prompts import (
    analyst_graph_compilation_prompt,
    analyst_graph_critique_prompt,
    analyst_memo_augmentation_prompt,
    analyst_memo_foundation_prompt,
    analyst_pipeline_system_prompt,
)

if TYPE_CHECKING:
    from llm.extractor import LLMExtractor


@dataclass
class AnalystPipelineState:
    foundation_memo: AnalystBusinessModelMemo = field(default_factory=AnalystBusinessModelMemo)
    augmented_memo: AnalystBusinessModelMemo = field(default_factory=AnalystBusinessModelMemo)
    compiled_graph_extraction: KnowledgeGraphExtraction = field(default_factory=KnowledgeGraphExtraction)
    final_extraction: KnowledgeGraphExtraction = field(default_factory=KnowledgeGraphExtraction)
    foundation_memo_audit: dict[str, Any] = field(default_factory=dict)
    augmented_memo_audit: dict[str, Any] = field(default_factory=dict)
    compiled_graph_audit: dict[str, Any] = field(default_factory=dict)
    critique_audit: dict[str, Any] = field(default_factory=dict)
    raw_foundation_memo_response: str | None = None
    raw_augmented_memo_response: str | None = None
    raw_compiled_graph_response: str | None = None
    raw_critique_response: str | None = None
    foundation_memo_attempts_used: int = 0
    augmented_memo_attempts_used: int = 0
    compiled_graph_attempts_used: int = 0
    critique_attempts_used: int = 0

    def to_result(self, *, success: bool, error: str | None = None) -> AnalystPipelineResult:
        return AnalystPipelineResult(
            success=success,
            foundation_memo=self.foundation_memo,
            augmented_memo=self.augmented_memo,
            compiled_graph_extraction=self.compiled_graph_extraction,
            final_extraction=self.final_extraction,
            foundation_memo_audit=self.foundation_memo_audit,
            augmented_memo_audit=self.augmented_memo_audit,
            compiled_graph_audit=self.compiled_graph_audit,
            critique_audit=self.critique_audit,
            raw_foundation_memo_response=self.raw_foundation_memo_response,
            raw_augmented_memo_response=self.raw_augmented_memo_response,
            raw_compiled_graph_response=self.raw_compiled_graph_response,
            raw_critique_response=self.raw_critique_response,
            foundation_memo_attempts_used=self.foundation_memo_attempts_used,
            augmented_memo_attempts_used=self.augmented_memo_attempts_used,
            compiled_graph_attempts_used=self.compiled_graph_attempts_used,
            critique_attempts_used=self.critique_attempts_used,
            error=error,
        )


class AnalystPipelineRunner:
    def __init__(self, extractor: "LLMExtractor"):
        self.extractor = extractor

    def _stage_complete_details(self, payload: Any) -> list[tuple[str, Any]]:
        if isinstance(payload, AnalystBusinessModelMemo):
            return [
                ("memo chars", len(payload.content)),
                ("memo lines", len(payload.content.splitlines())),
            ]
        if isinstance(payload, KnowledgeGraphExtraction):
            return [("result", f"{len(payload.triples)} triples")]
        return []

    def _run_text_stage(
        self,
        *,
        index: int,
        title: str,
        user_prompt: str,
        system_prompt: str,
        max_retries: int,
        details: list[tuple[str, Any]] | None = None,
    ) -> tuple[AnalystBusinessModelMemo, str, int, dict[str, Any]]:
        self.extractor._emit_progress(
            "stage_start",
            index=index,
            title=title,
            details=details,
        )
        memo_text, attempts_used, audit = self.extractor._call_text_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_retries=max_retries,
        )
        memo = AnalystBusinessModelMemo(content=memo_text)
        self.extractor._emit_progress("stage_complete", details=self._stage_complete_details(memo))
        return memo, memo_text, attempts_used, audit

    def _run_structured_stage(
        self,
        *,
        index: int,
        title: str,
        user_prompt: str,
        system_prompt: str,
        schema_name: str,
        schema_model: type[Any],
        fallback_payload: str,
        max_retries: int,
        details: list[tuple[str, Any]] | None = None,
    ) -> tuple[Any, str | None, int, dict[str, Any]]:
        self.extractor._emit_progress(
            "stage_start",
            index=index,
            title=title,
            details=details,
        )
        parsed_payload, raw_response, attempts_used, audit = self.extractor._call_structured_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            schema_name=schema_name,
            schema_model=schema_model,
            fallback_payload=fallback_payload,
            max_retries=max_retries,
            ontology_version="canonical",
        )
        self.extractor._emit_progress("stage_complete", details=self._stage_complete_details(parsed_payload))
        return parsed_payload, raw_response, attempts_used, audit

    def run(
        self,
        *,
        full_text: str,
        company_name: str | None = None,
        max_retries: int = 2,
        stop_after_pass1: bool = False,
    ) -> AnalystPipelineResult:
        if stop_after_pass1:
            raise ExtractionError(
                "Pipeline 'analyst' does not support stop_after_pass1 because pass 1 produces an analyst memo, not a loadable graph."
            )

        state = AnalystPipelineState()
        pipeline_system_prompt = analyst_pipeline_system_prompt(full_text)

        try:
            (
                state.foundation_memo,
                state.raw_foundation_memo_response,
                state.foundation_memo_attempts_used,
                state.foundation_memo_audit,
            ) = self._run_text_stage(
                index=2,
                title="Analyst memo 1 - Core structure",
                user_prompt=analyst_memo_foundation_prompt(company_name),
                system_prompt=pipeline_system_prompt,
                max_retries=max_retries,
                details=[("artifact", "AnalystBusinessModelMemo")],
            )
        except ExtractionError as exc:
            self.extractor._emit_progress("stage_failed", error=str(exc))
            return state.to_result(success=False, error=str(exc))

        foundation_memo_text = state.foundation_memo.content

        try:
            (
                state.augmented_memo,
                state.raw_augmented_memo_response,
                state.augmented_memo_attempts_used,
                state.augmented_memo_audit,
            ) = self._run_text_stage(
                index=3,
                title="Analyst memo 2 - Detail augmentation",
                user_prompt=analyst_memo_augmentation_prompt(company_name, foundation_memo_text),
                system_prompt=pipeline_system_prompt,
                max_retries=max_retries,
                details=[("artifact", "AnalystBusinessModelMemo")],
            )
        except ExtractionError as exc:
            self.extractor._emit_progress("stage_failed", error=str(exc))
            return state.to_result(success=False, error=str(exc))

        augmented_memo_text = state.augmented_memo.content

        try:
            (
                state.compiled_graph_extraction,
                state.raw_compiled_graph_response,
                state.compiled_graph_attempts_used,
                state.compiled_graph_audit,
            ) = self._run_structured_stage(
                index=4,
                title="Graph compilation",
                user_prompt=analyst_graph_compilation_prompt(company_name, augmented_memo_text),
                system_prompt=pipeline_system_prompt,
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Analyst graph compilation failed.","triples":[]}',
                max_retries=max_retries,
                details=[("input", "AnalystBusinessModelMemo")],
            )
        except ExtractionError as exc:
            self.extractor._emit_progress("stage_failed", error=str(exc))
            return state.to_result(success=False, error=str(exc))

        self.extractor._emit_progress(
            "stage_start",
            index=5,
            title="Critique - Overreach review",
            details=[("triples in", self.extractor._triple_count(state.compiled_graph_extraction))],
        )
        (
            state.final_extraction,
            state.raw_critique_response,
            state.critique_attempts_used,
            state.critique_audit,
        ) = self.extractor.reflect_extraction(
            full_text=full_text,
            current_extraction=state.compiled_graph_extraction,
            company_name=company_name,
            max_retries=max_retries,
            strict=False,
            system_prompt=pipeline_system_prompt,
            user_prompt=analyst_graph_critique_prompt(
                company_name,
                augmented_memo_text,
                self.extractor._compact_json(state.compiled_graph_extraction.model_dump(mode="json")),
            ),
            stage_label="Analyst critique",
            ontology_version="canonical",
        )
        self.extractor._emit_progress(
            "stage_complete",
            details=self.extractor._triple_delta_details(
                state.compiled_graph_extraction,
                state.final_extraction,
            ),
        )

        return state.to_result(success=True)
