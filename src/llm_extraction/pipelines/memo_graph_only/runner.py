from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llm_extraction.models import (
    AnalystBusinessModelMemo,
    ExtractionError,
    KnowledgeGraphExtraction,
    MemoGraphOnlyPipelineResult,
)
from llm_extraction.pipelines.memo_graph_only.prompts import (
    memo_graph_only_graph_compilation_prompt,
    memo_graph_only_graph_system_prompt,
    memo_graph_only_memo_foundation_prompt,
    memo_graph_only_pipeline_system_prompt,
)

if TYPE_CHECKING:
    from llm.extractor import LLMExtractor


@dataclass
class MemoGraphOnlyPipelineState:
    foundation_memo: AnalystBusinessModelMemo = field(default_factory=AnalystBusinessModelMemo)
    compiled_graph_extraction: KnowledgeGraphExtraction = field(default_factory=KnowledgeGraphExtraction)
    foundation_memo_audit: dict[str, Any] = field(default_factory=dict)
    compiled_graph_audit: dict[str, Any] = field(default_factory=dict)
    raw_foundation_memo_response: str | None = None
    raw_compiled_graph_response: str | None = None
    foundation_memo_attempts_used: int = 0
    compiled_graph_attempts_used: int = 0

    def to_result(self, *, success: bool, error: str | None = None) -> MemoGraphOnlyPipelineResult:
        return MemoGraphOnlyPipelineResult(
            success=success,
            foundation_memo=self.foundation_memo,
            compiled_graph_extraction=self.compiled_graph_extraction,
            final_extraction=self.compiled_graph_extraction,
            foundation_memo_audit=self.foundation_memo_audit,
            compiled_graph_audit=self.compiled_graph_audit,
            raw_foundation_memo_response=self.raw_foundation_memo_response,
            raw_compiled_graph_response=self.raw_compiled_graph_response,
            foundation_memo_attempts_used=self.foundation_memo_attempts_used,
            compiled_graph_attempts_used=self.compiled_graph_attempts_used,
            error=error,
        )


class MemoGraphOnlyPipelineRunner:
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

    def run(
        self,
        *,
        full_text: str,
        company_name: str | None = None,
        max_retries: int = 2,
        stop_after_pass1: bool = False,
    ) -> MemoGraphOnlyPipelineResult:
        if stop_after_pass1:
            raise ExtractionError(
                "Pipeline 'memo_graph_only' does not support stop_after_pass1 because the first stage produces a memo, not a loadable graph."
            )

        state = MemoGraphOnlyPipelineState()
        memo_system_prompt = memo_graph_only_pipeline_system_prompt(full_text)
        graph_system_prompt = memo_graph_only_graph_system_prompt()

        self.extractor._emit_progress(
            "stage_start",
            index=2,
            title="Memo graph-only memo - Core structure",
            details=[("artifact", "AnalystBusinessModelMemo")],
        )
        try:
            (
                memo_text,
                state.foundation_memo_attempts_used,
                state.foundation_memo_audit,
            ) = self.extractor._call_text_messages(
                messages=[
                    {"role": "system", "content": memo_system_prompt},
                    {"role": "user", "content": memo_graph_only_memo_foundation_prompt(company_name)},
                ],
                max_retries=max_retries,
            )
        except ExtractionError as exc:
            self.extractor._emit_progress("stage_failed", error=str(exc))
            return state.to_result(success=False, error=str(exc))

        state.raw_foundation_memo_response = memo_text
        state.foundation_memo = AnalystBusinessModelMemo(content=memo_text)
        self.extractor._emit_progress("stage_complete", details=self._stage_complete_details(state.foundation_memo))

        self.extractor._emit_progress(
            "stage_start",
            index=3,
            title="Memo graph-only graph compilation",
            details=[("input", "AnalystBusinessModelMemo")],
        )
        try:
            (
                state.compiled_graph_extraction,
                state.raw_compiled_graph_response,
                state.compiled_graph_attempts_used,
                state.compiled_graph_audit,
            ) = self.extractor._call_structured_messages(
                messages=[
                    {"role": "system", "content": graph_system_prompt},
                    {
                        "role": "user",
                        "content": memo_graph_only_graph_compilation_prompt(company_name, state.foundation_memo.content),
                    },
                ],
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Memo graph-only graph compilation failed.","triples":[]}',
                max_retries=max_retries,
            )
        except ExtractionError as exc:
            self.extractor._emit_progress("stage_failed", error=str(exc))
            return state.to_result(success=False, error=str(exc))

        self.extractor._emit_progress(
            "stage_complete",
            details=self._stage_complete_details(state.compiled_graph_extraction),
        )
        return state.to_result(success=True)
