from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llm_extraction.models import ExtractionError, KnowledgeGraphExtraction, ZeroShotPipelineResult
from llm_extraction.pipelines.zero_shot.prompts import zero_shot_extraction_prompt

if TYPE_CHECKING:
    from llm.extractor import LLMExtractor


@dataclass
class ZeroShotPipelineState:
    zero_shot_extraction: KnowledgeGraphExtraction = field(default_factory=KnowledgeGraphExtraction)
    zero_shot_audit: dict[str, Any] = field(default_factory=dict)
    raw_zero_shot_response: str | None = None
    zero_shot_attempts_used: int = 0

    def to_result(self, *, success: bool, error: str | None = None) -> ZeroShotPipelineResult:
        return ZeroShotPipelineResult(
            success=success,
            zero_shot_extraction=self.zero_shot_extraction,
            zero_shot_audit=self.zero_shot_audit,
            raw_zero_shot_response=self.raw_zero_shot_response,
            zero_shot_attempts_used=self.zero_shot_attempts_used,
            final_extraction=self.zero_shot_extraction,
            error=error,
        )


class ZeroShotPipelineRunner:
    def __init__(self, extractor: "LLMExtractor"):
        self.extractor = extractor

    def run(
        self,
        *,
        full_text: str,
        company_name: str | None = None,
        max_retries: int = 2,
        stop_after_pass1: bool = False,
    ) -> ZeroShotPipelineResult:
        if stop_after_pass1:
            raise ExtractionError(
                "Pipeline 'zero-shot' does not support stop_after_pass1 because it uses one full-graph extraction prompt."
            )

        state = ZeroShotPipelineState()
        self.extractor._emit_progress(
            "stage_start",
            index=2,
            title="Zero-shot extraction",
            details=[("prompt", "single-pass ontology extraction")],
        )

        try:
            (
                parsed_payload,
                state.raw_zero_shot_response,
                state.zero_shot_attempts_used,
                state.zero_shot_audit,
            ) = self.extractor.generate_structured_output(
                messages=[
                    {
                        "role": "user",
                        "content": zero_shot_extraction_prompt(full_text, company_name),
                    }
                ],
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Zero-shot extraction failed.","triples":[]}',
                max_retries=max_retries,
            )
        except ExtractionError as exc:
            self.extractor._emit_progress("stage_failed", error=str(exc))
            return state.to_result(success=False, error=str(exc))

        state.zero_shot_extraction = parsed_payload
        self.extractor._emit_progress(
            "stage_complete",
            details=[("result", f"{len(state.zero_shot_extraction.triples)} triples")],
        )
        return state.to_result(success=True)
