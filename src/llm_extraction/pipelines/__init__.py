from typing import TYPE_CHECKING, Any

from llm_extraction.models import ExtractionError
from llm_extraction.pipelines.analyst.runner import AnalystPipelineRunner
from llm_extraction.pipelines.canonical.runner import CanonicalPipelineRunner

if TYPE_CHECKING:
    from llm.extractor import LLMExtractor


KNOWN_PIPELINES: tuple[str, ...] = ("canonical", "analyst")
IMPLEMENTED_PIPELINES: tuple[str, ...] = ("canonical", "analyst")
PASS1_SUPPORTED_PIPELINES: frozenset[str] = frozenset({"canonical"})
PIPELINE_STAGE_COUNTS: dict[str, dict[str, int]] = {
    "canonical": {"full": 10, "pass1": 4},
    "analyst": {"full": 7},
}


def known_pipeline_names() -> tuple[str, ...]:
    return KNOWN_PIPELINES


def implemented_pipeline_names() -> tuple[str, ...]:
    return IMPLEMENTED_PIPELINES


def pipeline_supports_stop_after_pass1(pipeline: str) -> bool:
    if pipeline not in KNOWN_PIPELINES:
        raise ExtractionError(f"Unknown extraction pipeline: {pipeline}")
    return pipeline in PASS1_SUPPORTED_PIPELINES


def pipeline_stage_count(pipeline: str, *, stop_after_pass1: bool = False) -> int:
    if pipeline not in KNOWN_PIPELINES:
        raise ExtractionError(f"Unknown extraction pipeline: {pipeline}")
    if stop_after_pass1:
        if not pipeline_supports_stop_after_pass1(pipeline):
            raise ExtractionError(f"Pipeline '{pipeline}' does not support stop_after_pass1.")
        return PIPELINE_STAGE_COUNTS[pipeline]["pass1"]
    return PIPELINE_STAGE_COUNTS[pipeline]["full"]


def build_pipeline_runner(pipeline: str, extractor: "LLMExtractor") -> Any:
    if pipeline == "canonical":
        return CanonicalPipelineRunner(extractor)
    if pipeline == "analyst":
        return AnalystPipelineRunner(extractor)
    raise ExtractionError(f"Unknown extraction pipeline: {pipeline}")


def run_extraction_pipeline(
    *,
    pipeline: str,
    extractor: "LLMExtractor",
    full_text: str,
    company_name: str | None = None,
    max_retries: int = 2,
    stop_after_pass1: bool = False,
) -> Any:
    runner = build_pipeline_runner(pipeline, extractor)
    return runner.run(
        full_text=full_text,
        company_name=company_name,
        max_retries=max_retries,
        stop_after_pass1=stop_after_pass1,
    )


__all__ = [
    "KNOWN_PIPELINES",
    "IMPLEMENTED_PIPELINES",
    "build_pipeline_runner",
    "implemented_pipeline_names",
    "known_pipeline_names",
    "pipeline_stage_count",
    "pipeline_supports_stop_after_pass1",
    "run_extraction_pipeline",
]
