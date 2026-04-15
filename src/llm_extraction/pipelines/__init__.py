from typing import TYPE_CHECKING, Any

from llm_extraction.models import ExtractionError
from llm_extraction.pipelines.analyst.runner import AnalystPipelineRunner
from llm_extraction.pipelines.canonical.runner import CanonicalPipelineRunner

if TYPE_CHECKING:
    from llm.extractor import LLMExtractor


KNOWN_PIPELINES: tuple[str, ...] = ("canonical", "analyst")
IMPLEMENTED_PIPELINES: tuple[str, ...] = ("canonical",)


def known_pipeline_names() -> tuple[str, ...]:
    return KNOWN_PIPELINES


def implemented_pipeline_names() -> tuple[str, ...]:
    return IMPLEMENTED_PIPELINES


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
    "run_extraction_pipeline",
]
