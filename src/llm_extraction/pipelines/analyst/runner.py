from typing import TYPE_CHECKING

from llm_extraction.models import ExtractionError

if TYPE_CHECKING:
    from llm.extractor import LLMExtractor


class AnalystPipelineRunner:
    def __init__(self, extractor: "LLMExtractor"):
        self.extractor = extractor

    def run(
        self,
        *,
        full_text: str,
        company_name: str | None = None,
        max_retries: int = 2,
        stop_after_pass1: bool = False,
    ):
        raise ExtractionError(
            "Pipeline 'analyst' is not implemented yet. Use the scaffold under "
            "'src/llm_extraction/pipelines/analyst/' and 'prompts/analyst/' when you add it."
        )
