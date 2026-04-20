import unittest
from unittest.mock import patch

from llm.extractor import LLMExtractor
from llm_extraction.models import ExtractionError, KnowledgeGraphExtraction, Triple
from llm_extraction.pipelines.zero_shot.runner import ZeroShotPipelineRunner
from runtime.main import PipelineConsole


class ZeroShotPipelineRunnerTests(unittest.TestCase):
    def test_zero_shot_runner_rejects_stop_after_pass1(self):
        extractor = LLMExtractor(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="local-model",
            provider="local",
            api_mode="chat_completions",
        )

        with self.assertRaises(ExtractionError) as ctx:
            ZeroShotPipelineRunner(extractor).run(
                full_text="filing",
                company_name="Microsoft",
                stop_after_pass1=True,
            )

        self.assertIn("does not support stop_after_pass1", str(ctx.exception))

    def test_zero_shot_runner_uses_single_prompt_and_returns_graph(self):
        extractor = LLMExtractor(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="local-model",
            provider="local",
            api_mode="chat_completions",
        )
        lines: list[str] = []
        console = PipelineConsole(total_stages=4, printer=lines.append)
        extractor.progress_callback = console.handle_progress

        extracted_graph = KnowledgeGraphExtraction(
            extraction_notes="Single-pass graph.",
            triples=[
                Triple(
                    subject="Microsoft",
                    subject_type="Company",
                    relation="HAS_SEGMENT",
                    object="Intelligent Cloud",
                    object_type="BusinessSegment",
                ),
                Triple(
                    subject="Intelligent Cloud",
                    subject_type="BusinessSegment",
                    relation="OFFERS",
                    object="Azure",
                    object_type="Offering",
                ),
            ],
        )

        captured_messages: list[list[dict[str, str]]] = []

        def structured_side_effect(**kwargs):
            captured_messages.append(kwargs["messages"])
            return extracted_graph, "raw graph", 2, {"kept_triple_count": 2, "raw_triple_count": 2}

        with patch.object(extractor, "generate_structured_output", side_effect=structured_side_effect):
            result = ZeroShotPipelineRunner(extractor).run(
                full_text="filing",
                company_name="Microsoft",
                max_retries=2,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.zero_shot_extraction.model_dump(), extracted_graph.model_dump())
        self.assertEqual(result.final_extraction.model_dump(), extracted_graph.model_dump())
        self.assertEqual(result.zero_shot_attempts_used, 2)

        self.assertEqual(len(captured_messages), 1)
        self.assertEqual(len(captured_messages[0]), 1)
        self.assertEqual(captured_messages[0][0]["role"], "user")
        self.assertIn("Build the full canonical business-model knowledge graph from the filing in one pass.", captured_messages[0][0]["content"])
        self.assertIn("Start from zero and use the ontology as the target structure.", captured_messages[0][0]["content"])
        self.assertIn("<source_filing>", captured_messages[0][0]["content"])
        self.assertIn("approved macro regions", captured_messages[0][0]["content"])
        self.assertIn("[02/04] Zero-shot extraction", lines)
        self.assertTrue(any("prompt:" in line and "single-pass ontology extraction" in line for line in lines))
        self.assertTrue(any("result:" in line and "2 triples" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
