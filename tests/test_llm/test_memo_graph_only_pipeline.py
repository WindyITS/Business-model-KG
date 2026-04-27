import unittest
from unittest.mock import patch

from llm.extractor import LLMExtractor
from llm_extraction.models import (
    AnalystBusinessModelMemo,
    ExtractionError,
    KnowledgeGraphExtraction,
    Triple,
)
from llm_extraction.pipelines.memo_graph_only.runner import MemoGraphOnlyPipelineRunner
from runtime.main import PipelineConsole


class MemoGraphOnlyPipelineRunnerTests(unittest.TestCase):
    def test_memo_graph_only_runner_rejects_stop_after_pass1(self):
        extractor = LLMExtractor(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="local-model",
            provider="local",
            api_mode="chat_completions",
        )

        with self.assertRaises(ExtractionError) as ctx:
            MemoGraphOnlyPipelineRunner(extractor).run(
                full_text="filing",
                company_name="Microsoft",
                stop_after_pass1=True,
            )

        self.assertIn("does not support stop_after_pass1", str(ctx.exception))

    def test_memo_graph_only_runner_builds_first_memo_then_graph(self):
        extractor = LLMExtractor(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="local-model",
            provider="local",
            api_mode="chat_completions",
        )
        lines: list[str] = []
        console = PipelineConsole(total_stages=5, printer=lines.append)
        extractor.progress_callback = console.handle_progress

        foundation_memo = AnalystBusinessModelMemo(
            content=(
                "ANALYTICAL FRAME\n"
                "Summary:\nMicrosoft organizes the business around cloud and software franchises.\n"
            ),
        )
        compiled_graph = KnowledgeGraphExtraction(
            extraction_notes="Compiled from first memo.",
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

        def text_side_effect(**kwargs):
            captured_messages.append(kwargs["messages"])
            return foundation_memo.content, 1, {"format": "text", "content_length": len(foundation_memo.content)}

        def structured_side_effect(**kwargs):
            captured_messages.append(kwargs["messages"])
            return compiled_graph, "raw graph", 2, {"kept_triple_count": 2, "raw_triple_count": 2}

        with patch.object(extractor, "_call_text_messages", side_effect=text_side_effect) as mock_text, patch.object(
            extractor, "_call_structured_messages", side_effect=structured_side_effect
        ) as mock_structured, patch.object(extractor, "reflect_extraction") as mock_reflect:
            result = MemoGraphOnlyPipelineRunner(extractor).run(
                full_text="filing",
                company_name="Microsoft",
                max_retries=2,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.foundation_memo.content, foundation_memo.content)
        self.assertEqual(result.compiled_graph_extraction.model_dump(), compiled_graph.model_dump())
        self.assertEqual(result.final_extraction.model_dump(), compiled_graph.model_dump())
        self.assertEqual(result.foundation_memo_attempts_used, 1)
        self.assertEqual(result.compiled_graph_attempts_used, 2)
        mock_text.assert_called_once()
        mock_structured.assert_called_once()
        mock_reflect.assert_not_called()

        self.assertEqual(len(captured_messages), 2)
        self.assertIn("Return a structured markdown memo, not JSON", captured_messages[0][1]["content"])
        self.assertIn("Convert the analyst memo into an in-depth ontology-valid business-model graph.", captured_messages[1][1]["content"])
        self.assertIn(foundation_memo.content, captured_messages[1][1]["content"])
        self.assertIn("[02/05] Memo graph-only memo - Core structure", lines)
        self.assertIn("[03/05] Memo graph-only graph compilation", lines)
        self.assertTrue(any("memo chars:" in line for line in lines))
        self.assertTrue(any("result:" in line and "2 triples" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
