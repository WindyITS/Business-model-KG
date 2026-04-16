import unittest
from unittest.mock import patch

from llm.extractor import LLMExtractor
from llm_extraction.models import (
    AnalystBusinessModelMemo,
    ExtractionError,
    KnowledgeGraphExtraction,
    Triple,
)
from llm_extraction.pipelines.analyst.runner import AnalystPipelineRunner
from runtime.main import PipelineConsole


class AnalystPipelineRunnerTests(unittest.TestCase):
    def test_analyst_runner_rejects_stop_after_pass1(self):
        extractor = LLMExtractor(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="local-model",
            provider="local",
            api_mode="chat_completions",
        )

        with self.assertRaises(ExtractionError) as ctx:
            AnalystPipelineRunner(extractor).run(
                full_text="filing",
                company_name="Microsoft",
                stop_after_pass1=True,
            )

        self.assertIn("does not support stop_after_pass1", str(ctx.exception))

    def test_analyst_runner_builds_memo_then_graph_then_critique(self):
        extractor = LLMExtractor(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="local-model",
            provider="local",
            api_mode="chat_completions",
        )
        lines: list[str] = []
        console = PipelineConsole(total_stages=7, printer=lines.append)
        extractor.progress_callback = console.handle_progress

        foundation_memo = AnalystBusinessModelMemo(
            content=(
                "ANALYTICAL FRAME\n"
                "Summary:\nMicrosoft organizes the business around major software and cloud franchises.\n"
                "Explicit Support:\n- Azure is described inside the cloud business.\n"
                "Analyst Inference / Synthesis:\n- none\n"
                "Uncertainty / Ambiguity:\n- none\n"
            ),
        )
        augmented_memo = AnalystBusinessModelMemo(
            content=(
                foundation_memo.content
                + "\nSEGMENTS\n"
                + "[Segment] Intelligent Cloud\n"
                + "Role in Business Model:\nRuns the cloud infrastructure and platform business.\n"
                + "Explicit Support:\n- Azure is described inside the cloud business.\n"
                + "Analyst Inference / Synthesis:\n- none\n"
                + "Uncertainty / Ambiguity:\n- none\n"
                + "Customer Types:\n- none\n"
                + "Channels:\n- resellers | Partners extend enterprise distribution.\n"
            )
        )

        compiled_graph = KnowledgeGraphExtraction(
            extraction_notes="Compiled from analyst memo.",
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
                Triple(
                    subject="Intelligent Cloud",
                    subject_type="BusinessSegment",
                    relation="SELLS_THROUGH",
                    object="resellers",
                    object_type="Channel",
                ),
            ],
        )
        final_graph = KnowledgeGraphExtraction(
            extraction_notes="Removed a weak channel edge during critique.",
            triples=compiled_graph.triples[:2],
        )

        captured_messages: list[list[dict[str, str]]] = []
        text_call_count = 0

        def text_side_effect(**kwargs):
            nonlocal text_call_count
            captured_messages.append(kwargs["messages"])
            text_call_count += 1
            memo = foundation_memo if text_call_count == 1 else augmented_memo
            return memo.content, text_call_count, {"format": "text", "content_length": len(memo.content)}

        def structured_side_effect(**kwargs):
            captured_messages.append(kwargs["messages"])
            return compiled_graph, "raw graph", 1, {"kept_triple_count": 3}

        critique_prompts: dict[str, str] = {}

        def critique_side_effect(**kwargs):
            critique_prompts["system"] = kwargs["system_prompt"]
            critique_prompts["user"] = kwargs["user_prompt"]
            return final_graph, '{"extraction_notes":"critique","triples":[]}', 2, {"kept_triple_count": 2}

        with patch.object(extractor, "_call_text_messages", side_effect=text_side_effect), patch.object(
            extractor, "_call_structured_messages", side_effect=structured_side_effect
        ), patch.object(extractor, "reflect_extraction", side_effect=critique_side_effect):
            result = AnalystPipelineRunner(extractor).run(
                full_text="filing",
                company_name="Microsoft",
                max_retries=2,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.foundation_memo.content, foundation_memo.content)
        self.assertEqual(result.augmented_memo.content, augmented_memo.content)
        self.assertEqual(result.compiled_graph_extraction.model_dump(), compiled_graph.model_dump())
        self.assertEqual(result.final_extraction.model_dump(), final_graph.model_dump())
        self.assertEqual(result.critique_attempts_used, 2)

        self.assertEqual(len(captured_messages), 3)
        self.assertIn("not literal paragraph extraction", captured_messages[0][0]["content"])
        self.assertIn("Return a structured plain-text memo, not JSON", captured_messages[0][1]["content"])
        self.assertIn("<current_memo>", captured_messages[1][1]["content"])
        self.assertIn("keep the artifact as plain memo text, not JSON", captured_messages[1][1]["content"])
        self.assertIn("inference-only claims may be used only", captured_messages[2][1]["content"])
        self.assertIn("focus on reducing overreach", critique_prompts["user"])

        self.assertIn("[02/07] Analyst memo 1 - Core structure", lines)
        self.assertIn("[05/07] Critique - Overreach review", lines)
        self.assertTrue(any("memo chars:" in line for line in lines))
        self.assertTrue(any("memo lines:" in line for line in lines))
        self.assertTrue(any("triples added:" in line and "0" in line for line in lines))
        self.assertTrue(any("triples removed:" in line and "1" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
