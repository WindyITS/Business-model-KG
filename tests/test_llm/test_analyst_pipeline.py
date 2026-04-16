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
        self.assertIn("high-accountability analytical task", captured_messages[0][0]["content"])
        self.assertIn("missing a material offering family", captured_messages[0][0]["content"])
        self.assertIn("shared family and the distinct child offerings", captured_messages[0][0]["content"])
        self.assertIn("distinct named franchise should not be demoted", captured_messages[0][0]["content"])
        self.assertIn("follow the output template and structural instructions in the user prompt exactly", captured_messages[0][0]["content"])
        self.assertIn("keep distribution-channel structure on BusinessSegment by default", captured_messages[0][0]["content"])
        self.assertIn("stable named variants, tiers, editions, plans, form factors, or line families", captured_messages[0][0]["content"])
        self.assertIn("normalize annual, release-specific, or version-specific names", captured_messages[0][0]["content"])
        self.assertIn("prefer a clean global footprint such as Worldwide", captured_messages[0][0]["content"])
        self.assertIn("support, analyst inference, and ambiguity distinct", captured_messages[0][0]["content"])
        self.assertIn("there can be 2+ layers", captured_messages[0][0]["content"])
        self.assertNotIn("comparability", captured_messages[0][0]["content"])
        self.assertNotIn("<json_rules>", captured_messages[0][0]["content"])
        self.assertIn("Return a structured markdown memo, not JSON", captured_messages[0][1]["content"])
        self.assertIn("# ANALYTICAL FRAME", captured_messages[0][1]["content"])
        self.assertIn("## Summary:", captured_messages[0][1]["content"])
        self.assertIn("Child Offerings:", captured_messages[0][1]["content"])
        self.assertIn("must have its own full `[Offering]` block", captured_messages[0][1]["content"])
        self.assertIn("stable named variants, tiers, editions, plans, form factors, or line families", captured_messages[0][1]["content"])
        self.assertIn("do not populate `Channels` on a segment-anchored offering", captured_messages[0][1]["content"])
        self.assertIn("prefer `Worldwide`", captured_messages[0][1]["content"])
        self.assertNotIn("if a named offering family, suite, platform, or franchise is itself one of the core organizing units", captured_messages[0][1]["content"])
        self.assertIn("<current_memo>", captured_messages[1][1]["content"])
        self.assertIn("This is an expansion pass, not a correctness-review pass", captured_messages[1][1]["content"])
        self.assertIn("Return the full memo again as structured markdown, not JSON", captured_messages[1][1]["content"])
        self.assertIn("actively recover named offering families", captured_messages[1][1]["content"])
        self.assertIn("for each top-level offering or offering family already in the memo, deepen it", captured_messages[1][1]["content"])
        self.assertIn("do not collapse important offering families into segment prose", captured_messages[1][1]["content"])
        self.assertIn("Child Offerings` on the parent and `Parent Offering` on the child", captured_messages[1][1]["content"])
        self.assertIn("deepen any family-level offering that still needs decomposition", captured_messages[1][1]["content"])
        self.assertIn("split any franchise that is still being represented as one blended node", captured_messages[1][1]["content"])
        self.assertIn("do not demote it under a broader cloud, services, or reporting bucket", captured_messages[1][1]["content"])
        self.assertIn("broad descriptive categories", captured_messages[1][1]["content"])
        self.assertIn("durable variants, tiers, editions, plans, form factors, or line families", captured_messages[1][1]["content"])
        self.assertIn("normalize annual, release-specific, or version-specific names", captured_messages[1][1]["content"])
        self.assertIn("remove offering-level `Channels` that merely restate inherited segment distribution", captured_messages[1][1]["content"])
        self.assertIn("prefer a clean corporate-scope geography such as `Worldwide`", captured_messages[1][1]["content"])
        self.assertIn("your role here is to improve and deepen the memo", captured_messages[1][1]["content"])
        self.assertIn("support, analyst inference, and ambiguity are clearly separated", captured_messages[1][1]["content"])
        self.assertIn("add any missing material family-level offerings", captured_messages[1][1]["content"])
        self.assertIn("restore any distinct named franchise", captured_messages[1][1]["content"])
        self.assertIn("keep the artifact as markdown memo text, not JSON", captured_messages[1][1]["content"])
        self.assertIn("Return ONLY the full markdown memo.", captured_messages[1][1]["content"])
        self.assertIn("Convert the analyst memo into an in-depth ontology-valid business-model graph.", captured_messages[2][1]["content"])
        self.assertIn("the memo may contain both support and analyst inference; support is preferred", captured_messages[2][1]["content"])
        self.assertIn("ambiguity field does not signal material ambiguity", captured_messages[2][1]["content"])
        self.assertIn("if ambiguity suggests the relation may be wrong or overly neat, omit it", captured_messages[2][1]["content"])
        self.assertIn("preserve depth when the memo supports it", captured_messages[2][1]["content"])
        self.assertIn("`Parent Offering` or `Child Offerings` makes the hierarchy supportable", captured_messages[2][1]["content"])
        self.assertIn("do not flatten parent-child offering structure into only leaf offerings", captured_messages[2][1]["content"])
        self.assertIn("anchored to that segment and represented as its own full `[Offering]` block", captured_messages[2][1]["content"])
        self.assertIn("do not materialize a parent offering node", captured_messages[2][1]["content"])
        self.assertIn("offering lacks a segment anchor or the memo shows a differentiated channel pattern", captured_messages[2][1]["content"])
        self.assertIn("do not compile offering-level channels that merely restate inherited segment distribution", captured_messages[2][1]["content"])
        self.assertIn("preserve that cleaner geography instead of exploding it", captured_messages[2][1]["content"])
        self.assertIn('"subject_type": "EXACT_NODE_TYPE"', captured_messages[2][1]["content"])
        self.assertIn("Do not use:", captured_messages[2][1]["content"])
        self.assertIn("`predicate` instead of `relation`", captured_messages[2][1]["content"])
        self.assertIn("high-accountability analytical task", captured_messages[2][0]["content"])
        self.assertIn("preserve split-franchise hierarchy", captured_messages[2][0]["content"])
        self.assertIn("preserve distinct named franchises as standalone nodes", captured_messages[2][0]["content"])
        self.assertIn("preserve durable family and variant substructure", captured_messages[2][0]["content"])
        self.assertIn("keep channel structure on BusinessSegment by default", captured_messages[2][0]["content"])
        self.assertIn("keep corporate geography clean", captured_messages[2][0]["content"])
        self.assertNotIn("comparability", captured_messages[2][0]["content"])
        self.assertNotIn("<source_filing>", captured_messages[2][0]["content"])
        self.assertIn("focus on reducing overreach", critique_prompts["user"])
        self.assertIn("do not prune defensible parent-child offering hierarchy", critique_prompts["user"])
        self.assertIn("preserve split-franchise decomposition", critique_prompts["user"])
        self.assertIn("remove parent nodes that were materialized only from `Parent Offering` references", critique_prompts["user"])
        self.assertIn("remove child offerings that are only broad bucket labels", critique_prompts["user"])
        self.assertIn("preserve durable family and variant hierarchy", critique_prompts["user"])
        self.assertIn("remove offering-level `SELLS_THROUGH` edges", critique_prompts["user"])
        self.assertIn("prune it back toward the memo-supported macro geography such as `Worldwide`", critique_prompts["user"])
        self.assertIn('"subject_type": "EXACT_NODE_TYPE"', critique_prompts["user"])
        self.assertIn("tuple or array triples", critique_prompts["user"])
        self.assertIn("high-accountability analytical task", critique_prompts["system"])
        self.assertIn("preserve split-franchise hierarchy", critique_prompts["system"])
        self.assertIn("preserve distinct named franchises as standalone nodes", critique_prompts["system"])
        self.assertIn("keep channel structure on BusinessSegment by default", critique_prompts["system"])
        self.assertIn("prefer memo-supported global scope such as Worldwide", critique_prompts["system"])
        self.assertNotIn("comparability", critique_prompts["system"])
        self.assertNotIn("<source_filing>", critique_prompts["system"])

        self.assertIn("[02/07] Analyst memo 1 - Core structure", lines)
        self.assertIn("[05/07] Critique - Overreach review", lines)
        self.assertTrue(any("memo chars:" in line for line in lines))
        self.assertTrue(any("memo lines:" in line for line in lines))
        self.assertTrue(any("triples added:" in line and "0" in line for line in lines))
        self.assertTrue(any("triples removed:" in line and "1" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
