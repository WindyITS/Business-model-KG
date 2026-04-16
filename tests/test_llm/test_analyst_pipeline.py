import unittest
from unittest.mock import patch

from llm.extractor import LLMExtractor
from llm_extraction.models import (
    AnalystBusinessModelMemo,
    AnalystCanonicalLabelClaim,
    AnalystEvidence,
    AnalystNamedClaim,
    AnalystOffering,
    AnalystSegment,
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

        support = AnalystEvidence(explicit_support=["The filing describes Azure as part of the cloud business."])
        foundation_memo = AnalystBusinessModelMemo(
            company_name="Microsoft",
            analytical_frame="Microsoft organizes the business around major software and cloud franchises.",
            frame_support=support,
            segments=[
                AnalystSegment(
                    name="Intelligent Cloud",
                    role_in_business_model="Runs the cloud infrastructure and platform business.",
                    support=support,
                )
            ],
            offerings=[
                AnalystOffering(
                    name="Azure",
                    role_in_business_model="Cloud platform offering.",
                    support=support,
                    segment_anchors=[
                        AnalystNamedClaim(
                            name="Intelligent Cloud",
                            rationale="Azure is described inside the cloud business.",
                            support=support,
                        )
                    ],
                    revenue_models=[
                        AnalystCanonicalLabelClaim(
                            label="subscription",
                            rationale="Cloud contracts recur over time.",
                            support=support,
                        )
                    ],
                )
            ],
        )
        augmented_memo = foundation_memo.model_copy(deep=True)
        augmented_memo.segments[0].channels.append(
            AnalystCanonicalLabelClaim(
                label="resellers",
                rationale="Partners extend enterprise distribution.",
                support=AnalystEvidence(
                    explicit_support=["The filing references partner-led selling."],
                    analyst_inference=["Partner-led selling maps most closely to resellers."],
                ),
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
        memo_call_count = 0

        def structured_side_effect(**kwargs):
            nonlocal memo_call_count
            captured_messages.append(kwargs["messages"])
            if kwargs["schema_model"] is AnalystBusinessModelMemo:
                memo_call_count += 1
                memo = foundation_memo if memo_call_count == 1 else augmented_memo
                return memo, f"raw memo {memo_call_count}", memo_call_count, {"schema_name": "AnalystBusinessModelMemo"}
            return compiled_graph, "raw graph", 1, {"kept_triple_count": 3}

        critique_prompts: dict[str, str] = {}

        def critique_side_effect(**kwargs):
            critique_prompts["system"] = kwargs["system_prompt"]
            critique_prompts["user"] = kwargs["user_prompt"]
            return final_graph, '{"extraction_notes":"critique","triples":[]}', 2, {"kept_triple_count": 2}

        with patch.object(extractor, "_call_structured_messages", side_effect=structured_side_effect), patch.object(
            extractor, "reflect_extraction", side_effect=critique_side_effect
        ):
            result = AnalystPipelineRunner(extractor).run(
                full_text="filing",
                company_name="Microsoft",
                max_retries=2,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.foundation_memo.model_dump(mode="json"), foundation_memo.model_dump(mode="json"))
        self.assertEqual(result.augmented_memo.model_dump(mode="json"), augmented_memo.model_dump(mode="json"))
        self.assertEqual(result.compiled_graph_extraction.model_dump(), compiled_graph.model_dump())
        self.assertEqual(result.final_extraction.model_dump(), final_graph.model_dump())
        self.assertEqual(result.critique_attempts_used, 2)

        self.assertEqual(len(captured_messages), 3)
        self.assertIn("not literal paragraph extraction", captured_messages[0][0]["content"])
        self.assertIn("do not use your own knowledge to recover missing business-model details", captured_messages[0][1]["content"])
        self.assertIn("<current_memo>", captured_messages[1][1]["content"])
        self.assertIn("do not create artificial completeness", captured_messages[1][1]["content"])
        self.assertIn("inference-only claims may be used only", captured_messages[2][1]["content"])
        self.assertIn("focus on reducing overreach", critique_prompts["user"])

        self.assertIn("[02/07] Analyst memo 1 - Core structure", lines)
        self.assertIn("[05/07] Critique - Overreach review", lines)
        self.assertTrue(any("segments:" in line and "1" in line for line in lines))
        self.assertTrue(any("offerings:" in line and "1" in line for line in lines))
        self.assertTrue(any("triples added:" in line and "0" in line for line in lines))
        self.assertTrue(any("triples removed:" in line and "1" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
