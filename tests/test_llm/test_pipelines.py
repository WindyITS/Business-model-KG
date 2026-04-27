import unittest
from types import SimpleNamespace

from llm_extraction.models import ExtractionError
from llm_extraction.pipelines import (
    build_pipeline_runner,
    implemented_pipeline_names,
    known_pipeline_names,
    pipeline_stage_count,
    pipeline_supports_stop_after_pass1,
    run_extraction_pipeline,
)
from llm_extraction.pipelines.analyst.runner import AnalystPipelineRunner
from llm_extraction.pipelines.memo_graph_only.runner import MemoGraphOnlyPipelineRunner
from llm_extraction.pipelines.zero_shot.runner import ZeroShotPipelineRunner
from llm_extraction.prompting import pipeline_prompt_dir


class ExtractionPipelineRegistryTests(unittest.TestCase):
    def test_known_pipelines_include_analyst(self):
        self.assertEqual(known_pipeline_names(), ("analyst", "memo_graph_only", "zero-shot"))
        self.assertEqual(implemented_pipeline_names(), ("analyst", "memo_graph_only", "zero-shot"))

    def test_analyst_pipeline_dispatches_runner(self):
        runner = build_pipeline_runner("analyst", SimpleNamespace())

        self.assertIsInstance(runner, AnalystPipelineRunner)

    def test_zero_shot_pipeline_dispatches_runner(self):
        runner = build_pipeline_runner("zero-shot", SimpleNamespace())

        self.assertIsInstance(runner, ZeroShotPipelineRunner)

    def test_memo_graph_only_pipeline_dispatches_runner(self):
        runner = build_pipeline_runner("memo_graph_only", SimpleNamespace())

        self.assertIsInstance(runner, MemoGraphOnlyPipelineRunner)

    def test_pipeline_stage_metadata_tracks_analyst(self):
        self.assertFalse(pipeline_supports_stop_after_pass1("analyst"))
        self.assertFalse(pipeline_supports_stop_after_pass1("memo_graph_only"))
        self.assertFalse(pipeline_supports_stop_after_pass1("zero-shot"))
        self.assertEqual(pipeline_stage_count("analyst"), 7)
        self.assertEqual(pipeline_stage_count("memo_graph_only"), 5)
        self.assertEqual(pipeline_stage_count("zero-shot"), 4)

        with self.assertRaises(ExtractionError) as ctx:
            pipeline_stage_count("analyst", stop_after_pass1=True)

        self.assertIn("does not support stop_after_pass1", str(ctx.exception))

    def test_unknown_pipeline_still_raises_clear_error(self):
        with self.assertRaises(ExtractionError) as ctx:
            run_extraction_pipeline(
                pipeline="unknown",
                extractor=SimpleNamespace(),
                full_text="filing",
                company_name="Microsoft",
            )

        self.assertIn("Unknown extraction pipeline", str(ctx.exception))

    def test_analyst_prompt_assets_live_under_top_level_prompts_dir(self):
        prompt_dir = pipeline_prompt_dir("analyst")

        self.assertEqual(prompt_dir.parts[-2:], ("prompts", "analyst"))
        self.assertTrue((prompt_dir / "system.txt").is_file())
        self.assertTrue((prompt_dir / "memo_foundation.txt").is_file())

    def test_zero_shot_prompt_assets_live_under_top_level_prompts_dir(self):
        prompt_dir = pipeline_prompt_dir("zero-shot")

        self.assertEqual(prompt_dir.parts[-2:], ("prompts", "zero-shot"))
        self.assertTrue((prompt_dir / "extract.txt").is_file())

    def test_memo_graph_only_prompt_assets_live_under_top_level_prompts_dir(self):
        prompt_dir = pipeline_prompt_dir("memo_graph_only")

        self.assertEqual(prompt_dir.parts[-2:], ("prompts", "memo_graph_only"))
        self.assertTrue((prompt_dir / "system.txt").is_file())
        self.assertTrue((prompt_dir / "memo_foundation.txt").is_file())
        self.assertTrue((prompt_dir / "graph_system.txt").is_file())
        self.assertTrue((prompt_dir / "graph_compilation.txt").is_file())
