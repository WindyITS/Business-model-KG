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
from llm_extraction.prompting import pipeline_prompt_dir


class ExtractionPipelineRegistryTests(unittest.TestCase):
    def test_known_pipelines_include_analyst(self):
        self.assertEqual(known_pipeline_names(), ("canonical", "analyst"))
        self.assertEqual(implemented_pipeline_names(), ("canonical", "analyst"))

    def test_analyst_pipeline_dispatches_runner(self):
        runner = build_pipeline_runner("analyst", SimpleNamespace())

        self.assertIsInstance(runner, AnalystPipelineRunner)

    def test_pipeline_stage_metadata_tracks_analyst(self):
        self.assertTrue(pipeline_supports_stop_after_pass1("canonical"))
        self.assertFalse(pipeline_supports_stop_after_pass1("analyst"))
        self.assertEqual(pipeline_stage_count("canonical"), 10)
        self.assertEqual(pipeline_stage_count("canonical", stop_after_pass1=True), 4)
        self.assertEqual(pipeline_stage_count("analyst"), 7)

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

    def test_canonical_prompt_assets_live_under_top_level_prompts_dir(self):
        prompt_dir = pipeline_prompt_dir("canonical")

        self.assertEqual(prompt_dir.parts[-2:], ("prompts", "canonical"))
        self.assertTrue((prompt_dir / "system.txt").is_file())

    def test_analyst_prompt_assets_live_under_top_level_prompts_dir(self):
        prompt_dir = pipeline_prompt_dir("analyst")

        self.assertEqual(prompt_dir.parts[-2:], ("prompts", "analyst"))
        self.assertTrue((prompt_dir / "system.txt").is_file())
        self.assertTrue((prompt_dir / "memo_foundation.txt").is_file())
