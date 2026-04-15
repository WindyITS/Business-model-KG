import unittest
from types import SimpleNamespace

from llm_extraction.models import ExtractionError
from llm_extraction.pipelines import implemented_pipeline_names, known_pipeline_names, run_extraction_pipeline
from llm_extraction.prompting import pipeline_prompt_dir


class ExtractionPipelineRegistryTests(unittest.TestCase):
    def test_known_pipelines_include_analyst_scaffold(self):
        self.assertEqual(known_pipeline_names(), ("canonical", "analyst"))
        self.assertEqual(implemented_pipeline_names(), ("canonical",))

    def test_analyst_pipeline_scaffold_raises_clear_error(self):
        with self.assertRaises(ExtractionError) as ctx:
            run_extraction_pipeline(
                pipeline="analyst",
                extractor=SimpleNamespace(),
                full_text="filing",
                company_name="Microsoft",
            )

        self.assertIn("Pipeline 'analyst' is not implemented yet", str(ctx.exception))

    def test_canonical_prompt_assets_live_under_top_level_prompts_dir(self):
        prompt_dir = pipeline_prompt_dir("canonical")

        self.assertEqual(prompt_dir.parts[-2:], ("prompts", "canonical"))
        self.assertTrue((prompt_dir / "system.txt").is_file())
