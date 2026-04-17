import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_extraction import prompting


class PromptLoadingTests(unittest.TestCase):
    def test_prompt_root_prefers_explicit_override_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_root = Path(tmp_dir)
            (prompt_root / "canonical").mkdir()
            (prompt_root / "canonical" / "system.txt").write_text("override", encoding="utf-8")

            with patch.dict(os.environ, {prompting.PROMPTS_OVERRIDE_ENV: str(prompt_root)}, clear=False):
                self.assertEqual(prompting.prompt_root(), prompt_root.resolve())
                self.assertEqual(prompting.pipeline_prompt_dir("canonical"), prompt_root.resolve() / "canonical")

    def test_prompt_root_falls_back_to_bundled_prompts_when_repo_prompts_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_repo_root = Path(tmp_dir) / "missing-kg-prompts-dir"

            with patch.dict(os.environ, {prompting.PROMPTS_OVERRIDE_ENV: ""}, clear=False), patch.object(
                prompting, "REPO_PROMPTS_ROOT", missing_repo_root
            ):
                prompt_dir = prompting.pipeline_prompt_dir("canonical")

            self.assertEqual(prompt_dir.parts[-2:], ("_bundled_prompts", "canonical"))
            self.assertTrue((prompt_dir / "system.txt").is_file())

    def test_bundled_prompt_assets_match_repo_prompt_assets(self):
        repo_files = {
            path.relative_to(prompting.REPO_PROMPTS_ROOT): path.read_text(encoding="utf-8")
            for path in prompting.REPO_PROMPTS_ROOT.rglob("*")
            if path.is_file()
        }
        bundled_files = {
            path.relative_to(prompting.BUNDLED_PROMPTS_ROOT): path.read_text(encoding="utf-8")
            for path in prompting.BUNDLED_PROMPTS_ROOT.rglob("*")
            if path.is_file()
        }

        self.assertEqual(bundled_files, repo_files)
