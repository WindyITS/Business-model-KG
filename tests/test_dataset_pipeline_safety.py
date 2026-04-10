import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts"))
sys.path.insert(0, str(ROOT_DIR / "src"))

import run_batch
from augment_finreflectkg_stage3 import ensure_teacher_call_succeeded, teacher_error_relations


class DatasetPipelineSafetyTests(unittest.TestCase):
    def test_teacher_error_relations_detects_exhausted_relation_calls(self):
        call_log = {
            "chunk_key": {"ticker": "MSFT", "chunk_id": "c1"},
            "relation_reports": {
                "SERVES": {"valid_triples": []},
                "SELLS_THROUGH": {"error": "Connection error."},
                "MONETIZES_VIA": {"error": "Connection error."},
            },
        }

        self.assertEqual(teacher_error_relations(call_log), ["SELLS_THROUGH", "MONETIZES_VIA"])
        with self.assertRaisesRegex(RuntimeError, "Teacher augmentation failed"):
            ensure_teacher_call_succeeded(call_log)

    def test_batch_runner_passes_local_parquet_dataset_args(self):
        args = run_batch.build_dataset_source_args(
            hf_dataset="domyn/FinReflectKG",
            split="train",
            cache_dir="data/external/huggingface",
            parquet_files=["data/external/finreflectkg/data/train.parquet"],
            streaming=False,
        )

        self.assertIn("--parquet-file", args)
        self.assertIn("data/external/finreflectkg/data/train.parquet", args)
        self.assertIn("--no-streaming", args)
        self.assertNotIn("--hf-dataset", args)

    def test_batch_runner_passes_batch_window_to_stage2_and_stage3(self):
        with patch.object(run_batch, "run_cmd") as run_cmd:
            run_batch.run_stage2(2, Path("projected.jsonl"), 80000, 0.3, [])
            stage2_args = run_cmd.call_args.args[0]
            self.assertEqual(stage2_args[stage2_args.index("--limit-chunks") + 1], "80000")
            self.assertEqual(stage2_args[stage2_args.index("--skip-chunks") + 1], "80000")

            run_cmd.reset_mock()
            run_batch.run_stage3(
                2,
                Path("projected.jsonl"),
                Path("empty.jsonl"),
                model="google/gemma-4-26b-a4b",
                prompt_profile="default",
                empty_ratio=0.3,
                chunks_per_batch=80000,
                dataset_source_args=[],
            )
            stage3_args = run_cmd.call_args.args[0]
            self.assertEqual(stage3_args[stage3_args.index("--limit-chunks") + 1], "80000")
            self.assertEqual(stage3_args[stage3_args.index("--skip-chunks") + 1], "80000")

    def test_merge_requires_all_batches_unless_partial_merge_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            stage3_dir = temp_root / "outputs" / "batch_1" / "stage3"
            stage3_dir.mkdir(parents=True)
            example = {
                "metadata": {"chunk_key": {"ticker": "MSFT", "chunk_id": "c1"}},
                "output": {"triples": []},
            }
            (stage3_dir / "training_examples.jsonl").write_text(
                json.dumps(example, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (stage3_dir / "stage3_report.json").write_text(
                json.dumps({"relation_counts": {"SERVES": 1}}, ensure_ascii=False),
                encoding="utf-8",
            )

            with patch.object(run_batch, "ROOT_DIR", temp_root):
                with patch("sys.stdout", new_callable=io.StringIO):
                    self.assertEqual(run_batch.merge_batches(2), 1)
                self.assertFalse((temp_root / "outputs" / "merged" / "merge_report.json").exists())

                with patch("sys.stdout", new_callable=io.StringIO):
                    self.assertEqual(run_batch.merge_batches(2, allow_partial=True), 0)
                merge_report = json.loads(
                    (temp_root / "outputs" / "merged" / "merge_report.json").read_text(encoding="utf-8")
                )

        self.assertEqual(merge_report["batches_merged"], 1)
        self.assertEqual(merge_report["total_examples"], 1)


if __name__ == "__main__":
    unittest.main()
