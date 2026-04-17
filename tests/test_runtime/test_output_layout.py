import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from runtime.output_layout import (
    company_pipeline_root,
    discover_output_company_states,
    finalize_failed_run,
    finalize_successful_run,
    iter_latest_run_dirs,
    manifest_path,
    migrate_legacy_output_layout,
    prepare_output_layout,
    refresh_output_manifests,
    resolve_company_run_dir,
    slugify_company_name,
)


class OutputLayoutTests(unittest.TestCase):
    def test_slugify_company_name_normalizes_punctuation_and_case(self):
        self.assertEqual(slugify_company_name("Apple Inc."), "apple_inc")
        self.assertEqual(slugify_company_name("L'Oréal Groupe"), "l_oreal_groupe")

    def test_finalize_successful_run_replaces_latest_only_after_success(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            latest_dir = output_dir / "apple" / "canonical" / "latest"
            latest_dir.mkdir(parents=True, exist_ok=True)
            (latest_dir / "run_summary.json").write_text(json.dumps({"run_dir": str(latest_dir)}), encoding="utf-8")
            (latest_dir / "marker.txt").write_text("old", encoding="utf-8")

            layout = prepare_output_layout(
                output_dir=output_dir,
                company_name="Apple",
                pipeline="canonical",
                keep_current_output=False,
                started_at=datetime(2026, 4, 17, 10, 0, 0, tzinfo=timezone.utc),
            )
            (layout.staging_dir / "run_summary.json").write_text(json.dumps({"run_dir": str(layout.staging_dir)}), encoding="utf-8")
            (layout.staging_dir / "marker.txt").write_text("new", encoding="utf-8")

            final_dir = finalize_successful_run(layout)

            self.assertEqual(final_dir, latest_dir)
            self.assertEqual((latest_dir / "marker.txt").read_text(encoding="utf-8"), "new")
            self.assertFalse((output_dir / "apple" / "canonical" / ".previous_latest").exists())
            self.assertFalse(layout.staging_dir.exists())
            manifest = json.loads(manifest_path(output_dir, "Apple", "canonical").read_text(encoding="utf-8"))
            self.assertTrue(manifest["latest_available"])
            self.assertEqual(manifest["latest_run_dir"], str(latest_dir))

    def test_finalize_successful_run_can_preserve_current_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            latest_dir = output_dir / "microsoft" / "analyst" / "latest"
            latest_dir.mkdir(parents=True, exist_ok=True)
            (latest_dir / "marker.txt").write_text("stable", encoding="utf-8")

            layout = prepare_output_layout(
                output_dir=output_dir,
                company_name="Microsoft",
                pipeline="analyst",
                keep_current_output=True,
                started_at=datetime(2026, 4, 17, 10, 0, 0, tzinfo=timezone.utc),
            )
            (layout.staging_dir / "run_summary.json").write_text(json.dumps({"run_dir": str(layout.staging_dir)}), encoding="utf-8")
            (layout.staging_dir / "marker.txt").write_text("candidate", encoding="utf-8")

            final_dir = finalize_successful_run(layout)

            self.assertEqual(final_dir, layout.preserved_run_dir)
            self.assertEqual((latest_dir / "marker.txt").read_text(encoding="utf-8"), "stable")
            self.assertEqual((final_dir / "marker.txt").read_text(encoding="utf-8"), "candidate")

    def test_finalize_failed_run_moves_staging_into_failed_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            layout = prepare_output_layout(
                output_dir=output_dir,
                company_name="Palantir",
                pipeline="canonical",
                keep_current_output=False,
                started_at=datetime(2026, 4, 17, 10, 0, 0, tzinfo=timezone.utc),
            )
            (layout.staging_dir / "run_summary.json").write_text(json.dumps({"run_dir": str(layout.staging_dir)}), encoding="utf-8")
            (layout.staging_dir / "marker.txt").write_text("failed", encoding="utf-8")

            final_dir = finalize_failed_run(layout)

            self.assertEqual(final_dir, output_dir / "palantir" / "canonical" / "failed" / layout.run_token)
            self.assertEqual((final_dir / "marker.txt").read_text(encoding="utf-8"), "failed")
            self.assertFalse(layout.staging_dir.exists())
            manifest = json.loads(manifest_path(output_dir, "Palantir", "canonical").read_text(encoding="utf-8"))
            self.assertFalse(manifest["latest_available"])
            self.assertEqual(manifest["failed_run_tokens"], [layout.run_token])

    def test_iter_latest_run_dirs_returns_sorted_pipeline_latest_dirs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            (company_pipeline_root(output_dir, "Google", "analyst") / "latest").mkdir(parents=True, exist_ok=True)
            (company_pipeline_root(output_dir, "Apple", "analyst") / "latest").mkdir(parents=True, exist_ok=True)
            (company_pipeline_root(output_dir, "Apple", "canonical") / "latest").mkdir(parents=True, exist_ok=True)

            latest_dirs = iter_latest_run_dirs(output_dir, "analyst")

            self.assertEqual(
                latest_dirs,
                [
                    output_dir / "apple" / "analyst" / "latest",
                    output_dir / "google" / "analyst" / "latest",
                ],
            )

    def test_resolve_company_run_dir_defaults_to_latest_and_supports_run_tokens(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            latest_dir = company_pipeline_root(output_dir, "Apple", "analyst") / "latest"
            latest_dir.mkdir(parents=True, exist_ok=True)
            run_dir = company_pipeline_root(output_dir, "Apple", "analyst") / "runs" / "20260417T101500Z"
            run_dir.mkdir(parents=True, exist_ok=True)

            self.assertEqual(resolve_company_run_dir(output_dir, "Apple", "analyst"), latest_dir)
            self.assertEqual(
                resolve_company_run_dir(output_dir, "Apple", "analyst", "20260417T101500Z"),
                run_dir,
            )

    def test_migrate_legacy_output_layout_moves_flat_directories_and_updates_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            legacy_dir = output_dir / "apple_10k_analyst_pipeline_20260416T201638Z"
            legacy_dir.mkdir(parents=True, exist_ok=True)
            (legacy_dir / "run_summary.json").write_text(json.dumps({"run_dir": str(legacy_dir)}), encoding="utf-8")
            (legacy_dir / "artifact.txt").write_text("legacy", encoding="utf-8")

            migrations = migrate_legacy_output_layout(output_dir)

            target_dir = output_dir / "apple" / "analyst" / "latest"
            self.assertEqual(migrations, [(legacy_dir, target_dir)])
            self.assertTrue(target_dir.is_dir())
            self.assertEqual((target_dir / "artifact.txt").read_text(encoding="utf-8"), "legacy")
            summary = json.loads((target_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["run_dir"], str(target_dir))

    def test_discover_output_company_states_uses_saved_run_summary_when_latest_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            run_dir = company_pipeline_root(output_dir, "Apple Inc.", "analyst") / "runs" / "20260417T101500Z"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_summary.json").write_text(
                json.dumps(
                    {
                        "company_name": "Apple Inc.",
                        "source_file": "data/apple_10k.txt",
                        "run_dir": str(run_dir),
                    }
                ),
                encoding="utf-8",
            )

            states = discover_output_company_states(output_dir, "analyst")

            self.assertEqual(len(states), 1)
            self.assertEqual(states[0].company_name, "Apple Inc.")
            self.assertFalse(states[0].latest_available)
            self.assertEqual(states[0].available_run_tokens, ["20260417T101500Z"])

    def test_refresh_output_manifests_writes_manifest_for_existing_latest_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            latest_dir = company_pipeline_root(output_dir, "Google", "analyst") / "latest"
            latest_dir.mkdir(parents=True, exist_ok=True)
            (latest_dir / "run_summary.json").write_text(
                json.dumps({"company_name": "Google", "source_file": "data/google_10k.txt", "run_dir": str(latest_dir)}),
                encoding="utf-8",
            )

            manifests = refresh_output_manifests(output_dir, "analyst")

            manifest_file = manifest_path(output_dir, "Google", "analyst")
            self.assertEqual(manifests, [manifest_file])
            payload = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["company_name"], "Google")
            self.assertTrue(payload["latest_available"])


if __name__ == "__main__":
    unittest.main()
