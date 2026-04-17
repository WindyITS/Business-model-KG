from __future__ import annotations

import json
import re
import shutil
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


LEGACY_OUTPUT_DIR_RE = re.compile(
    r"^(?P<source_stem>.+)_(?P<pipeline>canonical|analyst)_pipeline_(?P<run_token>\d{8}T\d{6}Z(?:_\d+)?)$"
)
COMPANY_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify_company_name(company_name: str) -> str:
    normalized = unicodedata.normalize("NFKD", company_name)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = COMPANY_SLUG_RE.sub("_", ascii_text.casefold()).strip("_")
    return slug or "company"


def infer_company_name_from_source_stem(source_stem: str) -> str:
    return source_stem.replace("_10k", "").replace("_", " ").strip().title()


def company_pipeline_root(output_dir: Path, company_name: str, pipeline: str) -> Path:
    return output_dir / slugify_company_name(company_name) / pipeline


def iter_latest_run_dirs(output_dir: Path, pipeline: str) -> list[Path]:
    if not output_dir.exists():
        return []

    latest_dirs: list[Path] = []
    for company_dir in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        latest_dir = company_dir / pipeline / "latest"
        if latest_dir.is_dir():
            latest_dirs.append(latest_dir)
    return latest_dirs


def resolve_company_run_dir(output_dir: Path, company_name: str, pipeline: str, run_selector: str | None = None) -> Path:
    pipeline_root = company_pipeline_root(output_dir, company_name, pipeline)
    if run_selector is None or run_selector == "latest":
        return pipeline_root / "latest"

    explicit_path = Path(run_selector)
    if explicit_path.is_absolute():
        return explicit_path

    if "/" in run_selector or "\\" in run_selector:
        return (pipeline_root / explicit_path).resolve()

    runs_candidate = pipeline_root / "runs" / run_selector
    if runs_candidate.exists():
        return runs_candidate

    failed_candidate = pipeline_root / "failed" / run_selector
    if failed_candidate.exists():
        return failed_candidate

    return pipeline_root / explicit_path


def _run_token(started_at: datetime) -> str:
    return started_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _unique_child_path(parent: Path, base_name: str) -> Path:
    candidate = parent / base_name
    suffix = 2
    while candidate.exists():
        candidate = parent / f"{base_name}_{suffix}"
        suffix += 1
    return candidate


def _cleanup_empty_dir(path: Path) -> None:
    current = path
    while current.exists():
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _rewrite_run_summary_path(run_dir: Path) -> None:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        return
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["run_dir"] = str(run_dir)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class OutputLayout:
    company_name: str
    company_slug: str
    pipeline: str
    run_token: str
    root_dir: Path
    latest_dir: Path
    runs_dir: Path
    failed_dir: Path
    staging_root: Path
    staging_dir: Path
    preserved_run_dir: Path
    keep_current_output: bool

    @property
    def planned_output_dir(self) -> Path:
        if self.keep_current_output:
            return self.preserved_run_dir
        return self.latest_dir


def prepare_output_layout(
    *,
    output_dir: Path,
    company_name: str,
    pipeline: str,
    keep_current_output: bool,
    started_at: datetime,
) -> OutputLayout:
    company_slug = slugify_company_name(company_name)
    run_token = _run_token(started_at)
    root_dir = output_dir / company_slug / pipeline
    latest_dir = root_dir / "latest"
    runs_dir = root_dir / "runs"
    failed_dir = root_dir / "failed"
    staging_root = root_dir / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = _unique_child_path(staging_root, run_token)
    staging_dir.mkdir(parents=True, exist_ok=False)
    preserved_run_dir = runs_dir / staging_dir.name
    return OutputLayout(
        company_name=company_name,
        company_slug=company_slug,
        pipeline=pipeline,
        run_token=staging_dir.name,
        root_dir=root_dir,
        latest_dir=latest_dir,
        runs_dir=runs_dir,
        failed_dir=failed_dir,
        staging_root=staging_root,
        staging_dir=staging_dir,
        preserved_run_dir=preserved_run_dir,
        keep_current_output=keep_current_output,
    )


def finalize_successful_run(layout: OutputLayout) -> Path:
    if layout.keep_current_output:
        layout.runs_dir.mkdir(parents=True, exist_ok=True)
        layout.staging_dir.rename(layout.preserved_run_dir)
        _cleanup_empty_dir(layout.staging_root)
        return layout.preserved_run_dir

    layout.root_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = layout.root_dir / ".previous_latest"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    if layout.latest_dir.exists():
        layout.latest_dir.rename(backup_dir)

    try:
        layout.staging_dir.rename(layout.latest_dir)
    except Exception:
        if backup_dir.exists() and not layout.latest_dir.exists():
            backup_dir.rename(layout.latest_dir)
        raise

    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    _cleanup_empty_dir(layout.staging_root)
    return layout.latest_dir


def finalize_failed_run(layout: OutputLayout) -> Path:
    layout.failed_dir.mkdir(parents=True, exist_ok=True)
    failed_run_dir = layout.failed_dir / layout.run_token
    layout.staging_dir.rename(failed_run_dir)
    _cleanup_empty_dir(layout.staging_root)
    return failed_run_dir


def migrate_legacy_output_layout(output_dir: Path) -> list[tuple[Path, Path]]:
    migrations: list[tuple[Path, Path]] = []
    if not output_dir.exists():
        return migrations
    for child in sorted(output_dir.iterdir()):
        if not child.is_dir():
            continue
        match = LEGACY_OUTPUT_DIR_RE.match(child.name)
        if not match:
            continue

        source_stem = match.group("source_stem")
        pipeline = match.group("pipeline")
        run_token = match.group("run_token")
        company_name = infer_company_name_from_source_stem(source_stem)
        company_slug = slugify_company_name(company_name)
        pipeline_root = output_dir / company_slug / pipeline
        latest_dir = pipeline_root / "latest"
        destination = latest_dir if not latest_dir.exists() else _unique_child_path(pipeline_root / "runs", run_token)
        destination.parent.mkdir(parents=True, exist_ok=True)
        child.rename(destination)
        _rewrite_run_summary_path(destination)
        migrations.append((child, destination))
    return migrations
