from __future__ import annotations

import json
import re
import shutil
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMPANY_SLUG_RE = re.compile(r"[^a-z0-9]+")
PIPELINE_NAMES = ("analyst", "memo_graph_only", "zero-shot")
LEGACY_OUTPUT_DIR_RE = re.compile(
    rf"^(?P<source_stem>.+)_(?P<pipeline>{'|'.join(re.escape(name) for name in PIPELINE_NAMES)})_pipeline_(?P<run_token>\d{{8}}T\d{{6}}Z(?:_\d+)?)$"
)
MANIFEST_FILENAME = "manifest.json"


def slugify_company_name(company_name: str) -> str:
    normalized = unicodedata.normalize("NFKD", company_name)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = COMPANY_SLUG_RE.sub("_", ascii_text.casefold()).strip("_")
    return slug or "company"


def infer_company_name_from_source_stem(source_stem: str) -> str:
    return source_stem.replace("_10k", "").replace("_", " ").strip().title()


def company_pipeline_root(output_dir: Path, company_name: str, pipeline: str) -> Path:
    return output_dir / slugify_company_name(company_name) / pipeline


def manifest_path(output_dir: Path, company_name: str, pipeline: str) -> Path:
    return company_pipeline_root(output_dir, company_name, pipeline) / MANIFEST_FILENAME


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fallback_company_name(company_slug: str) -> str:
    return company_slug.replace("_", " ").title()


def _company_name_from_summary(summary: dict[str, Any], company_slug: str) -> str:
    company_name = summary.get("company_name")
    if isinstance(company_name, str) and company_name.strip():
        return company_name.strip()

    source_file = summary.get("source_file")
    if isinstance(source_file, str) and source_file.strip():
        return infer_company_name_from_source_stem(Path(source_file).stem)

    return _fallback_company_name(company_slug)


@dataclass
class OutputCompanyState:
    company_name: str
    company_slug: str
    pipeline: str
    root_dir: Path
    latest_dir: Path | None
    latest_available: bool
    available_run_tokens: list[str]
    failed_run_tokens: list[str]
    manifest_file: Path


def _build_output_company_state(root_dir: Path, company_slug: str, pipeline: str) -> OutputCompanyState | None:
    if not root_dir.is_dir():
        return None

    latest_dir = root_dir / "latest"
    latest_available = latest_dir.is_dir()
    runs_dir = root_dir / "runs"
    failed_dir = root_dir / "failed"
    manifest_file = root_dir / MANIFEST_FILENAME

    available_run_tokens = sorted(path.name for path in runs_dir.iterdir() if path.is_dir()) if runs_dir.is_dir() else []
    failed_run_tokens = sorted(path.name for path in failed_dir.iterdir() if path.is_dir()) if failed_dir.is_dir() else []

    if not latest_available and not available_run_tokens and not failed_run_tokens and not manifest_file.is_file():
        return None

    company_name: str | None = None
    latest_summary_path = latest_dir / "run_summary.json"
    if latest_summary_path.is_file():
        company_name = _company_name_from_summary(_load_json(latest_summary_path), company_slug)
    elif manifest_file.is_file():
        payload = _load_json(manifest_file)
        manifest_company_name = payload.get("company_name")
        if isinstance(manifest_company_name, str) and manifest_company_name.strip():
            company_name = manifest_company_name.strip()
    else:
        for candidate_parent in (runs_dir, failed_dir):
            if not candidate_parent.is_dir():
                continue
            for candidate_dir in sorted(path for path in candidate_parent.iterdir() if path.is_dir()):
                summary_path = candidate_dir / "run_summary.json"
                if summary_path.is_file():
                    company_name = _company_name_from_summary(_load_json(summary_path), company_slug)
                    break
            if company_name:
                break

    return OutputCompanyState(
        company_name=company_name or _fallback_company_name(company_slug),
        company_slug=company_slug,
        pipeline=pipeline,
        root_dir=root_dir,
        latest_dir=latest_dir if latest_available else None,
        latest_available=latest_available,
        available_run_tokens=available_run_tokens,
        failed_run_tokens=failed_run_tokens,
        manifest_file=manifest_file,
    )


def _write_output_manifest_for_root(root_dir: Path, company_slug: str, pipeline: str) -> Path | None:
    state = _build_output_company_state(root_dir, company_slug, pipeline)
    if state is None:
        return None

    payload = {
        "company_name": state.company_name,
        "company_slug": state.company_slug,
        "pipeline": state.pipeline,
        "latest_available": state.latest_available,
        "latest_run_dir": str(state.latest_dir) if state.latest_dir else None,
        "available_run_tokens": state.available_run_tokens,
        "failed_run_tokens": state.failed_run_tokens,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    state.manifest_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return state.manifest_file


def write_output_manifest(output_dir: Path, company_name: str, pipeline: str) -> Path | None:
    company_slug = slugify_company_name(company_name)
    return _write_output_manifest_for_root(output_dir / company_slug / pipeline, company_slug, pipeline)


def discover_output_company_states(output_dir: Path, pipeline: str) -> list[OutputCompanyState]:
    if not output_dir.exists():
        return []

    states: list[OutputCompanyState] = []
    for company_dir in sorted(path for path in output_dir.iterdir() if path.is_dir() and not path.name.startswith(".")):
        state = _build_output_company_state(company_dir / pipeline, company_dir.name, pipeline)
        if state is not None:
            states.append(state)
    return states


def refresh_output_manifests(output_dir: Path, pipeline: str | None = None) -> list[Path]:
    if not output_dir.exists():
        return []

    manifest_files: list[Path] = []
    pipelines = (pipeline,) if pipeline else PIPELINE_NAMES
    for company_dir in sorted(path for path in output_dir.iterdir() if path.is_dir() and not path.name.startswith(".")):
        for pipeline_name in pipelines:
            manifest_file = _write_output_manifest_for_root(company_dir / pipeline_name, company_dir.name, pipeline_name)
            if manifest_file is not None:
                manifest_files.append(manifest_file)
    return manifest_files


def iter_latest_run_dirs(output_dir: Path, pipeline: str) -> list[Path]:
    return [state.latest_dir for state in discover_output_company_states(output_dir, pipeline) if state.latest_dir is not None]


def _resolve_relative_run_selector(pipeline_root: Path, explicit_path: Path) -> Path:
    if explicit_path.is_absolute():
        raise ValueError("--run must stay within the selected company/pipeline output folder.")

    resolved_root = pipeline_root.resolve()
    resolved_candidate = (pipeline_root / explicit_path).resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("--run must stay within the selected company/pipeline output folder.") from exc
    return resolved_candidate


def resolve_company_run_dir(output_dir: Path, company_name: str, pipeline: str, run_selector: str | None = None) -> Path:
    pipeline_root = company_pipeline_root(output_dir, company_name, pipeline)
    if run_selector is None or run_selector == "latest":
        return pipeline_root / "latest"

    explicit_path = Path(run_selector)
    if explicit_path.is_absolute():
        raise ValueError("--run must stay within the selected company/pipeline output folder.")

    if "/" in run_selector or "\\" in run_selector:
        return _resolve_relative_run_selector(pipeline_root, explicit_path)

    runs_candidate = pipeline_root / "runs" / run_selector
    if runs_candidate.exists():
        return runs_candidate

    failed_candidate = pipeline_root / "failed" / run_selector
    if failed_candidate.exists():
        return failed_candidate

    return _resolve_relative_run_selector(pipeline_root, explicit_path)


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
        _write_output_manifest_for_root(layout.root_dir, layout.company_slug, layout.pipeline)
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
    _write_output_manifest_for_root(layout.root_dir, layout.company_slug, layout.pipeline)
    return layout.latest_dir


def finalize_failed_run(layout: OutputLayout) -> Path:
    layout.failed_dir.mkdir(parents=True, exist_ok=True)
    failed_run_dir = layout.failed_dir / layout.run_token
    layout.staging_dir.rename(failed_run_dir)
    _cleanup_empty_dir(layout.staging_root)
    _write_output_manifest_for_root(layout.root_dir, layout.company_slug, layout.pipeline)
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
        _write_output_manifest_for_root(pipeline_root, company_slug, pipeline)
        migrations.append((child, destination))
    return migrations
