from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError


QUERY_STACK_BUNDLE_DIR_ENV = "KG_QUERY_STACK_BUNDLE_DIR"
MANIFEST_FILENAME = "manifest.json"
SUPPORTED_QUERY_STACK_BUNDLE_FORMAT_VERSION = 1


class RouterBundleManifest(BaseModel):
    model_dir: str
    thresholds_path: str
    base_model: str
    max_length: int


class PlannerBundleManifest(BaseModel):
    base_model: str
    adapter_dir: str
    max_tokens: int
    system_prompt_path: str | None = None


class QueryStackBundleManifest(BaseModel):
    bundle_format_version: int
    router: RouterBundleManifest
    planner: PlannerBundleManifest


@dataclass(frozen=True)
class ResolvedQueryStackBundle:
    root_dir: Path
    manifest_path: Path
    manifest: QueryStackBundleManifest
    router_model_dir: Path
    router_thresholds_path: Path
    planner_adapter_dir: Path
    planner_system_prompt_path: Path | None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_query_stack_bundle_dir(root_dir: Path | None = None) -> Path:
    resolved_root = root_dir or repo_root()
    return resolved_root / "runtime_assets" / "query_stack" / "current"


def resolve_query_stack_bundle_dir(explicit_path: str | Path | None = None, *, root_dir: Path | None = None) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()

    env_path = os.getenv(QUERY_STACK_BUNDLE_DIR_ENV)
    if env_path:
        return Path(env_path).expanduser().resolve()

    return default_query_stack_bundle_dir(root_dir).resolve()


def manifest_path(bundle_dir: Path) -> Path:
    return bundle_dir / MANIFEST_FILENAME


def _resolve_bundle_path(bundle_dir: Path, relative_or_absolute: str) -> Path:
    configured = Path(relative_or_absolute).expanduser()
    if configured.is_absolute():
        return configured.resolve()
    return (bundle_dir / configured).resolve()


def load_query_stack_bundle(
    explicit_path: str | Path | None = None,
    *,
    root_dir: Path | None = None,
) -> ResolvedQueryStackBundle:
    bundle_dir = resolve_query_stack_bundle_dir(explicit_path, root_dir=root_dir)
    resolved_manifest_path = manifest_path(bundle_dir)
    if not resolved_manifest_path.is_file():
        raise FileNotFoundError(f"Query-stack manifest not found: {resolved_manifest_path}")

    try:
        payload = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Query-stack manifest is not valid JSON: {resolved_manifest_path}") from exc

    try:
        manifest = QueryStackBundleManifest.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Query-stack manifest has an invalid shape: {resolved_manifest_path}") from exc
    if manifest.bundle_format_version != SUPPORTED_QUERY_STACK_BUNDLE_FORMAT_VERSION:
        raise ValueError(
            "Query-stack manifest uses an unsupported bundle format version: "
            f"{manifest.bundle_format_version} (expected {SUPPORTED_QUERY_STACK_BUNDLE_FORMAT_VERSION})"
        )

    return ResolvedQueryStackBundle(
        root_dir=bundle_dir,
        manifest_path=resolved_manifest_path,
        manifest=manifest,
        router_model_dir=_resolve_bundle_path(bundle_dir, manifest.router.model_dir),
        router_thresholds_path=_resolve_bundle_path(bundle_dir, manifest.router.thresholds_path),
        planner_adapter_dir=_resolve_bundle_path(bundle_dir, manifest.planner.adapter_dir),
        planner_system_prompt_path=(
            _resolve_bundle_path(bundle_dir, manifest.planner.system_prompt_path)
            if manifest.planner.system_prompt_path
            else None
        ),
    )


__all__ = [
    "MANIFEST_FILENAME",
    "QUERY_STACK_BUNDLE_DIR_ENV",
    "PlannerBundleManifest",
    "QueryStackBundleManifest",
    "ResolvedQueryStackBundle",
    "RouterBundleManifest",
    "SUPPORTED_QUERY_STACK_BUNDLE_FORMAT_VERSION",
    "default_query_stack_bundle_dir",
    "load_query_stack_bundle",
    "manifest_path",
    "repo_root",
    "resolve_query_stack_bundle_dir",
]
