from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config, repo_root
from .frozen_prompt import FROZEN_QUERY_SYSTEM_PROMPT
from .json_utils import compact_json
from .paths import planner_adapter_dir, router_eval_dir, router_model_dir


DEFAULT_PUBLISH_DIR = repo_root() / "runtime_assets" / "query_stack" / "current"
MANIFEST_FILENAME = "manifest.json"


def _resolve_destination(destination: str | None) -> Path:
    if destination is None:
        return DEFAULT_PUBLISH_DIR.resolve()
    return Path(destination).expanduser().resolve()


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Directory not found: {source}")
    shutil.copytree(source, destination)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _replace_directory(staging_dir: Path, destination_dir: Path) -> None:
    backup_dir = destination_dir.parent / f".{destination_dir.name}.bak"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if destination_dir.exists():
        destination_dir.rename(backup_dir)
    try:
        staging_dir.rename(destination_dir)
    except Exception:
        if backup_dir.exists() and not destination_dir.exists():
            backup_dir.rename(destination_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def publish_query_stack(config_path: str | None = None, *, destination: str | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    destination_dir = _resolve_destination(destination)
    staging_dir = destination_dir.parent / f".{destination_dir.name}.staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=False)

    router_source_dir = router_model_dir(config)
    router_thresholds_path = router_eval_dir(config) / "thresholds.json"
    planner_source_dir = planner_adapter_dir(config)
    if not router_thresholds_path.is_file():
        raise FileNotFoundError(f"Router thresholds file not found: {router_thresholds_path}")

    try:
        _copy_tree(router_source_dir, staging_dir / "router" / "model")
        (staging_dir / "router").mkdir(parents=True, exist_ok=True)
        shutil.copy2(router_thresholds_path, staging_dir / "router" / "thresholds.json")
        _copy_tree(planner_source_dir, staging_dir / "planner" / "adapter")
        (staging_dir / "planner" / "system_prompt.txt").write_text(
            FROZEN_QUERY_SYSTEM_PROMPT + "\n",
            encoding="utf-8",
        )

        manifest = {
            "bundle_format_version": 1,
            "published_at": datetime.now(timezone.utc).isoformat(),
            "router": {
                "base_model": config.router.base_model,
                "max_length": config.router.max_length,
                "model_dir": "router/model",
                "thresholds_path": "router/thresholds.json",
            },
            "planner": {
                "base_model": config.planner.base_model,
                "max_tokens": config.planner.max_tokens,
                "adapter_dir": "planner/adapter",
                "system_prompt_path": "planner/system_prompt.txt",
            },
        }
        _write_json(staging_dir / MANIFEST_FILENAME, manifest)
        destination_dir.parent.mkdir(parents=True, exist_ok=True)
        _replace_directory(staging_dir, destination_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return {
        "destination_dir": str(destination_dir),
        "manifest_path": str(destination_dir / MANIFEST_FILENAME),
        "router_model_dir": str(destination_dir / "router" / "model"),
        "router_thresholds_path": str(destination_dir / "router" / "thresholds.json"),
        "planner_adapter_dir": str(destination_dir / "planner" / "adapter"),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish the fine-tuned query stack as a main-runtime deployment bundle.")
    parser.add_argument("--config", type=str, default=None, help="Path to the fine-tuning JSON config.")
    parser.add_argument("--dest", type=str, default=None, help="Optional destination override for the published bundle.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = publish_query_stack(args.config, destination=args.dest)
    print(compact_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
