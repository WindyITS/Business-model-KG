from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

from graph.neo4j_loader import Neo4jLoader
from llm_extraction.pipelines import implemented_pipeline_names
from runtime.output_layout import discover_output_company_states
from runtime.query_stack import local_stack_src, resolve_local_stack_config, resolve_local_stack_python


@dataclass
class HealthCheckResult:
    name: str
    status: str
    detail: str
    hint: str | None = None


def _project_root() -> Path:
    cwd = Path.cwd()
    if _looks_like_repo_root(cwd):
        return cwd
    return Path(__file__).resolve().parents[2]


def _looks_like_repo_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src").is_dir()


def _check_python_version() -> HealthCheckResult:
    version = sys.version_info
    if version >= (3, 10):
        return HealthCheckResult("python", "ok", f"{version.major}.{version.minor}.{version.micro}")
    return HealthCheckResult("python", "fail", f"{version.major}.{version.minor}.{version.micro}", "Python 3.10+ is required.")


def _check_repo_venv(root_dir: Path) -> HealthCheckResult:
    python_bin = root_dir / "venv" / "bin" / "python"
    if python_bin.exists():
        return HealthCheckResult("repo venv", "ok", str(python_bin))
    return HealthCheckResult(
        "repo venv",
        "warn",
        f"expected {python_bin}",
        "Run ./scripts/bootstrap_dev.sh, or set KG_PYTHON for the wrapper scripts.",
    )


def _check_env_example(root_dir: Path) -> HealthCheckResult:
    path = root_dir / ".env.example"
    if path.is_file():
        return HealthCheckResult("env template", "ok", str(path))
    return HealthCheckResult("env template", "fail", "missing .env.example", "Add .env.example so local defaults are visible.")


def _check_packaging_tools() -> HealthCheckResult:
    missing: list[str] = []
    versions: list[str] = []
    for name in ("pip", "setuptools", "wheel"):
        try:
            module = importlib.import_module(name)
        except Exception:
            missing.append(name)
            continue
        versions.append(f"{name}={getattr(module, '__version__', 'installed')}")

    if missing:
        return HealthCheckResult(
            "packaging tools",
            "fail",
            f'missing {", ".join(missing)}',
            "Run ./scripts/bootstrap_dev.sh to refresh pip/setuptools/wheel in the repo environment.",
        )
    return HealthCheckResult("packaging tools", "ok", ", ".join(versions))


def _check_query_stack(root_dir: Path) -> HealthCheckResult:
    python_path = Path(resolve_local_stack_python(root_dir=root_dir))
    config_value = resolve_local_stack_config(root_dir=root_dir)
    config_path = Path(config_value) if config_value else None
    src_dir = local_stack_src(root_dir)
    hint = "Routed queries fall back to the hosted planner automatically; fix this only if you want the local Qwen stack available."

    if not python_path.exists():
        return HealthCheckResult("query stack", "warn", f"local stack python missing at {python_path}", hint)
    if not src_dir.is_dir():
        return HealthCheckResult("query stack", "warn", f"local stack source missing at {src_dir}", hint)
    if config_path is not None and not config_path.is_file():
        return HealthCheckResult("query stack", "warn", f"local stack config missing at {config_path}", hint)

    config_detail = str(config_path) if config_path is not None else "stack default"
    return HealthCheckResult(
        "query stack",
        "ok",
        f"python={python_path}; src={src_dir}; config={config_detail}",
    )


def _check_prompts(root_dir: Path) -> HealthCheckResult:
    repo_prompts = root_dir / "prompts"
    bundled_prompts = root_dir / "src" / "llm_extraction" / "_bundled_prompts"
    if repo_prompts.is_dir() and bundled_prompts.is_dir():
        return HealthCheckResult("prompts", "ok", "repo prompts and bundled prompts are present")
    if repo_prompts.is_dir():
        return HealthCheckResult("prompts", "warn", "repo prompts found but bundled prompts missing", "Run scripts/sync_bundled_prompts.py before packaging.")
    if bundled_prompts.is_dir():
        return HealthCheckResult("prompts", "warn", "bundled prompts found but repo prompts missing", "Restore the editable prompts/ directory.")
    return HealthCheckResult("prompts", "fail", "no prompt assets found", "Check prompts/ and bundled prompt assets.")


def _check_ontology(root_dir: Path) -> HealthCheckResult:
    ontology_path = root_dir / "src" / "ontology" / "ontology.json"
    if ontology_path.is_file():
        return HealthCheckResult("ontology", "ok", str(ontology_path))
    return HealthCheckResult("ontology", "fail", "missing src/ontology/ontology.json", "Restore the canonical ontology file.")


def _check_outputs(root_dir: Path, output_dir: Path, pipeline: str) -> HealthCheckResult:
    resolved_output_dir = output_dir if output_dir.is_absolute() else (root_dir / output_dir)
    states = discover_output_company_states(resolved_output_dir, pipeline)
    latest_count = sum(1 for state in states if state.latest_available)
    archived_count = sum(len(state.available_run_tokens) for state in states)
    if latest_count:
        detail = f'{latest_count} compan{"y" if latest_count == 1 else "ies"} with latest "{pipeline}" output'
        if archived_count:
            detail += f"; {archived_count} archived run(s)"
        return HealthCheckResult("outputs", "ok", detail)
    if states:
        return HealthCheckResult(
            "outputs",
            "warn",
            f'no latest "{pipeline}" outputs; {archived_count} archived run(s) found',
            "Run the pipeline or load an existing saved run explicitly.",
        )
    return HealthCheckResult(
        "outputs",
        "warn",
        f'no "{pipeline}" outputs found under {resolved_output_dir}',
        "Run the pipeline to create outputs, or check --output-dir.",
    )


def _check_neo4j(uri: str, user: str, password: str, *, require: bool) -> HealthCheckResult:
    loader = Neo4jLoader(uri=uri, user=user, password=password)
    try:
        counts = loader.graph_counts()
    except Exception as exc:  # pragma: no cover - real connection failures are environment-dependent.
        status = "fail" if require else "warn"
        hint = "Start Neo4j with docker compose up -d, or use --skip-neo4j/--require-neo4j depending on your workflow."
        return HealthCheckResult("neo4j", status, str(exc), hint)
    finally:
        loader.close()

    return HealthCheckResult(
        "neo4j",
        "ok",
        f'{counts["node_count"]} node(s), {counts["relationship_count"]} relationship(s)',
    )


def _render_result(result: HealthCheckResult) -> str:
    label = result.status.upper().ljust(4)
    text = f"[{label}] {result.name}: {result.detail}"
    if result.hint:
        text += f" | hint: {result.hint}"
    return text


def _summary(results: list[HealthCheckResult]) -> tuple[int, int, int]:
    ok_count = sum(result.status == "ok" for result in results)
    warn_count = sum(result.status == "warn" for result in results)
    fail_count = sum(result.status == "fail" for result in results)
    return ok_count, warn_count, fail_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check whether the local repo and optional Neo4j service are ready to use.")
    parser.add_argument("--project-root", type=Path, default=None, help="Optional repo root override.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Root outputs directory to inspect.")
    parser.add_argument(
        "--pipeline",
        choices=implemented_pipeline_names(),
        default="analyst",
        help="Pipeline output family to inspect when checking saved outputs.",
    )
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687", help="Neo4j connection URI.")
    parser.add_argument("--neo4j-user", type=str, default="neo4j", help="Neo4j username.")
    parser.add_argument("--neo4j-password", type=str, default="password", help="Neo4j password.")
    parser.add_argument("--skip-neo4j", action="store_true", help="Skip the Neo4j connectivity check.")
    parser.add_argument(
        "--require-neo4j",
        action="store_true",
        help="Treat a Neo4j connectivity problem as a failing check instead of a warning.",
    )
    args = parser.parse_args(argv)

    root_dir = args.project_root.resolve() if args.project_root else _project_root()
    results = [
        _check_python_version(),
        _check_repo_venv(root_dir),
        _check_packaging_tools(),
        _check_env_example(root_dir),
        _check_query_stack(root_dir),
        _check_prompts(root_dir),
        _check_ontology(root_dir),
        _check_outputs(root_dir, args.output_dir, args.pipeline),
    ]
    if not args.skip_neo4j:
        results.append(_check_neo4j(args.neo4j_uri, args.neo4j_user, args.neo4j_password, require=args.require_neo4j))

    print("REPO HEALTH CHECK", flush=True)
    for result in results:
        print(_render_result(result), flush=True)

    ok_count, warn_count, fail_count = _summary(results)
    print(f"Summary: {ok_count} ok, {warn_count} warning(s), {fail_count} failing check(s).", flush=True)
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
