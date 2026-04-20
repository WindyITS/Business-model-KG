from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from graph.neo4j_loader import Neo4jLoader
from llm_extraction.models import Triple
from runtime.output_layout import (
    infer_company_name_from_source_stem,
    iter_latest_run_dirs,
    refresh_output_manifests,
    resolve_company_run_dir,
)


def _is_interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


@dataclass
class LoadTarget:
    company_name: str
    run_dir: Path
    triples: list[Triple]


@dataclass
class LoadFailure:
    company_name: str
    run_dir: Path
    error: str


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_company_name_from_run_summary(run_dir: Path, summary: dict[str, object]) -> str:
    company_name = summary.get("company_name")
    if isinstance(company_name, str) and company_name.strip():
        return company_name.strip()

    source_file = summary.get("source_file")
    if isinstance(source_file, str) and source_file.strip():
        return infer_company_name_from_source_stem(Path(source_file).stem)

    if run_dir.name == "latest" and len(run_dir.parents) >= 3:
        return run_dir.parents[1].name.replace("_", " ").title()
    if run_dir.parent.name in {"runs", "failed"} and len(run_dir.parents) >= 4:
        return run_dir.parents[2].name.replace("_", " ").title()
    if len(run_dir.parents) >= 2:
        return run_dir.parents[1].name.replace("_", " ").title()
    raise ValueError(f"Could not infer company name for run directory: {run_dir}")


def _load_triples_from_run_dir(run_dir: Path) -> list[Triple]:
    resolved_path = run_dir / "resolved_triples.json"
    if resolved_path.is_file():
        payload = _load_json(resolved_path)
        triples_payload = payload.get("triples", [])
    else:
        validation_path = run_dir / "validation_report.json"
        if not validation_path.is_file():
            raise FileNotFoundError(
                f"Run directory {run_dir} does not contain resolved_triples.json or validation_report.json."
            )
        payload = _load_json(validation_path)
        triples_payload = payload.get("valid_triples", [])

    triples = [Triple(**triple) for triple in triples_payload]
    if not triples:
        raise ValueError(f"Run directory {run_dir} does not contain any loadable triples.")
    return triples


def _load_target_from_run_dir(run_dir: Path) -> LoadTarget:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Run directory {run_dir} does not contain run_summary.json.")

    summary = _load_json(summary_path)
    company_name = _infer_company_name_from_run_summary(run_dir, summary)
    triples = _load_triples_from_run_dir(run_dir)
    return LoadTarget(company_name=company_name, run_dir=run_dir, triples=triples)


def _discover_targets(
    *,
    output_dir: Path,
    pipeline: str,
    company_name: str | None,
    run_selector: str | None,
) -> list[LoadTarget]:
    if company_name:
        run_dir = resolve_company_run_dir(output_dir, company_name, pipeline, run_selector)
        return [_load_target_from_run_dir(run_dir)]

    if run_selector is not None:
        raise ValueError("--run requires --company.")

    latest_dirs = iter_latest_run_dirs(output_dir, pipeline)
    if not latest_dirs:
        return []
    return [_load_target_from_run_dir(run_dir) for run_dir in latest_dirs]


def _confirm_bulk_load(
    *,
    counts: dict[str, int],
    target_count: int,
    pipeline: str,
    input_reader: Callable[[str], str],
    is_interactive: Callable[[], bool],
) -> bool:
    if counts["node_count"] == 0 and counts["relationship_count"] == 0:
        return True

    warning = (
        f"Neo4j already contains {counts['node_count']} nodes and {counts['relationship_count']} relationships. "
        f'Load the latest "{pipeline}" outputs for {target_count} companies anyway? '
        "This will replace matching companies and keep unrelated graph data. [y/N] "
    )
    if not is_interactive():
        print(
            "Refusing to bulk-load into a non-empty Neo4j database without confirmation. Re-run with --yes.",
            file=sys.stderr,
            flush=True,
        )
        print(warning, file=sys.stderr, flush=True)
        return False

    response = input_reader(warning).strip().casefold()
    return response in {"y", "yes"}


def _print_target_summary(target: LoadTarget, unload_summary: dict[str, int | str], loaded_triple_count: int) -> None:
    replaced_count = sum(
        int(unload_summary[key])
        for key in (
            "scoped_nodes_deleted",
            "scoped_relationships_deleted",
            "company_relationships_deleted",
            "company_node_deleted",
            "orphan_nodes_deleted",
        )
    )
    if replaced_count:
        replacement_text = f"replaced {replaced_count} existing company graph items"
    else:
        replacement_text = "no prior company graph found"

    print(
        f'loaded {target.company_name} from {target.run_dir} '
        f"({loaded_triple_count} triples; {replacement_text})",
        flush=True,
    )


def _company_is_loaded(counts: dict[str, int]) -> bool:
    return any(counts.values())


def _confirm_company_reload(
    *,
    target: LoadTarget,
    counts: dict[str, int],
    input_reader: Callable[[str], str],
    is_interactive: Callable[[], bool],
) -> bool:
    warning = (
        f'Neo4j already contains data for company "{target.company_name}" '
        f'({counts["company_node_count"]} company node(s), '
        f'{counts["scoped_node_count"]} scoped node(s), '
        f'{counts["relationship_count"]} relationship(s)). '
        f"Replace it with {target.run_dir}? [y/N] "
    )
    if not is_interactive():
        print(
            "Refusing to replace an already-loaded company in a non-interactive terminal without confirmation. "
            "Re-run with --yes.",
            file=sys.stderr,
            flush=True,
        )
        print(warning, file=sys.stderr, flush=True)
        return False

    response = input_reader(warning).strip().casefold()
    return response in {"y", "yes"}


def _print_load_failure(failure: LoadFailure) -> None:
    print(
        f'Failed to load {failure.company_name} from {failure.run_dir}: {failure.error}',
        file=sys.stderr,
        flush=True,
    )


def _print_final_summary(
    *,
    pipeline: str,
    total_targets: int,
    successful_targets: int,
    total_loaded: int,
    failures: list[LoadFailure],
) -> None:
    if failures:
        print(
            f'Loaded {successful_targets} of {total_targets} "{pipeline}" output(s) into Neo4j '
            f"({total_loaded} triples total). {len(failures)} compan"
            f'{"y" if len(failures) == 1 else "ies"} could not be loaded.',
            flush=True,
        )
        return

    print(
        f'Loaded {successful_targets} "{pipeline}" output(s) into Neo4j ({total_loaded} triples total).',
        flush=True,
    )


def main(
    argv: list[str] | None = None,
    *,
    input_reader: Callable[[str], str] = input,
    is_interactive: Callable[[], bool] = _is_interactive_terminal,
) -> int:
    parser = argparse.ArgumentParser(description="Load saved pipeline outputs into Neo4j.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Root outputs directory.")
    parser.add_argument(
        "--pipeline",
        choices=["literal", "analyst"],
        default="analyst",
        help="Pipeline output family to load. Defaults to analyst latest outputs; use literal explicitly for the staged literal extractor outputs.",
    )
    parser.add_argument("--company", type=str, default=None, help="Load only one company's saved output.")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Optional run selector used with --company. Use latest, a run token under runs/ or failed/, or a relative path inside the company/pipeline folder.",
    )
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687", help="Neo4j connection URI.")
    parser.add_argument("--neo4j-user", type=str, default="neo4j", help="Neo4j username.")
    parser.add_argument("--neo4j-password", type=str, default="password", help="Neo4j password.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts when bulk-loading into a non-empty database or replacing an already-loaded company.",
    )
    args = parser.parse_args(argv)

    try:
        refresh_output_manifests(args.output_dir, pipeline=args.pipeline)
        targets = _discover_targets(
            output_dir=args.output_dir,
            pipeline=args.pipeline,
            company_name=args.company,
            run_selector=args.run,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1

    if not targets:
        if args.company:
            print(f'No saved "{args.pipeline}" output was found for company "{args.company}".', file=sys.stderr, flush=True)
        else:
            print(f'No latest "{args.pipeline}" outputs were found under {args.output_dir}.', file=sys.stderr, flush=True)
        return 1

    loader = Neo4jLoader(uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password)
    try:
        if args.company is None:
            counts = loader.graph_counts()
            if not args.yes and not _confirm_bulk_load(
                counts=counts,
                target_count=len(targets),
                pipeline=args.pipeline,
                input_reader=input_reader,
                is_interactive=is_interactive,
            ):
                print("Aborted; nothing was loaded.", file=sys.stderr, flush=True)
                return 1
        else:
            target = targets[0]
            if not args.yes:
                company_counts = loader.company_graph_counts(target.company_name)
                if _company_is_loaded(company_counts) and not _confirm_company_reload(
                    target=target,
                    counts=company_counts,
                    input_reader=input_reader,
                    is_interactive=is_interactive,
                ):
                    print("Aborted; nothing was loaded.", file=sys.stderr, flush=True)
                    return 1

        loader.setup_constraints()

        total_loaded = 0
        failures: list[LoadFailure] = []
        successful_targets = 0
        for target in targets:
            try:
                unload_summary, loaded_triple_count = loader.replace_company_triples(
                    target.triples,
                    company_name=target.company_name,
                )
            except Exception as exc:
                failure = LoadFailure(company_name=target.company_name, run_dir=target.run_dir, error=str(exc))
                failures.append(failure)
                _print_load_failure(failure)
                if args.company is not None:
                    return 1
                continue

            successful_targets += 1
            total_loaded += loaded_triple_count
            _print_target_summary(target, unload_summary, loaded_triple_count)
    finally:
        loader.close()

    _print_final_summary(
        pipeline=args.pipeline,
        total_targets=len(targets),
        successful_targets=successful_targets,
        total_loaded=total_loaded,
        failures=failures,
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
