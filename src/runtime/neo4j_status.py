from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from graph.neo4j_loader import Neo4jLoader
from runtime.output_layout import OutputCompanyState, discover_output_company_states, refresh_output_manifests


@dataclass
class Neo4jCompanyStatus:
    company_name: str
    loaded: bool
    latest_available: bool
    latest_dir: Path | None
    available_run_tokens: list[str]
    failed_run_tokens: list[str]


def _build_company_statuses(
    *,
    output_states: list[OutputCompanyState],
    loaded_companies: list[str],
) -> list[Neo4jCompanyStatus]:
    state_by_company = {state.company_name: state for state in output_states}
    loaded_set = set(loaded_companies)
    company_names = sorted(loaded_set | set(state_by_company))
    statuses: list[Neo4jCompanyStatus] = []
    for company_name in company_names:
        state = state_by_company.get(company_name)
        statuses.append(
            Neo4jCompanyStatus(
                company_name=company_name,
                loaded=company_name in loaded_set,
                latest_available=bool(state and state.latest_available),
                latest_dir=state.latest_dir if state else None,
                available_run_tokens=state.available_run_tokens if state else [],
                failed_run_tokens=state.failed_run_tokens if state else [],
            )
        )
    return statuses


def _availability_text(status: Neo4jCompanyStatus, pipeline: str) -> str:
    if status.latest_available and status.latest_dir is not None:
        return f'latest "{pipeline}" output available at {status.latest_dir}'

    details: list[str] = [f'no latest "{pipeline}" output available']
    if status.available_run_tokens:
        details.append(f'{len(status.available_run_tokens)} archived run(s)')
    if status.failed_run_tokens:
        details.append(f'{len(status.failed_run_tokens)} failed run(s)')
    return "; ".join(details)


def _print_status_report(*, pipeline: str, statuses: list[Neo4jCompanyStatus]) -> None:
    loaded = [status for status in statuses if status.loaded]
    not_loaded = [status for status in statuses if not status.loaded]

    print("NEO4J STATUS", flush=True)
    print(f"pipeline: {pipeline}", flush=True)
    print("", flush=True)

    print(f"Loaded companies: {len(loaded)}", flush=True)
    if loaded:
        for status in loaded:
            print(f"- {status.company_name}: {_availability_text(status, pipeline)}", flush=True)
    else:
        print("- none", flush=True)

    print("", flush=True)
    print(f"Not loaded companies: {len(not_loaded)}", flush=True)
    if not_loaded:
        for status in not_loaded:
            print(f"- {status.company_name}: {_availability_text(status, pipeline)}", flush=True)
    else:
        print("- none", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show which companies are loaded in Neo4j and which saved outputs are available.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Root outputs directory.")
    parser.add_argument(
        "--pipeline",
        choices=["canonical", "analyst"],
        default="analyst",
        help="Pipeline output family to compare against Neo4j. Defaults to analyst.",
    )
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687", help="Neo4j connection URI.")
    parser.add_argument("--neo4j-user", type=str, default="neo4j", help="Neo4j username.")
    parser.add_argument("--neo4j-password", type=str, default="password", help="Neo4j password.")
    args = parser.parse_args(argv)

    refresh_output_manifests(args.output_dir, pipeline=args.pipeline)
    output_states = discover_output_company_states(args.output_dir, args.pipeline)

    loader = Neo4jLoader(uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password)
    try:
        loaded_companies = loader.list_loaded_companies()
    finally:
        loader.close()

    statuses = _build_company_statuses(output_states=output_states, loaded_companies=loaded_companies)
    _print_status_report(pipeline=args.pipeline, statuses=statuses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
