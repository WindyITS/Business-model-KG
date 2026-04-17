from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from graph.neo4j_loader import Neo4jLoader


def _is_interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _confirm_unload(
    *,
    company_name: str,
    input_reader: Callable[[str], str],
    is_interactive: Callable[[], bool],
) -> bool:
    if not is_interactive():
        print(
            "Refusing to unload from Neo4j without confirmation in a non-interactive terminal. Re-run with --yes.",
            file=sys.stderr,
            flush=True,
        )
        return False

    response = input_reader(f'Unload Neo4j graph footprint for company "{company_name}"? [y/N] ').strip().casefold()
    return response in {"y", "yes"}


def _print_summary(summary: dict[str, int | str]) -> None:
    total_removed = sum(
        int(summary[key])
        for key in (
            "scoped_nodes_deleted",
            "scoped_relationships_deleted",
            "company_relationships_deleted",
            "company_node_deleted",
            "orphan_nodes_deleted",
        )
    )

    if total_removed == 0:
        print(f'No Neo4j graph footprint was found for company "{summary["company_name"]}".', flush=True)
        return

    print("NEO4J COMPANY UNLOAD", flush=True)
    print(f'company: {summary["company_name"]}', flush=True)
    print(f'scoped nodes deleted: {summary["scoped_nodes_deleted"]}', flush=True)
    print(f'scoped relationships deleted: {summary["scoped_relationships_deleted"]}', flush=True)
    print(f'company relationships deleted: {summary["company_relationships_deleted"]}', flush=True)
    print(f'company node deleted: {summary["company_node_deleted"]}', flush=True)
    print(f'orphan shared nodes deleted: {summary["orphan_nodes_deleted"]}', flush=True)


def main(
    argv: list[str] | None = None,
    *,
    input_reader: Callable[[str], str] = input,
    is_interactive: Callable[[], bool] = _is_interactive_terminal,
) -> int:
    parser = argparse.ArgumentParser(description="Unload one company's graph footprint from Neo4j.")
    parser.add_argument("--company", required=True, help="Canonical company name to unload from Neo4j.")
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687", help="Neo4j connection URI.")
    parser.add_argument("--neo4j-user", type=str, default="neo4j", help="Neo4j username.")
    parser.add_argument("--neo4j-password", type=str, default="password", help="Neo4j password.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt and unload immediately.",
    )
    args = parser.parse_args(argv)

    if not args.yes and not _confirm_unload(
        company_name=args.company,
        input_reader=input_reader,
        is_interactive=is_interactive,
    ):
        print("Aborted; nothing was deleted.", file=sys.stderr, flush=True)
        return 1

    loader = Neo4jLoader(uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password)
    try:
        summary = loader.unload_company(args.company)
    finally:
        loader.close()

    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
