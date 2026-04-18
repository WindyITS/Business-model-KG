from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

from llm.extractor import LLMExtractor
from llm_extraction.models import ExtractionError
from neo4j import GraphDatabase

from .cypher_validation import normalize_neo4j_uri
from .model_provider import resolve_model_settings
from .query_planner import QueryPlanEnvelope, QueryResult, compile_query_plan, validate_compiled_query
from .query_prompt import QUERY_SYSTEM_PROMPT

QUERY_FALLBACK_PAYLOAD = '{"answerable": false, "reason": "beyond_local_coverage"}'
PARAM_REF_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
DEFAULT_QUERY_MAX_OUTPUT_TOKENS = 1200


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return str(value)


def _question_text(question_parts: list[str]) -> str:
    question = " ".join(part.strip() for part in question_parts).strip()
    if not question:
        raise ValueError("Question must not be empty.")
    return question


def _query_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": QUERY_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def generate_query(
    *,
    question: str,
    extractor: LLMExtractor,
    max_retries: int,
) -> tuple[QueryResult, str | None, int, dict[str, Any]]:
    plan, raw_response, attempts_used, audit = extractor.generate_structured_output(
        messages=_query_messages(question),
        schema_name="QueryPlanEnvelope",
        schema_model=QueryPlanEnvelope,
        fallback_payload=QUERY_FALLBACK_PAYLOAD,
        max_retries=max_retries,
        temperature=0.0,
    )
    result = compile_query_plan(plan)
    return result, raw_response, attempts_used, audit


def validate_generated_query(result: QueryResult) -> list[str]:
    return validate_compiled_query(result)


def execute_live_query(
    *,
    cypher: str,
    params: dict[str, Any],
    neo4j_uri: str | None,
    neo4j_user: str,
    neo4j_password: str,
) -> tuple[list[str], list[dict[str, Any]], str]:
    normalized_uri = normalize_neo4j_uri(neo4j_uri)
    driver = GraphDatabase.driver(normalized_uri, auth=(neo4j_user, neo4j_password))
    try:
        with driver.session() as session:
            result = session.run(cypher, **params)
            columns = list(result.keys())
            records = [_json_safe(record.data()) for record in result]
    finally:
        driver.close()
    return columns, records, normalized_uri


def preflight_live_query(
    *,
    cypher: str,
    params: dict[str, Any],
    neo4j_uri: str | None,
    neo4j_user: str,
    neo4j_password: str,
) -> str:
    normalized_uri = normalize_neo4j_uri(neo4j_uri)
    driver = GraphDatabase.driver(normalized_uri, auth=(neo4j_user, neo4j_password))
    try:
        with driver.session() as session:
            session.run(f"EXPLAIN {cypher}", **params).consume()
    finally:
        driver.close()
    return normalized_uri


def _print_status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _print_output(message: str) -> None:
    print(message, file=sys.stdout, flush=True)


def _format_cell_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _format_cypher_literal(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_format_cypher_literal(item) for item in value) + "]"
    if isinstance(value, tuple):
        return "[" + ", ".join(_format_cypher_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        items = [f"{key}: {_format_cypher_literal(item)}" for key, item in value.items()]
        return "{" + ", ".join(items) + "}"
    return _format_cypher_literal(str(value))


def _render_runnable_query(cypher: str, params: dict[str, Any]) -> str:
    if not params:
        return cypher

    def replace_param(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in params:
            return match.group(0)
        return _format_cypher_literal(params[name])

    return PARAM_REF_RE.sub(replace_param, cypher)


def _render_query_results(columns: list[str], rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""

    if len(columns) == 1:
        column = columns[0]
        return "\n".join(_format_cell_value(row.get(column)) for row in rows)

    rendered_rows = ["\t".join(columns)]
    for row in rows:
        rendered_rows.append("\t".join(_format_cell_value(row.get(column)) for column in columns))
    return "\n".join(rendered_rows)


def _build_parser(*, execute: bool) -> argparse.ArgumentParser:
    description = (
        "Generate read-only Cypher from a natural-language question and run it against the current Neo4j database."
        if execute
        else "Generate read-only Cypher from a natural-language question for the current business-model graph."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("question", nargs="+", help="Natural-language question about the current knowledge graph.")
    parser.add_argument(
        "--provider",
        choices=["local", "opencode-go"],
        default="local",
        help="Provider preset. Use local for the same local server used by the extraction runtime.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL root for the selected API.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or ID to use. If omitted, the provider default is used.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the selected provider. If omitted, provider-specific environment variables are used.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Maximum completion tokens per model call.",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum LLM retries per call.")
    parser.add_argument(
        "--repair-attempts",
        type=int,
        default=0,
        help="Reserved for compatibility. Planner-based queries refuse instead of attempting plan repair.",
    )
    if execute:
        parser.add_argument("--neo4j-uri", type=str, default=None, help="Neo4j connection URI.")
        parser.add_argument("--neo4j-user", type=str, default="neo4j", help="Neo4j username.")
        parser.add_argument("--neo4j-password", type=str, default="password", help="Neo4j password.")
    return parser


def _run(argv: list[str] | None, *, execute: bool) -> int:
    parser = _build_parser(execute=execute)
    args = parser.parse_args(argv)
    question = _question_text(args.question)

    try:
        model_settings = resolve_model_settings(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            max_output_tokens=args.max_output_tokens,
        )
        extractor = LLMExtractor(
            base_url=model_settings.base_url,
            api_key=model_settings.api_key,
            model=model_settings.model,
            provider=model_settings.provider,
            api_mode=model_settings.api_mode,
            max_output_tokens=model_settings.max_output_tokens or DEFAULT_QUERY_MAX_OUTPUT_TOKENS,
        )
        _print_status("Generating query plan...")
        query_result, _raw_response, _attempts_used, _audit = generate_query(
            question=question,
            extractor=extractor,
            max_retries=args.max_retries,
        )
    except (ExtractionError, ValueError) as exc:
        _print_status(f"Error: {exc}")
        return 1

    if not query_result.answerable:
        _print_status(f"Unsupported request: {query_result.reason}")
        return 0

    validation_failures = validate_generated_query(query_result)
    if validation_failures:
        _print_status("Unsupported request: beyond_local_coverage")
        return 0

    if not execute:
        _print_output(_render_runnable_query(query_result.cypher or "", _json_safe(query_result.params)))
        return 0

    try:
        preflight_live_query(
            cypher=query_result.cypher or "",
            params=query_result.params,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
        )
    except Exception as exc:
        _print_status(f"Neo4j preflight error: {exc}")
        return 1

    try:
        _print_status("Running query on Neo4j...")
        columns, rows, normalized_uri = execute_live_query(
            cypher=query_result.cypher or "",
            params=query_result.params,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
        )
    except Exception as exc:
        _print_status(f"Neo4j execution error: {exc}")
        return 1

    rendered_rows = _render_query_results(columns, rows)
    if rendered_rows:
        _print_output(rendered_rows)
    else:
        _print_status(f"No rows returned from {normalized_uri}.")
    return 0


def main_query(argv: list[str] | None = None) -> int:
    return _run(argv, execute=True)


def main_query_cypher(argv: list[str] | None = None) -> int:
    return _run(argv, execute=False)


if __name__ == "__main__":
    raise SystemExit(main_query())
