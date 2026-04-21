from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Literal

from llm.extractor import LLMExtractor
from llm_extraction.models import ExtractionError
from neo4j import GraphDatabase

from .cypher_validation import normalize_neo4j_uri
from .local_query_stack import run_local_query_stack
from .model_provider import resolve_model_settings
from .query_planner import QueryPlanEnvelope, QueryResult, compile_query_plan, validate_compiled_query
from .query_prompt import QUERY_SYSTEM_PROMPT

QUERY_FALLBACK_PAYLOAD = '{"answerable": false, "reason": "beyond_local_coverage"}'
PARAM_REF_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
DEFAULT_QUERY_MAX_OUTPUT_TOKENS = 1200
DEFAULT_ROUTED_FALLBACK_PROVIDER = "opencode-go"
ROUTED_STACK_MODE = "routed"
FALLBACK_STACK_MODE = "fallback"
QueryStackSelection = Literal["routed", "fallback"]

WRITE_REQUEST_RE = re.compile(r"\b(add|create|delete|drop|insert|load|merge|remove|set|update|upsert|write)\b", re.I)
TIME_REQUEST_RE = re.compile(r"\b(\d{4}|quarter|q[1-4]|month|week|year|yoy|over time|trend|historic)\b", re.I)


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


def _tail(text: str, *, limit: int = 300) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"...{compact[-(limit - 3):]}"


def _run_local_stack(
    *,
    question: str,
    local_stack_bundle_dir: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return run_local_query_stack(question, bundle_dir=local_stack_bundle_dir), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _infer_refusal_reason(question: str) -> str:
    if WRITE_REQUEST_RE.search(question):
        return "write_request"
    if TIME_REQUEST_RE.search(question):
        return "unsupported_time"
    return "beyond_local_coverage"


def _result_from_local_stack(payload: dict[str, Any], *, question: str) -> tuple[QueryResult | None, str]:
    decision = str(payload.get("decision", "")).strip().casefold()
    if decision == "refuse":
        return QueryResult(answerable=False, reason=_infer_refusal_reason(question)), decision
    if decision != "local":
        return None, decision

    compiled = payload.get("compiled")
    if not isinstance(compiled, dict):
        return None, decision
    cypher = compiled.get("cypher")
    params = compiled.get("params")
    if not isinstance(cypher, str) or not isinstance(params, dict):
        return None, decision
    return QueryResult(answerable=True, cypher=cypher, params=params), decision


def _local_planner_error(payload: dict[str, Any]) -> str | None:
    decision = str(payload.get("decision", "")).strip().casefold()
    if decision != "api_fallback":
        return None
    planner = payload.get("planner")
    if not isinstance(planner, dict):
        return None
    planner_error = planner.get("error")
    if isinstance(planner_error, str) and planner_error.strip():
        return planner_error.strip()
    return None


def _build_extractor(
    *,
    provider: str,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    max_output_tokens: int | None,
) -> LLMExtractor:
    model_settings = resolve_model_settings(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_output_tokens=max_output_tokens,
    )
    return LLMExtractor(
        base_url=model_settings.base_url,
        api_key=model_settings.api_key,
        model=model_settings.model,
        provider=model_settings.provider,
        api_mode=model_settings.api_mode,
        max_output_tokens=model_settings.max_output_tokens or DEFAULT_QUERY_MAX_OUTPUT_TOKENS,
    )


def _generate_query_with_provider(
    *,
    question: str,
    provider: str,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    max_output_tokens: int | None,
    max_retries: int,
) -> QueryResult:
    extractor = _build_extractor(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_output_tokens=max_output_tokens,
    )
    query_result, _raw_response, _attempts_used, _audit = generate_query(
        question=question,
        extractor=extractor,
        max_retries=max_retries,
    )
    return query_result


def _retry_question_with_error_context(question: str, error: Exception) -> str:
    return "\n".join(
        [
            question,
            "",
            "Planner retry context:",
            f"Previous fallback planner call failed with: {_tail(str(error), limit=500)}",
            "Return only compact valid JSON matching the required planner schema.",
        ]
    )


def _generate_fallback_query_with_retry(
    *,
    question: str,
    provider: str,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    max_output_tokens: int | None,
    max_retries: int,
) -> QueryResult:
    try:
        return _generate_query_with_provider(
            question=question,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
        )
    except Exception as first_error:  # noqa: BLE001
        _print_status(f"Fallback planner error (attempt 1): {_tail(str(first_error))}")
        _print_status("Retrying fallback planner once with error context...")
        retry_question = _retry_question_with_error_context(question, first_error)
        try:
            return _generate_query_with_provider(
                question=retry_question,
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
                max_output_tokens=max_output_tokens,
                max_retries=max_retries,
            )
        except Exception as second_error:  # noqa: BLE001
            _print_status("Warning: fallback planner failed twice in a row.")
            raise ValueError(
                "Fallback planner failed twice. "
                f"first={_tail(str(first_error), limit=220)}; "
                f"second={_tail(str(second_error), limit=220)}"
            ) from second_error


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


def _report_local_stack_error(error_detail: str) -> None:
    _print_status(f"Local query stack unavailable: {_tail(error_detail)}")
    _print_status("Falling back to hosted planner.")


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


def _resolve_query_result(
    *,
    question: str,
    args: argparse.Namespace,
) -> QueryResult:
    force_fallback = bool(getattr(args, "skip_local_stack", False)) or getattr(args, "stack", ROUTED_STACK_MODE) == FALLBACK_STACK_MODE
    if not force_fallback:
        _print_status("Routing with local deployed query stack...")
        local_payload, local_error = _run_local_stack(
            question=question,
            local_stack_bundle_dir=args.local_stack_bundle_dir,
        )
        if local_payload is not None:
            local_result, decision = _result_from_local_stack(local_payload, question=question)
            if local_result is not None:
                if local_result.answerable:
                    _print_status("Router decision: local (using local Qwen planner output).")
                else:
                    _print_status(f"Router decision: refuse ({local_result.reason}).")
                return local_result
            planner_error = _local_planner_error(local_payload)
            if planner_error:
                _report_local_stack_error(planner_error)
            elif decision == "api_fallback":
                _print_status("Router decision: api_fallback (using hosted planner fallback).")
            else:
                _print_status(f"Router decision: {decision or 'unknown'} (using hosted planner fallback).")
        elif local_error:
            _report_local_stack_error(local_error)
    else:
        _print_status("Using hosted planner only.")

    _print_status("Generating hosted fallback query plan...")
    return _generate_fallback_query_with_retry(
        question=question,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_output_tokens=args.max_output_tokens,
        max_retries=args.max_retries,
    )


def _build_parser(*, execute: bool) -> argparse.ArgumentParser:
    description = (
        "Route query with the published local query stack first, then use hosted planner fallback when needed, and run on Neo4j."
        if execute
        else "Route query with the published local query stack first, then use hosted planner fallback when needed, and render Cypher."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("question", nargs="+", help="Natural-language question about the current knowledge graph.")
    parser.add_argument(
        "--stack",
        choices=["routed", "fallback"],
        default="routed",
        help="Execution stack: routed (default) or fallback (skip the local query stack).",
    )
    parser.add_argument(
        "--provider",
        "--fallback-provider",
        dest="provider",
        choices=["opencode-go"],
        default=DEFAULT_ROUTED_FALLBACK_PROVIDER,
        help="Hosted fallback planner provider.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for the hosted fallback planner API.",
    )
    parser.add_argument(
        "--skip-local-stack",
        action="store_true",
        help="Bypass the published local query stack and always use hosted fallback planner generation.",
    )
    parser.add_argument(
        "--local-stack-bundle-dir",
        type=str,
        default=None,
        help="Optional override for the published local query-stack bundle directory.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name/ID for the hosted fallback planner.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the hosted fallback planner.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Maximum completion tokens per planner call.",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum planner retries per call.")
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
        query_result = _resolve_query_result(question=question, args=args)
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
