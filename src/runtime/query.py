from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

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
DEFAULT_ROUTED_FALLBACK_PROVIDER = "opencode-go"
DEFAULT_JOLLY_PROVIDER = "local"
ROUTED_STACK_MODE = "routed"
JOLLY_STACK_MODE = "jolly"
QueryStackMode = Literal["routed", "jolly"]
QueryStackSelection = Literal["routed", "fallback", "jolly"]
LOCAL_STACK_MODULE = "kg_query_planner_ft.local_stack"

WRITE_REQUEST_RE = re.compile(r"\b(add|create|delete|drop|insert|load|merge|remove|set|update|upsert|write)\b", re.I)
TIME_REQUEST_RE = re.compile(r"\b(\d{4}|quarter|q[1-4]|month|week|year|yoy|over time|trend|historic)\b", re.I)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_local_stack_python() -> str:
    return str(_repo_root() / "finetuning" / ".venv" / "bin" / "python")


def _default_local_stack_config() -> str:
    return str(_repo_root() / "finetuning" / "config" / "default.json")


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


def _extract_first_json_object(raw_text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw_text):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(raw_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("No JSON object found in local stack output.")


def _tail(text: str, *, limit: int = 300) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"...{compact[-(limit - 3):]}"


def _is_interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _prompt_yes_no(prompt: str, *, default_yes: bool = True) -> bool:
    while True:
        response = input(prompt).strip().casefold()
        if not response:
            return default_yes
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        _print_status("Please answer Y or n.")


def _confirm_api_fallback_after_local_error(error_detail: str) -> bool:
    _print_status(f"Local model error: {_tail(error_detail)}")
    if not _is_interactive_terminal():
        _print_status("Cannot prompt for API fallback in non-interactive mode; aborting.")
        return False
    return _prompt_yes_no("Use API fallback instead? [Y/n] ", default_yes=True)


def _resolve_local_stack_python(explicit_path: str | None) -> str:
    if explicit_path:
        return explicit_path
    env_path = os.getenv("KG_QUERY_LOCAL_STACK_PYTHON")
    if env_path:
        return env_path
    return _default_local_stack_python()


def _resolve_local_stack_config(explicit_path: str | None) -> str | None:
    if explicit_path:
        return explicit_path
    env_path = os.getenv("KG_QUERY_LOCAL_STACK_CONFIG")
    if env_path:
        return env_path
    default_path = _default_local_stack_config()
    return default_path if Path(default_path).exists() else None


def _run_local_stack(
    *,
    question: str,
    local_stack_python: str | None,
    local_stack_config: str | None,
    timeout_seconds: int,
) -> tuple[dict[str, Any] | None, str | None]:
    python_executable = _resolve_local_stack_python(local_stack_python)
    config_path = _resolve_local_stack_config(local_stack_config)
    command = [python_executable, "-m", LOCAL_STACK_MODULE, question]
    if config_path:
        command.extend(["--config", config_path])

    runtime_env = dict(os.environ)
    finetuning_src = _repo_root() / "finetuning" / "src"
    existing_pythonpath = runtime_env.get("PYTHONPATH")
    runtime_env["PYTHONPATH"] = (
        f"{finetuning_src}:{existing_pythonpath}" if existing_pythonpath else str(finetuning_src)
    )

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout_seconds,
            env=runtime_env,
        )
    except FileNotFoundError:
        return None, f"local stack python not found at {python_executable}"
    except subprocess.TimeoutExpired:
        return None, f"local stack timed out after {timeout_seconds}s"
    except OSError as exc:
        return None, f"local stack failed to start: {exc}"

    if completed.returncode != 0:
        detail = _tail(completed.stderr or completed.stdout or "no output")
        return None, f"local stack exited with {completed.returncode}: {detail}"

    try:
        payload = _extract_first_json_object(completed.stdout)
    except ValueError as exc:
        return None, f"{exc} Raw output: {_tail(completed.stdout)}"

    return payload, None


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
    mode: QueryStackMode,
) -> QueryResult:
    if mode == JOLLY_STACK_MODE:
        _print_status("LM Studio jolly mode: generating query plan directly...")
        return _generate_query_with_provider(
            question=question,
            provider=DEFAULT_JOLLY_PROVIDER,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
        )

    force_fallback = bool(getattr(args, "skip_local_stack", False)) or getattr(args, "stack", ROUTED_STACK_MODE) == "fallback"
    if not force_fallback:
        _print_status("Routing with local router + Qwen planner...")
        local_payload, local_error = _run_local_stack(
            question=question,
            local_stack_python=args.local_stack_python,
            local_stack_config=args.local_stack_config,
            timeout_seconds=args.local_stack_timeout,
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
            if planner_error and not _confirm_api_fallback_after_local_error(planner_error):
                raise ValueError("API fallback declined after local model error.")
            if decision == "api_fallback":
                _print_status("Router decision: api_fallback (using remote planner fallback).")
            else:
                _print_status(f"Router decision: {decision or 'unknown'} (falling back to remote planner).")
        elif local_error:
            if not _confirm_api_fallback_after_local_error(local_error):
                raise ValueError("API fallback declined after local model error.")
            _print_status("Falling back to remote planner.")
    else:
        _print_status("Using fallback planner only.")

    _print_status("Generating fallback query plan...")
    return _generate_fallback_query_with_retry(
        question=question,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_output_tokens=args.max_output_tokens,
        max_retries=args.max_retries,
    )


def _build_parser(*, execute: bool, mode: QueryStackMode) -> argparse.ArgumentParser:
    if mode == JOLLY_STACK_MODE:
        description = (
            "LM Studio jolly mode: generate read-only Cypher directly from a local model and run it against Neo4j."
            if execute
            else "LM Studio jolly mode: generate read-only Cypher directly from a local model."
        )
    else:
        description = (
            "Route query with local router+Qwen first, then use remote planner fallback when needed, and run on Neo4j."
            if execute
            else "Route query with local router+Qwen first, then use remote planner fallback when needed, and render Cypher."
        )

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("question", nargs="+", help="Natural-language question about the current knowledge graph.")
    if mode == JOLLY_STACK_MODE:
        parser.add_argument(
            "--base-url",
            type=str,
            default=None,
            help="Base URL for LM Studio's OpenAI-compatible API.",
        )
    else:
        parser.add_argument(
            "--stack",
            choices=["routed", "fallback", "jolly"],
            default="routed",
            help="Execution stack: routed (default), fallback (skip local router+Qwen), or jolly (LM Studio direct).",
        )
        parser.add_argument(
            "--provider",
            "--fallback-provider",
            dest="provider",
            choices=["local", "opencode-go"],
            default=DEFAULT_ROUTED_FALLBACK_PROVIDER,
            help="Planner fallback provider used when local routing decides api_fallback or is unavailable.",
        )
        parser.add_argument(
            "--base-url",
            type=str,
            default=None,
            help="Base URL for selected stack API (fallback provider or LM Studio jolly).",
        )
        parser.add_argument(
            "--skip-local-stack",
            action="store_true",
            help="Bypass local router+Qwen and always use fallback planner generation.",
        )
        parser.add_argument(
            "--local-stack-python",
            type=str,
            default=None,
            help="Python executable for local routing/planning (defaults to finetuning/.venv/bin/python).",
        )
        parser.add_argument(
            "--local-stack-config",
            type=str,
            default=None,
            help="Config path passed to the local router/planner stack.",
        )
        parser.add_argument(
            "--local-stack-timeout",
            type=int,
            default=120,
            help="Timeout in seconds for local router+Qwen planning.",
        )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "LM Studio model name/ID for jolly mode."
            if mode == JOLLY_STACK_MODE
            else "Model name/ID for fallback planner or LM Studio jolly stack."
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "API key for LM Studio's OpenAI-compatible API."
            if mode == JOLLY_STACK_MODE
            else "API key for fallback provider or LM Studio jolly stack."
        ),
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


def _effective_mode(*, mode: QueryStackMode, args: argparse.Namespace) -> QueryStackMode:
    if mode == JOLLY_STACK_MODE:
        return JOLLY_STACK_MODE
    stack = getattr(args, "stack", ROUTED_STACK_MODE)
    if stack == JOLLY_STACK_MODE:
        return JOLLY_STACK_MODE
    return ROUTED_STACK_MODE


def _run(argv: list[str] | None, *, execute: bool, mode: QueryStackMode) -> int:
    parser = _build_parser(execute=execute, mode=mode)
    args = parser.parse_args(argv)
    question = _question_text(args.question)
    effective_mode = _effective_mode(mode=mode, args=args)

    try:
        query_result = _resolve_query_result(question=question, args=args, mode=effective_mode)
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
    return _run(argv, execute=True, mode=ROUTED_STACK_MODE)


def main_query_cypher(argv: list[str] | None = None) -> int:
    return _run(argv, execute=False, mode=ROUTED_STACK_MODE)


def main_query_jolly(argv: list[str] | None = None) -> int:
    return _run(argv, execute=True, mode=JOLLY_STACK_MODE)


def main_query_cypher_jolly(argv: list[str] | None = None) -> int:
    return _run(argv, execute=False, mode=JOLLY_STACK_MODE)


if __name__ == "__main__":
    raise SystemExit(main_query())
