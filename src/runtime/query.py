from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

from llm.extractor import LLMExtractor
from llm_extraction.models import ExtractionError
from neo4j import GraphDatabase
from pydantic import BaseModel, Field, model_validator
from text2cypher.prompting import TEXT2CYPHER_REPAIR_SYSTEM_PROMPT, TEXT2CYPHER_SYSTEM_PROMPT
from text2cypher.validation import normalize_neo4j_uri, validate_params_match, validate_read_only_cypher

from .model_provider import resolve_model_settings

QUERY_FALLBACK_PAYLOAD = '{"answerable": false, "reason": "generation_failed"}'
PARAM_REF_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
DEFAULT_REPAIR_ATTEMPTS = 2
DEFAULT_QUERY_MAX_OUTPUT_TOKENS = 1200
DEFAULT_REPAIR_MAX_OUTPUT_TOKENS = 700
SEMANTIC_REPAIR_PARAM_PREFIXES = ("customer_type", "channel", "revenue_model", "offering")
BOOLEAN_HINT_TOKENS = (" and ", " or ", " both ", " either ")


class Text2CypherQueryResult(BaseModel):
    answerable: bool
    cypher: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None

    @model_validator(mode="after")
    def _validate_contract(self) -> "Text2CypherQueryResult":
        if self.answerable:
            if not self.cypher or not self.cypher.strip():
                raise ValueError("Answerable responses must include a non-empty cypher string.")
            if self.reason is not None:
                raise ValueError("Answerable responses must not include a refusal reason.")
            return self

        if not self.reason or not self.reason.strip():
            raise ValueError("Refusal responses must include a non-empty reason.")
        if self.cypher is not None:
            raise ValueError("Refusal responses must not include cypher.")
        if self.params:
            raise ValueError("Refusal responses must not include params.")
        return self


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
        {"role": "system", "content": TEXT2CYPHER_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def generate_text2cypher_query(
    *,
    question: str,
    extractor: LLMExtractor,
    max_retries: int,
) -> tuple[Text2CypherQueryResult, str | None, int, dict[str, Any]]:
    result, raw_response, attempts_used, audit = extractor.generate_structured_output(
        messages=_query_messages(question),
        schema_name="Text2CypherQueryResult",
        schema_model=Text2CypherQueryResult,
        fallback_payload=QUERY_FALLBACK_PAYLOAD,
        max_retries=max_retries,
        temperature=0.0,
    )
    return result, raw_response, attempts_used, audit


def _bounded_extractor(extractor: LLMExtractor, *, max_output_tokens: int) -> LLMExtractor:
    current_limit = extractor.max_output_tokens
    if current_limit is not None and current_limit <= max_output_tokens:
        return extractor
    return LLMExtractor(
        base_url=extractor.base_url,
        api_key=extractor.api_key,
        model=extractor.model,
        provider=extractor.provider,
        api_mode=extractor.api_mode,
        max_output_tokens=max_output_tokens,
        progress_callback=extractor.progress_callback,
    )


def repair_text2cypher_query(
    *,
    question: str,
    previous_result: Text2CypherQueryResult,
    error_message: str,
    extractor: LLMExtractor,
    max_retries: int,
) -> tuple[Text2CypherQueryResult, str | None, int, dict[str, Any]]:
    previous_payload = json.dumps(
        _json_safe(previous_result.model_dump(mode="json", exclude_none=True)),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    repair_instruction = (
        "The previous answer for the same user question failed. "
        "Fix it and return a corrected JSON response only.\n\n"
        f"User question:\n{question}\n\n"
        f"Previous answer:\n{previous_payload}\n\n"
        f"Failure details:\n{error_message}"
    )
    repair_extractor = _bounded_extractor(extractor, max_output_tokens=DEFAULT_REPAIR_MAX_OUTPUT_TOKENS)
    result, raw_response, attempts_used, audit = repair_extractor.generate_structured_output(
        messages=[
            {"role": "system", "content": TEXT2CYPHER_REPAIR_SYSTEM_PROMPT},
            {"role": "user", "content": repair_instruction},
        ],
        schema_name="Text2CypherQueryResultRepair",
        schema_model=Text2CypherQueryResult,
        fallback_payload=QUERY_FALLBACK_PAYLOAD,
        max_retries=max_retries,
        temperature=0.0,
    )
    return result, raw_response, attempts_used, audit


def validate_generated_query(result: Text2CypherQueryResult) -> list[str]:
    if not result.answerable:
        return []

    cypher = result.cypher or ""
    failures = list(validate_read_only_cypher(cypher))
    failures.extend(validate_params_match(cypher, result.params))
    return failures


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


def _has_repeated_semantic_slot(params: dict[str, Any]) -> bool:
    counts: dict[str, int] = {}
    for key in params:
        for prefix in SEMANTIC_REPAIR_PARAM_PREFIXES:
            if key == prefix or key.startswith(f"{prefix}_"):
                counts[prefix] = counts.get(prefix, 0) + 1
    return any(count > 1 for count in counts.values())


def _should_attempt_empty_result_repair(question: str, result: Text2CypherQueryResult) -> bool:
    normalized_question = f" {question.casefold()} "
    return _has_repeated_semantic_slot(result.params) or any(
        token in normalized_question for token in BOOLEAN_HINT_TOKENS
    )


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
        default=DEFAULT_REPAIR_ATTEMPTS,
        help="Maximum automatic query repair attempts after validation or Neo4j errors.",
    )
    if execute:
        parser.add_argument("--neo4j-uri", type=str, default=None, help="Neo4j connection URI.")
        parser.add_argument("--neo4j-user", type=str, default="neo4j", help="Neo4j username.")
        parser.add_argument("--neo4j-password", type=str, default="password", help="Neo4j password.")
    return parser


def _print_output(message: str) -> None:
    print(message, file=sys.stdout, flush=True)


def _repair_or_none(
    *,
    question: str,
    current_result: Text2CypherQueryResult,
    error_message: str,
    extractor: LLMExtractor,
    max_retries: int,
    repair_attempt_index: int,
    repair_attempt_limit: int,
) -> Text2CypherQueryResult | None:
    if repair_attempt_index >= repair_attempt_limit:
        return None

    _print_status(
        f"Repairing query after error (attempt {repair_attempt_index + 1}/{repair_attempt_limit})..."
    )
    repaired_result, _raw_response, _attempts_used, _audit = repair_text2cypher_query(
        question=question,
        previous_result=current_result,
        error_message=error_message,
        extractor=extractor,
        max_retries=max_retries,
    )
    return repaired_result


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
        _print_status("Generating query...")
        query_result, _raw_response, attempts_used, _audit = generate_text2cypher_query(
            question=question,
            extractor=extractor,
            max_retries=args.max_retries,
        )
    except (ExtractionError, ValueError) as exc:
        _print_status(f"Error: {exc}")
        return 1

    repair_attempt_index = 0
    empty_result_repair_attempted = False
    while True:
        if not query_result.answerable:
            _print_status(f"Unsupported request: {query_result.reason}")
            return 0

        validation_failures = validate_generated_query(query_result)
        if validation_failures:
            error_message = "Validation failed:\n" + "\n".join(f"- {failure}" for failure in validation_failures)
            repaired_result = _repair_or_none(
                question=question,
                current_result=query_result,
                error_message=error_message,
                extractor=extractor,
                max_retries=args.max_retries,
                repair_attempt_index=repair_attempt_index,
                repair_attempt_limit=args.repair_attempts,
            )
            if repaired_result is not None:
                query_result = repaired_result
                repair_attempt_index += 1
                continue

            _print_status("Generated query failed validation.")
            if query_result.cypher:
                _print_status(query_result.cypher)
            for failure in validation_failures:
                _print_status(f"- {failure}")
            return 1

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
            repaired_result = _repair_or_none(
                question=question,
                current_result=query_result,
                error_message=f"Neo4j EXPLAIN error:\n{exc}",
                extractor=extractor,
                max_retries=args.max_retries,
                repair_attempt_index=repair_attempt_index,
                repair_attempt_limit=args.repair_attempts,
            )
            if repaired_result is not None:
                query_result = repaired_result
                repair_attempt_index += 1
                continue

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
            repaired_result = _repair_or_none(
                question=question,
                current_result=query_result,
                error_message=f"Neo4j execution error:\n{exc}",
                extractor=extractor,
                max_retries=args.max_retries,
                repair_attempt_index=repair_attempt_index,
                repair_attempt_limit=args.repair_attempts,
            )
            if repaired_result is not None:
                query_result = repaired_result
                repair_attempt_index += 1
                continue

            _print_status(f"Neo4j execution error: {exc}")
            return 1

        if (
            not rows
            and not empty_result_repair_attempted
            and _should_attempt_empty_result_repair(question, query_result)
        ):
            repaired_result = _repair_or_none(
                question=question,
                current_result=query_result,
                error_message=(
                    "Neo4j query returned zero rows.\n"
                    "Re-check canonical closed-label normalization against the ontology label lists. "
                    "Preserve the user's boolean logic. If the request asks about companies rather "
                    "than one segment or one offering, do not require a single segment or offering "
                    "to satisfy every condition unless the request says so explicitly."
                ),
                extractor=extractor,
                max_retries=args.max_retries,
                repair_attempt_index=repair_attempt_index,
                repair_attempt_limit=args.repair_attempts,
            )
            if repaired_result is not None:
                query_result = repaired_result
                repair_attempt_index += 1
                empty_result_repair_attempted = True
                continue

        break

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
