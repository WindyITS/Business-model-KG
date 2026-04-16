import json
import logging
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from llm_extraction.audit import aggregate_extraction_audits, audit_knowledge_graph_payload, normalize_lenient_payload
from llm_extraction.models import (
    AnalystPipelineResult,
    CanonicalPipelineResult,
    ExtractionError,
    ExtractionPipelineResult,
    KnowledgeGraphExtraction,
    Triple,
)
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


class LLMExtractor:
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        model: str = "local-model",
        provider: str = "local",
        api_mode: str = "chat_completions",
        max_output_tokens: int | None = None,
        progress_callback: Callable[..., None] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.api_mode = api_mode
        self.max_output_tokens = max_output_tokens
        self.progress_callback = progress_callback
        self.client = None

        if api_mode != "messages":
            from openai import OpenAI

            self.client = OpenAI(base_url=self.base_url, api_key=api_key)

    def _emit_progress(self, event: str, **payload: Any) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback(event, **payload)

    @staticmethod
    def _compact_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _truncate_debug_text(value: Any, *, max_length: int = 1200) -> str:
        compact = " ".join(str(value).split())
        if len(compact) <= max_length:
            return compact
        return f"{compact[: max_length - 3]}..."

    @staticmethod
    def _compact_headers(headers: Any, *, max_items: int = 20) -> str | None:
        if headers is None:
            return None

        try:
            header_items = list(headers.items())
        except Exception:
            return None

        snapshot: dict[str, str] = {}
        for index, (key, value) in enumerate(header_items):
            if index >= max_items:
                snapshot["__truncated__"] = f"{len(header_items) - max_items} more headers"
                break
            snapshot[str(key).lower()] = str(value)

        if not snapshot:
            return None
        return json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _debug_value_text(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return str(value)

    @classmethod
    def _format_http_diagnostics(
        cls,
        *,
        method: str | None = None,
        url: Any = None,
        status: int | None = None,
        headers: Any = None,
        parsed_body: Any = None,
        raw_body: str | None = None,
    ) -> str:
        details: list[str] = []

        if method or url:
            details.append(f"request={method or '?'} {url or '?'}")
        if status is not None:
            details.append(f"status={status}")

        header_text = cls._compact_headers(headers)
        if header_text:
            details.append(f"headers={header_text}")

        parsed_body_text = cls._debug_value_text(parsed_body)
        if parsed_body_text:
            details.append(f"parsed_error={cls._truncate_debug_text(parsed_body_text, max_length=800)}")

        if raw_body:
            normalized_raw = cls._truncate_debug_text(raw_body, max_length=1200)
            normalized_parsed = cls._truncate_debug_text(parsed_body_text, max_length=1200) if parsed_body_text else None
            if normalized_raw != normalized_parsed:
                details.append(f"raw_response={normalized_raw}")

        return "; ".join(details)

    @classmethod
    def _http_exception_diagnostics(cls, exc: Exception) -> str | None:
        response = getattr(exc, "response", None)
        if response is None:
            return None

        request = getattr(response, "request", None) or getattr(exc, "request", None)
        method = getattr(request, "method", None)
        url = getattr(request, "url", None)
        status = getattr(response, "status_code", None)
        headers = getattr(response, "headers", None)

        raw_body = None
        try:
            raw_body = response.text
        except Exception:
            raw_body = None

        diagnostics = cls._format_http_diagnostics(
            method=method,
            url=url,
            status=status,
            headers=headers,
            parsed_body=getattr(exc, "body", None),
            raw_body=raw_body,
        )
        return diagnostics or None

    @staticmethod
    def _strip_code_fence(content: str) -> str:
        match = CODE_FENCE_RE.match(content.strip())
        if match:
            return match.group(1).strip()
        return content.strip()

    @staticmethod
    def _prepare_messages_for_provider(messages: list[dict[str, str]], provider: str) -> list[dict[str, str]]:
        if provider != "opencode-go":
            return list(messages)

        # OpenCode Go is handled more conservatively: keep the pipeline structure,
        # but send system instructions as user messages for compatibility.
        return [
            {
                **message,
                "role": "user" if message.get("role") == "system" else message.get("role", "user"),
            }
            for message in messages
        ]

    @staticmethod
    def _merge_serves_into_base(
        base_extraction: KnowledgeGraphExtraction,
        serves_extraction: KnowledgeGraphExtraction,
    ) -> KnowledgeGraphExtraction:
        return LLMExtractor._merge_relation_subset_into_base(
            base_extraction,
            serves_extraction,
            allowed_relations={"SERVES"},
        )

    @staticmethod
    def _merge_relation_subset_into_base(
        base_extraction: KnowledgeGraphExtraction,
        subset_extraction: KnowledgeGraphExtraction,
        *,
        allowed_relations: set[str],
    ) -> KnowledgeGraphExtraction:
        merged_triples: list[Triple] = []
        seen: set[tuple[str, str, str, str, str]] = set()

        for triple in base_extraction.triples:
            if triple.relation in allowed_relations:
                continue
            key = (triple.subject, triple.subject_type, triple.relation, triple.object, triple.object_type)
            if key in seen:
                continue
            seen.add(key)
            merged_triples.append(triple)

        for triple in subset_extraction.triples:
            if triple.relation not in allowed_relations:
                continue
            key = (triple.subject, triple.subject_type, triple.relation, triple.object, triple.object_type)
            if key in seen:
                continue
            seen.add(key)
            merged_triples.append(triple)

        return KnowledgeGraphExtraction(
            extraction_notes=subset_extraction.extraction_notes,
            triples=merged_triples,
        )

    @staticmethod
    def _triple_key(triple: Triple) -> tuple[str, str, str, str, str]:
        return (
            " ".join(triple.subject.split()).casefold(),
            triple.subject_type,
            triple.relation,
            " ".join(triple.object.split()).casefold(),
            triple.object_type,
        )

    @classmethod
    def _triple_count(cls, extraction: KnowledgeGraphExtraction) -> int:
        return len({cls._triple_key(triple) for triple in extraction.triples})

    @classmethod
    def _triple_delta_details(
        cls,
        before_extraction: KnowledgeGraphExtraction,
        after_extraction: KnowledgeGraphExtraction,
    ) -> list[tuple[str, int]]:
        before_keys = {cls._triple_key(triple) for triple in before_extraction.triples}
        after_keys = {cls._triple_key(triple) for triple in after_extraction.triples}
        return [
            ("triples out", len(after_keys)),
            ("triples added", len(after_keys - before_keys)),
            ("triples removed", len(before_keys - after_keys)),
        ]

    @staticmethod
    def _load_json_payload(content: str, fallback_payload: str) -> tuple[dict[str, Any], bool, bool]:
        content = content.strip()
        if not content:
            raise ExtractionError("Empty response from model.")

        try:
            return json.loads(content), False, False
        except json.JSONDecodeError:
            pass

        fenced_content = LLMExtractor._strip_code_fence(content)
        if fenced_content != content:
            try:
                return json.loads(fenced_content), True, False
            except json.JSONDecodeError:
                pass

        match = JSON_OBJECT_RE.search(content)
        if match:
            json_object_text = match.group(0)
            if json_object_text != content:
                try:
                    return json.loads(json_object_text), True, False
                except json.JSONDecodeError:
                    pass

        likely_truncated = (
            content.count("{") > content.count("}")
            or content.count("[") > content.count("]")
            or not content.endswith(("}", "]"))
        )
        if likely_truncated:
            logger.warning("Model response may be truncated. Attempting to salvage JSON prefix...")
            last_object_end = content.rfind("}")
            last_array_end = content.rfind("]")
            last_json_end = max(last_object_end, last_array_end)
            truncated_candidate = content[: last_json_end + 1] if last_json_end != -1 else ""
            if truncated_candidate:
                try:
                    return json.loads(truncated_candidate), True, False
                except json.JSONDecodeError:
                    pass

        logger.warning("Model response was not exact raw JSON. Falling back to the recovery payload.")
        return json.loads(fallback_payload), True, True

    @staticmethod
    def _responses_refusal_text(response: Any) -> str | None:
        for output_item in getattr(response, "output", []) or []:
            for content_item in getattr(output_item, "content", []) or []:
                if getattr(content_item, "type", None) == "refusal":
                    refusal = getattr(content_item, "refusal", None)
                    if refusal:
                        return str(refusal)
        return None

    @staticmethod
    def _responses_output_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text)

        text_parts: list[str] = []
        for output_item in getattr(response, "output", []) or []:
            for content_item in getattr(output_item, "content", []) or []:
                if getattr(content_item, "type", None) != "output_text":
                    continue
                text = getattr(content_item, "text", None)
                if text:
                    text_parts.append(str(text))
        return "".join(text_parts)

    @staticmethod
    def _messages_request_payload(
        messages: list[dict[str, str]],
        *,
        model: str,
        max_output_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        anthropic_messages = [
            {
                "role": message.get("role", "user"),
                "content": message.get("content", ""),
            }
            for message in messages
            if message.get("role", "user") in {"user", "assistant"}
        ]
        return {
            "model": model,
            "max_tokens": max_output_tokens,
            "messages": anthropic_messages,
            "temperature": temperature,
        }

    @staticmethod
    def _messages_output_text(response_payload: dict[str, Any]) -> str:
        text_parts: list[str] = []
        for content_item in response_payload.get("content", []) or []:
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") != "text":
                continue
            text = content_item.get("text")
            if text:
                text_parts.append(str(text))
        return "".join(text_parts)

    def _call_messages_api(
        self,
        *,
        request_messages: list[dict[str, str]],
        temperature: float,
        call_label: str,
        attempt: int,
        max_retries: int,
        return_token_count: bool = False,
    ) -> str | tuple[str, int | None]:
        if self.max_output_tokens is None:
            raise ExtractionError("Messages API requires max_output_tokens to be set.")

        request_payload = self._messages_request_payload(
            request_messages,
            model=self.model,
            max_output_tokens=self.max_output_tokens,
            temperature=temperature,
        )
        request = urllib.request.Request(
            url=f"{self.base_url}/messages",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw_response = response.read().decode("utf-8")
                response_headers = response.headers
                response_status = getattr(response, "status", None)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            diagnostics = self._format_http_diagnostics(
                method="POST",
                url=request.full_url,
                status=exc.code,
                headers=exc.headers,
                raw_body=error_body,
            )
            if diagnostics:
                logger.warning(
                    "Messages API HTTP diagnostics for %s attempt %s/%s: %s",
                    call_label,
                    attempt,
                    max_retries,
                    diagnostics,
                )
            raise ExtractionError(f"Messages API returned HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise ExtractionError(f"Messages API request failed: {exc.reason}") from exc

        try:
            response_payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            diagnostics = self._format_http_diagnostics(
                method="POST",
                url=request.full_url,
                status=response_status,
                headers=response_headers,
                raw_body=raw_response,
            )
            if diagnostics:
                logger.warning(
                    "Messages API non-JSON diagnostics for %s attempt %s/%s: %s",
                    call_label,
                    attempt,
                    max_retries,
                    diagnostics,
                )
            raise ExtractionError(
                f"Messages API returned non-JSON content: {self._truncate_debug_text(raw_response, max_length=800)}"
            ) from exc
        stop_reason = response_payload.get("stop_reason")
        usage = response_payload.get("usage") or {}
        output_tokens = usage.get("output_tokens")
        logger.info(
            "Structured call %s attempt %s/%s stop_reason=%s output_tokens=%s",
            call_label,
            attempt,
            max_retries,
            stop_reason,
            output_tokens,
        )

        content = self._messages_output_text(response_payload)
        if not content:
            raise ExtractionError("Messages API returned no text content.")
        if return_token_count:
            return content, output_tokens
        return content

    def _call_structured_messages(
        self,
        *,
        messages: list[dict[str, str]],
        schema_name: str,
        schema_model: type[BaseModel],
        fallback_payload: str,
        max_retries: int,
        temperature: float = 0.0,
        ontology_version: str = "canonical",
    ) -> tuple[BaseModel, str | None, int, dict[str, Any]]:
        request_messages = self._prepare_messages_for_provider(messages, self.provider)
        call_label = schema_name
        last_error: Exception | None = None
        for message in reversed(request_messages):
            if message.get("role") != "user":
                continue
            for line in message.get("content", "").splitlines():
                stripped = line.strip()
                if stripped:
                    call_label = stripped[:120]
                    break
            if call_label != schema_name:
                break

        for attempt in range(1, max_retries + 1):
            self._emit_progress("llm_call_start", attempt=attempt, max_retries=max_retries)
            try:
                content = ""
                token_count: int | None = None
                if self.api_mode == "responses":
                    call_kwargs = {
                        "model": self.model,
                        "input": request_messages,
                        "temperature": temperature,
                    }
                    response = self.client.responses.create(**call_kwargs)
                    status = getattr(response, "status", None)
                    usage = getattr(response, "usage", None)
                    output_tokens = getattr(usage, "output_tokens", None) if usage is not None else None
                    token_count = output_tokens
                    logger.info(
                        "Structured call %s attempt %s/%s status=%s output_tokens=%s",
                        call_label,
                        attempt,
                        max_retries,
                        status,
                        output_tokens,
                    )
                    refusal_text = self._responses_refusal_text(response)
                    if refusal_text:
                        raise ExtractionError(f"Model refused request: {refusal_text}")
                    content = self._responses_output_text(response)
                elif self.api_mode == "messages":
                    content, token_count = self._call_messages_api(
                        request_messages=request_messages,
                        temperature=temperature,
                        call_label=call_label,
                        attempt=attempt,
                        max_retries=max_retries,
                        return_token_count=True,
                    )
                else:
                    call_kwargs = {
                        "model": self.model,
                        "messages": request_messages,
                        "temperature": temperature,
                    }
                    if self.max_output_tokens is not None:
                        call_kwargs["max_tokens"] = self.max_output_tokens
                    response = self.client.chat.completions.create(**call_kwargs)
                    choice = response.choices[0]
                    finish_reason = getattr(choice, "finish_reason", None)
                    usage = getattr(response, "usage", None)
                    completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
                    token_count = completion_tokens
                    logger.info(
                        "Structured call %s attempt %s/%s finish_reason=%s completion_tokens=%s",
                        call_label,
                        attempt,
                        max_retries,
                        finish_reason,
                        completion_tokens,
                    )
                    refusal_text = getattr(choice.message, "refusal", None)
                    if refusal_text:
                        raise ExtractionError(f"Model refused request: {refusal_text}")
                    content = choice.message.content or ""
                parsed_payload, payload_parse_recovered, used_recovery_payload = self._load_json_payload(
                    content or "",
                    fallback_payload,
                )
                if used_recovery_payload:
                    raise ExtractionError("Model response was not recoverable as JSON.")
                parsed_model, audit = self._lenient_model_from_payload(
                    schema_model,
                    parsed_payload,
                    ontology_version=ontology_version,
                )
                audit["payload_parse_recovered"] = payload_parse_recovered
                self._emit_progress(
                    "llm_call_complete",
                    attempt=attempt,
                    max_retries=max_retries,
                    tokens=token_count,
                )
                return parsed_model, content, attempt, audit
            except (json.JSONDecodeError, ValidationError, ExtractionError) as exc:
                last_error = exc
                self._emit_progress(
                    "llm_call_error",
                    attempt=attempt,
                    max_retries=max_retries,
                    error=str(exc),
                    will_retry=attempt < max_retries,
                )
                logger.warning(
                    "Structured call %s failed on attempt %s/%s: %s",
                    call_label,
                    attempt,
                    max_retries,
                    exc,
                )
            except Exception as exc:
                last_error = exc
                diagnostics = self._http_exception_diagnostics(exc)
                if diagnostics:
                    logger.warning(
                        "HTTP diagnostics for %s attempt %s/%s: %s",
                        call_label,
                        attempt,
                        max_retries,
                        diagnostics,
                    )
                self._emit_progress(
                    "llm_call_error",
                    attempt=attempt,
                    max_retries=max_retries,
                    error=str(exc),
                    will_retry=attempt < max_retries,
                )
                logger.warning("LLM API error for %s on attempt %s/%s: %s", call_label, attempt, max_retries, exc)

        if last_error is not None:
            raise ExtractionError(f"Failed after {max_retries} attempts. Last error: {last_error}")
        raise ExtractionError(f"Failed after {max_retries} attempts")

    def _call_text_messages(
        self,
        *,
        messages: list[dict[str, str]],
        max_retries: int,
        temperature: float = 0.0,
    ) -> tuple[str, int, dict[str, Any]]:
        request_messages = self._prepare_messages_for_provider(messages, self.provider)
        call_label = "text"
        last_error: Exception | None = None
        for message in reversed(request_messages):
            if message.get("role") != "user":
                continue
            for line in message.get("content", "").splitlines():
                stripped = line.strip()
                if stripped:
                    call_label = stripped[:120]
                    break
            if call_label != "text":
                break

        for attempt in range(1, max_retries + 1):
            self._emit_progress("llm_call_start", attempt=attempt, max_retries=max_retries)
            try:
                content = ""
                token_count: int | None = None
                if self.api_mode == "responses":
                    call_kwargs = {
                        "model": self.model,
                        "input": request_messages,
                        "temperature": temperature,
                    }
                    response = self.client.responses.create(**call_kwargs)
                    status = getattr(response, "status", None)
                    usage = getattr(response, "usage", None)
                    output_tokens = getattr(usage, "output_tokens", None) if usage is not None else None
                    token_count = output_tokens
                    logger.info(
                        "Text call %s attempt %s/%s status=%s output_tokens=%s",
                        call_label,
                        attempt,
                        max_retries,
                        status,
                        output_tokens,
                    )
                    refusal_text = self._responses_refusal_text(response)
                    if refusal_text:
                        raise ExtractionError(f"Model refused request: {refusal_text}")
                    content = self._responses_output_text(response)
                elif self.api_mode == "messages":
                    content, token_count = self._call_messages_api(
                        request_messages=request_messages,
                        temperature=temperature,
                        call_label=call_label,
                        attempt=attempt,
                        max_retries=max_retries,
                        return_token_count=True,
                    )
                else:
                    call_kwargs = {
                        "model": self.model,
                        "messages": request_messages,
                        "temperature": temperature,
                    }
                    if self.max_output_tokens is not None:
                        call_kwargs["max_tokens"] = self.max_output_tokens
                    response = self.client.chat.completions.create(**call_kwargs)
                    choice = response.choices[0]
                    finish_reason = getattr(choice, "finish_reason", None)
                    usage = getattr(response, "usage", None)
                    completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
                    token_count = completion_tokens
                    logger.info(
                        "Text call %s attempt %s/%s finish_reason=%s completion_tokens=%s",
                        call_label,
                        attempt,
                        max_retries,
                        finish_reason,
                        completion_tokens,
                    )
                    refusal_text = getattr(choice.message, "refusal", None)
                    if refusal_text:
                        raise ExtractionError(f"Model refused request: {refusal_text}")
                    content = choice.message.content or ""

                content = self._strip_code_fence(content or "").strip()
                if not content:
                    raise ExtractionError("Model returned empty text content.")
                audit = {
                    "content_length": len(content),
                    "line_count": len(content.splitlines()),
                    "format": "text",
                }
                self._emit_progress(
                    "llm_call_complete",
                    attempt=attempt,
                    max_retries=max_retries,
                    tokens=token_count,
                )
                return content, attempt, audit
            except ExtractionError as exc:
                last_error = exc
                self._emit_progress(
                    "llm_call_error",
                    attempt=attempt,
                    max_retries=max_retries,
                    error=str(exc),
                    will_retry=attempt < max_retries,
                )
                logger.warning(
                    "Text call %s failed on attempt %s/%s: %s",
                    call_label,
                    attempt,
                    max_retries,
                    exc,
                )
            except Exception as exc:
                last_error = exc
                diagnostics = self._http_exception_diagnostics(exc)
                if diagnostics:
                    logger.warning(
                        "HTTP diagnostics for %s attempt %s/%s: %s",
                        call_label,
                        attempt,
                        max_retries,
                        diagnostics,
                    )
                self._emit_progress(
                    "llm_call_error",
                    attempt=attempt,
                    max_retries=max_retries,
                    error=str(exc),
                    will_retry=attempt < max_retries,
                )
                logger.warning("LLM API error for %s on attempt %s/%s: %s", call_label, attempt, max_retries, exc)

        if last_error is not None:
            raise ExtractionError(f"Failed after {max_retries} attempts. Last error: {last_error}")
        raise ExtractionError(f"Failed after {max_retries} attempts")

    @staticmethod
    def _lenient_model_from_payload(
        schema_model: type[BaseModel],
        payload: Any,
        *,
        ontology_version: str = "canonical",
    ) -> tuple[BaseModel, dict[str, Any]]:
        normalized_payload = normalize_lenient_payload(payload)
        valid_triples, audit = audit_knowledge_graph_payload(normalized_payload, ontology_version=ontology_version)
        extraction_notes = str(
            normalized_payload.get("extraction_notes", normalized_payload.get("chain_of_thought_reasoning", "")) or ""
        )
        triple_objects = [Triple(**triple) for triple in valid_triples]

        if schema_model is KnowledgeGraphExtraction:
            model = KnowledgeGraphExtraction(
                extraction_notes=extraction_notes,
                triples=triple_objects,
            )
            return model, audit

        validation_error: ValidationError | None = None
        candidate_payloads = [payload]
        if normalized_payload is not payload:
            candidate_payloads.append(normalized_payload)

        for index, candidate_payload in enumerate(candidate_payloads):
            try:
                model = schema_model.model_validate(candidate_payload)
                return model, {
                    "schema_name": schema_model.__name__,
                    "used_normalized_payload": index > 0,
                }
            except ValidationError as exc:
                validation_error = exc

        if validation_error is not None:
            raise validation_error
        raise TypeError(f"Unsupported schema model for lenient payload parsing: {schema_model!r}")

    def _call_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema_model: type[BaseModel],
        fallback_payload: str,
        max_retries: int,
        temperature: float = 0.0,
        ontology_version: str = "canonical",
    ) -> tuple[BaseModel, str | None, int, dict[str, Any]]:
        return self._call_structured_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            schema_name=schema_name,
            schema_model=schema_model,
            fallback_payload=fallback_payload,
            max_retries=max_retries,
            temperature=temperature,
            ontology_version=ontology_version,
        )

    def generate_structured_output(
        self,
        *,
        messages: list[dict[str, str]],
        schema_name: str,
        schema_model: type[BaseModel],
        fallback_payload: str,
        max_retries: int,
        temperature: float = 0.0,
        ontology_version: str = "canonical",
    ) -> tuple[BaseModel, str | None, int, dict[str, Any]]:
        return self._call_structured_messages(
            messages=messages,
            schema_name=schema_name,
            schema_model=schema_model,
            fallback_payload=fallback_payload,
            max_retries=max_retries,
            temperature=temperature,
            ontology_version=ontology_version,
        )

    def reflect_extraction(
        self,
        *,
        full_text: str,
        current_extraction: KnowledgeGraphExtraction,
        company_name: str | None = None,
        max_retries: int = 2,
        strict: bool = True,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        stage_label: str = "Reflection",
        ontology_version: str = "canonical",
    ) -> tuple[KnowledgeGraphExtraction, str | None, int, dict[str, Any]]:
        _ = full_text
        if system_prompt is None or user_prompt is None:
            raise ValueError(
                "reflect_extraction requires explicit system_prompt and user_prompt; pipeline-specific reflection prompts live with the pipeline."
            )

        try:
            final_extraction, raw_response, attempts_used, audit = self._call_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Reflection failed.","triples":[]}',
                max_retries=max_retries,
                ontology_version=ontology_version,
            )
            if not final_extraction.triples and current_extraction.triples:
                logger.warning("%s returned no triples. Falling back to prior graph.", stage_label)
                _, fallback_audit = audit_knowledge_graph_payload(
                    current_extraction.model_dump(mode="json"),
                    ontology_version=ontology_version,
                )
                return current_extraction, raw_response, attempts_used, fallback_audit
            return final_extraction, raw_response, attempts_used, audit
        except ExtractionError:
            if strict:
                raise
            logger.warning("%s failed. Falling back to prior graph.", stage_label)
            _, fallback_audit = audit_knowledge_graph_payload(
                current_extraction.model_dump(mode="json"),
                ontology_version=ontology_version,
            )
            return current_extraction, None, max_retries, fallback_audit


__all__ = [
    "AnalystPipelineResult",
    "CanonicalPipelineResult",
    "ExtractionError",
    "ExtractionPipelineResult",
    "KnowledgeGraphExtraction",
    "LLMExtractor",
    "Triple",
    "aggregate_extraction_audits",
    "audit_knowledge_graph_payload",
    "normalize_lenient_payload",
]
