from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .frozen_prompt import FROZEN_QUERY_SYSTEM_PROMPT


class PlannerGenerator:
    def __init__(self, *, model_path: str, adapter_path: str | None = None):
        try:
            from mlx_lm.generate import generate
            from mlx_lm.sample_utils import make_sampler
            from mlx_lm.utils import load
        except ImportError as exc:  # pragma: no cover - exercised only in the training env
            raise RuntimeError(
                "Planner generation requires mlx-lm in the fine-tuning environment."
            ) from exc
        self._generate = generate
        # Use the MLX sampler API for deterministic greedy decoding. Older
        # mlx-lm versions do not accept temp/top_p directly on generate().
        self._sampler = make_sampler(temp=0.0)
        load_kwargs: dict[str, Any] = {
            "tokenizer_config": {"trust_remote_code": True},
        }
        if adapter_path is not None:
            load_kwargs["adapter_path"] = adapter_path
        self._model, self._tokenizer = load(model_path, **load_kwargs)

    def generate(self, question: str, *, max_tokens: int) -> str:
        prompt = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": FROZEN_QUERY_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return self._generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            verbose=False,
            max_tokens=max_tokens,
            sampler=self._sampler,
        )


class LMStudioPlannerGenerator:
    def __init__(
        self,
        *,
        model_name: str,
        base_url: str = "http://localhost:1234/v1",
        api_key: str | None = None,
    ):
        self._model_name = model_name
        self._base_url = self._normalize_base_url(base_url)
        self._api_key = api_key or os.getenv("LM_STUDIO_API_KEY") or "lm-studio"

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized[: -len("/chat/completions")]
        return normalized

    @staticmethod
    def _extract_message_text(payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("LM Studio response did not include any choices.")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("LM Studio response choice had an unexpected shape.")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("LM Studio response did not include a message payload.")
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            fragments = [
                str(item.get("text", "")).strip()
                for item in content
                if isinstance(item, dict) and str(item.get("text", "")).strip()
            ]
            if fragments:
                return "\n".join(fragments)
        raise RuntimeError("LM Studio response content was empty.")

    @staticmethod
    def _http_error_text(exc: urllib.error.HTTPError) -> str:
        try:
            raw = exc.read().decode("utf-8")
        except Exception:
            raw = ""
        if raw:
            return f"HTTP {exc.code}: {raw}"
        return f"HTTP {exc.code}: {exc.reason}"

    def generate(self, question: str, *, max_tokens: int) -> str:
        request_payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": FROZEN_QUERY_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        request = urllib.request.Request(
            url=f"{self._base_url}/chat/completions",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"LM Studio planner call failed: {self._http_error_text(exc)}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LM Studio planner call failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("LM Studio planner call returned non-JSON output.") from exc
        return self._extract_message_text(payload)
