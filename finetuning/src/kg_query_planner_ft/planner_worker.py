from __future__ import annotations

from typing import Any

from .frozen_prompt import FROZEN_QUERY_SYSTEM_PROMPT


class PlannerGenerator:
    def __init__(self, *, model_path: str, adapter_path: str):
        try:
            from mlx_lm.generate import generate
            from mlx_lm.utils import load
        except ImportError as exc:  # pragma: no cover - exercised only in the training env
            raise RuntimeError(
                "Planner generation requires mlx-lm in the fine-tuning environment."
            ) from exc
        self._generate = generate
        self._model, self._tokenizer = load(
            model_path,
            tokenizer_config={"trust_remote_code": True},
            adapter_path=adapter_path,
        )

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
            temp=0.0,
            top_p=1.0,
        )
