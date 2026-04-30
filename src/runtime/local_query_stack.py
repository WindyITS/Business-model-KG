from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from .query_planner import QueryPlanEnvelope, compile_query_plan, validate_compiled_query
from .query_prompt import LOCAL_QUERY_SYSTEM_PROMPT
from .query_stack import ResolvedQueryStackBundle, load_query_stack_bundle


ROUTER_LABELS = ("api_fallback", "local", "refuse")
LOCAL_DECISION_THRESHOLD = 0.97


class LocalQueryStackError(RuntimeError):
    pass


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
    raise ValueError("No JSON object found in deployed planner output.")


def _label_to_id(label: str) -> int:
    return ROUTER_LABELS.index(label)


def _softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    max_logit = max(logits)
    exp = [math.exp(value - max_logit) for value in logits]
    denom = sum(exp) or 1.0
    return [value / denom for value in exp]


def _apply_temperature(logits: list[float], temperature: float) -> list[float]:
    safe_temperature = max(float(temperature), 1e-3)
    return _softmax([value / safe_temperature for value in logits])


def _load_thresholds(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LocalQueryStackError(f"Router thresholds file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LocalQueryStackError(f"Router thresholds file is not valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise LocalQueryStackError(f"Router thresholds file has an invalid shape: {path}")
    return payload


def _decide_router_outcome(probabilities: dict[str, float], _thresholds: dict[str, Any]) -> str:
    if probabilities["local"] >= LOCAL_DECISION_THRESHOLD:
        return "local"
    return "refuse" if probabilities["refuse"] >= probabilities["api_fallback"] else "api_fallback"


def _system_prompt(bundle: ResolvedQueryStackBundle) -> str:
    prompt_path = bundle.planner_system_prompt_path
    if prompt_path is None:
        return LOCAL_QUERY_SYSTEM_PROMPT
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise LocalQueryStackError(f"Planner system prompt file not found: {prompt_path}") from exc


class DeployedRouterPredictor:
    def __init__(self, *, model_dir: Path, max_length: int, temperature: float = 1.0):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise LocalQueryStackError(
                'Local query routing requires the optional runtime dependencies. Install with pip install -e ".[query-stack]".'
            ) from exc

        self._torch = torch
        self._temperature = temperature
        self._max_length = max_length
        self._device = self._resolve_device(torch)
        # The slow DeBERTa tokenizer avoids an upstream fast-tokenizer regex warning
        # without changing the sentencepiece IDs used for routing.
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self._model.to(self._device)
        self._model.eval()

    @staticmethod
    def _resolve_device(torch: Any) -> Any:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def predict(self, question: str) -> dict[str, float]:
        encoded = self._tokenizer(
            [question],
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.no_grad():
            logits = self._model(**encoded).logits.detach().cpu().tolist()[0]
        probabilities = _apply_temperature([float(value) for value in logits], self._temperature)
        return {label: probabilities[_label_to_id(label)] for label in ROUTER_LABELS}


class DeployedPlannerGenerator:
    def __init__(self, *, model_path: str, adapter_path: str, system_prompt: str):
        try:
            from mlx_lm.generate import generate
            from mlx_lm.sample_utils import make_sampler
            from mlx_lm.utils import load
        except ImportError as exc:
            raise LocalQueryStackError(
                'Local planner inference requires the optional runtime dependencies. Install with pip install -e ".[query-stack]".'
            ) from exc
        self._generate = generate
        self._sampler = make_sampler(temp=0.0)
        self._system_prompt = system_prompt
        self._model, self._tokenizer = load(
            model_path,
            adapter_path=adapter_path,
            tokenizer_config={"trust_remote_code": True},
        )

    def generate(self, question: str, *, max_tokens: int) -> str:
        prompt = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self._system_prompt},
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


@lru_cache(maxsize=4)
def _router_predictor_for(model_dir: str, max_length: int, temperature: float) -> DeployedRouterPredictor:
    return DeployedRouterPredictor(model_dir=Path(model_dir), max_length=max_length, temperature=temperature)


@lru_cache(maxsize=4)
def _planner_generator_for(model_path: str, adapter_path: str, system_prompt: str) -> DeployedPlannerGenerator:
    return DeployedPlannerGenerator(model_path=model_path, adapter_path=adapter_path, system_prompt=system_prompt)


def run_local_query_stack(question: str, *, bundle_dir: str | Path | None = None) -> dict[str, Any]:
    bundle = load_query_stack_bundle(bundle_dir)
    thresholds = _load_thresholds(bundle.router_thresholds_path)
    temperature = float(thresholds.get("temperature", 1.0))
    predictor = _router_predictor_for(
        str(bundle.router_model_dir),
        bundle.manifest.router.max_length,
        temperature,
    )
    probabilities = predictor.predict(question)
    decision = _decide_router_outcome(probabilities, thresholds)
    output: dict[str, Any] = {
        "decision": decision,
        "router": {
            "probabilities": probabilities,
            "thresholds": thresholds,
        },
        "planner": None,
        "plan": None,
        "compiled": None,
    }

    if decision != "local":
        return output

    if not bool(thresholds.get("planner_gate_open", False)):
        output["decision"] = "api_fallback"
        output["router"]["gate_reason"] = "planner_gate_closed"
        return output

    try:
        generator = _planner_generator_for(
            bundle.manifest.planner.base_model,
            str(bundle.planner_adapter_dir),
            _system_prompt(bundle),
        )
        generated_text = generator.generate(question, max_tokens=bundle.manifest.planner.max_tokens)
        parsed_json = _extract_first_json_object(generated_text)
        validated = QueryPlanEnvelope.model_validate(parsed_json)
        compiled = compile_query_plan(validated)
        failures = validate_compiled_query(compiled)
        if not compiled.answerable or failures:
            raise ValueError(compiled.reason or "; ".join(failures))
    except Exception as exc:  # noqa: BLE001
        output["decision"] = "api_fallback"
        output["planner"] = {"error": str(exc)}
        return output

    output["planner"] = {"generated_text": generated_text}
    output["plan"] = validated.model_dump(mode="json", exclude_none=True)
    output["compiled"] = {
        "cypher": compiled.cypher,
        "params": compiled.params,
    }
    return output


__all__ = [
    "DeployedPlannerGenerator",
    "DeployedRouterPredictor",
    "LocalQueryStackError",
    "run_local_query_stack",
]
