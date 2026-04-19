from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from .config import load_config
from .json_utils import compact_json, extract_first_json_object
from .paths import planner_adapter_dir, router_eval_dir, router_model_dir
from .planner_worker import PlannerGenerator
from .router_eval import decide_router_outcome, load_thresholds, predict_router_probabilities
from .runtime_compat import load_runtime_contract


def _run_planner(
    question: str,
    *,
    base_model: str,
    adapter_path: Path,
    max_tokens: int,
) -> dict[str, Any]:
    generator = PlannerGenerator(model_path=base_model, adapter_path=str(adapter_path))
    generated_text = generator.generate(question, max_tokens=max_tokens)
    parsed_json = extract_first_json_object(generated_text)
    QueryPlanEnvelope, compile_query_plan, validate_compiled_query = load_runtime_contract()
    validated = QueryPlanEnvelope.model_validate(parsed_json)
    compiled = compile_query_plan(validated)
    failures = validate_compiled_query(compiled)
    if not compiled.answerable or failures:
        raise ValueError(compiled.reason or "; ".join(failures))
    return {
        "generated_text": generated_text,
        "plan": validated.model_dump(mode="json", exclude_none=True),
        "compiled": {
            "cypher": compiled.cypher,
            "params": compiled.params,
        },
    }


def run_local_stack(
    question: str,
    config_path: str | None = None,
    *,
    router_predictor: Callable[[str], dict[str, float]] | None = None,
    planner_runner: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    thresholds = load_thresholds(router_eval_dir(config))
    if router_predictor is None:
        def router_predictor(text: str) -> dict[str, float]:
            return predict_router_probabilities(
                text,
                model_dir=router_model_dir(config),
                max_length=config.router.max_length,
                temperature=float(thresholds.get("temperature", 1.0)),
            )

    probabilities = router_predictor(question)
    initial_decision = decide_router_outcome(probabilities, thresholds)
    output: dict[str, Any] = {
        "decision": initial_decision,
        "router": {
            "probabilities": probabilities,
            "thresholds": thresholds,
        },
        "planner": None,
        "plan": None,
        "compiled": None,
    }
    if initial_decision != "local":
        return output
    if not bool(thresholds.get("planner_gate_open", False)):
        output["decision"] = "api_fallback"
        output["router"]["gate_reason"] = "planner_gate_closed"
        return output

    if planner_runner is None:
        def planner_runner(text: str) -> dict[str, Any]:
            return _run_planner(
                text,
                base_model=config.planner.base_model,
                adapter_path=planner_adapter_dir(config),
                max_tokens=config.planner.max_tokens,
            )

    try:
        planner_payload = planner_runner(question)
    except Exception as exc:  # noqa: BLE001
        output["decision"] = "api_fallback"
        output["planner"] = {"error": str(exc)}
        return output

    output["planner"] = {"generated_text": planner_payload["generated_text"]}
    output["plan"] = planner_payload["plan"]
    output["compiled"] = planner_payload["compiled"]
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local router + planner stack.")
    parser.add_argument("question", nargs="+", help="Question to route and, if safe, plan locally.")
    parser.add_argument("--config", type=str, default=None, help="Path to the fine-tuning JSON config.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    question = " ".join(part.strip() for part in args.question).strip()
    result = run_local_stack(question, args.config)
    print(compact_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
