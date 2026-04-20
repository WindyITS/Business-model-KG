from __future__ import annotations

import argparse
from typing import Any, Callable

from .config import load_config
from .json_utils import compact_json
from .paths import router_eval_dir, router_model_dir
from .router_eval import decide_router_outcome, load_thresholds, predict_router_probabilities


def run_local_router(
    question: str,
    config_path: str | None = None,
    *,
    router_predictor: Callable[[str], dict[str, float]] | None = None,
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
    decision = decide_router_outcome(probabilities, thresholds)
    return {
        "decision": decision,
        "router": {
            "probabilities": probabilities,
            "thresholds": thresholds,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local router only (no planner).")
    parser.add_argument("question", nargs="+", help="Question to route locally.")
    parser.add_argument("--config", type=str, default=None, help="Path to the fine-tuning JSON config.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    question = " ".join(part.strip() for part in args.question).strip()
    result = run_local_router(question, args.config)
    print(compact_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
