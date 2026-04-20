from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import load_config
from .json_utils import compact_json, extract_first_json_object, read_jsonl, write_json, write_jsonl
from .paths import planner_adapter_dir, planner_eval_dir, prepared_planner_raw_dir
from .planner_worker import LMStudioPlannerGenerator, PlannerGenerator
from .progress import StepProgress, track
from .runtime_compat import load_runtime_contract


def _family_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    bucket: dict[str, dict[str, int]] = defaultdict(lambda: Counter())  # type: ignore[assignment]
    for row in rows:
        family = row["family"]
        bucket[family]["count"] += 1
        if row["json_parse_ok"]:
            bucket[family]["json_parse_ok"] += 1
        if row["schema_valid"]:
            bucket[family]["schema_valid"] += 1
        if row["family_correct"]:
            bucket[family]["family_correct"] += 1
        if row["exact_match"]:
            bucket[family]["exact_match"] += 1
        if row["compile_success"]:
            bucket[family]["compile_success"] += 1
    return {family: dict(sorted(counts.items())) for family, counts in sorted(bucket.items())}


def _evaluate_split(
    rows: list[dict[str, Any]],
    generator: PlannerGenerator,
    *,
    max_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    QueryPlanEnvelope, compile_query_plan, validate_compiled_query = load_runtime_contract()
    predictions: list[dict[str, Any]] = []
    for row in track(rows, total=len(rows), desc="planner eval", unit="row"):
        generated_text = generator.generate(row["question"], max_tokens=max_tokens)
        parsed_json: Any | None = None
        json_parse_ok = False
        schema_valid = False
        family_correct = False
        exact_match = False
        compile_success = False
        normalized_predicted_plan: dict[str, Any] | None = None
        compile_reason: str | None = None
        try:
            parsed_json = extract_first_json_object(generated_text)
            json_parse_ok = True
            validated = QueryPlanEnvelope.model_validate(parsed_json)
            normalized_predicted_plan = validated.model_dump(mode="json", exclude_none=True)
            schema_valid = True
            family_correct = normalized_predicted_plan.get("family") == row["gold_plan"].get("family")
            exact_match = compact_json(normalized_predicted_plan) == compact_json(row["gold_plan"])
            compiled = compile_query_plan(validated)
            compile_success = compiled.answerable and not validate_compiled_query(compiled)
            if not compile_success:
                compile_reason = compiled.reason
        except Exception as exc:  # noqa: BLE001
            compile_reason = str(exc)
        predictions.append(
            {
                "question": row["question"],
                "family": row["family"],
                "gold_plan": row["gold_plan"],
                "generated_text": generated_text,
                "parsed_json": parsed_json,
                "json_parse_ok": json_parse_ok,
                "schema_valid": schema_valid,
                "family_correct": family_correct,
                "exact_match": exact_match,
                "compile_success": compile_success,
                "compile_reason": compile_reason,
            }
        )

    count = len(predictions)
    metrics = {
        "count": count,
        "json_parse_rate": sum(1 for row in predictions if row["json_parse_ok"]) / count if count else 0.0,
        "schema_valid_rate": sum(1 for row in predictions if row["schema_valid"]) / count if count else 0.0,
        "family_accuracy": sum(1 for row in predictions if row["family_correct"]) / count if count else 0.0,
        "exact_plan_match_rate": sum(1 for row in predictions if row["exact_match"]) / count if count else 0.0,
        "compile_success_rate": sum(1 for row in predictions if row["compile_success"]) / count if count else 0.0,
        "per_family": _family_summary(predictions),
    }
    return metrics, predictions


def evaluate_planner(
    config_path: str | None = None,
    *,
    base_only: bool = False,
    backend: str = "mlx",
    lmstudio_model: str | None = None,
    lmstudio_base_url: str = "http://localhost:1234/v1",
    lmstudio_api_key: str | None = None,
) -> dict[str, Any]:
    if backend != "mlx" and base_only:
        raise ValueError("--base-only is supported only for the mlx backend.")

    with StepProgress(total=4, desc="eval-planner") as progress:
        config = load_config(config_path)
        if backend == "mlx":
            generator = PlannerGenerator(
                model_path=config.planner.base_model,
                adapter_path=None if base_only else str(planner_adapter_dir(config)),
            )
            eval_dir = planner_eval_dir(config, base_only=base_only, backend=backend)
            summary = {
                "backend": "mlx",
                "mode": "base_model" if base_only else "adapter",
                "base_model": config.planner.base_model,
                "adapter_path": None if base_only else str(planner_adapter_dir(config)),
            }
        elif backend == "lmstudio":
            resolved_lmstudio_model = lmstudio_model or config.planner.base_model
            generator = LMStudioPlannerGenerator(
                model_name=resolved_lmstudio_model,
                base_url=lmstudio_base_url,
                api_key=lmstudio_api_key,
            )
            eval_dir = planner_eval_dir(
                config,
                backend=backend,
                model_name=resolved_lmstudio_model,
            )
            summary = {
                "backend": "lmstudio",
                "mode": "lmstudio_model",
                "base_model": config.planner.base_model,
                "adapter_path": None,
                "served_model": resolved_lmstudio_model,
                "lmstudio_base_url": lmstudio_base_url,
            }
        else:
            raise ValueError(f"Unsupported planner eval backend: {backend}")

        prepared_dir = prepared_planner_raw_dir(config)
        validation_rows = read_jsonl(prepared_dir / "valid.jsonl")
        test_rows = read_jsonl(prepared_dir / "test.jsonl")
        progress.advance("loaded planner evaluation splits")

        validation_metrics, validation_predictions = _evaluate_split(
            validation_rows,
            generator,
            max_tokens=config.planner.max_tokens,
        )
        progress.advance("evaluated validation split")
        test_metrics, test_predictions = _evaluate_split(
            test_rows,
            generator,
            max_tokens=config.planner.max_tokens,
        )
        progress.advance("evaluated release eval split")
        eval_dir.mkdir(parents=True, exist_ok=True)
        write_json(eval_dir / "validation_metrics.json", validation_metrics)
        write_json(eval_dir / "release_eval_metrics.json", test_metrics)
        write_jsonl(eval_dir / "validation_predictions.jsonl", validation_predictions)
        write_jsonl(eval_dir / "release_eval_predictions.jsonl", test_predictions)
        summary["validation"] = validation_metrics
        summary["release_eval"] = test_metrics
        write_json(eval_dir / "summary.json", summary)
        progress.advance("wrote planner evaluation artifacts")
        return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the local planner with either the MLX fine-tuning stack or an LM Studio-served model."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to the fine-tuning JSON config.")
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Evaluate the base planner model without loading any adapter weights.",
    )
    parser.add_argument(
        "--backend",
        choices=("mlx", "lmstudio"),
        default="mlx",
        help="Planner generation backend: the MLX fine-tuning stack or an LM Studio-served model.",
    )
    parser.add_argument(
        "--lmstudio-model",
        type=str,
        default=None,
        help="LM Studio model name/ID to evaluate when --backend lmstudio.",
    )
    parser.add_argument(
        "--lmstudio-base-url",
        type=str,
        default="http://localhost:1234/v1",
        help="Base URL for LM Studio's OpenAI-compatible API when --backend lmstudio.",
    )
    parser.add_argument(
        "--lmstudio-api-key",
        type=str,
        default=None,
        help="API key for LM Studio's OpenAI-compatible API when --backend lmstudio.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = evaluate_planner(
        args.config,
        base_only=args.base_only,
        backend=args.backend,
        lmstudio_model=args.lmstudio_model,
        lmstudio_base_url=args.lmstudio_base_url,
        lmstudio_api_key=args.lmstudio_api_key,
    )
    print(compact_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
