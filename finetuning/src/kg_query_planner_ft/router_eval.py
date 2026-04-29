from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .cli_output import render_router_eval_summary
from .constants import ROUTER_LABELS
from .json_utils import compact_json, read_jsonl, write_json, write_jsonl
from .paths import prepared_router_dir, router_eval_dir, router_model_dir
from .progress import StepProgress, track
from .router_metrics import (
    LOCAL_DECISION_THRESHOLD,
    apply_router_policy,
    apply_temperature,
    id_to_label,
    label_to_id,
    summarize_predictions,
)


def _optional_router_deps():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only in the training env
        raise RuntimeError(
            "Router evaluation requires torch and transformers in the fine-tuning environment."
        ) from exc
    return torch, AutoModelForSequenceClassification, AutoTokenizer


def _torch_device(torch: Any) -> Any:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _fit_temperature(logits: np.ndarray, label_ids: list[int]) -> float:
    torch, _, _ = _optional_router_deps()
    device = _torch_device(torch)
    logits_tensor = torch.tensor(logits, dtype=torch.float32, device=device)
    labels_tensor = torch.tensor(label_ids, dtype=torch.long, device=device)
    temperature = torch.nn.Parameter(torch.ones(1, dtype=torch.float32, device=device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion = torch.nn.CrossEntropyLoss()

    def closure() -> Any:
        optimizer.zero_grad()
        loss = criterion(logits_tensor / torch.clamp(temperature, min=1e-3), labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(temperature.detach().cpu(), min=1e-3).item())


def _load_split_rows(prepared_dir: Path, split_name: str) -> list[dict[str, Any]]:
    return read_jsonl(prepared_dir / f"{split_name}.jsonl")


def _load_router_bundle(model_dir: Path) -> tuple[Any, Any, Any]:
    torch, AutoModelForSequenceClassification, AutoTokenizer = _optional_router_deps()
    # DeBERTa V2's fast tokenizer can emit a misleading regex warning in recent transformers;
    # the slow tokenizer yields the same sentencepiece IDs we expect for router scoring.
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = _torch_device(torch)
    model.to(device)
    model.eval()
    return torch, tokenizer, model


def collect_router_logits(
    model_dir: Path,
    rows: list[dict[str, Any]],
    *,
    max_length: int,
    batch_size: int,
    desc: str = "score router",
) -> np.ndarray:
    torch, tokenizer, model = _load_router_bundle(model_dir)
    device = _torch_device(torch)
    logits_batches: list[np.ndarray] = []
    batch_starts = range(0, len(rows), batch_size)
    for start in track(
        batch_starts,
        total=len(rows) // batch_size + int(len(rows) % batch_size > 0),
        desc=desc,
        unit="batch",
    ):
        batch_rows = rows[start : start + batch_size]
        encoded = tokenizer(
            [row["question"] for row in batch_rows],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits.detach().cpu().numpy()
        logits_batches.append(logits)
    return np.concatenate(logits_batches, axis=0) if logits_batches else np.zeros((0, len(ROUTER_LABELS)))


def _argmax_labels(logits: np.ndarray) -> list[str]:
    return [id_to_label(int(index)) for index in logits.argmax(axis=1)]


def build_router_policy(validation_rows: list[dict[str, Any]], validation_logits: np.ndarray) -> dict[str, Any]:
    validation_truth = [row["label"] for row in validation_rows]
    label_ids = [label_to_id(label) for label in validation_truth]
    temperature = _fit_temperature(validation_logits, label_ids)
    calibrated = apply_temperature(validation_logits, temperature)
    thresholded = apply_router_policy(calibrated)
    policy_metrics = summarize_predictions(validation_truth, thresholded)
    local_metrics = policy_metrics["per_label"]["local"]
    return {
        "temperature": temperature,
        "local_threshold": {
            "threshold": LOCAL_DECISION_THRESHOLD,
            "precision": local_metrics["precision"],
            "recall": local_metrics["recall"],
            "support": int(policy_metrics["counts"].get("local", 0)),
        },
        "policy": "local_if_probability_at_least_0.95_else_best_nonlocal",
        "validation_policy_metrics": policy_metrics,
    }


def evaluate_router(config_path: str | None = None) -> dict[str, Any]:
    with StepProgress(total=5, desc="eval-router") as progress:
        config = load_config(config_path)
        prepared_dir = prepared_router_dir(config)
        model_dir = router_model_dir(config)
        eval_dir = router_eval_dir(config)
        eval_dir.mkdir(parents=True, exist_ok=True)

        validation_rows = _load_split_rows(prepared_dir, "valid")
        test_rows = _load_split_rows(prepared_dir, "test")
        progress.advance("loaded prepared evaluation splits")

        validation_logits = collect_router_logits(
            model_dir,
            validation_rows,
            max_length=config.router.max_length,
            batch_size=config.router.eval_batch_size,
            desc="score router validation",
        )
        test_logits = collect_router_logits(
            model_dir,
            test_rows,
            max_length=config.router.max_length,
            batch_size=config.router.eval_batch_size,
            desc="score router release eval",
        )
        progress.advance("scored router logits")

        calibration = build_router_policy(validation_rows, validation_logits)
        progress.advance("fit calibration and fixed router policy")

        validation_truth = [row["label"] for row in validation_rows]
        test_truth = [row["label"] for row in test_rows]
        validation_argmax = _argmax_labels(validation_logits)
        test_argmax = _argmax_labels(test_logits)

        validation_probs = apply_temperature(validation_logits, calibration["temperature"])
        test_probs = apply_temperature(test_logits, calibration["temperature"])

        validation_policy = apply_router_policy(validation_probs)
        test_policy = apply_router_policy(test_probs)

        validation_payload = {
            "argmax_metrics": summarize_predictions(validation_truth, validation_argmax),
            "policy_metrics": summarize_predictions(validation_truth, validation_policy),
        }
        test_payload = {
            "argmax_metrics": summarize_predictions(test_truth, test_argmax),
            "policy_metrics": summarize_predictions(test_truth, test_policy),
        }

        validation_predictions = []
        for row, probs, argmax_label, policy_label in zip(validation_rows, validation_probs, validation_argmax, validation_policy):
            validation_predictions.append(
                {
                    "question": row["question"],
                    "label": row["label"],
                    "argmax_label": argmax_label,
                    "policy_label": policy_label,
                    "probabilities": {label: float(probs[label_to_id(label)]) for label in ROUTER_LABELS},
                }
            )
        test_predictions = []
        for row, probs, argmax_label, policy_label in zip(test_rows, test_probs, test_argmax, test_policy):
            test_predictions.append(
                {
                    "question": row["question"],
                    "label": row["label"],
                    "argmax_label": argmax_label,
                    "policy_label": policy_label,
                    "probabilities": {label: float(probs[label_to_id(label)]) for label in ROUTER_LABELS},
                }
            )
        progress.advance("built evaluation reports")

        thresholds = {
            "temperature": calibration["temperature"],
            "local_threshold": calibration["local_threshold"],
            "policy": calibration["policy"],
            "planner_gate_open": planner_gate_is_open(
                validation_payload["policy_metrics"],
                min_local_precision=config.router.local_precision_min,
            ),
        }
        write_json(eval_dir / "thresholds.json", thresholds)
        write_json(eval_dir / "validation_metrics.json", validation_payload)
        write_json(eval_dir / "release_eval_metrics.json", test_payload)
        write_jsonl(eval_dir / "validation_predictions.jsonl", validation_predictions)
        write_jsonl(eval_dir / "release_eval_predictions.jsonl", test_predictions)

        summary = {
            "model_dir": str(model_dir),
            "eval_dir": str(eval_dir),
            "thresholds": thresholds,
            "validation": validation_payload,
            "release_eval": test_payload,
        }
        write_json(eval_dir / "summary.json", summary)
        progress.advance("wrote router evaluation artifacts")
        return summary


def load_thresholds(eval_dir: Path) -> dict[str, Any]:
    import json

    return json.loads((eval_dir / "thresholds.json").read_text(encoding="utf-8"))


def predict_router_probabilities(
    question: str,
    *,
    model_dir: Path,
    max_length: int,
    temperature: float = 1.0,
) -> dict[str, float]:
    logits = collect_router_logits(
        model_dir,
        [{"question": question}],
        max_length=max_length,
        batch_size=1,
        desc="score router question",
    )
    probabilities = apply_temperature(logits, temperature)[0]
    return {label: float(probabilities[label_to_id(label)]) for label in ROUTER_LABELS}


def decide_router_outcome(probabilities: dict[str, float], _thresholds: dict[str, Any]) -> str:
    if probabilities["local"] >= LOCAL_DECISION_THRESHOLD:
        return "local"
    return "refuse" if probabilities["refuse"] >= probabilities["api_fallback"] else "api_fallback"


def planner_gate_is_open(policy_metrics: dict[str, Any], *, min_local_precision: float) -> bool:
    local_predictions = int(policy_metrics.get("counts", {}).get("local", 0))
    local_precision = float(policy_metrics["per_label"]["local"]["precision"])
    return local_predictions > 0 and local_precision >= min_local_precision


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the router classifier and export fixed routing policy metadata.")
    parser.add_argument("--config", type=str, default=None, help="Path to the fine-tuning JSON config.")
    parser.add_argument("--json", action="store_true", help="Print the final summary as compact JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = evaluate_router(args.config)
    write_json(router_eval_dir(load_config(args.config)) / "latest.json", summary)
    print(compact_json(summary) if args.json else render_router_eval_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
