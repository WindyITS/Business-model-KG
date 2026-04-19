from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

from .constants import ROUTER_LABELS


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    safe_temperature = max(float(temperature), 1e-3)
    return softmax(logits / safe_temperature)


def label_to_id(label: str) -> int:
    return ROUTER_LABELS.index(label)


def id_to_label(index: int) -> str:
    return ROUTER_LABELS[index]


def metrics_for_label(y_true: Iterable[str], y_pred: Iterable[str], label: str) -> dict[str, float]:
    truth = list(y_true)
    pred = list(y_pred)
    tp = sum(1 for actual, guess in zip(truth, pred) if actual == label and guess == label)
    fp = sum(1 for actual, guess in zip(truth, pred) if actual != label and guess == label)
    fn = sum(1 for actual, guess in zip(truth, pred) if actual == label and guess != label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": sum(1 for actual in truth if actual == label),
    }


def macro_f1(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    return float(np.mean([metrics_for_label(y_true, y_pred, label)["f1"] for label in ROUTER_LABELS]))


def confusion_matrix(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, dict[str, int]]:
    matrix = {actual: {pred: 0 for pred in ROUTER_LABELS} for actual in ROUTER_LABELS}
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual][predicted] += 1
    return matrix


def summarize_predictions(y_true: list[str], y_pred: list[str]) -> dict[str, object]:
    per_label = {
        label: metrics_for_label(y_true, y_pred, label)
        for label in ROUTER_LABELS
    }
    accuracy = sum(1 for actual, guess in zip(y_true, y_pred) if actual == guess) / len(y_true)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1(y_true, y_pred),
        "counts": dict(sorted(Counter(y_pred).items())),
        "per_label": per_label,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def choose_binary_threshold(
    scores: np.ndarray,
    truth_is_positive: np.ndarray,
    min_precision: float,
) -> dict[str, float]:
    best: dict[str, float] | None = None
    for threshold in sorted(set(float(score) for score in scores), reverse=True):
        predicted_positive = scores >= threshold
        tp = int(np.sum(predicted_positive & truth_is_positive))
        fp = int(np.sum(predicted_positive & ~truth_is_positive))
        fn = int(np.sum(~predicted_positive & truth_is_positive))
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        support = int(np.sum(predicted_positive))
        candidate = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "support": support,
        }
        if precision < min_precision:
            continue
        if best is None:
            best = candidate
            continue
        if recall > best["recall"]:
            best = candidate
            continue
        if recall == best["recall"] and threshold < best["threshold"]:
            best = candidate
    if best is not None:
        return best
    return {
        "threshold": 1.0,
        "precision": 1.0,
        "recall": 0.0,
        "support": 0,
    }


def apply_router_policy(
    probabilities: np.ndarray,
    local_threshold: float,
    refuse_threshold: float,
) -> list[str]:
    local_index = label_to_id("local")
    refuse_index = label_to_id("refuse")
    decisions: list[str] = []
    for row in probabilities:
        if float(row[local_index]) >= local_threshold:
            decisions.append("local")
        elif float(row[refuse_index]) >= refuse_threshold:
            decisions.append("refuse")
        else:
            decisions.append("api_fallback")
    return decisions
