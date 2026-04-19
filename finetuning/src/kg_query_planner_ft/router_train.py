from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .constants import ROUTER_LABELS
from .json_utils import compact_json, read_jsonl, write_json
from .paths import prepared_router_dir, router_model_dir
from .progress import StepProgress


def _optional_router_deps():
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - exercised only in the training env
        raise RuntimeError(
            "Router training requires torch, datasets, and transformers in the fine-tuning environment."
        ) from exc
    return (
        torch,
        Dataset,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )


def _label_mapping() -> tuple[dict[str, int], dict[int, str]]:
    label2id = {label: index for index, label in enumerate(ROUTER_LABELS)}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label


def _dataset_from_rows(rows: list[dict[str, Any]], dataset_cls: Any, label2id: dict[str, int]) -> Any:
    return dataset_cls.from_list(
        [{"question": row["question"], "labels": label2id[row["label"]]} for row in rows]
    )


def train_router(config_path: str | None = None) -> dict[str, Any]:
    with StepProgress(total=6, desc="train-router") as progress:
        (
            torch,
            Dataset,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        ) = _optional_router_deps()
        config = load_config(config_path)
        prepared_dir = prepared_router_dir(config)
        output_dir = router_model_dir(config)
        output_dir.mkdir(parents=True, exist_ok=True)

        label2id, id2label = _label_mapping()
        train_rows = read_jsonl(prepared_dir / "train.jsonl")
        valid_rows = read_jsonl(prepared_dir / "valid.jsonl")
        progress.advance("loaded prepared router splits")

        tokenizer = AutoTokenizer.from_pretrained(config.router.base_model)
        train_dataset = _dataset_from_rows(train_rows, Dataset, label2id)
        valid_dataset = _dataset_from_rows(valid_rows, Dataset, label2id)

        def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
            return tokenizer(
                batch["question"],
                truncation=True,
                max_length=config.router.max_length,
            )

        train_dataset = train_dataset.map(tokenize, batched=True, desc="tokenize router train")
        valid_dataset = valid_dataset.map(tokenize, batched=True, desc="tokenize router valid")
        columns_to_keep = ["input_ids", "attention_mask", "labels"]
        train_dataset = train_dataset.remove_columns([name for name in train_dataset.column_names if name not in columns_to_keep])
        valid_dataset = valid_dataset.remove_columns([name for name in valid_dataset.column_names if name not in columns_to_keep])
        progress.advance("tokenized router datasets")

        counts = Counter(row["label"] for row in train_rows)
        total = sum(counts.values())
        class_weights = torch.tensor(
            [total / (len(ROUTER_LABELS) * counts[label]) for label in ROUTER_LABELS],
            dtype=torch.float32,
        )

        class WeightedTrainer(Trainer):
            def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any) -> Any:
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(outputs.logits.device))
                loss = loss_fct(outputs.logits, labels)
                return (loss, outputs) if return_outputs else loss

        def compute_metrics(eval_prediction: Any) -> dict[str, float]:
            logits, labels = eval_prediction
            predictions = logits.argmax(axis=1)
            macro_scores = []
            for label_id in range(len(ROUTER_LABELS)):
                tp = np.sum((predictions == label_id) & (labels == label_id))
                fp = np.sum((predictions == label_id) & (labels != label_id))
                fn = np.sum((predictions != label_id) & (labels == label_id))
                precision = tp / (tp + fp) if tp + fp else 1.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                macro_scores.append(f1)
            accuracy = float(np.mean(predictions == labels))
            return {"macro_f1": float(np.mean(macro_scores)), "accuracy": accuracy}

        model = AutoModelForSequenceClassification.from_pretrained(
            config.router.base_model,
            num_labels=len(ROUTER_LABELS),
            label2id=label2id,
            id2label=id2label,
        )
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            learning_rate=config.router.learning_rate,
            num_train_epochs=config.router.epochs,
            per_device_train_batch_size=config.router.train_batch_size,
            per_device_eval_batch_size=config.router.eval_batch_size,
            weight_decay=config.router.weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_macro_f1",
            greater_is_better=True,
            save_total_limit=2,
            report_to=[],
            seed=config.router.seed,
            disable_tqdm=False,
        )
        progress.advance("initialized router model and trainer")

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config.router.early_stopping_patience)],
        )
        progress.advance("starting router fine-tune")
        train_result = trainer.train()
        progress.advance("router fine-tune complete")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        summary = {
            "output_dir": str(output_dir),
            "train_examples": len(train_rows),
            "valid_examples": len(valid_rows),
            "label_counts": dict(sorted(counts.items())),
            "train_metrics": {key: float(value) for key, value in train_result.metrics.items()},
        }
        write_json(output_dir / "training_summary.json", summary)
        progress.advance("saved router artifacts")
        return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the router classifier.")
    parser.add_argument("--config", type=str, default=None, help="Path to the fine-tuning JSON config.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = train_router(args.config)
    print(compact_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
