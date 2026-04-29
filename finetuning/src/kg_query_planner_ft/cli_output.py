from __future__ import annotations

from typing import Any


def _fmt_int(value: Any) -> str:
    return f"{int(value):,}"


def _fmt_float(value: Any, *, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def _fmt_pct(value: Any) -> str:
    return f"{float(value) * 100:.2f}%"


def _fmt_counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{key}={_fmt_int(value)}" for key, value in sorted(counts.items()))


def _fmt_rate(value: Any, total: Any) -> str:
    total_int = int(total)
    if total_int <= 0:
        return "0/0 (0.00%)"
    value_int = int(value)
    return f"{value_int}/{total_int} ({(value_int / total_int) * 100:.2f}%)"


def _fmt_named_counts(counts: dict[str, Any], *, sort_by_value: bool = False) -> str:
    items = list(counts.items())
    if sort_by_value:
        items.sort(key=lambda item: (-int(item[1]), str(item[0])))
    else:
        items.sort(key=lambda item: str(item[0]))
    return ", ".join(f"{key}={_fmt_int(value)}" for key, value in items)


def _metric_line(label: str, metrics: dict[str, Any]) -> str:
    return (
        f"{label}: accuracy={_fmt_pct(metrics['accuracy'])}, "
        f"macro_f1={_fmt_pct(metrics['macro_f1'])}, "
        f"counts={_fmt_counts(metrics['counts'])}"
    )


def render_router_training_summary(summary: dict[str, Any]) -> str:
    train_metrics = summary["train_metrics"]
    lines = [
        "Router Training Summary",
        f"Output dir: {summary['output_dir']}",
        f"Train examples: {_fmt_int(summary['train_examples'])}",
        f"Validation examples: {_fmt_int(summary['valid_examples'])}",
        f"Label counts: {_fmt_counts(summary['label_counts'])}",
        "Train metrics:",
        f"  epochs={_fmt_float(train_metrics['epoch'], digits=1)}",
        f"  loss={_fmt_float(train_metrics['train_loss'], digits=6)}",
        f"  runtime={_fmt_float(train_metrics['train_runtime'], digits=2)}s",
        f"  samples_per_second={_fmt_float(train_metrics['train_samples_per_second'], digits=2)}",
        f"  steps_per_second={_fmt_float(train_metrics['train_steps_per_second'], digits=2)}",
    ]
    return "\n".join(lines)


def render_router_eval_summary(summary: dict[str, Any]) -> str:
    thresholds = summary["thresholds"]
    local_threshold = thresholds["local_threshold"]
    policy_name = thresholds.get("policy", "local_if_probability_at_least_0.95_else_best_nonlocal")
    validation = summary["validation"]
    release_eval = summary["release_eval"]
    lines = [
        "Router Evaluation Summary",
        f"Thresholds dir: {summary['eval_dir']}",
        f"Router model dir: {summary['model_dir']}",
        "Router policy:",
        f"  planner_gate_open={'yes' if thresholds['planner_gate_open'] else 'no'}",
        f"  temperature={_fmt_float(thresholds['temperature'])}",
        f"  policy={policy_name}",
        (
            "  local: "
            f"threshold={_fmt_float(local_threshold['threshold'])}, "
            f"precision={_fmt_pct(local_threshold['precision'])}, "
            f"recall={_fmt_pct(local_threshold['recall'])}, "
            f"support={_fmt_int(local_threshold['support'])}"
        ),
        "Validation split:",
        f"  {_metric_line('argmax', validation['argmax_metrics'])}",
        f"  {_metric_line('policy', validation['policy_metrics'])}",
        "Release eval split:",
        f"  {_metric_line('argmax', release_eval['argmax_metrics'])}",
        f"  {_metric_line('policy', release_eval['policy_metrics'])}",
    ]
    return "\n".join(lines)


def render_publish_query_stack_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Query Stack Publish Summary",
        f"Destination dir: {summary['destination_dir']}",
        f"Manifest: {summary['manifest_path']}",
        f"Router model dir: {summary['router_model_dir']}",
        f"Router thresholds: {summary['router_thresholds_path']}",
        f"Planner adapter dir: {summary['planner_adapter_dir']}",
    ]
    return "\n".join(lines)


def render_prepare_data_summary(summary: dict[str, Any]) -> str:
    router = summary["router"]
    planner_raw = summary["planner_raw"]
    planner_balanced = summary["planner_balanced"]
    lines = [
        "Prepare Data Summary",
        f"Source root: {summary['source_root']}",
        "Router dataset:",
        f"  Output dir: {router['output_dir']}",
        f"  Split counts: {_fmt_named_counts(router['counts_by_split'])}",
        "  Labels by split:",
    ]
    for split_name, counts in sorted(router["label_counts_by_split"].items()):
        lines.append(f"    {split_name}: {_fmt_named_counts(counts)}")

    lines.extend(
        [
            "Planner raw dataset:",
            f"  Output dir: {planner_raw['output_dir']}",
            f"  Split counts: {_fmt_named_counts(planner_raw['counts_by_split'])}",
            f"  Train augmentations: {_fmt_int(planner_raw.get('train_augmentation_rows', 0))}",
            "  Families by split:",
        ]
    )
    for split_name, counts in sorted(planner_raw["family_counts_by_split"].items()):
        lines.append(f"    {split_name}: {_fmt_named_counts(counts, sort_by_value=True)}")
    if planner_raw.get("train_augmentation_family_counts"):
        lines.append(
            "  Augmentation families: "
            f"{_fmt_named_counts(planner_raw['train_augmentation_family_counts'], sort_by_value=True)}"
        )

    lines.extend(
        [
            "Planner balanced dataset:",
            f"  Output dir: {planner_balanced['output_dir']}",
            f"  Split counts: {_fmt_named_counts(planner_balanced['counts_by_split'])}",
            "  Families by split:",
        ]
    )
    for split_name, counts in sorted(planner_balanced["family_counts_by_split"].items()):
        lines.append(f"    {split_name}: {_fmt_named_counts(counts, sort_by_value=True)}")

    return "\n".join(lines)


def _planner_metric_line(label: str, metrics: dict[str, Any]) -> str:
    return (
        f"{label}: json_parse={_fmt_pct(metrics['json_parse_rate'])}, "
        f"contract_valid={_fmt_pct(metrics['contract_valid_rate'])}, "
        f"family_accuracy={_fmt_pct(metrics['family_accuracy'])}, "
        f"exact_match={_fmt_pct(metrics['exact_plan_match_rate'])}"
    )


def _worst_family_lines(metrics: dict[str, Any], *, limit: int = 5) -> list[str]:
    per_family = metrics.get("per_family", {})
    ranked: list[tuple[str, float, dict[str, Any]]] = []
    for family, family_metrics in per_family.items():
        count = int(family_metrics.get("count", 0))
        if count <= 0:
            continue
        contract_valid = int(family_metrics.get("contract_valid", 0))
        exact_match = int(family_metrics.get("exact_match", 0))
        family_correct = int(family_metrics.get("family_correct", 0))
        contract_rate = contract_valid / count
        ranked.append((family, contract_rate, {"count": count, "contract_valid": contract_valid, "exact_match": exact_match, "family_correct": family_correct}))
    ranked.sort(key=lambda item: (item[1], item[0]))
    lines: list[str] = []
    for family, _rate, family_metrics in ranked[:limit]:
        lines.append(
            "  "
            f"{family}: "
            f"contract_valid={_fmt_rate(family_metrics['contract_valid'], family_metrics['count'])}, "
            f"family_correct={_fmt_rate(family_metrics['family_correct'], family_metrics['count'])}, "
            f"exact_match={_fmt_rate(family_metrics['exact_match'], family_metrics['count'])}"
        )
    return lines


def render_planner_eval_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Planner Evaluation Summary",
        f"Mode: {summary['mode']}",
        f"Backend: {summary['backend']}",
        f"Base model: {summary['base_model']}",
    ]
    artifact_dir = summary.get("artifact_dir")
    if artifact_dir:
        lines.append(f"Artifact dir: {artifact_dir}")
    if summary.get("adapter_path"):
        lines.append(f"Adapter path: {summary['adapter_path']}")
    if summary.get("served_model"):
        lines.append(f"Served model: {summary['served_model']}")
    if summary.get("lmstudio_base_url"):
        lines.append(f"LM Studio URL: {summary['lmstudio_base_url']}")

    validation = summary["validation"]
    release_eval = summary["release_eval"]
    lines.extend(
        [
            "Validation split:",
            f"  count={_fmt_int(validation['count'])}",
            f"  {_planner_metric_line('metrics', validation)}",
            "  weakest families by contract validity:",
            *_worst_family_lines(validation),
            "Release eval split:",
            f"  count={_fmt_int(release_eval['count'])}",
            f"  {_planner_metric_line('metrics', release_eval)}",
            "  weakest families by contract validity:",
            *_worst_family_lines(release_eval),
        ]
    )
    return "\n".join(lines)


def render_planner_training_summary(summary: dict[str, Any]) -> str:
    grad_checkpoint = bool(summary.get("grad_checkpoint", False))
    lines = [
        "Planner Training Summary",
        f"Data dir: {summary['data_dir']}",
        f"Adapter dir: {summary['adapter_dir']}",
        f"Train examples: {_fmt_int(summary['train_examples'])}",
        f"Steps per epoch: {_fmt_int(summary['steps_per_epoch'])}",
        f"Total iters: {_fmt_int(summary['total_iters'])}",
        f"Checkpoint every: {_fmt_int(summary['checkpoint_every'])}",
        f"Gradient checkpointing: {'enabled' if grad_checkpoint else 'disabled'}",
        f"Effective batch size: {_fmt_int(summary['effective_batch_size'])}",
        f"Config path: {summary['config_path']}",
    ]
    if summary.get("checkpoint_root_dir"):
        lines.insert(3, f"Checkpoint root dir: {summary['checkpoint_root_dir']}")
    if summary.get("resume_adapter_file"):
        lines.append(f"Resume adapter file: {summary['resume_adapter_file']}")
    else:
        lines.append("Resume adapter file: none (training from scratch)")
    if summary.get("resume_checkpoint_dir"):
        lines.append(f"Resume checkpoint dir: {summary['resume_checkpoint_dir']}")
    if summary.get("latest_resume_checkpoint_dir"):
        lines.append(f"Latest resume checkpoint dir: {summary['latest_resume_checkpoint_dir']}")
    return "\n".join(lines)


__all__ = [
    "render_prepare_data_summary",
    "render_planner_eval_summary",
    "render_planner_training_summary",
    "render_publish_query_stack_summary",
    "render_router_eval_summary",
    "render_router_training_summary",
]
