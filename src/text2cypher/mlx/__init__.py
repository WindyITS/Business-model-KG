from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = ROOT / "datasets" / "text2cypher" / "v3"
DEFAULT_TRAIN_MESSAGES_PATH = DATASET_ROOT / "training" / "train_messages.jsonl"
DEFAULT_TEST_MESSAGES_PATH = DATASET_ROOT / "evaluation" / "test_messages.jsonl"
DEFAULT_TEST_EXAMPLES_PATH = DATASET_ROOT / "evaluation" / "test_examples.jsonl"
DEFAULT_FIXTURES_PATH = DATASET_ROOT / "source" / "fixture_instances.jsonl"

DEFAULT_PIPELINE_ROOT = ROOT / "outputs" / "text2cypher_mlx" / "gemma4_e4b"
DEFAULT_PREPARED_DATA_ROOT = DEFAULT_PIPELINE_ROOT / "dataset"
DEFAULT_ADAPTER_PATH = DEFAULT_PIPELINE_ROOT / "adapters"
DEFAULT_EVAL_OUTPUT_ROOT = DEFAULT_PIPELINE_ROOT / "evaluation"

DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_TRAIN_ITERS = 5000
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUMULATION_STEPS = 4
DEFAULT_NUM_LAYERS = 8
DEFAULT_MAX_TOKENS = 512
DEFAULT_ENABLE_THINKING = False

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
_CYTHER_WHITESPACE_RE = re.compile(r"\s+")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False))
            handle.write("\n")


def _prepared_mlx_rows(message_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared_rows: list[dict[str, Any]] = []
    for row in message_rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"SFT row {row.get('sft_example_id')!r} is missing messages")
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"SFT row {row.get('sft_example_id')!r} must end with an assistant message")

        prepared_rows.append(
            {
                "id": row["sft_example_id"],
                "messages": messages,
                "prompt": row["prompt"],
                "completion": row["completion"],
                "split": row["split"],
                "training_example_ids": row["training_example_ids"],
                "metadata": row["metadata"],
            }
        )
    return prepared_rows


def prepare_mlx_chat_dataset(
    train_messages_path: Path = DEFAULT_TRAIN_MESSAGES_PATH,
    test_messages_path: Path = DEFAULT_TEST_MESSAGES_PATH,
    output_root: Path = DEFAULT_PREPARED_DATA_ROOT,
    *,
    force: bool = False,
) -> dict[str, Any]:
    train_rows = _prepared_mlx_rows(load_jsonl(train_messages_path))
    test_rows = _prepared_mlx_rows(load_jsonl(test_messages_path))

    if output_root.exists():
        if not force:
            raise FileExistsError(
                f"Prepared MLX dataset directory already exists: {output_root}. Re-run with force=True to overwrite."
            )
        for child in output_root.iterdir():
            if child.is_dir():
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_file():
                        nested.unlink()
                    elif nested.is_dir():
                        nested.rmdir()
                child.rmdir()
            else:
                child.unlink()
    output_root.mkdir(parents=True, exist_ok=True)

    train_output_path = output_root / "train.jsonl"
    test_output_path = output_root / "test.jsonl"
    write_jsonl(train_output_path, train_rows)
    write_jsonl(test_output_path, test_rows)

    manifest = {
        "dataset_root": str(output_root),
        "source_files": {
            "train_messages_path": str(train_messages_path),
            "test_messages_path": str(test_messages_path),
        },
        "counts": {
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
        },
        "format": "chat",
        "notes": {
            "mask_prompt_recommended": True,
            "final_message_is_completion": True,
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_mlx_lora_command(
    *,
    model: str,
    data_dir: Path,
    adapter_path: Path,
    iters: int = DEFAULT_TRAIN_ITERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accumulation_steps: int = DEFAULT_GRAD_ACCUMULATION_STEPS,
    num_layers: int = DEFAULT_NUM_LAYERS,
    fine_tune_type: str = "lora",
    mask_prompt: bool = True,
    grad_checkpoint: bool = True,
    resume_adapter_file: Path | None = None,
    report_to: str | None = None,
    project_name: str | None = None,
    python_bin: str | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    command = [
        python_bin or sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        model,
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapter_path),
        "--iters",
        str(iters),
        "--batch-size",
        str(batch_size),
        "--num-layers",
        str(num_layers),
        "--grad-accumulation-steps",
        str(grad_accumulation_steps),
    ]
    if fine_tune_type != "lora":
        command.extend(["--fine-tune-type", fine_tune_type])
    if mask_prompt:
        command.append("--mask-prompt")
    if grad_checkpoint:
        command.append("--grad-checkpoint")
    if resume_adapter_file is not None:
        command.extend(["--resume-adapter-file", str(resume_adapter_file)])
    if report_to:
        command.extend(["--report-to", report_to])
    if project_name:
        command.extend(["--project-name", project_name])
    command.extend(extra_args)
    return command


def build_mlx_test_command(
    *,
    model: str,
    data_dir: Path,
    adapter_path: Path,
    python_bin: str | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    command = [
        python_bin or sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        model,
        "--adapter-path",
        str(adapter_path),
        "--data",
        str(data_dir),
        "--test",
    ]
    command.extend(extra_args)
    return command


def run_command(command: Sequence[str], *, cwd: Path | None = None, dry_run: bool = False) -> subprocess.CompletedProcess[str] | None:
    if dry_run:
        return None
    return subprocess.run(command, cwd=cwd, check=True, text=True)


def _balanced_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    start_index: int | None = None
    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_index is not None:
                candidates.append(text[start_index : index + 1])
                start_index = None

    return candidates


def extract_json_dict(text: str) -> tuple[dict[str, Any] | None, str | None]:
    stripped = text.strip()
    candidate_texts: list[str] = []
    if stripped:
        candidate_texts.append(stripped)
        candidate_texts.extend(match.strip() for match in _JSON_FENCE_RE.findall(stripped) if match.strip())
        candidate_texts.extend(_balanced_json_candidates(stripped))

    seen: set[str] = set()
    for candidate in reversed(candidate_texts):
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, candidate
    return None, None


def normalize_cypher(cypher: str) -> str:
    return _CYTHER_WHITESPACE_RE.sub(" ", cypher).strip()


def normalize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [normalize_json_value(item) for item in value]
    return value


def normalize_completion_payload(payload: dict[str, Any]) -> dict[str, Any]:
    answerable = payload.get("answerable")
    normalized: dict[str, Any] = {"answerable": answerable}
    if answerable is True:
        normalized["cypher"] = normalize_cypher(str(payload.get("cypher", "")))
        normalized["params"] = normalize_json_value(payload.get("params", {}))
    elif answerable is False:
        normalized["reason"] = payload.get("reason")
    return normalized


def score_prediction_payload(
    raw_output_text: str,
    *,
    gold_completion: str,
) -> dict[str, Any]:
    gold_payload = json.loads(gold_completion)
    parsed_payload, extracted_json = extract_json_dict(raw_output_text)

    metrics: dict[str, Any] = {
        "valid_json": parsed_payload is not None,
        "json_extraction_used": parsed_payload is not None and extracted_json != raw_output_text.strip(),
        "answerable_match": False,
        "reason_match": False,
        "params_exact_match": False,
        "cypher_exact_match": False,
        "cypher_normalized_match": False,
        "structured_match": False,
        "parsed_payload": parsed_payload,
        "extracted_json": extracted_json,
    }
    if parsed_payload is None:
        return metrics

    metrics["answerable_match"] = parsed_payload.get("answerable") == gold_payload.get("answerable")
    metrics["structured_match"] = normalize_completion_payload(parsed_payload) == normalize_completion_payload(gold_payload)

    if gold_payload.get("answerable") is False:
        metrics["reason_match"] = parsed_payload.get("reason") == gold_payload.get("reason")
        return metrics

    predicted_params = parsed_payload.get("params")
    gold_params = gold_payload.get("params")
    if isinstance(predicted_params, dict) and isinstance(gold_params, dict):
        metrics["params_exact_match"] = normalize_json_value(predicted_params) == normalize_json_value(gold_params)

    predicted_cypher = parsed_payload.get("cypher")
    gold_cypher = gold_payload.get("cypher")
    if isinstance(predicted_cypher, str) and isinstance(gold_cypher, str):
        metrics["cypher_exact_match"] = predicted_cypher == gold_cypher
        metrics["cypher_normalized_match"] = normalize_cypher(predicted_cypher) == normalize_cypher(gold_cypher)

    return metrics


def _format_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def summarize_prediction_metrics(row_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total_rows = len(row_results)
    valid_json_rows = sum(result["metrics"]["valid_json"] for result in row_results)
    extraction_rows = sum(result["metrics"]["json_extraction_used"] for result in row_results)
    answerable_rows = [result for result in row_results if result["gold_payload"]["answerable"]]
    refusal_rows = [result for result in row_results if not result["gold_payload"]["answerable"]]

    answerable_match_rows = sum(result["metrics"]["answerable_match"] for result in row_results)
    reason_match_rows = sum(result["metrics"]["reason_match"] for result in refusal_rows)
    params_match_rows = sum(result["metrics"]["params_exact_match"] for result in answerable_rows)
    cypher_exact_rows = sum(result["metrics"]["cypher_exact_match"] for result in answerable_rows)
    cypher_normalized_rows = sum(result["metrics"]["cypher_normalized_match"] for result in answerable_rows)
    structured_match_rows = sum(result["metrics"]["structured_match"] for result in row_results)

    execution_rows = [result for result in row_results if result.get("execution") is not None]
    execution_match_rows = sum(bool(result["execution"]["matched"]) for result in execution_rows)

    return {
        "rows": total_rows,
        "valid_json_rows": valid_json_rows,
        "valid_json_rate": _format_rate(valid_json_rows, total_rows),
        "json_extraction_rows": extraction_rows,
        "json_extraction_rate": _format_rate(extraction_rows, total_rows),
        "answerable_match_rows": answerable_match_rows,
        "answerable_match_rate": _format_rate(answerable_match_rows, total_rows),
        "reason_match_rows": reason_match_rows,
        "reason_match_rate": _format_rate(reason_match_rows, len(refusal_rows)),
        "params_exact_match_rows": params_match_rows,
        "params_exact_match_rate": _format_rate(params_match_rows, len(answerable_rows)),
        "cypher_exact_match_rows": cypher_exact_rows,
        "cypher_exact_match_rate": _format_rate(cypher_exact_rows, len(answerable_rows)),
        "cypher_normalized_match_rows": cypher_normalized_rows,
        "cypher_normalized_match_rate": _format_rate(cypher_normalized_rows, len(answerable_rows)),
        "structured_match_rows": structured_match_rows,
        "structured_match_rate": _format_rate(structured_match_rows, total_rows),
        "execution_rows": len(execution_rows),
        "execution_match_rows": execution_match_rows,
        "execution_match_rate": _format_rate(execution_match_rows, len(execution_rows)),
    }


def generate_model_output_for_messages(
    *,
    model_path: str,
    adapter_path: Path,
    messages: Sequence[dict[str, Any]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    enable_thinking: bool = DEFAULT_ENABLE_THINKING,
) -> str:
    model, tokenizer, generate = load_mlx_model_and_tokenizer(model_path=model_path, adapter_path=adapter_path)
    return generate_output_with_loaded_model(
        model=model,
        tokenizer=tokenizer,
        generate_fn=generate,
        messages=messages,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )


def load_mlx_model_and_tokenizer(
    *,
    model_path: str,
    adapter_path: Path,
) -> tuple[Any, Any, Any]:
    try:
        from mlx_lm import generate, load
    except ImportError as exc:  # pragma: no cover - exercised only in a configured MLX environment.
        raise RuntimeError(
            "mlx-lm is not installed. Install it with `pip install \"mlx-lm[train]\"` before running the MLX pipeline."
        ) from exc

    model, tokenizer = load(model_path, adapter_path=str(adapter_path))
    return model, tokenizer, generate


def generate_output_with_loaded_model(
    *,
    model: Any,
    tokenizer: Any,
    generate_fn: Any,
    messages: Sequence[dict[str, Any]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    enable_thinking: bool = DEFAULT_ENABLE_THINKING,
) -> str:
    prompt_messages = list(messages[:-1])
    prompt = _apply_chat_template(tokenizer, prompt_messages, enable_thinking=enable_thinking)
    return generate_fn(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


def _apply_chat_template(tokenizer: Any, messages: Sequence[dict[str, Any]], *, enable_thinking: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _normalize_record(record: dict[str, Any]) -> str:
    return json.dumps(normalize_json_value(record), ensure_ascii=False, sort_keys=True)


def evaluate_execution_matches(
    *,
    row_results: Sequence[dict[str, Any]],
    example_rows: Sequence[dict[str, Any]],
    fixture_rows: Sequence[dict[str, Any]],
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
) -> list[dict[str, Any]]:
    from text2cypher.validation import (
        SyntheticGraphLoader,
        validate_params_match,
        validate_read_only_cypher,
        validate_result,
    )

    examples_by_training_id = {row["training_example_id"]: row for row in example_rows}
    fixtures_by_graph_id = {row["graph_id"]: row for row in fixture_rows}
    results_by_graph_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in row_results:
        training_ids = result["message_row"]["training_example_ids"]
        if len(training_ids) != 1:
            raise ValueError(
                f"MLX evaluation expects one training example per held-out SFT row, got {training_ids!r}"
            )
        training_id = training_ids[0]
        example = examples_by_training_id[training_id]
        results_by_graph_id[example["graph_id"]].append({**result, "example_row": example})

    loader = SyntheticGraphLoader(neo4j_uri, neo4j_user, neo4j_password)
    try:
        loader.setup_constraints()
        for graph_id, graph_results in results_by_graph_id.items():
            loader.clear_graph()
            loader.load_graph(fixtures_by_graph_id[graph_id])

            gold_cache: dict[str, tuple[list[str], list[dict[str, Any]]]] = {}
            for graph_result in graph_results:
                example = graph_result["example_row"]
                if example["answerable"]:
                    gold_cache[example["training_example_id"]] = loader.run_query(example["gold_cypher"], example["params"])

            for graph_result in graph_results:
                graph_result["execution"] = _evaluate_execution_match(
                    loader=loader,
                    prediction_result=graph_result,
                    gold_result_cache=gold_cache,
                    validate_params_match_fn=validate_params_match,
                    validate_read_only_cypher_fn=validate_read_only_cypher,
                    validate_result_fn=validate_result,
                )
    finally:
        loader.close()
    return list(row_results)


def _evaluate_execution_match(
    *,
    loader: Any,
    prediction_result: dict[str, Any],
    gold_result_cache: dict[str, tuple[list[str], list[dict[str, Any]]]],
    validate_params_match_fn: Any,
    validate_read_only_cypher_fn: Any,
    validate_result_fn: Any,
) -> dict[str, Any]:
    example = prediction_result["example_row"]
    metrics = prediction_result["metrics"]
    parsed_payload = metrics["parsed_payload"]

    if not example["answerable"]:
        return {
            "matched": bool(parsed_payload and parsed_payload.get("answerable") is False),
            "skipped": True,
            "reason": "gold_refusal",
            "failures": [],
        }

    failures: list[str] = []
    if parsed_payload is None:
        failures.append("Prediction was not valid JSON")
        return {"matched": False, "skipped": False, "reason": "invalid_json", "failures": failures}
    if parsed_payload.get("answerable") is not True:
        failures.append("Prediction refused an answerable held-out question")
        return {"matched": False, "skipped": False, "reason": "predicted_refusal", "failures": failures}

    cypher = parsed_payload.get("cypher")
    params = parsed_payload.get("params")
    if not isinstance(cypher, str) or not cypher.strip():
        failures.append("Prediction omitted a non-empty cypher string")
    if not isinstance(params, dict):
        failures.append("Prediction omitted a params object")
        params = {}

    if failures:
        return {"matched": False, "skipped": False, "reason": "invalid_payload_shape", "failures": failures}

    failures.extend(validate_read_only_cypher_fn(cypher))
    failures.extend(validate_params_match_fn(cypher, params))

    if failures:
        return {"matched": False, "skipped": False, "reason": "query_contract_failure", "failures": failures}

    try:
        predicted_columns, predicted_records = loader.run_query(cypher, params)
    except Exception as exc:  # pragma: no cover - exercised only with a live Neo4j instance.
        failures.append(f"Execution failed: {exc}")
        return {"matched": False, "skipped": False, "reason": "execution_failed", "failures": failures}

    failures.extend(
        validate_result_fn(example["intent_id"], example["result_shape"], predicted_columns, predicted_records)
    )

    gold_columns, gold_records = gold_result_cache[example["training_example_id"]]
    predicted_record_set = sorted(_normalize_record(record) for record in predicted_records)
    gold_record_set = sorted(_normalize_record(record) for record in gold_records)
    matched = predicted_columns == gold_columns and predicted_record_set == gold_record_set and not failures

    return {
        "matched": matched,
        "skipped": False,
        "reason": "compared",
        "failures": failures,
        "predicted_columns": predicted_columns,
        "gold_columns": gold_columns,
        "predicted_row_count": len(predicted_records),
        "gold_row_count": len(gold_records),
    }
