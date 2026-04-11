import argparse
import json
import math
import sys
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from batch_repair import build_repair_plan, collect_window_chunk_key_texts, example_chunk_key_text
from finreflectkg_projection import (
    discover_trusted_segments,
    load_finreflectkg_rows,
    load_jsonl_records,
    project_dataset_rows,
    sample_empty_examples,
    write_jsonl,
)
from finreflectkg_stage3 import (
    STAGE3_RELATIONS,
    Stage3TeacherAugmentor,
    finalize_stage3_dataset,
    refill_and_finalize_stage3_dataset,
)
from stage3_prompt_profiles import DEFAULT_PROMPT_PROFILE, list_prompt_profiles


DEFAULT_MODEL = "google/gemma-4-26b-a4b"


class RepairTeacherError(RuntimeError):
    pass


def ensure_output_root_is_available(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"Output root already exists and is not empty: {output_root}")


def require_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return load_jsonl_records(path)


def require_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def teacher_error_relations(call_log: dict) -> list[str]:
    relation_reports = call_log.get("relation_reports", {})
    return [
        str(relation)
        for relation, report in relation_reports.items()
        if isinstance(report, dict) and report.get("error")
    ]


def record_teacher_log_or_raise(call_log: dict, teacher_logs: list[dict], teacher_log_path: Path) -> None:
    teacher_logs.append(call_log)
    write_jsonl(teacher_log_path, teacher_logs)
    failed_relations = teacher_error_relations(call_log)
    if not failed_relations:
        return
    relation_list = ", ".join(failed_relations)
    chunk_key = json.dumps(call_log.get("chunk_key", {}), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    raise RepairTeacherError(f"Teacher augmentation failed for {chunk_key}: {relation_list}")


def load_legacy_artifacts(source_batch_dir: Path) -> dict[str, list[dict] | dict]:
    stage1_dir = source_batch_dir / "stage1"
    stage2_dir = source_batch_dir / "stage2"
    stage3_dir = source_batch_dir / "stage3"
    return {
        "projected_examples": require_jsonl(stage1_dir / "projected_examples.jsonl"),
        "stage2_empty_examples": require_jsonl(stage2_dir / "empty_examples.jsonl"),
        "augmented_positive_examples": require_jsonl(stage3_dir / "augmented_positive_examples.jsonl"),
        "augmented_empty_candidates": require_jsonl(stage3_dir / "augmented_empty_candidates.jsonl"),
        "teacher_logs": require_jsonl(stage3_dir / "teacher_logs.jsonl"),
        "stage3_report": require_json(stage3_dir / "stage3_report.json"),
    }


def sort_examples(examples: list[dict]) -> list[dict]:
    return sorted(examples, key=example_chunk_key_text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair an existing batch by adding deterministic HAS_SEGMENT/PART_OF structure without overwriting the original outputs.",
    )
    parser.add_argument("--batch", type=int, required=True, help="Which existing batch to repair.")
    parser.add_argument("--chunks-per-batch", type=int, default=80000, help="Batch window size to repair.")
    parser.add_argument("--source-batch-dir", type=str, default=None, help="Existing batch directory to repair. Defaults to outputs/batch_N.")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Destination root for repaired outputs. Defaults to outputs/repairs/batch_N_segment_fix.",
    )
    parser.add_argument("--plan-only", action="store_true", help="Build repaired Stage 1/2 outputs and a Stage 3 rerun plan without calling the teacher.")
    parser.add_argument("--skip-refill", action="store_true", help="Skip Stage 3 empty refill rounds.")
    parser.add_argument("--hf-dataset", type=str, default="domyn/FinReflectKG", help="Hugging Face dataset id.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split.")
    parser.add_argument("--cache-dir", type=str, default="data/external/huggingface", help="Local cache dir for dataset shards.")
    parser.add_argument("--parquet-file", action="append", default=[], help="Optional local parquet file path. Repeat for multiple files.")
    parser.add_argument("--no-streaming", action="store_true", help="Disable streaming mode.")
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional raw-row cap for debugging.")
    parser.add_argument("--empty-ratio", type=float, default=0.3, help="Target empty ratio for the repaired dataset.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model id for teacher reruns (default: {DEFAULT_MODEL}).")
    parser.add_argument("--base-url", type=str, default="http://localhost:1234/v1", help="Local teacher base URL.")
    parser.add_argument("--api-key", type=str, default="lm-studio", help="Local teacher API key.")
    parser.add_argument("--prompt-profile", type=str, default=DEFAULT_PROMPT_PROFILE, choices=list_prompt_profiles(), help="Stage 3 prompt profile.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per teacher relation.")
    parser.add_argument("--no-schema", action="store_true", default=False, help="Disable JSON Schema response enforcement.")
    parser.add_argument("--no-think", action="store_true", default=False, help="Disable model thinking mode.")
    parser.add_argument("--max-completion-tokens", type=int, default=1024, help="Max tokens per teacher response.")
    args = parser.parse_args()

    skip_chunks = (args.batch - 1) * args.chunks_per_batch
    source_batch_dir = Path(args.source_batch_dir) if args.source_batch_dir else (ROOT_DIR / "outputs" / f"batch_{args.batch}")
    output_root = (
        Path(args.output_root)
        if args.output_root
        else (ROOT_DIR / "outputs" / "repairs" / f"batch_{args.batch}_segment_fix")
    )

    ensure_output_root_is_available(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stage1_dir = output_root / "stage1"
    stage2_dir = output_root / "stage2"
    stage3_dir = output_root / "stage3"

    legacy = load_legacy_artifacts(source_batch_dir)

    def load_rows():
        return load_finreflectkg_rows(
            hf_dataset=None if args.parquet_file else args.hf_dataset,
            split=args.split,
            cache_dir=args.cache_dir,
            parquet_files=args.parquet_file or None,
            streaming=not args.no_streaming,
            limit_rows=args.limit_rows,
        )

    window_rows = load_rows()
    window_chunk_key_texts = collect_window_chunk_key_texts(
        window_rows,
        limit_chunks=args.chunks_per_batch,
        skip_chunks=skip_chunks,
    )

    discovery_rows = load_rows()
    trusted_segments_by_filing, trusted_segment_report = discover_trusted_segments(
        discovery_rows,
        limit_chunks=args.chunks_per_batch,
        skip_chunks=skip_chunks,
    )

    stage1_rows = load_rows()
    repaired_projected_examples, stage1_report = project_dataset_rows(
        stage1_rows,
        limit_chunks=args.chunks_per_batch,
        skip_chunks=skip_chunks,
        trusted_segments_by_filing=trusted_segments_by_filing,
    )
    repaired_projected_examples = sort_examples(repaired_projected_examples)
    stage1_report["trusted_segment_discovery"] = trusted_segment_report
    write_jsonl(stage1_dir / "projected_examples.jsonl", repaired_projected_examples)
    (stage1_dir / "projection_report.json").write_text(
        json.dumps(stage1_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    stage2_rows = load_rows()
    repaired_empty_examples, stage2_report = sample_empty_examples(
        stage2_rows,
        positive_example_count=len(repaired_projected_examples),
        empty_ratio=args.empty_ratio,
        limit_chunks=args.chunks_per_batch,
        skip_chunks=skip_chunks,
        trusted_segments_by_filing=trusted_segments_by_filing,
    )
    repaired_empty_examples = sort_examples(repaired_empty_examples)
    write_jsonl(stage2_dir / "empty_examples.jsonl", repaired_empty_examples)
    write_jsonl(stage2_dir / "training_examples.jsonl", repaired_projected_examples + repaired_empty_examples)
    stage2_report_payload = {
        "source_dataset": "domyn/FinReflectKG",
        "trusted_segment_discovery": trusted_segment_report,
        "positive_example_count": len(repaired_projected_examples),
        **stage2_report,
        "merged_example_count": len(repaired_projected_examples) + len(repaired_empty_examples),
    }
    (stage2_dir / "empty_sampling_report.json").write_text(
        json.dumps(stage2_report_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    repair_plan = build_repair_plan(
        repaired_projected_examples=repaired_projected_examples,
        repaired_sampled_empty_examples=repaired_empty_examples,
        legacy_projected_examples=legacy["projected_examples"],
        legacy_empty_origin_examples=legacy["stage2_empty_examples"] + legacy["augmented_empty_candidates"],
        legacy_verified_empty_examples=[
            example
            for example in legacy["augmented_empty_candidates"]
            if not example.get("output", {}).get("triples")
        ],
        legacy_augmented_positive_examples=legacy["augmented_positive_examples"],
        legacy_teacher_logs=legacy["teacher_logs"],
        window_chunk_key_texts=window_chunk_key_texts,
    )

    reused_augmented_positive_examples = sort_examples(repair_plan["reused_augmented_positive_examples"])
    reused_verified_empty_examples = sort_examples(repair_plan["reused_verified_empty_examples"])
    positive_examples_to_rerun = sort_examples(repair_plan["positive_examples_to_rerun"])
    new_empty_candidates_to_run = sort_examples(repair_plan["new_empty_candidates_to_run"])
    write_jsonl(stage3_dir / "reused_augmented_positive_examples.jsonl", reused_augmented_positive_examples)
    write_jsonl(stage3_dir / "reused_verified_empty_examples.jsonl", reused_verified_empty_examples)
    write_jsonl(stage3_dir / "pending_positive_examples.jsonl", positive_examples_to_rerun)
    write_jsonl(stage3_dir / "pending_empty_candidates.jsonl", new_empty_candidates_to_run)

    repair_plan_payload = {
        "batch": args.batch,
        "source_batch_dir": str(source_batch_dir),
        "output_root": str(output_root),
        "chunks_per_batch": args.chunks_per_batch,
        "skip_chunks": skip_chunks,
        "trusted_segment_discovery": trusted_segment_report,
        **repair_plan["report"],
    }
    (stage3_dir / "repair_plan.json").write_text(
        json.dumps(repair_plan_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.plan_only:
        print(f"Built repair plan in {output_root}")
        print(f"  Repaired positives: {len(repaired_projected_examples)}")
        print(f"  Reused Stage 3 positives: {len(reused_augmented_positive_examples)}")
        print(f"  Positive reruns needed: {len(positive_examples_to_rerun)}")
        print(f"  Reused verified empties (no rerun): {len(reused_verified_empty_examples)}")
        print(f"  New empty candidates to run: {len(new_empty_candidates_to_run)}")
        return 0

    augmentor = Stage3TeacherAugmentor(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        prompt_profile=args.prompt_profile,
        use_schema=not args.no_schema,
        disable_thinking=args.no_think,
        max_completion_tokens=args.max_completion_tokens,
    )

    teacher_logs = list(repair_plan["reused_teacher_logs"])
    teacher_log_path = stage3_dir / "teacher_logs.jsonl"
    token_limit_exceeded_count = 0
    augmented_positive_examples = list(reused_augmented_positive_examples)

    for example in tqdm(positive_examples_to_rerun, desc="Repair positives", unit="example"):
        augmented_example, call_log = augmentor.augment_example(example, relations=STAGE3_RELATIONS, max_retries=args.max_retries)
        record_teacher_log_or_raise(call_log, teacher_logs, teacher_log_path)
        if call_log.get("token_limit_exceeded"):
            token_limit_exceeded_count += 1
            continue
        augmented_positive_examples.append(augmented_example)

    # Run teacher on new empty candidates until we reach the target empty count.
    # Legacy verified empties are carried forward as-is and count toward the target.
    target_empty_count = math.ceil(len(repaired_projected_examples) * args.empty_ratio / (1 - args.empty_ratio))
    verified_empty_count = len(reused_verified_empty_examples)
    new_augmented_empty_candidates = []
    for example in tqdm(new_empty_candidates_to_run, desc="Verify new empties", unit="example"):
        if verified_empty_count >= target_empty_count:
            break
        augmented_example, call_log = augmentor.augment_example(example, relations=STAGE3_RELATIONS, max_retries=args.max_retries)
        record_teacher_log_or_raise(call_log, teacher_logs, teacher_log_path)
        if call_log.get("token_limit_exceeded"):
            token_limit_exceeded_count += 1
            continue
        new_augmented_empty_candidates.append(augmented_example)
        if not augmented_example.get("output", {}).get("triples"):
            verified_empty_count += 1

    # Combine reused verified empties with newly verified candidates for finalization.
    initial_augmented_empty_candidates = reused_verified_empty_examples + new_augmented_empty_candidates

    augmented_positive_examples = sort_examples(augmented_positive_examples)
    if args.skip_refill:
        final_positive_examples, final_empty_examples, training_examples, final_report = finalize_stage3_dataset(
            augmented_positive_examples,
            initial_augmented_empty_candidates,
            empty_ratio=args.empty_ratio,
            augmented_relation_trigger_candidates=[],
        )
        all_augmented_empty_candidates = list(initial_augmented_empty_candidates)
        refill_logs = []
    else:
        final_positive_examples, final_empty_examples, training_examples, all_augmented_empty_candidates, final_report, refill_logs = refill_and_finalize_stage3_dataset(
            augmentor=augmentor,
            augmented_positive_examples=augmented_positive_examples,
            initial_augmented_empty_candidates=initial_augmented_empty_candidates,
            augmented_relation_trigger_candidates=[],
            empty_ratio=args.empty_ratio,
            hf_dataset=None if args.parquet_file else args.hf_dataset,
            split=args.split,
            cache_dir=args.cache_dir,
            parquet_files=args.parquet_file or None,
            streaming=not args.no_streaming,
            limit_rows=args.limit_rows,
            limit_chunks=args.chunks_per_batch,
            skip_chunks=skip_chunks,
            min_word_count=80,
            min_char_count=400,
            max_retries=args.max_retries,
            refill_pool_multiplier=2.0,
            max_refill_rounds=5,
            trusted_segments_by_filing=trusted_segments_by_filing,
            trusted_segment_report=trusted_segment_report,
        )
    teacher_logs.extend(refill_logs)
    write_jsonl(teacher_log_path, teacher_logs)

    all_augmented_empty_candidates = sort_examples(all_augmented_empty_candidates)
    final_positive_examples = sort_examples(final_positive_examples)
    final_empty_examples = sort_examples(final_empty_examples)
    training_examples = sort_examples(training_examples)

    write_jsonl(stage3_dir / "augmented_positive_examples.jsonl", augmented_positive_examples)
    write_jsonl(stage3_dir / "augmented_empty_candidates.jsonl", all_augmented_empty_candidates)
    write_jsonl(stage3_dir / "augmented_relation_trigger_candidates.jsonl", [])
    write_jsonl(stage3_dir / "final_positive_examples.jsonl", final_positive_examples)
    write_jsonl(stage3_dir / "final_empty_examples.jsonl", final_empty_examples)
    write_jsonl(stage3_dir / "training_examples.jsonl", training_examples)

    final_report_payload = {
        "batch": args.batch,
        "source_batch_dir": str(source_batch_dir),
        "output_root": str(output_root),
        "repair_mode": "legacy-verified-empty-union",
        "trusted_segment_discovery": trusted_segment_report,
        "repair_plan": repair_plan["report"],
        "token_limit_exceeded_count": token_limit_exceeded_count,
        **final_report,
    }
    (stage3_dir / "stage3_report.json").write_text(
        json.dumps(final_report_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Repaired batch written to {output_root}")
    print(f"  Training examples: {final_report.get('final_training_example_count', '?')}")
    print(f"  Positive: {final_report.get('final_positive_example_count', '?')}")
    print(f"  Empty: {final_report.get('selected_empty_count', '?')}")
    print(f"  Relations: {final_report.get('relation_counts', {})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
