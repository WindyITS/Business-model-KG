import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finreflectkg_projection import load_jsonl_records
from finreflectkg_stage3 import (
    STAGE3_RELATIONS,
    Stage3TeacherAugmentor,
    build_relation_trigger_candidate_pool,
    example_chunk_key_text,
    finalize_stage3_dataset,
    refill_and_finalize_stage3_dataset,
    write_jsonl,
)
from stage3_prompt_profiles import DEFAULT_PROMPT_PROFILE, list_prompt_profiles


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 3 teacher augmentation for missing ontology slices.")
    parser.add_argument("--base-url", type=str, default="http://localhost:1234/v1", help="Base URL for the local LLM endpoint.")
    parser.add_argument("--model", type=str, default="local-model", help="Model id sent to the local LLM endpoint.")
    parser.add_argument("--api-key", type=str, default="lm-studio", help="API key for the local LLM endpoint.")
    parser.add_argument("--prompt-profile", type=str, default=DEFAULT_PROMPT_PROFILE, choices=list_prompt_profiles(), help="Named Stage 3 prompt profile to use.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per relation-specific teacher call.")
    parser.add_argument("--debug-dir", type=str, default=None, help="Optional directory where exact Stage-3 prompts/responses are dumped.")
    parser.add_argument("--debug-chunk-filter", type=str, default=None, help="Optional chunk-key/text substring used to decide which example to dump for debugging.")
    parser.add_argument(
        "--projected-jsonl",
        type=str,
        default="outputs/finreflectkg_stage1/projected_examples.jsonl",
        help="Stage-1 projected positive examples.",
    )
    parser.add_argument(
        "--candidate-empty-jsonl",
        type=str,
        default="outputs/finreflectkg_stage2/empty_examples.jsonl",
        help="Stage-2 candidate empty examples.",
    )
    parser.add_argument("--limit-positive-examples", type=int, default=None, help="Optional cap on Stage-1 positive examples for smoke tests.")
    parser.add_argument("--limit-empty-examples", type=int, default=None, help="Optional cap on Stage-2 candidate empty examples for smoke tests.")
    parser.add_argument(
        "--relation-trigger-count",
        type=int,
        default=0,
        help="Optional number of additional non-Stage1 chunks to sample using strong relation trigger phrases.",
    )
    parser.add_argument("--empty-ratio", type=float, default=0.3, help="Final empty/non-empty ratio target.")
    parser.add_argument("--skip-refill", action="store_true", help="Skip sampling replacement empty candidates; useful for teacher smoke tests.")
    parser.add_argument("--refill-pool-multiplier", type=float, default=2.0, help="How aggressively to oversample replacement empty candidates per refill round.")
    parser.add_argument("--max-refill-rounds", type=int, default=5, help="Maximum rounds for replacing candidate empties that become non-empty.")
    parser.add_argument("--hf-dataset", type=str, default="domyn/FinReflectKG", help="Hugging Face dataset id.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split.")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/external/huggingface",
        help="Local cache dir for downloaded dataset shards.",
    )
    parser.add_argument(
        "--parquet-file",
        action="append",
        default=[],
        help="Optional local parquet file path. Repeat for multiple files.",
    )
    parser.add_argument("--no-streaming", action="store_true", help="Disable streaming mode.")
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional cap on raw rows read when sampling extra Stage-3 candidate pools.")
    parser.add_argument("--limit-chunks", type=int, default=None, help="Optional cap on processed chunks when sampling extra Stage-3 candidate pools.")
    parser.add_argument("--min-empty-words", type=int, default=80, help="Minimum words for an empty chunk candidate.")
    parser.add_argument("--min-empty-chars", type=int, default=400, help="Minimum chars for an empty chunk candidate.")
    parser.add_argument(
        "--augmented-positives-output",
        type=str,
        default="outputs/finreflectkg_stage3/augmented_positive_examples.jsonl",
        help="Where to write Stage-3 augmented positive examples.",
    )
    parser.add_argument(
        "--augmented-empty-output",
        type=str,
        default="outputs/finreflectkg_stage3/augmented_empty_candidates.jsonl",
        help="Where to write all augmented candidate empties.",
    )
    parser.add_argument(
        "--augmented-trigger-output",
        type=str,
        default="outputs/finreflectkg_stage3/augmented_relation_trigger_candidates.jsonl",
        help="Where to write Stage-3 augmented relation-trigger candidates.",
    )
    parser.add_argument(
        "--final-positive-output",
        type=str,
        default="outputs/finreflectkg_stage3/final_positive_examples.jsonl",
        help="Where to write final non-empty examples.",
    )
    parser.add_argument(
        "--final-empty-output",
        type=str,
        default="outputs/finreflectkg_stage3/final_empty_examples.jsonl",
        help="Where to write final verified empty examples.",
    )
    parser.add_argument(
        "--training-output",
        type=str,
        default="outputs/finreflectkg_stage3/training_examples.jsonl",
        help="Where to write the final merged Stage-3 training dataset.",
    )
    parser.add_argument(
        "--teacher-log-output",
        type=str,
        default="outputs/finreflectkg_stage3/teacher_logs.jsonl",
        help="Where to write relation-level teacher call logs.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="outputs/finreflectkg_stage3/stage3_report.json",
        help="Where to write the Stage-3 report.",
    )
    args = parser.parse_args()

    projected_examples = load_jsonl_records(Path(args.projected_jsonl))
    candidate_empty_path = Path(args.candidate_empty_jsonl)
    candidate_empty_examples = load_jsonl_records(candidate_empty_path) if candidate_empty_path.exists() else []
    if args.limit_positive_examples is not None:
        projected_examples = projected_examples[: max(0, args.limit_positive_examples)]
    if args.limit_empty_examples is not None:
        candidate_empty_examples = candidate_empty_examples[: max(0, args.limit_empty_examples)]

    augmentor = Stage3TeacherAugmentor(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        prompt_profile=args.prompt_profile,
        debug_dir=args.debug_dir,
        debug_chunk_filter=args.debug_chunk_filter,
    )

    relation_trigger_pool_report = {
        "processed_chunk_count": 0,
        "eligible_relation_trigger_chunk_count": 0,
        "excluded_chunk_count": 0,
        "sampled_relation_trigger_chunk_count": 0,
        "target_relation_trigger_count": max(0, args.relation_trigger_count),
        "matched_relation_counts": {},
        "min_word_count": args.min_empty_words,
        "min_char_count": args.min_empty_chars,
    }
    relation_trigger_candidates = []
    if args.relation_trigger_count > 0:
        exclude_chunk_key_texts = {
            example_chunk_key_text(example)
            for example in projected_examples + candidate_empty_examples
        }
        relation_trigger_candidates, relation_trigger_pool_report = build_relation_trigger_candidate_pool(
            hf_dataset=None if args.parquet_file else args.hf_dataset,
            split=args.split,
            cache_dir=args.cache_dir,
            parquet_files=args.parquet_file or None,
            streaming=not args.no_streaming,
            limit_rows=args.limit_rows,
            limit_chunks=args.limit_chunks,
            target_candidate_count=args.relation_trigger_count,
            exclude_chunk_key_texts=exclude_chunk_key_texts,
            min_word_count=args.min_empty_words,
            min_char_count=args.min_empty_chars,
        )

    augmented_positive_examples = []
    teacher_logs = []
    for example in tqdm(projected_examples, desc="Stage 3 positives", unit="example"):
        augmented_example, call_log = augmentor.augment_example(example, relations=STAGE3_RELATIONS, max_retries=args.max_retries)
        augmented_positive_examples.append(augmented_example)
        teacher_logs.append(call_log)

    initial_augmented_empty_candidates = []
    for example in tqdm(candidate_empty_examples, desc="Stage 3 empties", unit="example"):
        augmented_example, call_log = augmentor.augment_example(example, relations=STAGE3_RELATIONS, max_retries=args.max_retries)
        initial_augmented_empty_candidates.append(augmented_example)
        teacher_logs.append(call_log)

    augmented_relation_trigger_candidates = []
    for example in tqdm(relation_trigger_candidates, desc="Stage 3 triggers", unit="example"):
        augmented_example, call_log = augmentor.augment_example(example, relations=STAGE3_RELATIONS, max_retries=args.max_retries)
        augmented_relation_trigger_candidates.append(augmented_example)
        teacher_logs.append(call_log)

    if args.skip_refill:
        final_positive_examples, final_empty_examples, training_examples, final_report = finalize_stage3_dataset(
            augmented_positive_examples,
            initial_augmented_empty_candidates,
            empty_ratio=args.empty_ratio,
            augmented_relation_trigger_candidates=augmented_relation_trigger_candidates,
        )
        all_augmented_empty_candidates = list(initial_augmented_empty_candidates)
        final_report["refill_round_count"] = 0
        final_report["refill_reports"] = []
        final_report["final_candidate_empty_pool_count"] = len(all_augmented_empty_candidates)
        refill_logs = []
    else:
        final_positive_examples, final_empty_examples, training_examples, all_augmented_empty_candidates, final_report, refill_logs = refill_and_finalize_stage3_dataset(
            augmentor=augmentor,
            augmented_positive_examples=augmented_positive_examples,
            initial_augmented_empty_candidates=initial_augmented_empty_candidates,
            augmented_relation_trigger_candidates=augmented_relation_trigger_candidates,
            empty_ratio=args.empty_ratio,
            hf_dataset=None if args.parquet_file else args.hf_dataset,
            split=args.split,
            cache_dir=args.cache_dir,
            parquet_files=args.parquet_file or None,
            streaming=not args.no_streaming,
            limit_rows=args.limit_rows,
            limit_chunks=args.limit_chunks,
            min_word_count=args.min_empty_words,
            min_char_count=args.min_empty_chars,
            max_retries=args.max_retries,
            refill_pool_multiplier=args.refill_pool_multiplier,
            max_refill_rounds=args.max_refill_rounds,
        )
        teacher_logs.extend(refill_logs)

    write_jsonl(Path(args.augmented_positives_output), augmented_positive_examples)
    write_jsonl(Path(args.augmented_empty_output), all_augmented_empty_candidates)
    write_jsonl(Path(args.augmented_trigger_output), augmented_relation_trigger_candidates)
    write_jsonl(Path(args.final_positive_output), final_positive_examples)
    write_jsonl(Path(args.final_empty_output), final_empty_examples)
    write_jsonl(Path(args.training_output), training_examples)
    write_jsonl(Path(args.teacher_log_output), teacher_logs)

    report_payload = {
        "source_dataset": "domyn/FinReflectKG",
        "projected_jsonl": str(Path(args.projected_jsonl)),
        "candidate_empty_jsonl": str(candidate_empty_path),
        "teacher_relations": list(STAGE3_RELATIONS),
        "prompt_profile": args.prompt_profile,
        "projected_example_count": len(projected_examples),
        "initial_candidate_empty_count": len(candidate_empty_examples),
        "initial_relation_trigger_candidate_count": len(relation_trigger_candidates),
        "augmented_positive_example_count": len(augmented_positive_examples),
        "initial_augmented_candidate_empty_count": len(initial_augmented_empty_candidates),
        "augmented_relation_trigger_candidate_count": len(augmented_relation_trigger_candidates),
        "relation_trigger_pool_report": relation_trigger_pool_report,
        **final_report,
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote Stage 3 augmented positives to {args.augmented_positives_output}")
    print(f"Wrote Stage 3 augmented empties to {args.augmented_empty_output}")
    print(f"Wrote Stage 3 augmented relation-trigger candidates to {args.augmented_trigger_output}")
    print(f"Wrote final training dataset to {args.training_output}")
    print(f"Wrote Stage 3 report to {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
