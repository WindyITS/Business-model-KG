# Text2Cypher Dataset

This folder holds the prose documentation for the supervised text-to-Cypher dataset.

Machine-readable artifacts are built locally under `datasets/text2cypher/v3/` and exported to Hugging Face with `text2cypher-export-hf` or [`scripts/text2cypher/export_hf_text2cypher_dataset.py`](../../scripts/text2cypher/export_hf_text2cypher_dataset.py). The generated dataset directories are intentionally not tracked in git.

## Canonical Docs

- [Design plan](./design/dataset_plan.md)
- [Coverage grid](./design/coverage_grid.md)
- [Intent inventory](./design/intent_cases_by_family.md)
- [Fixture library](./design/fixture_library.md)
- [Readiness assessment: v3](./design/readiness_v3.md)
- [MLX LoRA fine-tuning guide](./fine_tuning_mlx.md)

## Local V3 Build Output

Run the local dataset build first:

```bash
./venv/bin/text2cypher-build
```

That writes the current `v3` artifact set under `datasets/text2cypher/v3/`, including:

- `source/fixture_instances.jsonl`
- `source/bound_seed_examples.jsonl`
- `reports/bound_seed_validation_report.json`
- `reports/sft_manifest.json`
- `reports/heldout_test_manifest.json`
- `reports/leakage_report.json`
- `training/training_examples.jsonl`
- `training/valid_examples.jsonl`
- `training/valid_messages.jsonl`
- `training/train_messages.jsonl`
- `evaluation/test_messages.jsonl`

## Release Surfaces

- local generated dataset workspace: `datasets/text2cypher/v3/`
- HF release bundle workspace: `dist/huggingface/text2cypher-v3/`
- public GitHub repo: KG pipeline, ontology, dataset docs, and the workflow narrative
- Hugging Face dataset repo: the published release consumed by trainers and evaluators

## Fine-Tuning Workflow

The current local fine-tuning path is:

1. build the local `v3` dataset workspace
2. prepare MLX-ready `chat` JSONL from the `v3` message exports
3. fine-tune `Qwen/Qwen3-4B` with LoRA on Apple Silicon via `mlx-lm`
4. score the held-out set by JSON validity, structured match, and optional Neo4j execution

See [MLX LoRA fine-tuning guide](./fine_tuning_mlx.md) for the exact commands and output locations.

## Workflow

This text-to-Cypher dataset was built through agent orchestration, but not through a fully autonomous self-generation loop. The semantics were authored and validated deliberately.

The workflow was:

- define query families and intent-level semantic tasks
- author synthetic graph fixtures where those tasks are valid, ambiguous, or unsupported on purpose
- write gold parameterized Cypher for each intent
- bind those intents to concrete synthetic values
- validate the gold queries against Neo4j-backed synthetic graphs
- expand the natural-language side with multiple user phrasings, including messier analyst-style prompts and refusal cases

In practice, agents handled orchestration, expansion, and verification, while the dataset logic itself stayed curated at the intent, fixture, and query-pattern level rather than accepted from blind auto-generation.

## Release Workflow

The repo keeps the workflow narrative public while keeping generated release material local:

- public GitHub repo: KG pipeline, ontology, dataset docs, builder/validator code, and the export workflow
- local dataset workspace: generated `datasets/text2cypher/v3/` artifacts, intentionally kept out of git
- local packaging templates and `dist/` export tree: generated release material, intentionally kept out of git
- Hugging Face dataset repo: the published corpus plus the generated SFT-ready training view

That keeps the provenance visible to anyone landing on the project while still keeping release-operation details out of version control.

## Tooling

Dataset validation entrypoint:

- preferred console entrypoint: `text2cypher-validate`
- implementation: [`src/text2cypher/validation.py`](../../src/text2cypher/validation.py)
- compatibility wrapper: [`src/validate_text2cypher_dataset.py`](../../src/validate_text2cypher_dataset.py)

Important runtime distinction:

- the text-to-Cypher validator still loads one synthetic fixture graph at a time into Neo4j
- the current harness now supports `company_name`-scoped `BusinessSegment` and `Offering` nodes, `node_id`-based relationship loading, and place helper arrays such as `within_places` and `includes_places`
- that means the validator is now much closer to the production query contract, even though it remains fixture-isolated rather than operating over a single shared multi-company database

## Dataset Root

The default local dataset root is `datasets/text2cypher/v3/`.
