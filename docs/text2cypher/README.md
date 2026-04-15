# Text2Cypher Dataset

This folder holds the prose documentation for the supervised text-to-Cypher dataset.

Machine-readable artifacts live under [`datasets/text2cypher/`](../../datasets/text2cypher/README.md), not in `docs/`. The canonical corpus and the generated `messages.jsonl` / split-specific message exports all live there.

## Canonical Docs

- [Design plan](./design/dataset_plan.md)
- [Coverage grid](./design/coverage_grid.md)
- [Intent inventory](./design/intent_cases_by_family.md)
- [Fixture library](./design/fixture_library.md)
- [Readiness assessment: v3](./design/readiness_v3.md)

## Active V3 Artifacts

- [Fixture instances](../../datasets/text2cypher/v3/source/fixture_instances.jsonl)
- [Bound seed examples](../../datasets/text2cypher/v3/source/bound_seed_examples.jsonl)
- [Validation report](../../datasets/text2cypher/v3/reports/bound_seed_validation_report.json)
- [Train manifest](../../datasets/text2cypher/v3/reports/sft_manifest.json)
- [Held-out manifest](../../datasets/text2cypher/v3/reports/heldout_test_manifest.json)
- [Leakage report](../../datasets/text2cypher/v3/reports/leakage_report.json)
- [Training corpus](../../datasets/text2cypher/v3/training/training_examples.jsonl)
- [Train messages](../../datasets/text2cypher/v3/training/train_messages.jsonl)
- [Held-out test messages](../../datasets/text2cypher/v3/evaluation/test_messages.jsonl)

## Release Surfaces

- active release: [`datasets/text2cypher/v3/`](../../datasets/text2cypher/README.md)
- train-facing SFT export: [`datasets/text2cypher/v3/training/train_messages.jsonl`](../../datasets/text2cypher/v3/training/train_messages.jsonl)
- held-out evaluation set: [`datasets/text2cypher/v3/evaluation/test_messages.jsonl`](../../datasets/text2cypher/v3/evaluation/test_messages.jsonl)
- public GitHub repo: KG pipeline, ontology, dataset docs, and the workflow narrative
- Hugging Face dataset repo: the release bundle consumed by trainers and evaluators

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

The repo keeps the workflow narrative public, while allowing the packaging notes/templates to stay local:

- public GitHub repo: KG pipeline, ontology, dataset docs, and the workflow narrative
- Hugging Face dataset repo: canonical corpus plus the generated SFT-ready training view
- local packaging templates and `dist/` export tree: generated release material, intentionally kept out of git

That keeps the provenance visible to anyone landing on the project while still keeping release-operation details out of version control.

## Tooling

Dataset validation entrypoint:

- [`src/validate_text2cypher_dataset.py`](../../src/validate_text2cypher_dataset.py)

Important runtime distinction:

- the text-to-Cypher validator still loads one synthetic fixture graph at a time into Neo4j
- the current harness now supports `company_name`-scoped `BusinessSegment` and `Offering` nodes, `node_id`-based relationship loading, and place helper arrays such as `within_places` and `includes_places`
- that means the validator is now much closer to the production query contract, even though it remains fixture-isolated rather than operating over a single shared multi-company database

## Dataset Root

The active machine-readable dataset release lives in `datasets/text2cypher/v3/`.
