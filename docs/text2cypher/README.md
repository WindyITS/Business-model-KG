# Text2Cypher Dataset

This folder holds the prose documentation for the supervised text-to-Cypher dataset.

Machine-readable artifacts live under [`datasets/text2cypher/`](../../datasets/text2cypher/README.md), not in `docs/`.

## Canonical Docs

- [Design plan](./design/dataset_plan.md)
- [Coverage grid](./design/coverage_grid.md)
- [Intent inventory](./design/intent_cases_by_family.md)
- [Fixture library](./design/fixture_library.md)
- [Readiness assessment](./design/readiness_v2.md)

## Canonical V2 Artifacts

- [Fixture instances](../../datasets/text2cypher/v2/source/fixture_instances.jsonl)
- [Bound seed examples](../../datasets/text2cypher/v2/source/bound_seed_examples.jsonl)
- [Validation report](../../datasets/text2cypher/v2/reports/bound_seed_validation_report.json)
- [Training corpus](../../datasets/text2cypher/v2/training/training_examples.jsonl)
- [Split manifest](../../datasets/text2cypher/v2/reports/training_split_manifest.json)
- [Train split](../../datasets/text2cypher/v2/training/train.jsonl)
- [Dev split](../../datasets/text2cypher/v2/training/dev.jsonl)
- [Test split](../../datasets/text2cypher/v2/training/test.jsonl)

## How It Was Built

This text-to-Cypher dataset/database was built through agent orchestration rather than a fully autonomous self-generation loop.

The workflow was:

- define query families and intent-level semantic tasks
- author synthetic graph fixtures where those tasks are valid, ambiguous, or unsupported on purpose
- write gold parameterized Cypher for each intent
- bind those intents to concrete synthetic values
- validate the gold queries against Neo4j-backed synthetic graphs
- expand the natural-language side with multiple user phrasings, including messier analyst-style prompts and refusal cases

In practice, the agents handled orchestration, expansion, and verification, but the dataset logic was still curated at the intent, fixture, and query-pattern level rather than accepted from blind auto-generation.

## Tooling

Dataset validation entrypoint:
- [`src/validate_text2cypher_dataset.py`](../../src/validate_text2cypher_dataset.py)

Graph comparison and Neo4j inspection:
- [`src/evaluate_graph.py`](../../src/evaluate_graph.py)

Important runtime distinction:
- the text-to-Cypher validator still loads one synthetic fixture graph at a time into Neo4j
- the current harness now supports `company_name`-scoped `BusinessSegment` and `Offering` nodes, `node_id`-based relationship loading, and place helper arrays such as `within_places` and `includes_places`
- that means the validator is now much closer to the production query contract, even though it remains fixture-isolated rather than operating over a single shared multi-company database

## Archive

Legacy prototype snapshots are kept in `datasets/text2cypher/archive/v0/` for provenance, and `v2/` is the active dataset root.
