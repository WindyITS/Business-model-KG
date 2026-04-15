# Text2Cypher Dataset

This folder holds the prose documentation for the supervised text-to-Cypher dataset.

Machine-readable artifacts live under [`datasets/text2cypher/`](../../datasets/text2cypher/README.md), not in `docs/`.

## Canonical Docs

- [Design plan](./design/dataset_plan.md)
- [Coverage grid](./design/coverage_grid.md)
- [Intent inventory](./design/intent_cases_by_family.md)
- [Fixture library](./design/fixture_library.md)
- [Readiness assessment](./design/readiness_v1.md)

## Canonical V1 Artifacts

- [Fixture instances](../../datasets/text2cypher/v1/source/fixture_instances.jsonl)
- [Bound seed examples](../../datasets/text2cypher/v1/source/bound_seed_examples.jsonl)
- [Validation report](../../datasets/text2cypher/v1/reports/bound_seed_validation_report.json)
- [Training corpus](../../datasets/text2cypher/v1/training/training_examples.jsonl)
- [Split manifest](../../datasets/text2cypher/v1/reports/training_split_manifest.json)
- [Train split](../../datasets/text2cypher/v1/training/train.jsonl)
- [Dev split](../../datasets/text2cypher/v1/training/dev.jsonl)
- [Test split](../../datasets/text2cypher/v1/training/test.jsonl)

## Tooling

Dataset validation entrypoint:
- [`src/validate_text2cypher_dataset.py`](../../src/validate_text2cypher_dataset.py)

Graph comparison and Neo4j inspection:
- [`src/evaluate_graph.py`](../../src/evaluate_graph.py)

Important runtime distinction:
- the text-to-Cypher validator loads one synthetic fixture graph at a time into Neo4j and uses name-only uniqueness inside that isolated graph
- the production extraction loader scopes `BusinessSegment` and `Offering` by `company_name`, so the dataset harness validates query behavior but does not reproduce the production multi-company identity model exactly

## Archive

Legacy prototype snapshots are kept in `datasets/text2cypher/archive/v0/` for provenance, but `v1/` is the active dataset root.
