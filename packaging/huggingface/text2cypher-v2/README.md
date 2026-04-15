---
pretty_name: Text2Cypher v2
language:
  - en
task_categories:
  - question-answering
  - text-to-sql
tags:
  - neo4j
  - cypher
  - text-to-cypher
  - synthetic-data
  - agent-orchestrated
annotations_creators:
  - machine-generated
size_categories:
  - 1K<n<10K
---

# Text2Cypher v2

`Text2Cypher v2` is a synthetic supervised dataset for training and evaluating models that translate natural-language requests into Neo4j Cypher.

It was built as a spec-first corpus rather than as a blind auto-generation run. The dataset was designed from query families and intent-level tasks, then grounded in synthetic knowledge graphs, validated gold Cypher, and expanded user-facing phrasings. The result is a dataset that teaches both the query shape and the language users actually use when asking for it.

## Dataset snapshot

- `24` synthetic fixtures
- `421` source examples
- `4,501` training rows
- `112` intents
- `31` query families
- `421/421` validation pass rate against the Neo4j-backed synthetic graphs

## What is in the release

The corpus covers the intended text-to-Cypher surface for this project, including:

- company-scoped inventory lookups for `BusinessSegment` and `Offering`
- hierarchical traversal and rollup queries
- geography-aware queries using `Place.within_places` and `Place.includes_places`
- multi-constraint compositions that combine inventory, geography, customer, channel, and revenue signals
- refusal cases for unsupported, ambiguous, or underspecified requests

The dataset was deliberately expanded beyond clean benchmark phrasing. Many examples use messier analyst-style wording so the model sees variation closer to real user prompts.

## How it was built

The build followed an agent-orchestrated, spec-first workflow:

1. define query families and intent-level semantic tasks
2. author synthetic graph fixtures where each task is answerable, ambiguous, or unsupported on purpose
3. write gold parameterized Cypher for each intent
4. bind those intents to concrete synthetic values
5. validate the Cypher against Neo4j-backed synthetic graphs
6. expand the natural-language side with multiple paraphrases, including noisy and conversational prompts

In other words, agents handled orchestration, expansion, and validation, while the dataset logic itself remained curated at the fixture, intent, and query-pattern level.

## Recommended use

This release is intended for:

- fine-tuning text-to-Cypher models
- evaluating query decomposition and graph reasoning
- testing refusal behavior for unsupported or ambiguous requests

It is not a substitute for a real production graph, and it does not aim to model every possible enterprise query. Its purpose is to teach the current query contract of this project as faithfully as possible.

## Data organization

The canonical machine-readable release is expected to live in a Hugging Face dataset repository with a simple root layout, typically:

- `source/fixture_instances.jsonl`
- `source/bound_seed_examples.jsonl`
- `reports/bound_seed_validation_report.json`
- `reports/training_split_manifest.json`
- `training/training_examples.jsonl`
- `training/train.jsonl`
- `training/dev.jsonl`
- `training/test.jsonl`

## Notes on provenance

The corpus was built from a synthetic graph specification and validated end to end before release. The important design idea is that the dataset is not just a pile of questions and answers: it is a checked mapping from intent to graph pattern to Cypher to user phrasing.

That makes the release easier to maintain over time, because new versions can be regenerated from the same high-level specification instead of patched by hand.
