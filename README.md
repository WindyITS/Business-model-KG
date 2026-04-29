# Business Model KG Evaluation Handoff

This branch is a lightweight evaluation handoff for Business Model KG, a local
pipeline for turning SEC 10-K business sections into standardized
business-model knowledge graphs.

The full project asks a practical research question: can a language model
extract a useful, comparable view of how companies make money, while still
respecting a fixed graph schema and leaving enough artifacts behind for
evaluation and debugging? The main repository answers that with extraction
pipelines, a shared ontology, saved outputs, an evaluation benchmark, optional
Neo4j loading, and an optional natural-language query layer.

This branch keeps only the parts needed for evaluation work.

## What Is Included Here

This handoff branch includes:

- evaluation scripts, benchmarks, and current evaluation results
- saved extraction outputs under `outputs/`
- source 10-K text files under `data/`
- prompt assets under `prompts/`
- project docs under `docs/`
- evaluation tests under `tests/test_evaluation/`
- the small `runtime.output_layout` helper needed by the evaluator

It intentionally excludes:

- fine-tuning code and artifacts
- query-stack model artifacts
- Neo4j/query runtime code that is not needed for evaluation
- curated query-planner training datasets

In other words, this branch is for benchmark review, metric work, and output
inspection. It is not the full extraction/query runtime checkout.

## Project Story

Public filings describe businesses in rich prose, but that prose is hard to
compare across companies. Business Model KG takes the business section of a
10-K and turns it into a graph centered on companies, business segments,
offerings, customer types, channels, revenue models, places, and named
partners.

The extraction is intentionally not just "ask a model for triples." The full
project keeps a fixed ontology, validates every output against that ontology,
stores intermediate artifacts for inspection, and evaluates the final resolved
graph against manually curated gold triples. This branch preserves the pieces
needed to work on that evaluation loop.

The evaluation-facing workflow is:

1. inspect source 10-K text files under `data/`
2. inspect saved extraction outputs under `outputs/`
3. compare `latest/resolved_triples.json` against gold benchmark triples
4. review unmatched triples in generated CSV files
5. compute strict and hand-matched metrics under `evaluation/results/`

## Graph Model

The graph is segment-centered. `Company` is the corporate shell,
`BusinessSegment` is the main business-model anchor, and `Offering` is the
product or service inventory layer.

The main relation pattern is:

- `Company -> HAS_SEGMENT -> BusinessSegment`
- `BusinessSegment -> OFFERS -> Offering`
- `BusinessSegment -> SERVES -> CustomerType`
- `BusinessSegment -> SELLS_THROUGH -> Channel`
- `Offering -> MONETIZES_VIA -> RevenueModel`
- `Company -> OPERATES_IN -> Place`
- `Company -> PARTNERS_WITH -> Company`

The complete schema, design principles, canonical labels, validation behavior,
and geography rules are documented in [`docs/ontology.md`](./docs/ontology.md).

## Evaluation

Evaluation compares post-resolution, post-validation triples from:

```text
outputs/<company>/<pipeline>/latest/resolved_triples.json
```

against manually curated benchmark triples under:

```text
evaluation/benchmarks/<split>/clean/
```

The primary score is strict normalized typed-triple precision, recall, and F1.
The secondary score is hand-matched precision, recall, and F1, where a reviewer
can explicitly pair otherwise unmatched gold and predicted rows in a review CSV.

Use [`evaluation/README.md`](./evaluation/README.md) for the evaluation
commands and output files. Use
[`docs/evaluation_contract.md`](./docs/evaluation_contract.md) for the scoring
contract, normalization rules, and interpretation notes.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Common Commands

Evaluate one split for one pipeline:

```bash
python -m evaluation.scripts.evaluate --pipeline analyst --split dev
python -m evaluation.scripts.evaluate --pipeline analyst --split test
```

Other pipeline names:

- `zero-shot`
- `memo_graph_only`
- `analyst`

Evaluate one company:

```bash
python -m evaluation.scripts.evaluate --pipeline analyst --company microsoft
```

Apply hand matches after editing review CSVs:

```bash
python -m evaluation.scripts.apply_hand_matches --results-dir evaluation/results/analyst/dev
```

Run the evaluation tests:

```bash
python -m pytest -q tests/test_evaluation
```

## Documentation Map

The docs are split by purpose:

- [`docs/project_walkthrough.md`](./docs/project_walkthrough.md): plain-language overview of the full project flow
- [`docs/ontology.md`](./docs/ontology.md): graph schema, canonical labels, validation behavior, and place rules
- [`docs/runtime_guide.md`](./docs/runtime_guide.md): full-runtime command reference from the main branch; useful context, though this handoff branch does not include every runtime command it mentions
- [`docs/repo_structure.md`](./docs/repo_structure.md): map of the full repository structure; useful context, though this handoff branch is intentionally smaller
- [`evaluation/README.md`](./evaluation/README.md): benchmark layout, evaluation commands, result files, and hand-match workflow
- [`docs/evaluation_contract.md`](./docs/evaluation_contract.md): exact evaluation target, matching rules, metric definitions, and interpretation notes
- [`docs/final_project_completion_plan.md`](./docs/final_project_completion_plan.md): historical project tracker and presentation-oriented notes

## Branch Scope

This branch exists so collaborators can work on evaluation without pulling the
fine-tuning island or local query-stack model artifacts. For extraction,
Neo4j/query runtime work, or fine-tuning, use the main project branch instead.
