# Business Model KG

Business Model KG is a local pipeline for turning SEC 10-K business sections
into standardized business-model knowledge graphs.

The project is built around a practical research question: can a language model
extract a useful, comparable view of how companies make money, while still
respecting a fixed graph schema and leaving enough artifacts behind for
evaluation and debugging? The repo answers that question with three extraction
pipelines, a shared ontology, saved output artifacts, an evaluation benchmark,
and an optional Neo4j query layer.

The maintained surface is the extraction and runtime stack for building,
validating, evaluating, loading, and querying business-model graphs.

## Project Story

Public filings describe businesses in rich prose, but that prose is hard to
compare across companies. This project takes the business section of a 10-K and
turns it into a graph centered on companies, business segments, offerings,
customer types, channels, revenue models, places, and named partners.

The extraction is intentionally not just "ask a model for triples." The repo
keeps a fixed ontology, validates every output against that ontology, stores
intermediate artifacts for inspection, and evaluates the final resolved graph
against manually curated gold triples. That makes the project useful both as a
working local tool and as an experiment about extraction quality.

The full workflow is:

1. read a 10-K business section from `data/`
2. run one of the extraction pipelines
3. resolve surface forms and validate triples against the ontology
4. save the run under `outputs/<company>/<pipeline>/`
5. compare outputs against the benchmark under `evaluation/`
6. optionally load saved graphs into Neo4j
7. optionally query the live graph with natural language or rendered Cypher

For a plain-language walkthrough of how these pieces fit together, start with
[`docs/project_walkthrough.md`](./docs/project_walkthrough.md).

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

`CustomerType`, `Channel`, and `RevenueModel` use closed canonical labels so
different companies can be compared consistently. Geography is kept canonical
at `Company -> OPERATES_IN -> Place`, with downstream helper metadata for
broader or narrower place matching in Neo4j.

The complete schema, design principles, canonical labels, validation behavior,
and geography rules are documented in [`docs/ontology.md`](./docs/ontology.md).

## Extraction Pipelines

The repo includes three supported extraction pipelines:

- `analyst`: builds a structured analyst memo, augments it, compiles it into a graph, then runs a critique pass
- `memo_graph_only`: builds the first analyst memo and compiles it into a graph, skipping later memo augmentation and critique
- `zero-shot`: directly asks for the graph in a single pass

These pipelines share the same downstream resolver, validator, output layout,
and evaluation target. That keeps the comparison focused on extraction strategy
rather than different post-processing rules.

Editable prompt assets live under [`prompts/`](./prompts/). Packaged fallback
copies live under `src/llm_extraction/_bundled_prompts/`, so the project can
still run after installation.

## Evaluation

Evaluation compares post-resolution, post-validation triples from:

```text
outputs/<company>/<pipeline>/latest/resolved_triples.json
```

against manually curated benchmark triples under:

```text
evaluation/benchmarks/<split>/clean/
```

The primary score is exact normalized 3-field edge agreement over
`subject`, `relation`, and `object`, reported as micro precision/recall/F1 and
macro-F1 by company. Strict 5-field typed-triple matching is kept as a
diagnostic view, and relaxed graph-aware F1 gives partial credit for documented
company aliases and hierarchy/roll-up alignments.

Use [`evaluation/README.md`](./evaluation/README.md) for the evaluation
commands and output files. Use
[`docs/evaluation_contract.md`](./docs/evaluation_contract.md) for the scoring
contract, normalization rules, and interpretation notes.

## Published Artifacts

The public datasets and model bundle are published on the Hugging Face profile
[`WindyITS`](https://huggingface.co/WindyITS).

The uploaded artifacts are:

- [`WindyITS/business-model-kg-benchmark-outputs`](https://huggingface.co/datasets/WindyITS/business-model-kg-benchmark-outputs): benchmark triples, generated extraction outputs, and evaluation documentation
- [`WindyITS/business-model-kg-query-planner-data`](https://huggingface.co/datasets/WindyITS/business-model-kg-query-planner-data): curated query-router/planner fine-tuning data
- [`WindyITS/business-model-kg-query-stack`](https://huggingface.co/WindyITS/business-model-kg-query-stack): deployable local query stack with the fine-tuned DeBERTa router and MLX planner adapter

## Query Runtime

The runtime can load saved outputs into Neo4j and answer read-only questions
against the live graph.

There are two query paths behind the same CLI surface:

- a routed local stack that uses a published router/planner bundle only when
  the router assigns `local` with at least `0.97` confidence
- a hosted fallback path that generates guarded read-only Cypher when the local
  stack is unavailable, errors, or the router selects `api_fallback`

If the router selects `refuse`, the command returns an unsupported-request
result instead of calling the hosted fallback.

The query layer is optional. You can run extraction and evaluation without the
Neo4j/query-stack pieces.

For command details, provider options, Neo4j behavior, output layout, prompt
loading, and maintenance commands, see
[`docs/runtime_guide.md`](./docs/runtime_guide.md).

## Quickstart

Requirements:

- Python 3.10+
- an OpenAI-compatible local endpoint such as LM Studio, unless you only run evaluation
- optionally Neo4j for graph loading and querying
- optionally an OpenCode Go API key for hosted model runs

Bootstrap the source checkout:

```bash
./scripts/bootstrap_dev.sh
```

Manual equivalent:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

If you want the runtime to execute the local query stack, install the optional
query extras:

```bash
pip install -e ".[query-stack]"
```

Copy `.env.example` if you want a local environment template.

## Common Commands

Run an extraction without touching Neo4j:

```bash
./scripts/kg-pipeline data/microsoft_10k.txt --skip-neo4j
```

Choose a specific extraction pipeline:

```bash
./scripts/kg-pipeline data/microsoft_10k.txt --pipeline zero-shot --skip-neo4j
```

Start Neo4j and run an extraction that loads the result:

```bash
docker compose up -d
./scripts/kg-pipeline data/microsoft_10k.txt
```

Render a read-only Cypher query:

```bash
./scripts/kg-query-cypher "Which company segments sell through marketplaces?"
```

Run a natural-language query against Neo4j:

```bash
./scripts/kg-query "Which company segments sell through marketplaces?"
```

Evaluate a pipeline on one benchmark split:

```bash
./venv/bin/python -m evaluation.scripts.evaluate --pipeline analyst --split dev
```

Run the main checks:

```bash
bash ./scripts/check_repo.sh
```

## Documentation Map

The docs are split by purpose:

- [`docs/project_walkthrough.md`](./docs/project_walkthrough.md): the best first read after this README; explains the project flow in plain language
- [`docs/ontology.md`](./docs/ontology.md): the graph schema, design principles, canonical labels, validation behavior, and place rules
- [`docs/runtime_guide.md`](./docs/runtime_guide.md): CLI commands, provider settings, Neo4j load/unload/status behavior, query runtime, output artifacts, and prompt workflow
- [`docs/repo_structure.md`](./docs/repo_structure.md): a detailed map of the repository and what each major directory is for
- [`evaluation/README.md`](./evaluation/README.md): benchmark layout, evaluation commands, metrics, result files, and reference datasets
- [`docs/evaluation_contract.md`](./docs/evaluation_contract.md): the exact evaluation target, matching rules, metric definitions, and interpretation notes
- [`finetuning/README.md`](./finetuning/README.md): the isolated fine-tuning island for the local query router/planner

## Repo Map

The main areas are:

- `src/runtime/`: extraction CLI, query CLI, output layout, provider resolution, and Neo4j helper commands
- `src/llm_extraction/pipelines/`: pipeline-specific extraction orchestration
- `src/llm/`: model calls, retries, JSON recovery, parsing, and auditing
- `src/ontology/`: machine-readable ontology, canonical labels, place hierarchy, and validation
- `src/graph/`: Neo4j load, replace, and unload logic
- `prompts/`: editable prompt assets
- `evaluation/`: benchmark files, evaluator scripts, and result artifacts
- `finetuning/`: isolated router/planner training workflow
- `data/`: 10-K text files and curated query-planner datasets

For a more detailed layout, see [`docs/repo_structure.md`](./docs/repo_structure.md).

## Tests

Run the main test suite:

```bash
./venv/bin/python -m unittest discover -s tests
```

Run only the evaluation tests:

```bash
./venv/bin/python -m pytest -q tests/test_evaluation
```
