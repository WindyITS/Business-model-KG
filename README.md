# Business Model KG

A local pipeline for turning SEC 10-K business sections into standardized
business-model knowledge graphs.

The maintained surface is the extraction and runtime stack for building,
validating, loading, and querying business-model graphs.

## What It Does

The project reads a filing, runs one of the supported extraction pipelines,
validates the result against a fixed ontology, saves run artifacts, and can
load the graph into Neo4j for querying.

At a high level:

1. read a 10-K business section
2. extract a business-model graph
3. resolve and validate entities and triples
4. save outputs under `outputs/`
5. optionally load the result into Neo4j
6. query the live graph with natural language or Cypher

For the plain-language architecture walkthrough, see
[`docs/project_walkthrough.md`](./docs/project_walkthrough.md).

## Graph Model

The graph is centered on:

- `Company`
- `BusinessSegment`
- `Offering`

It also uses canonical labels for:

- `CustomerType`
- `Channel`
- `RevenueModel`
- `Place`

The ontology is segment-centered:

- `Company -> HAS_SEGMENT -> BusinessSegment`
- `BusinessSegment -> OFFERS -> Offering`
- `BusinessSegment -> SERVES -> CustomerType`
- `BusinessSegment -> SELLS_THROUGH -> Channel`
- `Offering -> MONETIZES_VIA -> RevenueModel`
- `Company -> OPERATES_IN -> Place`
- `Company -> PARTNERS_WITH -> Company`

Full ontology details live in [`docs/ontology.md`](./docs/ontology.md).

## Extraction Pipelines

The repo ships three supported extraction pipelines:

- `analyst`: memo-first extraction, graph compilation, then critique
- `memo_graph_only`: first analyst memo plus graph compilation, without later memo augmentation or critique
- `zero-shot`: direct single-pass graph extraction baseline

Prompt assets are edited under [`prompts/`](./prompts/). Packaged fallback
copies live under [`src/llm_extraction/_bundled_prompts/`](./src/llm_extraction/_bundled_prompts/).

## Quickstart

Requirements:

- Python 3.10+
- an OpenAI-compatible local endpoint such as LM Studio
- optionally Neo4j
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

The `dev` extra installs test tooling. If you also want the runtime to execute
the published local query stack, install:

```bash
pip install -e ".[query-stack]"
```

Copy `.env.example` if you want a template for local environment variables.

## Common Commands

Run an extraction without touching Neo4j:

```bash
./scripts/kg-pipeline data/microsoft_10k.txt --skip-neo4j
```

Choose a specific pipeline:

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

Check local readiness:

```bash
./scripts/kg-health-check
```

Run maintainer checks:

```bash
bash ./scripts/check_repo.sh
```

For the full command and runtime guide, see
[`docs/runtime_guide.md`](./docs/runtime_guide.md).

## Repo Map

The most important folders are:

- [`src/runtime/`](./src/runtime/): CLI behavior, output layout, query stack, Neo4j helpers
- [`src/llm_extraction/pipelines/`](./src/llm_extraction/pipelines/): pipeline orchestration
- [`src/llm/`](./src/llm/): model transport, retries, parsing, and auditing
- [`src/ontology/`](./src/ontology/): ontology config, place hierarchy, and validation
- [`src/graph/`](./src/graph/): Neo4j loading and maintenance logic
- [`prompts/`](./prompts/): editable extraction prompts
- [`evaluation/`](./evaluation/): benchmark data, evaluation scripts, and results
- [`finetuning/`](./finetuning/): isolated local query-router/planner fine-tuning workflow
- [`data/query_planner_curated/`](./data/query_planner_curated/): preserved curated query-planner datasets

For a fuller layout, see [`docs/repo_structure.md`](./docs/repo_structure.md).

## Evaluation And Fine-Tuning

Evaluation compares post-resolution, post-validation output triples against
manually curated gold triples.

Start with:

- [`evaluation/README.md`](./evaluation/README.md) for benchmark and evaluation commands
- [`docs/evaluation_contract.md`](./docs/evaluation_contract.md) for the scoring contract

The fine-tuning workflow is intentionally isolated from the main runtime.
Start with [`finetuning/README.md`](./finetuning/README.md).

## Tests

Run the main test suite:

```bash
./venv/bin/python -m unittest discover -s tests
```

Run the broader maintainer check:

```bash
bash ./scripts/check_repo.sh
```
