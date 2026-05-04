# Business Model KG

Business Model KG turns SEC 10-K business sections into standardized
business-model knowledge graphs.

The project asks a practical research question: can a language model extract a
useful and comparable view of how companies make money while staying inside a
fixed graph schema and leaving enough artifacts behind for evaluation,
debugging, and reproduction?

The repository contains the maintained extraction stack, the ontology, the
evaluation benchmark interface, an optional Neo4j query runtime, and an isolated
fine-tuning workflow for the local query router/planner.

## How The Project Fits Together

At a high level, the workflow is:

1. read a 10-K business section from `data/`
2. run one of the extraction pipelines
3. resolve surface forms and validate triples against the ontology
4. save the run under `outputs/<company>/<pipeline>/`
5. evaluate saved outputs against curated benchmark triples
6. optionally load saved graphs into Neo4j
7. optionally query the live graph in natural language or rendered Cypher

The graph is segment-centered. A `Company` has `BusinessSegment` nodes, segments
offer `Offering` nodes, and the graph also records customer types, channels,
revenue models, operating geographies, and named partners.

The three extraction pipelines compare different prompting strategies while
sharing the same ontology, validation, output layout, and evaluator:

- `analyst`: memo-first extraction with augmentation, graph compilation, and critique
- `memo_graph_only`: memo-first extraction without later augmentation or critique
- `zero-shot`: direct one-pass graph extraction

For the conceptual tour, start with
[`docs/project_walkthrough.md`](./docs/project_walkthrough.md). For the schema,
see [`docs/ontology.md`](./docs/ontology.md).

## Quick Start

Requirements:

- Python 3.10+
- an OpenAI-compatible local endpoint such as LM Studio if you want to run new extraction calls
- optionally Docker and Neo4j for graph loading and querying
- optionally an OpenCode Go API key for hosted extraction or hosted query fallback

Bootstrap a source checkout:

```bash
./scripts/bootstrap_dev.sh
```

Run a light local readiness check:

```bash
./scripts/kg-health-check --skip-neo4j
```

Run an extraction without loading Neo4j. Start LM Studio or another
OpenAI-compatible local endpoint first, or use a hosted provider such as
OpenCode Go with an API key.

```bash
./scripts/kg-pipeline data/microsoft_10k.txt --skip-neo4j
```

Start Neo4j and load saved outputs when you want the graph query layer:

```bash
docker compose up -d
./scripts/kg-neo4j-load
./scripts/kg-neo4j-status
```

The full maintainer check is broader: it runs tests, fine-tuning tests,
compilation checks, wrapper checks, and package smoke installs.

```bash
bash ./scripts/check_repo.sh
```

## Public Artifacts

Generated outputs, fine-tuning data, and the local query-stack bundle are not
tracked in Git. They are either generated locally or downloaded from Hugging
Face.

The public artifacts are:

- [`WindyITS/business-model-kg-benchmark-outputs`](https://huggingface.co/datasets/WindyITS/business-model-kg-benchmark-outputs): benchmark triples and generated extraction outputs
- [`WindyITS/business-model-kg-query-planner-data`](https://huggingface.co/datasets/WindyITS/business-model-kg-query-planner-data): query-router/planner fine-tuning data
- [`WindyITS/business-model-kg-query-stack`](https://huggingface.co/WindyITS/business-model-kg-query-stack): deployable local query-stack bundle

Use [`docs/reproducibility.md`](./docs/reproducibility.md) as the canonical
reviewer guide for artifact setup and reproduction paths.

## Choose Your Path

If you want to understand the project, read
[`docs/project_walkthrough.md`](./docs/project_walkthrough.md).

If you want to reproduce results, read
[`docs/reproducibility.md`](./docs/reproducibility.md).

If you want command details for extraction, Neo4j, providers, query runtime,
and output layout, read [`docs/runtime_guide.md`](./docs/runtime_guide.md).

If you want evaluation details, read [`docs/evaluation.md`](./docs/evaluation.md).

If you want fine-tuning details, read [`finetuning/README.md`](./finetuning/README.md).

If you want to orient yourself in the repository, read
[`docs/project_map.md`](./docs/project_map.md).

## Maintained Surface

The maintained source packages are:

- `runtime`: extraction, query, Neo4j, output, and health-check CLIs
- `llm_extraction`: pipeline orchestration and prompt rendering
- `llm`: model calls, retries, JSON parsing, and audit helpers
- `ontology`: schema loading, canonical labels, validation, and place handling
- `graph`: Neo4j loading and graph maintenance helpers

The fine-tuning package under `finetuning/` is intentionally separate. It
produces the query-stack bundle used by the main runtime, but the main runtime
does not import fine-tuning code.
