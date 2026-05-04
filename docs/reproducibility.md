# Reproducibility Guide

This project has three reproducibility islands. They are intentionally separate
because they answer different questions:

1. Can the project be used locally to extract, load, and query business-model
   graphs?
2. Can the extraction evaluation be reproduced from the public benchmark and
   output artifacts?
3. Can the local query router/planner fine-tuning workflow be rerun from the
   published fine-tuning dataset?

The main codebase, the evaluation assets, and the fine-tuning workflow share
the same project story, but they should be reproduced independently.

## Island 1: Use The Project

This is the normal source-checkout workflow for running extraction and the
runtime tools.

Start with:

```bash
./scripts/bootstrap_dev.sh
bash ./scripts/check_repo.sh
```

Then run an extraction without Neo4j:

```bash
./scripts/kg-pipeline data/microsoft_10k.txt --skip-neo4j
```

If Neo4j is available, start it and load saved graph outputs:

```bash
docker compose up -d
./scripts/kg-neo4j-load
./scripts/kg-neo4j-status
```

For the narrative explanation of the runtime, read:

- [`project_walkthrough.md`](./project_walkthrough.md)
- [`runtime_guide.md`](./runtime_guide.md)
- [`ontology.md`](./ontology.md)

## Island 2: Reproduce Extraction Evaluation

This island measures the extraction pipelines against the curated benchmark.
It does not require rerunning the LLM extraction calls if the public generated
outputs are downloaded first.

The canonical guide is [`evaluation.md`](./evaluation.md). In short:

1. Download `WindyITS/business-model-kg-benchmark-outputs` from Hugging Face.
2. Copy `benchmarks/` into `evaluation/benchmarks/`.
3. Place each published `resolved_triples.json` under
   `outputs/<company>/<pipeline>/latest/`.
4. Run the evaluator for each pipeline and split.

Example:

```bash
for pipeline in zero-shot memo_graph_only analyst; do
  for split in dev test; do
    ./venv/bin/python -m evaluation.scripts.evaluate \
      --pipeline "$pipeline" \
      --split "$split" \
      --yes
  done
done
```

Generated reports are written under:

```text
evaluation/results/<pipeline>/<split>/
```

## Island 3: Reproduce Query-Stack Fine-Tuning

This island is deliberately isolated under `finetuning/`. It has its own
package, environment, config, tests, data-preparation command, training
commands, evaluation commands, and publish step.

Bootstrap only the fine-tuning environment:

```bash
bash finetuning/scripts/bootstrap_env.sh
```

Download the public query-planner dataset into the layout expected by the
default config:

```bash
bash finetuning/scripts/download_query_planner_data.sh
```

Activate the isolated environment:

```bash
source finetuning/.venv/bin/activate
```

Run the fine-tuning workflow:

```bash
prepare-data
train-router
eval-router
train-planner
eval-planner
publish-query-stack
```

The default config lives at:

```text
finetuning/config/default.json
```

It expects the downloaded dataset at:

```text
data/query_planner_curated/v1_final/
```

and writes training artifacts to:

```text
finetuning/artifacts/kg-query-planner/
```

The final publish step writes the runtime bundle consumed by the main query
runtime:

```text
runtime_assets/query_stack/
```

For the detailed fine-tuning narrative, read
[`../finetuning/README.md`](../finetuning/README.md).

## Installed Fine-Tuning Package Notes

The fine-tuning package can also be installed outside the source checkout. In
that mode, the package carries a fallback copy of the default config. Relative
paths in that config resolve from the current working directory unless
`KG_QUERY_PLANNER_FT_ROOT` is set.

Use this when running the fine-tuning CLIs from an installed wheel but keeping
data and artifacts in a project directory:

```bash
export KG_QUERY_PLANNER_FT_ROOT=/path/to/project-root
```

Then the default config paths resolve as:

```text
$KG_QUERY_PLANNER_FT_ROOT/data/query_planner_curated/v1_final/
$KG_QUERY_PLANNER_FT_ROOT/finetuning/artifacts/kg-query-planner/
```
