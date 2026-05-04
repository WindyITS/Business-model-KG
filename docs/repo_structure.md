# Repo Structure

This document is the detailed layout map for the project.

## Top Level

```text
kg-v0/
  README.md
  pyproject.toml
  docker-compose.yml
  .env.example
  data/
  docs/
  evaluation/
  finetuning/
  prompts/
  scripts/
  src/
  tests/
```

## Source Packages

```text
src/
  graph/
  llm/
  llm_extraction/
  ontology/
  runtime/
  training/
```

Important modules:

- `src/runtime/main.py`: extraction CLI runtime used by `kg-pipeline`
- `src/runtime/query.py`: natural-language query runtime used by `kg-query`
- `src/runtime/query_cypher.py`: query-to-Cypher entry point
- `src/runtime/query_planner.py`: local query-plan schema and deterministic compiler
- `src/runtime/query_stack.py`: loader for the published local query bundle
- `src/runtime/local_query_stack.py`: local router/planner orchestration
- `src/runtime/cypher_validation.py`: read-only Cypher checks and Neo4j URI normalization
- `src/runtime/model_provider.py`: provider, model, and API-mode resolution
- `src/runtime/entity_resolver.py`: light post-extraction surface-form normalization
- `src/runtime/output_layout.py`: staging, latest, runs, and failed output management
- `src/runtime/neo4j_load.py`: CLI for loading saved runs into Neo4j
- `src/runtime/neo4j_status.py`: CLI for reporting Neo4j vs output status
- `src/runtime/neo4j_admin.py`: CLI for company/full Neo4j unload operations
- `src/runtime/health_check.py`: repo, query-stack, and Neo4j readiness checks
- `src/llm/extractor.py`: model transport, retries, JSON recovery, parsing, and auditing
- `src/llm_extraction/models.py`: Pydantic models for triples, pipeline results, and memos
- `src/llm_extraction/prompting.py`: prompt loading and rendering helpers
- `src/llm_extraction/pipelines/`: pipeline-specific orchestration and prompt selection
- `src/ontology/ontology.json`: machine-readable ontology definition
- `src/ontology/config.py`: ontology loader and canonical label access
- `src/ontology/validator.py`: ontology validation, dedupe, and structural checks
- `src/ontology/place_hierarchy.py`: place normalization and geographic rollup helpers
- `src/graph/neo4j_loader.py`: Neo4j load, replace, and unload logic

The maintained import surface is package-based:

- `runtime.*`
- `graph.*`
- `ontology.*`
- `llm.*`
- `llm_extraction.*`

## Pipelines And Prompts

Editable prompt assets:

```text
prompts/
  analyst/
  memo_graph_only/
  zero-shot/
```

Bundled package fallback prompts:

```text
src/llm_extraction/_bundled_prompts/
  analyst/
  memo_graph_only/
  zero-shot/
```

Pipeline implementations:

```text
src/llm_extraction/pipelines/
  analyst/
  memo_graph_only/
  zero_shot/
```

Prompt loading order is:

1. `KG_PROMPTS_DIR`
2. repo `prompts/`
3. bundled package prompts

## Runtime Assets And Outputs

Runtime query-stack bundle:

```text
runtime_assets/
  query_stack/
    manifest.json
    router/
    planner/
```

Saved extraction outputs:

```text
outputs/
  <company>/
    analyst/
    memo_graph_only/
    zero-shot/
```

Both `runtime_assets/query_stack/` and `outputs/` are ignored by Git because
they are local runtime artifacts.

## Data

Input filing text files live at the top of `data/`, for example:

```text
data/
  microsoft_10k.txt
  apple_10k.txt
  google_10k.txt
```

The curated query-planner datasets live under:

```text
data/query_planner_curated/
  v1_baseline/
  v1_final/
```

The final curated dataset for the local query planner is:

```text
data/query_planner_curated/v1_final/
```

This dataset is preserved for fine-tuning and reproducibility. The main runtime
does not expose a dataset-construction CLI surface.

Training and export work should consume `supervision_target`, which makes
`local_safe`, `strong_model_candidate`, and `refuse` explicit in the serialized
supervision object.

## Evaluation

```text
evaluation/
  README.md
  benchmarks/
    dev/clean/
    test/clean/
  scripts/
  results/
```

The benchmark stores one JSONL file per company under `clean/`. Each row is a
typed triple with:

```text
subject; subject_type; relation; object; object_type
```

Evaluation compares gold triples against:

```text
outputs/<company>/<pipeline>/latest/resolved_triples.json
```

See [`evaluation.md`](./evaluation.md) and
[`../evaluation/README.md`](../evaluation/README.md).

## Fine-Tuning Island

```text
finetuning/
  README.md
  pyproject.toml
  config/
  scripts/
  src/kg_query_planner_ft/
  tests/
  artifacts/
```

This directory is intentionally separate from the main KG/runtime codebase.
The main repo does not import from the fine-tuning island. The only handoff back
to the main runtime is the published query-stack bundle under:

```text
runtime_assets/query_stack/
```

See [`../finetuning/README.md`](../finetuning/README.md).

## Scripts

```text
scripts/
  _run_repo_module.sh
  bootstrap_dev.sh
  check_repo.sh
  clean_local_artifacts.sh
  kg-pipeline
  kg-query
  kg-query-cypher
  kg-neo4j-load
  kg-neo4j-status
  kg-neo4j-unload
  kg-health-check
  sync_bundled_prompts.py
```

The wrapper scripts run source modules through the repo virtual environment.
They are the most reliable commands when working directly from a checkout.

## Tests

```text
tests/
  test_evaluation/
  test_graph/
  test_llm/
  test_ontology/
  test_runtime/
```

Fine-tuning has its own tests under:

```text
finetuning/tests/
```
