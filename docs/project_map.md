# Project Map

This map is for orientation. It explains what each major file and directory is
for, without going deep into implementation details.

## Top-Level Files

- `README.md`: the front door to the project. It explains the purpose,
  high-level workflow, public artifacts, and where to go next.
- `pyproject.toml`: packaging metadata and dependencies for the main
  extraction/runtime project.
- `docker-compose.yml`: local Neo4j service definition for loading and querying
  saved graphs.
- `.env.example`: optional environment-variable template for local model,
  hosted provider, prompt, query-stack, and Neo4j settings.
- `.gitignore`: records which local artifacts are intentionally not tracked,
  such as generated outputs, model bundles, virtual environments, and caches.

## Documentation

- `docs/project_walkthrough.md`: plain-language explanation of how the project
  works.
- `docs/reproducibility.md`: canonical reviewer guide for artifact setup and
  the three reproducibility islands.
- `docs/runtime_guide.md`: command reference for extraction, providers, Neo4j,
  querying, output layout, prompts, cleanup, and checks.
- `docs/evaluation.md`: evaluation method, public evaluation artifacts,
  metrics, and reproduction commands.
- `docs/ontology.md`: graph schema, modeling rules, canonical labels,
  validation behavior, and geography policy.
- `docs/project_map.md`: this orientation map.

## Data And Artifacts

- `data/`: checked-in example 10-K business-section text files. The ignored
  `data/query_planner_curated/` subfolder is used only by fine-tuning and can be
  downloaded from Hugging Face.
- `outputs/`: ignored local extraction outputs. A run writes resolved triples,
  validation reports, summaries, and intermediate pipeline artifacts here.
- `runtime_assets/`: local runtime bundles that are too large or specific to
  track in Git. The main expected bundle is `runtime_assets/query_stack/`.
- `evaluation/results/`: checked-in and regenerated evaluation reports.
- `finetuning/artifacts/`: ignored fine-tuning outputs such as prepared data,
  router models, planner adapters, and evaluation summaries.

## Main Source Packages

- `src/runtime/`: command-line entry points and operational logic. This is where
  extraction runs are launched, outputs are organized, Neo4j is loaded, query
  commands are handled, and health checks live.
- `src/llm_extraction/`: extraction pipeline orchestration, prompt loading, and
  shared graph/memo models.
- `src/llm/`: model-call helpers, retry behavior, JSON parsing, and audit
  support used by extraction.
- `src/ontology/`: machine-readable ontology, canonical labels, place handling,
  and validation rules.
- `src/graph/`: Neo4j loading and graph maintenance helpers.
- `src/training/`: reserved package surface for training-related compatibility.

## Prompts

- `prompts/`: human-edited prompt assets used during development.
- `src/llm_extraction/_bundled_prompts/`: packaged fallback copy of the prompts
  so installed builds can still run without the editable `prompts/` folder.
- `scripts/sync_bundled_prompts.py`: helper that syncs edited prompts into the
  bundled prompt copy before packaging.

## Evaluation Island

- `evaluation/benchmarks/`: curated benchmark triples and annotation-reliability
  inputs.
- `evaluation/scripts/evaluate.py`: evaluator used to compare saved extraction
  outputs against benchmark triples.
- `evaluation/README.md`: folder-local orientation. The full guide is
  `docs/evaluation.md`.

## Fine-Tuning Island

- `finetuning/README.md`: guide for the isolated query router/planner
  fine-tuning workflow.
- `finetuning/pyproject.toml`: separate package metadata and dependencies for
  fine-tuning.
- `finetuning/config/`: JSON configs for training and resume variants.
- `finetuning/scripts/bootstrap_env.sh`: creates the isolated fine-tuning
  environment.
- `finetuning/scripts/download_query_planner_data.sh`: downloads the public
  query-planner dataset into the expected local layout.
- `finetuning/src/kg_query_planner_ft/`: fine-tuning package code for preparing
  data, training/evaluating the router, training/evaluating the planner, and
  publishing the runtime query-stack bundle.
- `finetuning/tests/`: tests for the fine-tuning island.

## Scripts

- `scripts/bootstrap_dev.sh`: creates the main project virtual environment and
  installs development dependencies.
- `scripts/check_repo.sh`: full maintainer check covering tests, compilation,
  command wrappers, and package smoke installs.
- `scripts/clean_local_artifacts.sh`: removes caches and scratch artifacts
  without removing saved extraction outputs.
- `scripts/kg-pipeline`: source-checkout wrapper for extraction runs.
- `scripts/kg-query`: source-checkout wrapper for natural-language querying
  against Neo4j.
- `scripts/kg-query-cypher`: source-checkout wrapper for rendering read-only
  Cypher from natural language.
- `scripts/kg-neo4j-load`: loads saved outputs into Neo4j.
- `scripts/kg-neo4j-status`: compares local saved outputs with the live Neo4j
  graph.
- `scripts/kg-neo4j-unload`: removes graph data from Neo4j.
- `scripts/kg-health-check`: checks local readiness for the repo, prompts,
  ontology, outputs, query stack, and optional Neo4j service.

## Tests

- `tests/`: main project tests for runtime behavior, ontology validation, graph
  loading, LLM extraction helpers, and evaluation logic.
- `finetuning/tests/`: isolated tests for fine-tuning data preparation,
  training/evaluation helpers, publishing, and path handling.
