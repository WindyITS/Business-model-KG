# Business Model KG

A local pipeline for turning SEC 10-K business sections into a standardized business-model knowledge graph.

The repo focuses on one maintained surface:
- the extraction and runtime stack for building, validating, and querying the business-model graph

## What The Pipeline Extracts

The graph is built around:
- `Company`
- `BusinessSegment`
- `Offering`

with canonical closed labels for:
- `CustomerType`
- `Channel`
- `RevenueModel`

The ontology is segment-centered:
- structure lives on `BusinessSegment -> OFFERS -> Offering`
- `SERVES` lives on `BusinessSegment`
- `SELLS_THROUGH` is segment-first
- `MONETIZES_VIA` lives on `Offering`
- `OPERATES_IN` and `PARTNERS_WITH` stay company-level

When loading multiple companies into Neo4j, `BusinessSegment` and `Offering` nodes are scoped by
`company_name` so same-named nodes from different companies do not collapse into one shared node.
This keeps canonical shared vocabularies such as `Channel`, `CustomerType`, `RevenueModel`, and
`Place` global, while keeping company-owned inventory nodes distinct.

Full ontology spec: [`docs/ontology.md`](./docs/ontology.md)

## Pipeline Philosophy

The maintained extraction runtimes follow a few consistent rules:

- scope-first modeling: `Company` is the corporate shell, `BusinessSegment` is the primary semantic anchor, and `Offering` is the inventory layer
- canonical extraction over convenience duplication: the extractor does not materialize inherited or rollup facts just to make the graph denser
- closed semantic vocabularies: `CustomerType`, `Channel`, and `RevenueModel` must map to the approved canonical labels or be omitted
- precision-first semantic extraction: `SERVES`, `SELLS_THROUGH`, `MONETIZES_VIA`, and `PARTNERS_WITH` favor standardization and explicit support over aggressive recall
- broader but still text-grounded company geography capture: `OPERATES_IN` is more recall-friendly than the semantic business-model relations, but still constrained to meaningful company presence
- pipeline-specific control flow: the analyst runtime is memo-first and staged, while the zero-shot runtime is a single-pass baseline

This means the effective behavior of the maintained pipelines comes from three layers together:
- the formal schema in [`src/ontology/ontology.json`](./src/ontology/ontology.json)
- the extraction pipeline implementations under [`src/llm/`](./src/llm/) and [`src/llm_extraction/pipelines/`](./src/llm_extraction/pipelines/) with prompt assets in [`prompts/analyst/`](./prompts/analyst/) and [`prompts/zero-shot/`](./prompts/zero-shot/)
- the final normalization and structural enforcement in [`src/ontology/validator.py`](./src/ontology/validator.py)

## Extraction Pipelines

The repo ships two supported extraction pipelines:

- `analyst`: a memo-first runtime that builds a structured analyst memo from the full filing, then compiles that memo into the ontology graph and runs a short overreach critique pass
- `zero-shot`: a single-pass baseline that emits the ontology graph directly from the filing

### Analyst Extraction Pipeline

High-level flow:

1. Read the filing text, infer `company_name` from the input filename, and write `chunks.json`
2. `Analyst memo 1`: build the foundational business-model memo as structured plain text
3. `Analyst memo 2`: augment that memo with additional defensible detail, still as structured plain text
4. `Graph compilation`: convert the memo into ontology-valid triples
5. `Critique`: prune overreach and overly neat structure
6. Resolve surface forms, revalidate the final graph, and write final artifacts
7. Optionally load the graph into Neo4j

Runtime notes:
- the memo is a first-class structured plain-text artifact, not just hidden reasoning
- the memo explicitly separates filing support, analyst inference, and uncertainty
- the analyst runtime treats the ontology as the target graph structure, not as a literal paragraph-extraction cage

### Zero-Shot Extraction Pipeline

High-level flow:

1. Read the filing text, infer `company_name` from the input filename, and write `chunks.json`
2. `Zero-shot extraction`: build the full ontology graph in one pass
3. Resolve surface forms, revalidate the final graph, and write final artifacts
4. Optionally load the graph into Neo4j

Runtime notes:
- this path is the leanest extraction baseline in the repo
- the single-pass output is still audited, resolved, validated, and optionally loaded with the same runtime tooling as the analyst pipeline
- the final CLI validation does not require strict text grounding by default

## Query Interface

The runtime includes a read-only natural-language query path for the live Neo4j graph:

- `kg-query-cypher` and `kg-query` use the routed stack: a published local query-stack bundle first, with hosted free-form Cypher fallback when needed
- `kg-query` and `kg-query-cypher` also accept `--stack routed|fallback` so one command can either try the local stack first or force hosted fallback only

The local planner and hosted query prompts live in [`src/runtime/query_prompt.py`](./src/runtime/query_prompt.py), the deterministic compiler used by the local stack lives in [`src/runtime/query_planner.py`](./src/runtime/query_planner.py), and the read-only Cypher guards live in [`src/runtime/cypher_validation.py`](./src/runtime/cypher_validation.py).
The deployed local query runtime reads a published bundle from `runtime_assets/query_stack/current/` by default; override that with `--local-stack-bundle-dir` or `KG_QUERY_STACK_BUNDLE_DIR` when needed.

The final curated fine-tuning dataset for the query planner is preserved under [`data/query_planner_curated/v1_final`](./data/query_planner_curated/v1_final/). The repo intentionally does not keep a dataset-construction CLI surface.

For Neo4j maintenance, the repo also ships:

- `kg-neo4j-load` to load saved outputs into Neo4j, either in bulk or for one company/run
- `kg-neo4j-status` to show which companies are loaded in Neo4j and which saved outputs are available
- `kg-neo4j-unload` to unload Neo4j graph data (full dataset by default, or one company with `--company`)

## Repo Layout

```text
kg-v0/
  README.md                                  - main overview, setup, commands, and repo conventions
  docs/
    ontology.md                              - human-readable ontology specification
    project_walkthrough.md                   - plain-language architecture walkthrough
  data/
    query_planner_curated/                   - curated dataset for local query-stack finetuning
      v1_baseline/                           - earlier curated dataset release
      v1_final/                              - final curated dataset used by finetuning configs
  prompts/
    analyst/                                 - prompt assets for the memo-first extraction pipeline
    zero-shot/                               - prompt assets for the direct extraction pipeline
  runtime_assets/
    query_stack/
      current/                               - published local router/planner bundle used by runtime
  outputs/
    apple/                                   - saved extraction runs for Apple
    google/                                  - saved extraction runs for Google
    microsoft/                               - saved extraction runs for Microsoft
    palantir/                                - saved extraction runs for Palantir
  scripts/
    _run_repo_module.sh                      - shared launcher for running repo Python modules
    bootstrap_dev.sh                         - creates or refreshes the main repo virtualenv
    check_repo.sh                            - maintainer smoke-check script
    clean_local_artifacts.sh                 - removes caches and build noise without touching outputs
    kg-pipeline                              - wrapper for the extraction runtime
    kg-query                                 - wrapper for live natural-language querying
    kg-query-cypher                          - wrapper for query-to-Cypher rendering
    kg-neo4j-load                            - wrapper for loading saved outputs into Neo4j
    kg-neo4j-status                          - wrapper for comparing Neo4j vs saved outputs
    kg-neo4j-unload                          - wrapper for unloading Neo4j graph data
    kg-health-check                          - wrapper for local readiness checks
    sync_bundled_prompts.py                  - copies editable prompts into packaged bundled prompts
  src/
    graph/
      neo4j_loader.py                        - Neo4j load/replace/unload logic
    llm/
      extractor.py                           - generic LLM transport, retries, parsing, and auditing
    llm_extraction/
      models.py                              - Pydantic models for triples, pipeline results, and memos
      prompting.py                           - prompt loading and rendering helpers
      _bundled_prompts/                      - packaged fallback copy of prompt assets
        analyst/                             - bundled analyst prompts
        zero-shot/                           - bundled zero-shot prompts
      pipelines/
        analyst/                             - memo-first extraction runner and stages
        zero_shot/                           - single-pass extraction runner
    ontology/
      config.py                              - ontology loader and canonical label access
      ontology.json                          - machine-readable ontology definition
      place_hierarchy.py                     - place normalization and geographic rollup helpers
      validator.py                           - ontology validation, dedupe, and structural checks
    runtime/
      main.py                                - top-level extraction CLI runtime (`kg-pipeline`)
      query.py                               - top-level query runtime (`kg-query`)
      query_cypher.py                        - query-to-Cypher CLI entrypoint
      query_planner.py                       - local query-plan schema and deterministic compiler
      query_prompt.py                        - local and hosted query prompt contracts
      query_stack.py                         - loader for the published local query bundle
      local_query_stack.py                   - local router/planner orchestration
      cypher_validation.py                   - read-only Cypher checks and Neo4j URI normalization
      model_provider.py                      - provider/model/API-mode resolution
      entity_resolver.py                     - light post-extraction surface-form normalization
      output_layout.py                       - staging/latest/runs/failed output management
      neo4j_load.py                          - CLI for loading saved runs into Neo4j
      neo4j_status.py                        - CLI for reporting Neo4j vs output status
      neo4j_admin.py                         - CLI for company/full Neo4j unload operations
      health_check.py                        - repo/query-stack/Neo4j readiness checks
  tests/
    test_graph/                              - tests for graph and Neo4j utilities
    test_llm/                                - tests for LLM transport and parsing logic
    test_ontology/                           - tests for ontology and validation behavior
    test_runtime/                            - tests for CLI/runtime/query/output behavior
  finetuning/
    README.md                                - overview of the isolated finetuning workflow
    config/                                  - JSON configs for finetuning runs
    scripts/
      bootstrap_env.sh                       - creates the dedicated finetuning environment
    src/
      kg_query_planner_ft/
        cli_output.py                        - human-readable CLI summaries for finetuning commands
        config.py                            - finetuning config schema and loader
        constants.py                         - route labels and shared constants
        frozen_prompt.py                     - frozen planner prompt used during training and eval
        paths.py                             - path resolution for datasets and artifacts
        prepare_data.py                      - turns curated data into router/planner training splits
        router_train.py                      - trains the DeBERTa router classifier
        router_eval.py                       - calibrates and evaluates router, then writes thresholds
        planner_train.py                     - trains the local planner with MLX QLoRA
        planner_eval.py                      - evaluates planner JSON, contract, family, and exact-match quality
        planner_worker.py                    - generation backends for planner evaluation
        publish_query_stack.py               - publishes trained artifacts into the runtime bundle
        offline_contract.py                  - training-side validation of planner JSON shape
    tests/                                   - tests for the finetuning island
    artifacts/                               - local finetuning outputs, checkpoints, and eval reports
```

The maintained import surface is package-based: `runtime.*`, `graph.*`, `ontology.*`, and `llm.*`.
The old top-level `src/*.py` compatibility shims have been removed; new code should import the
package modules directly.

Pipeline structure notes:
- prompt files are edited under `prompts/`; packaged installs carry a bundled fallback copy under `src/llm_extraction/_bundled_prompts/`
- prompt loading order is: `KG_PROMPTS_DIR` override, then repo `prompts/`, then bundled packaged prompts
- [`src/llm/extractor.py`](./src/llm/extractor.py) handles transport, retries, JSON recovery, and parsing only
- pipeline-specific orchestration and prompt selection live under [`src/llm_extraction/pipelines/`](./src/llm_extraction/pipelines/)

## Quickstart

Requirements:
- Python 3.10+
- an OpenAI-compatible local endpoint such as LM Studio
- optionally Neo4j
- optionally an OpenCode Go API key if you want to run against hosted models instead of a local endpoint

Setup:

Recommended one-command setup from a cloned repo:

```bash
./scripts/bootstrap_dev.sh
```

Manual equivalent:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

The `dev` extra installs the test tooling used by `./scripts/check_repo.sh`. If you only want
the runtime CLI surface and not the maintainer checks, `pip install -e .` is still enough.
If you also want the main runtime to execute the published local query stack, install the optional runtime extras with `pip install -e ".[query-stack]"`.

That editable install creates the convenience commands in `venv/bin/`:
- `kg-pipeline`
- `kg-query`
- `kg-query-cypher`
- `kg-neo4j-load`
- `kg-neo4j-status`
- `kg-neo4j-unload`
- `kg-health-check`

If you already had the virtual environment before a new CLI command was added to the repo, rerun:

```bash
pip install -e .
```

That refreshes the `venv/bin/` entry points so the new commands appear there too.

If you want a template for local environment variables and defaults, copy from `.env.example`.

For day-to-day work from a source checkout, the most reliable commands are the wrapper scripts under `scripts/`:

- `./scripts/kg-pipeline`
- `./scripts/kg-query`
- `./scripts/kg-query-cypher`
- `./scripts/kg-neo4j-load`
- `./scripts/kg-neo4j-status`
- `./scripts/kg-neo4j-unload`
- `./scripts/kg-health-check`

These wrappers run the repo source directly with the repo virtual environment, so they still work even if the editable-install entry points have not been refreshed yet.
For direct module execution, use package entrypoints such as `python -m runtime.main`,
`python -m runtime.query`, and `python -m runtime.query_cypher`.
The retained query-planner training artifact lives under `data/query_planner_curated/v1_final/`. Training/export work should consume `supervision_target`, which makes `local_safe`, `strong_model_candidate`, and `refuse` explicit in the serialized supervision object.

For a plain-language overview of how the project fits together, see [docs/project_walkthrough.md](./docs/project_walkthrough.md).

To remove local caches and scratch artifacts without touching saved outputs, run:

```bash
./scripts/clean_local_artifacts.sh
```

To run the main repo safety checks end to end, run:

```bash
bash ./scripts/check_repo.sh
```

Prompt workflow notes:
- an editable install keeps using the repo-level `prompts/` directory, so prompt iteration stays fast
- a standard package install also works because the package now ships a bundled prompt fallback
- if you want to test an alternate prompt set without editing the repo copy, set `KG_PROMPTS_DIR=/path/to/prompts`
- if a reflection stage returns an empty graph or exhausts retries, an interactive terminal now asks whether to keep the last good graph from the current run; non-interactive runs keep it automatically and print that choice in the console output

Run the extraction pipeline:

```bash
./scripts/kg-pipeline data/microsoft_10k.txt --skip-neo4j
```

Optional explicit `zero-shot` pipeline flag:

```bash
./scripts/kg-pipeline data/microsoft_10k.txt --pipeline zero-shot --skip-neo4j
```

Render a Cypher query without executing it:

```bash
./scripts/kg-query-cypher "Which company segments sell through marketplaces?"
```

Run a query against Neo4j:

```bash
./scripts/kg-query "Which company segments sell through marketplaces?" \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password password
```

Force hosted fallback only (skip local router/local planner) with one switch:

```bash
./scripts/kg-query-cypher "Which company segments sell through marketplaces?" --stack fallback
```

Override the published local query-stack bundle directory:

```bash
./scripts/kg-query-cypher "Which company segments sell through marketplaces?" \
  --local-stack-bundle-dir /path/to/runtime_assets/query_stack/current
```

Run against OpenCode Go with a hosted open model:

```bash
export OPENCODE_GO_API_KEY=your_key_here
./scripts/kg-pipeline data/microsoft_10k.txt \
  --provider opencode-go \
  --model kimi-k2.5 \
  --skip-neo4j
```

Other supported OpenCode Go models use the same CLI shape:

```bash
./scripts/kg-pipeline data/microsoft_10k.txt --provider opencode-go --model mimo-v2-pro --skip-neo4j
./scripts/kg-pipeline data/microsoft_10k.txt --provider opencode-go --model minimax-m2.7 --skip-neo4j
```

Provider notes:
- `local` defaults to `http://localhost:1234/v1` and `local-model`
- `opencode-go` defaults to `https://opencode.ai/zen/go/v1` and `kimi-k2.5`
- `opencode-go` currently supports `kimi-k2.5`, `mimo-v2-pro`, and `minimax-m2.7` in this repo
- `kimi-k2.5` and `mimo-v2-pro` use OpenCode Go's `chat/completions` endpoint; `minimax-m2.7` is routed automatically to OpenCode Go's Anthropic-compatible `messages` endpoint
- the CLI also accepts friendly OpenCode Go model names such as `MiniMax M2.7`, plus prefixed IDs like `opencode-go/kimi-k2.5`
- the CLI accepts either a root base URL or a full documented endpoint like `.../chat/completions` and normalizes it automatically
- `local` reads `--api-key` first, then `LOCAL_LLM_API_KEY`, then `LM_STUDIO_API_KEY`, and otherwise falls back to `lm-studio`
- `opencode-go` reads `--api-key` first, then `OPENCODE_GO_API_KEY`, then `OPENCODE_API_KEY`
- for `opencode-go`, the runtime rewrites `system` messages to `user` messages for compatibility while keeping the rest of the pipeline flow unchanged
- `opencode-go` defaults to `--max-output-tokens 20000`; override it if needed
- the CLI exposes both the `analyst` and `zero-shot` pipelines
- every run writes `run_summary.json`; the console header shows pipeline, provider, and model, and LLM attempt summaries show token counts when available
- successful Neo4j loads now replace the previous graph footprint for that same company by default; use `kg-neo4j-unload --yes` when you intentionally want to wipe the full database first

Useful CLI flags:
- `--company-name`: override the inferred company name used for output folders and company-scoped Neo4j operations
- `--max-retries`: change the retry budget per LLM call
- `--base-url`: pass either an API root or a full endpoint URL; the runtime normalizes common suffixes
- `--api-key`: override environment-based key resolution
- `--max-output-tokens`: explicitly cap model output tokens
- `kg-query` / `kg-query-cypher --stack fallback`: single-switch way to skip the local router/planner and force hosted fallback query generation
- `kg-query` / `kg-query-cypher --local-stack-bundle-dir /path/to/query_stack/current`: point routed query commands to a specific published bundle
- `kg-query` / `kg-query-cypher --provider opencode-go`: choose the hosted fallback query-generation provider used only when local routing does not return a local-safe plan
- when the local query stack errors in routed mode, the CLI logs the problem and falls back automatically, even in non-interactive shells
- hosted fallback query generation retries once with error context; if both attempts fail, the CLI prints a warning
- `--keep-current-output`: keep the current `latest/` output untouched and store this successful run under `runs/` instead; requires `--skip-neo4j`
- `kg-neo4j-load`: load saved outputs into Neo4j; defaults to all `analyst/latest` outputs under `outputs/`
- `kg-neo4j-status`: report which companies are loaded in Neo4j and whether local latest outputs exist for them
- `kg-neo4j-unload`: unload the full Neo4j dataset
- `kg-neo4j-unload --company "Name"`: remove one company's loaded graph footprint while keeping other companies intact
- `kg-health-check`: inspect the local repo setup, saved outputs, and optional Neo4j connectivity

Load into Neo4j instead:

```bash
docker compose up -d
./scripts/kg-pipeline data/microsoft_10k.txt
```

Load the latest `analyst` outputs for every company into Neo4j:

```bash
./scripts/kg-neo4j-load
```

Load the latest `analyst` output for one company only:

```bash
./scripts/kg-neo4j-load --company "Microsoft"
```

Load one exact saved run for a company:

```bash
./scripts/kg-neo4j-load --company "Microsoft" --run 20260417T101500Z
```

You can also pass a relative path inside that company's pipeline folder, but `--run` is intentionally limited to that folder. It will not jump to another company or to an arbitrary filesystem path.

Use `--pipeline zero-shot` if you want the command to target zero-shot outputs instead of the default analyst outputs.
Use `--yes` to skip the bulk-load warning when Neo4j already contains data.
If you target one company with `--company` and that company is already loaded, the command now asks for confirmation before replacing that company graph.

Show which companies are currently loaded in Neo4j and which local outputs are ready to load:

```bash
./scripts/kg-neo4j-status
```

`kg-neo4j-status` is read-only: it reports the current state but does not rewrite outputs or load/unload anything.

Run a local environment health check:

```bash
./scripts/kg-health-check
```

If you only want repo checks and do not want to probe Neo4j, use:

```bash
./scripts/kg-health-check --skip-neo4j
```

Unload the full Neo4j dataset:

```bash
./scripts/kg-neo4j-unload
```

Unload only one company's graph footprint from Neo4j:

```bash
./scripts/kg-neo4j-unload --company "Microsoft"
```

If you want to skip the confirmation prompt in automation or scripts, add `--yes`.

## Neo4j Command Guide

These are the main commands that touch Neo4j, and they are meant for different moments in the workflow:

- `kg-pipeline <file>`: runs a new extraction. By default, if the run succeeds, it replaces that same company's currently loaded graph in Neo4j unless you add `--skip-neo4j`.
- `kg-neo4j-load`: loads saved outputs from `outputs/` into Neo4j. With no extra flags it targets every `outputs/<company>/analyst/latest/` directory, warns before bulk-loading into a non-empty database, and keeps going if one company fails while reporting the failures at the end.
- `kg-neo4j-load --company "Name"`: loads only that company's latest saved output for the selected pipeline. If that company is already loaded in Neo4j, the command asks before replacing it unless you pass `--yes`.
- `kg-neo4j-load --company "Name" --run <token>`: loads one exact saved run from `runs/<token>` or `failed/<token>`, or a relative path inside that company's pipeline folder, instead of the latest output. This is mainly for debugging, testing, or comparing older runs.
- `kg-neo4j-status`: compares Neo4j against the saved outputs for the selected pipeline. It tells you which companies are loaded, which are not, and, for the not-loaded ones, whether a latest output is ready to load. It is a reporting command only and does not modify outputs or Neo4j.
- `kg-neo4j-unload`: clears the full Neo4j dataset. It asks for confirmation unless you pass `--yes`.
- `kg-neo4j-unload --company "Name"`: removes only that company's graph footprint from Neo4j and leaves unrelated companies in place. It asks for confirmation unless you pass `--yes`.
- `kg-health-check`: checks whether the local repo setup looks usable. It reports Python, the repo venv, `.env.example`, the published query-stack bundle, prompt assets, ontology assets, saved outputs, and optionally Neo4j connectivity.

The most useful flags in practice are:

- `--company-name` on `kg-pipeline` when the filename is not the company identity you want to use for outputs and Neo4j replacement.
- `--pipeline analyst|zero-shot` on `kg-pipeline`, `kg-neo4j-load`, and `kg-neo4j-status` when you want to switch output families.
- `--keep-current-output --skip-neo4j` on `kg-pipeline` when you want to save a test run under `runs/` without replacing the current `latest/` output or the live Neo4j graph.
- `--yes` on `kg-neo4j-load` or `kg-neo4j-unload` when the command is running in automation or a non-interactive script.
- `--skip-neo4j` on `kg-health-check` when you want setup/output checks without treating a stopped Neo4j instance as part of the current task.

Neo4j Browser:
- `http://localhost:7474`
- default credentials: `neo4j / password`

Generate a read-only Cypher query from a natural-language question:

```bash
./scripts/kg-query-cypher "Which companies sell to developers through direct sales?"
```

Generate the query and run it against the current Neo4j database:

```bash
./scripts/kg-query "Which companies sell to developers through direct sales?"
```

These routed commands try the published local query-stack bundle first and fall back to hosted free-form Cypher generation automatically when the local stack is unavailable or declines to handle the request. The local stack stays compiler-based; the hosted fallback writes full Cypher JSON directly, still behind the read-only guards and Neo4j `EXPLAIN` preflight. Use `--stack fallback` when you want to bypass the local stack entirely.

`kg-query` returns rows from the live database as plain text, while `kg-query-cypher` returns a runnable
plain-text Cypher query with generated params already inlined. Progress and error messages are printed to
stderr so stdout stays easy to pipe or copy.

## Neo4j Node Scoping

When the loader writes to Neo4j:
- `Company`, `Channel`, `CustomerType`, `RevenueModel`, and `Place` remain globally keyed by `name`
- `BusinessSegment` and `Offering` are keyed by `(company_name, name)`

This prevents cross-company collisions such as two different companies each having an offering named
`Advertising`.

Example checks:

```cypher
MATCH (n)
WHERE n.company_name = "Apple" AND (n:BusinessSegment OR n:Offering)
RETURN labels(n)[0] AS label, n.name AS name, n.company_name AS company_name
ORDER BY label, name
LIMIT 100
```

```cypher
MATCH (o:Offering {name: "Advertising"})
RETURN o.name AS offering, o.company_name AS company_name
ORDER BY company_name
```

If your database already contains older unscoped `BusinessSegment` or `Offering` nodes from a
previous loader version, clear and reload the graph once so the scoped identity takes effect
consistently.

Unload modes:
- `kg-neo4j-unload` with no `--company` clears the full dataset
- `kg-neo4j-unload --company "Name"` is company-level, not run-level
- company unload removes the chosen company's scoped `BusinessSegment` and `Offering` nodes, its outgoing company-level graph links, and any shared nodes that become orphaned because of that unload
- company unload does not clear unrelated companies or act as an exact rollback for one historical run

## Geography

The extractor and Neo4j loader keep geography canonical at
`Company-[:OPERATES_IN]->Place`.

No derived place hierarchy relationships are materialized in Neo4j.

Instead, each extracted `Place` node can receive two query helper properties during load:
- `within_places`: broader canonical places that contain the place
- `includes_places`: narrower canonical places that the place contains

Examples:
- `Italy` can carry `within_places = ["Europe", "Western Europe", "EMEA", "European Union"]`
- `Europe` can carry `includes_places = ["Western Europe", "Eastern Europe", "Italy", "Germany", ...]`
- `United States` can carry `includes_places = ["Alabama", "Alaska", ..., "Wyoming"]`

This keeps the graph canonical while still letting Neo4j answer both directions of
geography matching without extra `WITHIN` edges.

Recommended Cypher pattern:

```cypher
MATCH (company:Company)-[:OPERATES_IN]->(place:Place)
WITH company, place.name AS matched_place,
     CASE
       WHEN matched_place = $place THEN 0
       WHEN $place IN coalesce(place.includes_places, []) THEN 1
       WHEN $place IN coalesce(place.within_places, []) THEN 2
       ELSE NULL
     END AS match_rank
WHERE match_rank IS NOT NULL
WITH company, MIN(match_rank) AS best_rank, collect(DISTINCT matched_place) AS matched_places
RETURN company.name AS company,
       CASE best_rank
         WHEN 0 THEN 'exact'
         WHEN 1 THEN 'narrower_place'
         ELSE 'broader_region'
       END AS geography_match,
       matched_places
ORDER BY best_rank, company
```

A company can match through more than one direct place tag. For example, a company tagged
to both `Europe` and `European Union` can match a query for `Italy` through both tags.
The `MIN(match_rank)` aggregation above collapses those into one row per company while
keeping the strongest match class.

## Output Artifacts

Outputs are now organized by company and pipeline:

```text
outputs/
  microsoft/
    analyst/
      latest/
        ...
      manifest.json
    zero-shot/
      latest/
        ...
      manifest.json
```

Default behavior:
- a successful run stages artifacts first, then replaces `latest/` only after the run succeeds
- the previous `latest/` output for that company and pipeline is removed on successful replacement
- failed runs do not overwrite `latest/`; their artifacts are moved into `failed/`
- `--keep-current-output --skip-neo4j` keeps the current `latest/` untouched and stores the successful run under `runs/` for testing or comparison
- each company/pipeline folder also keeps a `manifest.json` summary so the Neo4j helper commands can report what latest output, archived runs, and failed runs exist

The output folder name comes from the canonical company identity used for the run:
- by default this is inferred from the input filename
- you can override it with `--company-name`

Saved-output Neo4j load notes:
- `kg-neo4j-load` defaults to the `analyst/latest` outputs because those are the preferred saved outputs for reload
- with no `--company`, it loads every available `latest/` directory for the chosen pipeline
- with `--company`, it loads that company's `latest/` directory for the chosen pipeline
- with `--company --run <token>`, it loads an exact saved run under `runs/<token>` or `failed/<token>` for that company and pipeline, or a relative path inside that same company/pipeline folder
- the bulk `kg-neo4j-load` command warns before running if Neo4j already contains data
- if a bulk load hits a problem for one company, it keeps going for the others and reports which companies failed
- if a single-company load sees that the company is already present in Neo4j, it asks before replacing that company graph unless you pass `--yes`
- company replacement is transactional, so if the new load fails the previous live graph for that company is left untouched
- `kg-neo4j-status` compares Neo4j against the saved outputs and tells you which companies are loaded, which are not, and whether a latest output is available to load; it does not rewrite manifests or change the graph

Operational safety notes:
- `./scripts/kg-health-check` is the quick “is my local setup ready?” command
- `bash ./scripts/check_repo.sh` is the deeper maintainer check that runs tests, compile checks, wrapper smoke checks, and a package-install smoke test
- `.github/workflows/checks.yml` runs the same repo-check script on pushes and pull requests

Analyst pipeline runs write a different mix centered on:
- `analyst_memo_foundation.md`
- `analyst_memo_augmented.md`
- `analyst_graph_compilation.json`
- `analyst_graph_critique.json`
- `run_summary.json`
- `chunks.json`
- `resolved_triples.json`
- `validation_report.json`

Zero-shot pipeline runs write a smaller graph-only set centered on:
- `zero_shot_extraction.json`
- `run_summary.json`
- `chunks.json`
- `resolved_triples.json`
- `validation_report.json`

## Tests

Extraction/runtime tests:

```bash
./venv/bin/python -m unittest discover -s tests
```
