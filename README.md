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

The literal runtime follows a few consistent rules:

- scope-first modeling: `Company` is the corporate shell, `BusinessSegment` is the primary semantic anchor, and `Offering` is the inventory layer
- canonical extraction over convenience duplication: the extractor does not materialize inherited or rollup facts just to make the graph denser
- closed semantic vocabularies: `CustomerType`, `Channel`, and `RevenueModel` must map to the approved canonical labels or be omitted
- precision-first semantic extraction: `SERVES`, `SELLS_THROUGH`, `MONETIZES_VIA`, and `PARTNERS_WITH` favor standardization and explicit support over aggressive recall
- broader but still text-grounded company geography capture: `OPERATES_IN` is more recall-friendly than the semantic business-model relations, but still constrained to meaningful company presence
- staged extraction with supervision: the runtime extracts structure first, then relation families, then runs a rule-only reflection pass followed by a filing-aware reconciliation pass

This means the effective behavior of the literal pipeline comes from three layers together:
- the formal schema in [`src/ontology/ontology.json`](./src/ontology/ontology.json)
- the staged extraction and reflection pipeline under [`src/llm/`](./src/llm/) and [`src/llm_extraction/pipelines/`](./src/llm_extraction/pipelines/) with prompt assets in [`prompts/canonical/`](./prompts/canonical/) backing the `literal` pipeline
- the final normalization and structural enforcement in [`src/ontology/validator.py`](./src/ontology/validator.py)

## Extraction Pipelines

The repo ships two extraction pipelines:

- `literal`: the staged, ontology-constrained extraction runtime
- `analyst`: a sibling runtime that first builds a structured analyst memo from the full filing, then compiles that memo into the ontology graph and runs a short overreach critique pass

### Literal Extraction Pipeline

High-level flow:

1. Read the filing text, infer `company_name` from the input filename, and write `chunks.json`
2. `PASS 1`: build the structural skeleton with only `HAS_SEGMENT` and `OFFERS`
3. `PASS 2A`: extract only `SELLS_THROUGH`
4. `PASS 2B`: extract only `MONETIZES_VIA`
5. `PASS 3`: extract only `SERVES` from the structural graph
6. `PASS 4`: extract only company-level `OPERATES_IN` and `PARTNERS_WITH`
7. `Reflection 1`: enforce ontology and graph-rule compliance without adding new filing facts
8. `Reflection 2`: reconcile the draft graph against the full filing
9. Resolve surface forms, revalidate the final graph, and write final artifacts
10. Optionally load the graph into Neo4j

Runtime notes:
- each LLM step is relation-scoped and independently audited
- the pass outputs are merged incrementally rather than regenerated from scratch each time
- the final CLI validation is ontology- and structure-driven, with duplicate removal and place normalization
- the final CLI validation does not require strict text grounding by default
- if a reflection pass fails or returns an empty graph, the runtime falls back to the prior graph instead of hard-failing the whole run
- `--only-pass1` stops after the structural skeleton, then still resolves, validates, and can still load to Neo4j unless `--skip-neo4j` is also set

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
- `--only-pass1` is intentionally literal-only because the analyst pipeline's first pass is a memo rather than a loadable graph

## Query Interface

The runtime includes a read-only natural-language query path for the live Neo4j graph:

- `kg-query-cypher` and `kg-query` use the routed stack: local router + local Qwen planner first, with remote planner fallback
- `kg-query-cypher-jolly` and `kg-query-jolly` force LM Studio direct planning and bypass routing/fallback logic
- `kg-query` and `kg-query-cypher` also accept `--stack routed|fallback|jolly` so one command can select the behavior

The planner prompt lives in [`src/runtime/query_prompt.py`](./src/runtime/query_prompt.py), the deterministic compiler lives in [`src/runtime/query_planner.py`](./src/runtime/query_planner.py), and the read-only Cypher guards live in [`src/runtime/cypher_validation.py`](./src/runtime/cypher_validation.py).

The final curated fine-tuning dataset for the query planner is preserved under [`data/query_planner_curated/v1_final`](./data/query_planner_curated/v1_final/). The repo intentionally does not keep a dataset-construction CLI surface.

For Neo4j maintenance, the repo also ships:

- `kg-neo4j-load` to load saved outputs into Neo4j, either in bulk or for one company/run
- `kg-neo4j-status` to show which companies are loaded in Neo4j and which saved outputs are available
- `kg-neo4j-unload` to unload Neo4j graph data (full dataset by default, or one company with `--company`)

## Repo Layout

```text
src/
  runtime/
    main.py               runtime CLI implementation
    query.py              natural-language query CLI
    query_planner.py      family-based query-plan compiler
    neo4j_load.py         saved-output Neo4j load CLI
    neo4j_status.py       Neo4j vs saved-output status CLI
    neo4j_admin.py        Neo4j unload CLI (full dataset or one company)
    output_layout.py      company/pipeline output staging and promotion helpers
    query_prompt.py       prompt assets for query-plan generation and repair
    cypher_validation.py  read-only query guards and Neo4j URI normalization
    model_provider.py     provider/model resolution
    entity_resolver.py    light entity normalization
  llm/
    extractor.py          generic LLM calling and extraction facade
  llm_extraction/
    prompting.py          lightweight prompt loading/rendering helpers
    pipelines/__init__.py pipeline registry and runner dispatch
    pipelines/canonical/
                          literal pipeline orchestration
    pipelines/analyst/
                          analyst memo -> graph pipeline orchestration
  ontology/
    ontology.json         canonical ontology config
    config.py             canonical ontology loader
    validator.py          ontology validation and structural checks
    place_hierarchy.py    place normalization and hierarchy helpers
  graph/
    neo4j_loader.py       Neo4j loading and company-level unload
    evaluate_graph.py     graph evaluation utilities

  main.py                 compatibility CLI wrapper
  llm_extractor.py        compatibility extractor wrapper
  ontology_validator.py   compatibility validator wrapper

prompts/
  README.md               prompt asset overview
  canonical/              prompt assets backing the literal pipeline
  analyst/                analyst pipeline prompt assets

docs/
  ontology.md             canonical ontology specification
  project_walkthrough.md  plain-language architecture and workflow guide

scripts/
  bootstrap_dev.sh        create or refresh the local dev environment
  check_repo.sh           run the main local safety checks
  clean_local_artifacts.sh
                          remove caches and local scratch artifacts
  kg-pipeline             source-checkout pipeline wrapper
  kg-query                source-checkout query wrapper
  kg-query-cypher         source-checkout query-to-Cypher wrapper
  kg-query-jolly          source-checkout LM Studio direct query wrapper
  kg-query-cypher-jolly   source-checkout LM Studio direct query-to-Cypher wrapper
  kg-neo4j-load           source-checkout saved-output load wrapper
  kg-neo4j-status         source-checkout Neo4j status wrapper
  kg-neo4j-unload         source-checkout Neo4j unload wrapper
  kg-health-check         local repo and Neo4j health check

tests/
  test_runtime/
  test_llm/
  test_ontology/
  test_graph/
```

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

That editable install creates the convenience commands in `venv/bin/`:
- `kg-pipeline`
- `kg-evaluate-graph`
- `kg-query`
- `kg-query-cypher`
- `kg-query-jolly`
- `kg-query-cypher-jolly`
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
- `./scripts/kg-query-jolly`
- `./scripts/kg-query-cypher-jolly`
- `./scripts/kg-neo4j-load`
- `./scripts/kg-neo4j-status`
- `./scripts/kg-neo4j-unload`
- `./scripts/kg-health-check`

These wrappers run the repo source directly with the repo virtual environment, so they still work even if the editable-install entry points have not been refreshed yet.
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

Optional explicit `literal` pipeline flag:

```bash
./scripts/kg-pipeline data/microsoft_10k.txt --pipeline literal --skip-neo4j
```

Run only the structural skeleton:

```bash
./scripts/kg-pipeline data/microsoft_10k.txt --only-pass1 --skip-neo4j
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

Force fallback planner only (skip local router+Qwen) with one switch:

```bash
./scripts/kg-query-cypher "Which company segments sell through marketplaces?" --stack fallback
```

Force LM Studio direct planning (jolly mode, no routing/fallback):

```bash
./scripts/kg-query-cypher-jolly "Which company segments sell through marketplaces?"
./scripts/kg-query-jolly "Which company segments sell through marketplaces?"
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
- the CLI exposes both the `literal` and `analyst` pipelines
- every run writes `run_summary.json`; the console header shows pipeline, provider, and model, and LLM attempt summaries show token counts when available
- successful Neo4j loads now replace the previous graph footprint for that same company by default; use `--clear-neo4j` only when you truly want to wipe the entire database first

Useful CLI flags:
- `--only-pass1`: stop after structural extraction, then still resolve/validate and optionally load
- `--company-name`: override the inferred company name used for output folders and company-scoped Neo4j operations
- `--max-retries`: change the retry budget per LLM call
- `--base-url`: pass either an API root or a full endpoint URL; the runtime normalizes common suffixes
- `--api-key`: override environment-based key resolution
- `--max-output-tokens`: explicitly cap model output tokens
- `kg-query` / `kg-query-cypher --skip-local-stack`: skip local router+Qwen and force fallback planner generation
- `kg-query` / `kg-query-cypher --stack fallback`: preferred single-switch way to force fallback planner generation
- `kg-query` / `kg-query-cypher --stack jolly`: run LM Studio jolly mode from the same command surface
- `kg-query` / `kg-query-cypher --local-stack-python /path/to/python`: point routed query commands to a specific local stack environment
- `kg-query` / `kg-query-cypher --provider ...`: choose the fallback planner provider used only when local routing does not return a local-safe plan
- when local planner errors in routed mode, the CLI asks `Use API fallback instead? [Y/n]` before continuing
- fallback planner generation retries once with error context; if both attempts fail, the CLI prints a warning
- `--clear-neo4j`: clear the target database before loading
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

Use `--pipeline literal` if you want the command to target literal outputs instead of the default analyst outputs.
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
- `kg-health-check`: checks whether the local repo setup looks usable. It reports Python, the repo venv, `.env.example`, prompt assets, ontology assets, saved outputs, and optionally Neo4j connectivity.

The most useful flags in practice are:

- `--company-name` on `kg-pipeline` when the filename is not the company identity you want to use for outputs and Neo4j replacement.
- `--pipeline literal|analyst` on `kg-pipeline`, `kg-neo4j-load`, and `kg-neo4j-status` when you want to work with literal outputs instead of the default analyst ones.
- `--keep-current-output --skip-neo4j` on `kg-pipeline` when you want to save a test run under `runs/` without replacing the current `latest/` output or the live Neo4j graph.
- `--clear-neo4j` on `kg-pipeline` only when you intentionally want to wipe the entire Neo4j database before loading.
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

These routed commands try local router+Qwen first, then use fallback planner generation when needed.

If you want to bypass routing and force LM Studio direct planning:

```bash
./scripts/kg-query-cypher-jolly "Which companies sell to developers through direct sales?"
./scripts/kg-query-jolly "Which companies sell to developers through direct sales?"
```

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
    canonical/
      latest/
        ...
      runs/
        20260417T101500Z/
          ...
      failed/
        20260417T103000Z/
          ...
    analyst/
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

Canonical pipeline runs write artifacts such as:
- `run_summary.json`
- `chunks.json`
- `skeleton_extraction.json`
- `pass2_channels_extraction.json`
- `pass2_revenue_extraction.json`
- `pass2_commercial_extraction.json`
- `pass3_serves_extraction.json`
- `pass4_corporate_extraction.json`
- `pre_reflection_extraction.json`
- `rule_reflection_extraction.json`
- `reflection_extraction.json`
- `extractions.json`
- `extraction_audits.json`
- `final_output_validation_report.json`
- `resolved_triples.json`
- `validation_report.json`

`--only-pass1` writes the structural subset of these artifacts rather than the full multi-pass set.

Analyst pipeline runs write a different mix centered on:
- `analyst_memo_foundation.md`
- `analyst_memo_augmented.md`
- `analyst_graph_compilation.json`
- `analyst_graph_critique.json`
- `run_summary.json`
- `chunks.json`
- `resolved_triples.json`
- `validation_report.json`

## Evaluation Utilities

Compare a predicted extraction to a gold graph:

```bash
./scripts/kg-evaluate-graph compare path/to/predicted.json path/to/gold.json
```

Inspect the graph currently loaded in Neo4j:

```bash
./scripts/kg-evaluate-graph dump-neo4j
```

The evaluation utility accepts payloads wrapped as `triples`, `resolved_triples`, or `valid_triples`.

## Tests

Extraction/runtime tests:

```bash
./venv/bin/python -m unittest discover -s tests
```
