# Runtime Guide

This guide collects the command, output, and Neo4j details for day-to-day
runtime work.

## Source Checkout Commands

For day-to-day work from this repo, prefer the wrapper scripts under `scripts/`:

- `./scripts/kg-pipeline`
- `./scripts/kg-query`
- `./scripts/kg-query-cypher`
- `./scripts/kg-neo4j-load`
- `./scripts/kg-neo4j-status`
- `./scripts/kg-neo4j-unload`
- `./scripts/kg-health-check`

These wrappers run the repo source directly with the repo virtual environment.
They keep working even if editable-install entry points under `venv/bin/` have
not been refreshed.

For direct module execution, use package entry points such as:

```bash
python -m runtime.main
python -m runtime.query
python -m runtime.query_cypher
```

If you installed the package in editable mode, these commands are also exposed
under `venv/bin/`:

- `kg-pipeline`
- `kg-query`
- `kg-query-cypher`
- `kg-neo4j-load`
- `kg-neo4j-status`
- `kg-neo4j-unload`
- `kg-health-check`

If a new command was added after your virtual environment was created, refresh
the editable install:

```bash
pip install -e .
```

## Extraction

Run the default extraction pipeline without loading Neo4j:

```bash
./scripts/kg-pipeline data/microsoft_10k.txt --skip-neo4j
```

Run a specific pipeline:

```bash
./scripts/kg-pipeline data/microsoft_10k.txt --pipeline analyst --skip-neo4j
./scripts/kg-pipeline data/microsoft_10k.txt --pipeline memo_graph_only --skip-neo4j
./scripts/kg-pipeline data/microsoft_10k.txt --pipeline zero-shot --skip-neo4j
```

Useful flags:

- `--company-name`: override the inferred company name used for outputs and Neo4j replacement
- `--pipeline analyst|memo_graph_only|zero-shot`: choose the extraction pipeline
- `--max-retries`: change the retry budget per LLM call
- `--base-url`: pass either an API root or a full endpoint URL
- `--api-key`: override environment-based key resolution
- `--max-output-tokens`: explicitly cap model output tokens
- `--skip-neo4j`: save outputs without loading Neo4j
- `--keep-current-output --skip-neo4j`: keep `latest/` untouched and store the successful run under `runs/`

Every run writes `run_summary.json`. The console header shows pipeline,
provider, and model. LLM attempt summaries include token counts when available.

## Providers

The runtime supports a local OpenAI-compatible endpoint and OpenCode Go.

Local defaults:

- base URL: `http://localhost:1234/v1`
- model: `local-model`
- API key lookup: `--api-key`, then `LOCAL_LLM_API_KEY`, then `LM_STUDIO_API_KEY`, then `lm-studio`

OpenCode Go defaults:

- base URL: `https://opencode.ai/zen/go/v1`
- model: `kimi-k2.5`
- API key lookup: `--api-key`, then `OPENCODE_GO_API_KEY`, then `OPENCODE_API_KEY`

Run against OpenCode Go:

```bash
export OPENCODE_GO_API_KEY=your_key_here
./scripts/kg-pipeline data/microsoft_10k.txt \
  --provider opencode-go \
  --model kimi-k2.5 \
  --skip-neo4j
```

Other supported OpenCode Go model examples:

```bash
./scripts/kg-pipeline data/microsoft_10k.txt --provider opencode-go --model mimo-v2-pro --skip-neo4j
./scripts/kg-pipeline data/microsoft_10k.txt --provider opencode-go --model minimax-m2.7 --skip-neo4j
```

Notes:

- `opencode-go` supports `kimi-k2.5`, `mimo-v2-pro`, and `minimax-m2.7` in this repo
- friendly names such as `MiniMax M2.7` and prefixed IDs such as `opencode-go/kimi-k2.5` are accepted
- a root base URL or a full documented endpoint URL can be passed; the runtime normalizes common suffixes
- `kimi-k2.5` and `mimo-v2-pro` use OpenCode Go's `chat/completions` endpoint
- `minimax-m2.7` is routed automatically to OpenCode Go's Anthropic-compatible `messages` endpoint
- for `opencode-go`, the runtime rewrites `system` messages to `user` messages for compatibility
- `opencode-go` defaults to `--max-output-tokens 20000`

## Querying

The query commands can render read-only Cypher or run a natural-language
question against the current Neo4j graph.

`kg-query-cypher` can render a query without Neo4j. `kg-query` also needs Neo4j
running and loaded with saved graph outputs.

The routed query commands try the published local query-stack bundle first.
The router sends a query to the local planner only when its `local` probability
is at least `0.97`; below that fixed gate it chooses the stronger non-local
class, either `api_fallback` or `refuse`. The commands fall back to hosted
free-form Cypher generation when the local stack is unavailable, errors, or the
router selects `api_fallback`. If the router selects `refuse`, the command
returns an unsupported-request result instead of using the hosted fallback.

### Local Query-Stack Bundle

The local router/planner bundle is not tracked in Git. A fresh clone will not
have `runtime_assets/query_stack/` until you download the published bundle or
rebuild it from the fine-tuning island.

For local routed querying, download the published bundle and install the query
extras:

```bash
./venv/bin/python -m pip install "huggingface_hub[cli]"
./venv/bin/huggingface-cli download WindyITS/business-model-kg-query-stack \
  --local-dir runtime_assets/query_stack
./venv/bin/python -m pip install -e ".[query-stack]"
```

Without this bundle, routed query commands can still use hosted fallback when a
fallback provider and API key are configured, but the local planner path will
not be available.

For hosted fallback, configure an OpenCode Go API key:

```bash
export OPENCODE_GO_API_KEY=your_key_here
```

Render a read-only Cypher query with the routed stack:

```bash
./scripts/kg-query-cypher "Which companies sell to developers through direct sales?"
```

Force hosted fallback only:

```bash
./scripts/kg-query-cypher "Which company segments sell through marketplaces?" --stack fallback
```

Override the local query-stack bundle:

```bash
./scripts/kg-query-cypher "Which company segments sell through marketplaces?" \
  --local-stack-bundle-dir /path/to/runtime_assets/query_stack
```

After Neo4j is running and saved outputs have been loaded, run a
natural-language query against the graph:

```bash
./scripts/kg-query "Which companies sell to developers through direct sales?"
```

Query command notes:

- `kg-query` returns rows from the live database as plain text
- `kg-query-cypher` returns a runnable plain-text Cypher query with generated params already inlined
- progress and error messages go to stderr so stdout stays easy to pipe or copy
- hosted fallback query generation retries once with error context
- if both hosted attempts fail, the CLI prints a warning

## Neo4j

Start Neo4j:

```bash
docker compose up -d
```

Default browser:

- `http://localhost:7474`
- default credentials: `neo4j / password`

Run an extraction and load the result:

```bash
./scripts/kg-pipeline data/microsoft_10k.txt
```

Load latest `analyst` outputs for every company:

```bash
./scripts/kg-neo4j-load
```

Load latest `analyst` output for one company:

```bash
./scripts/kg-neo4j-load --company "Microsoft"
```

Load one exact saved run:

```bash
./scripts/kg-neo4j-load --company "Microsoft" --run 20260417T101500Z
```

Use `--pipeline zero-shot` or `--pipeline memo_graph_only` to target another
output family. Use `--yes` to skip confirmation prompts in automation.

Show loaded companies and available local outputs:

```bash
./scripts/kg-neo4j-status
```

Unload the full Neo4j dataset:

```bash
./scripts/kg-neo4j-unload
```

Unload one company's graph footprint:

```bash
./scripts/kg-neo4j-unload --company "Microsoft"
```

Neo4j command behavior:

- `kg-pipeline <file>` replaces that same company's loaded graph on success unless `--skip-neo4j` is passed
- `kg-neo4j-load` defaults to every `outputs/<company>/analyst/latest/` directory
- `kg-neo4j-load --company "Name"` loads only that company's latest saved output
- `kg-neo4j-load --company "Name" --run <token>` loads a saved run under `runs/<token>` or `failed/<token>`
- bulk load warns before running if Neo4j already contains data
- bulk load keeps going if one company fails and reports failures at the end
- single-company load asks before replacing that company unless `--yes` is passed
- company replacement is transactional, so a failed reload leaves the previous live graph untouched
- `kg-neo4j-status` is read-only
- `kg-neo4j-unload` clears the full dataset unless `--company` is passed

## Neo4j Node Scoping

When the loader writes to Neo4j:

- `Company`, `Channel`, `CustomerType`, `RevenueModel`, and `Place` are globally keyed by `name`
- `BusinessSegment` and `Offering` are keyed by `(company_name, name)`

This prevents cross-company collisions such as two companies each having an
offering named `Advertising`.

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

If your database contains older unscoped `BusinessSegment` or `Offering` nodes,
clear and reload the graph once so scoped identity takes effect consistently.

## Geography In Neo4j

The extractor and loader keep geography canonical at:

```text
Company-[:OPERATES_IN]->Place
```

No derived place hierarchy relationships are materialized in Neo4j. Instead,
each extracted `Place` node can receive query helper properties during load:

- `within_places`: broader canonical places that contain the place
- `includes_places`: narrower canonical places that the place contains

Examples:

- `Italy` can include `within_places = ["Europe", "Western Europe", "EMEA", "European Union"]`
- `Europe` can include `includes_places = ["Western Europe", "Eastern Europe", "Italy", "Germany", ...]`
- `United States` can include `includes_places = ["Alabama", "Alaska", ..., "Wyoming"]`

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

## Output Layout

Outputs are organized by company and pipeline:

```text
outputs/
  microsoft/
    analyst/
      latest/
      runs/
      failed/
      manifest.json
    memo_graph_only/
      latest/
      runs/
      failed/
      manifest.json
    zero-shot/
      latest/
      runs/
      failed/
      manifest.json
```

Folder meanings:

- `latest/`: current successful output for that company and pipeline
- `runs/`: successful runs kept without replacing `latest/`
- `failed/`: failed attempts kept for debugging
- `manifest.json`: summary consumed by helper commands

Default behavior:

- a successful run stages artifacts first, then replaces `latest/`
- the previous `latest/` output for that company and pipeline is removed on replacement
- failed runs do not overwrite `latest/`
- `--keep-current-output --skip-neo4j` stores a successful run under `runs/`
- the output folder name comes from the inferred company identity unless `--company-name` is passed

Analyst pipeline runs center on:

- `analyst_memo_foundation.md`
- `analyst_memo_augmented.md`
- `analyst_graph_compilation.json`
- `analyst_graph_critique.json`
- `run_summary.json`
- `chunks.json`
- `resolved_triples.json`
- `validation_report.json`

Memo graph-only runs center on:

- `memo_graph_only_memo_foundation.md`
- `memo_graph_only_graph_compilation.json`
- `run_summary.json`
- `chunks.json`
- `resolved_triples.json`
- `validation_report.json`

Zero-shot runs center on:

- `zero_shot_extraction.json`
- `run_summary.json`
- `chunks.json`
- `resolved_triples.json`
- `validation_report.json`

## Prompt Workflow

Prompt files are edited under `prompts/`. Packaged installs carry a bundled
fallback copy under `src/llm_extraction/_bundled_prompts/`.

Prompt loading order:

1. `KG_PROMPTS_DIR`
2. repo-level `prompts/`
3. bundled packaged prompts

Notes:

- editable installs keep using the repo-level `prompts/` directory
- package installs work because bundled prompts are included as package data
- `KG_PROMPTS_DIR=/path/to/prompts` can test an alternate prompt set
- if a reflection stage returns an empty graph or exhausts retries, interactive terminals ask whether to keep the last good graph from the current run
- non-interactive runs keep the last good graph automatically and print that choice

## Cleanup And Checks

Remove local caches and scratch artifacts without touching saved outputs:

```bash
./scripts/clean_local_artifacts.sh
```

Run a light local readiness check:

```bash
./scripts/kg-health-check
```

Skip Neo4j probing:

```bash
./scripts/kg-health-check --skip-neo4j
```

Run the full maintainer check:

```bash
bash ./scripts/check_repo.sh
```

The full maintainer check runs tests, fine-tuning tests, compilation checks,
wrapper checks, and package smoke installs. It is intentionally broader than a
quick reviewer smoke check.

`.github/workflows/checks.yml` runs the same repo-check script on pushes and
pull requests.
