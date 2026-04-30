# Project Walkthrough

This document explains how the project works in plain language.

## What The Project Does

The repo takes a business section from a 10-K filing, extracts a structured view of the company, validates that structure against a fixed ontology, writes saved outputs to disk, and can load the result into Neo4j for querying.

In short, the workflow is:

1. read a filing
2. run one of the extraction pipelines
3. resolve and validate the graph
4. save the run artifacts under `outputs/`
5. optionally load the result into Neo4j

## The Supported Pipelines

There are three supported pipelines:

- `analyst`: a memo-first pipeline that builds a structured analyst view and then compiles that into the graph
- `memo_graph_only`: an ablation pipeline that builds only the first analyst memo, compiles it into the graph, and skips memo augmentation and critique
- `zero-shot`: a single-pass baseline that emits the ontology graph directly from the filing

The default Neo4j reload commands prefer `analyst/latest` because that is the current preferred saved output for interactive reloads.

## The Main Runtime Flow

The runtime entry point is `runtime.main`.

At a high level it does this:

1. infer or accept the company name
2. create a staging output directory
3. run the selected extraction pipeline
4. resolve entities and validate triples against the ontology
5. promote the successful run into the right place under `outputs/`
6. optionally replace that same company in Neo4j

If a run fails, the current `latest/` output stays untouched and the failed attempt is stored under `failed/`.

## Output Layout

Outputs are organized by company and pipeline under `outputs/`.

The key folders mean:

- `latest/`: the current successful output for that company and pipeline
- `runs/`: successful runs that were kept without replacing `latest/`
- `failed/`: failed attempts kept for debugging
- `manifest.json`: a small summary that tells the helper commands what exists

For the full output tree and artifact list, see [`runtime_guide.md`](./runtime_guide.md#output-layout).

## Neo4j Lifecycle

There are three main Neo4j helper commands:

- `kg-neo4j-load`: load saved outputs into Neo4j
- `kg-neo4j-status`: compare Neo4j against the saved outputs
- `kg-neo4j-unload`: unload Neo4j graph data (full dataset by default, or one company with `--company`)

There is also one operational helper command:

- `kg-health-check`: check whether the local repo, routed query stack, saved outputs, and optional Neo4j service look ready to use

The important behavior is:

- company replacement is company-scoped, not whole-database by default
- single-company replacement is transactional, so a failed reload does not wipe the previous live graph for that company
- status is read-only
- unload supports full-dataset clear (no `--company`) and company-scoped removal (`--company`) that keeps unrelated shared graph state where possible

For natural-language querying, the repo now uses two query modes behind the same CLI surface:

- the router first decides `local`, `api_fallback`, or `refuse`
- `local` is allowed only when the router assigns at least `0.97` local confidence;
  otherwise the runtime chooses between the two non-local labels
- if the route is `local`, the published local planner returns an answerable-only compact plan that Python compiles into Cypher
- the hosted fallback returns full read-only Cypher JSON directly and retries once with error context if generation, validation, or Neo4j execution fails

The important design split is:

- the router owns refusal and fallback decisions
- the local planner only speaks in supported local plan space
- the hosted fallback is the broader query-authoring path

## Query Stack Results

The local query stack is useful, but it should be understood as a first
deployable routing/planning layer rather than a finished general query system.
The strongest result is that many graph-shaped questions can run fully local:
company, segment, offering, customer type, channel, revenue-model, geography,
partner, count, ranking, and offering-hierarchy lookups all have deterministic
Cypher compilers behind them.

The dataset is good in the sense that it has clear route labels, balanced top
level classes, reproducible splits, and local-safe examples that compile to
known Cypher and expected rows. It also deliberately separates three behaviors:
local graph plans, hosted-fallback candidates, and terminal refusals. That makes
the runtime architecture easy to reason about.

The dataset weakness is that the balance is mostly numerical, not semantic.
Some phrases become shortcuts. For example, `how many` appears only as supported
local graph counts, so the router can over-trust ordinary count questions that
are not about the graph. Refusals are also mostly business-domain refusals that
mention known graph entities, such as unsupported employees, suppliers, revenue,
or time-series facts. There are very few truly out-of-domain refusals, and some
valid paraphrases are thin or missing in validation and release-eval splits.

The router is the highest-leverage piece to improve next. When it routes a valid
local question to `api_fallback`, the local planner never gets a chance; when it
routes an unsupported question to `local`, the planner may fail noisily before
hosted fallback takes over. The most important observed router gaps are natural
paraphrases such as `sell to developers through direct sales` and hard negatives
such as ordinary-world count or inventory questions.

The planner problems are narrower. When the router sends a question local, the
planner usually only needs to choose one supported family and fill a compact
payload. It can still miss natural phrasings, especially possessive hierarchy
wording such as `offerings under Apple's iPhone`, where it should choose
`descendant_offerings_by_root`. It can also expose schema gaps, such as users
asking for companies that match both geography and segment filters when the
current local family returns company-plus-segment rows instead.

Practically, router fixes should come first. The hosted fallback can absorb many
planner misses, but it cannot help when the router incorrectly withholds a valid
local query from the planner or confidently sends a bad request into local mode.
The next query-stack iteration should therefore focus on a router dataset v2
with more hard local paraphrases, more hard refusals, and validation/release
coverage for the same patterns before spending much more time on planner tuning.

## Recommended Commands From A Source Checkout

When you are working directly from this repo, the most reliable commands are the wrapper scripts under `scripts/`.

Use:

- `./scripts/kg-pipeline`
- `./scripts/kg-query`
- `./scripts/kg-query-cypher`
- `./scripts/kg-neo4j-load`
- `./scripts/kg-neo4j-status`
- `./scripts/kg-neo4j-unload`

These wrappers run the repo source directly with the repo virtual environment and do not depend on refreshed `venv/bin/` entry points.

For the full command reference, see [`runtime_guide.md`](./runtime_guide.md).

## Bootstrap And Cleanup

To create or refresh the local development environment:

```bash
./scripts/bootstrap_dev.sh
```

To run a quick local health check:

```bash
./scripts/kg-health-check
```

To run the fuller maintainer check:

```bash
bash ./scripts/check_repo.sh
```

To remove common local noise without touching saved outputs:

```bash
./scripts/clean_local_artifacts.sh
```

That cleanup script removes cache folders, generated egg-info metadata, local logs, and similar scratch artifacts.

## Where To Look In The Code

If you want to navigate the codebase, these are the most important areas:

- `src/runtime/`: CLI behavior, output layout, Neo4j load/status/unload helpers
- `src/llm_extraction/pipelines/`: pipeline-specific orchestration
- `src/llm/`: model calling, retries, parsing, fallback behavior
- `src/ontology/`: ontology rules and validation
- `src/graph/`: Neo4j loading and graph maintenance helpers
- `prompts/`: editable prompt assets used during development
- `finetuning/`: the isolated training/export island for the local query router/planner
- `data/query_planner_curated/`: preserved curated datasets for that finetuning workflow, not a live runtime input path

For a more detailed layout map, see [`repo_structure.md`](./repo_structure.md).

The main runtime does not import from the finetuning island directly. The handoff back into production is the published local query bundle under `runtime_assets/query_stack/`.

## A Good Mental Model

The cleanest way to think about the project is:

- prompts define how extraction is asked for
- pipelines decide how extraction is staged
- ontology decides what is allowed
- outputs capture each run on disk
- Neo4j commands manage which saved company graphs are live
