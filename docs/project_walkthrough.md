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

There are two supported pipelines:

- `analyst`: a memo-first pipeline that builds a structured analyst view and then compiles that into the graph
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

Outputs are organized by company and pipeline:

```text
outputs/
  microsoft/
    analyst/
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

The key folders mean:

- `latest/`: the current successful output for that company and pipeline
- `runs/`: successful runs that were kept without replacing `latest/`
- `failed/`: failed attempts kept for debugging
- `manifest.json`: a small summary that tells the helper commands what exists

## Neo4j Lifecycle

There are three main Neo4j helper commands:

- `kg-neo4j-load`: load saved outputs into Neo4j
- `kg-neo4j-status`: compare Neo4j against the saved outputs
- `kg-neo4j-unload`: unload Neo4j graph data (full dataset by default, or one company with `--company`)

There is also one operational helper command:

- `kg-health-check`: check whether the local repo, saved outputs, and optional Neo4j service look ready to use

The important behavior is:

- company replacement is company-scoped, not whole-database by default
- single-company replacement is transactional, so a failed reload does not wipe the previous live graph for that company
- status is read-only
- unload supports full-dataset clear (no `--company`) and company-scoped removal (`--company`) that keeps unrelated shared graph state where possible

## Recommended Commands From A Source Checkout

When you are working directly from this repo, the most reliable commands are the wrapper scripts under `scripts/`.

Use:

- `./scripts/kg-pipeline`
- `./scripts/kg-query`
- `./scripts/kg-query-cypher`
- `./scripts/kg-query-jolly`
- `./scripts/kg-query-cypher-jolly`
- `./scripts/kg-neo4j-load`
- `./scripts/kg-neo4j-status`
- `./scripts/kg-neo4j-unload`

These wrappers run the repo source directly with the repo virtual environment and do not depend on refreshed `venv/bin/` entry points.

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
- `src/graph/`: Neo4j loading and graph evaluation helpers
- `prompts/`: editable prompt assets used during development

## A Good Mental Model

The cleanest way to think about the project is:

- prompts define how extraction is asked for
- pipelines decide how extraction is staged
- ontology decides what is allowed
- outputs capture each run on disk
- Neo4j commands manage which saved company graphs are live
