# Business Model KG

A local pipeline for turning SEC 10-K business sections into a standardized business-model knowledge graph.

The repo now contains only the canonical runtime stack:
- one canonical extraction pipeline
- one canonical ontology
- one validator aligned to that ontology
- optional Neo4j loading and graph evaluation utilities

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

Full ontology spec: [`docs/ontology.md`](./docs/ontology.md)

## Canonical Extraction Pipeline

The default runtime pipeline is the only supported extraction pipeline in the repo.

High-level flow:

1. `PASS 1`: build the structural skeleton
2. `PASS 2A`: extract channels
3. `PASS 2B`: extract revenue models
4. `PASS 3`: extract customer types from the structural graph
5. `PASS 4`: extract company-level geography and partnerships
6. Final reflection: reconcile the graph against the filing

Then the pipeline:
- validates the final triples against the ontology
- resolves duplicate surface forms
- optionally loads the graph into Neo4j

## Repo Layout

```text
src/
  llm_extractor.py        canonical prompt pipeline
  main.py                 CLI entrypoint
  ontology_config.py      canonical ontology loader
  ontology_validator.py   ontology validation and structural checks
  entity_resolver.py      light entity normalization
  neo4j_loader.py         Neo4j loading
  evaluate_graph.py       graph evaluation utilities

configs/
  ontology.json           canonical ontology config

docs/
  ontology.md             canonical ontology specification

tests/
  test_pipeline_components.py
  test_ontology_validator.py
```

## Quickstart

Requirements:
- Python 3.10+
- an OpenAI-compatible local endpoint such as LM Studio
- optionally Neo4j
- optionally an OpenCode Go API key if you want to run against hosted models instead of a local endpoint

Setup:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the extraction pipeline:

```bash
./venv/bin/python src/main.py data/microsoft_10k.txt --skip-neo4j
```

Optional explicit pipeline flag:

```bash
./venv/bin/python src/main.py data/microsoft_10k.txt --pipeline canonical --skip-neo4j
```

Run against OpenCode Go with a hosted open model:

```bash
export OPENCODE_GO_API_KEY=your_key_here
./venv/bin/python src/main.py data/microsoft_10k.txt \
  --provider opencode-go \
  --model kimi-k2.5 \
  --skip-neo4j
```

Other supported OpenCode Go models use the same CLI shape:

```bash
./venv/bin/python src/main.py data/microsoft_10k.txt --provider opencode-go --model mimo-v2-pro --skip-neo4j
./venv/bin/python src/main.py data/microsoft_10k.txt --provider opencode-go --model minimax-m2.7 --skip-neo4j
```

Provider notes:
- `local` defaults to `http://localhost:1234/v1` and `local-model`
- `opencode-go` defaults to `https://opencode.ai/zen/go/v1` and `kimi-k2.5`
- `opencode-go` currently supports `kimi-k2.5`, `mimo-v2-pro`, and `minimax-m2.7` in this repo
- `kimi-k2.5` and `mimo-v2-pro` use OpenCode Go's `chat/completions` endpoint; `minimax-m2.7` is routed automatically to OpenCode Go's Anthropic-compatible `messages` endpoint
- the CLI also accepts friendly OpenCode Go model names such as `MiniMax M2.7`, plus prefixed IDs like `opencode-go/kimi-k2.5`
- the CLI accepts either a root base URL or a full documented endpoint like `.../chat/completions` and normalizes it automatically
- for `opencode-go`, the runtime rewrites `system` messages to `user` messages for compatibility while keeping the rest of the pipeline flow unchanged
- `no-schema` is the default behavior for all providers; pass `--use-schema` to opt back into JSON Schema enforcement
- `opencode-go` defaults to `--max-output-tokens 20000`; override it if needed
- every run logs a settings line including pipeline, model, schema mode, and max output tokens

Load into Neo4j instead:

```bash
docker compose up -d
./venv/bin/python src/main.py data/microsoft_10k.txt
```

Neo4j Browser:
- `http://localhost:7474`
- default credentials: `neo4j / password`

## Output Artifacts

Each run writes a timestamped directory under `outputs/` with artifacts such as:
- `run_summary.json`
- `skeleton_extraction.json`
- `pass2_channels_extraction.json`
- `pass2_revenue_extraction.json`
- `pass3_serves_extraction.json`
- `pass4_corporate_extraction.json`
- `pre_reflection_extraction.json`
- `reflection_extraction.json`
- `resolved_triples.json`
- `validation_report.json`

## Tests

Extraction/runtime tests:

```bash
./venv/bin/python -m pytest -q tests/test_pipeline_components.py tests/test_ontology_validator.py
```
