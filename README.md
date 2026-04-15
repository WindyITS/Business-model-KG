# Business Model KG

A local pipeline for turning SEC 10-K business sections into a standardized business-model knowledge graph.

The repo is organized around two maintained surfaces:
- one canonical extraction/runtime stack
- one structured text-to-Cypher dataset for SFT and evaluation

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
  validate_text2cypher_dataset.py
                          validates gold Cypher against synthetic fixtures
  entity_resolver.py      light entity normalization
  place_hierarchy.py      place normalization and query metadata helpers
  neo4j_loader.py         Neo4j loading
  evaluate_graph.py       graph evaluation utilities

configs/
  ontology.json           canonical ontology config

docs/
  ontology.md             canonical ontology specification
  text2cypher/
    README.md             dataset documentation and canonical entrypoints
    design/               coverage, intent, fixture, and readiness docs

datasets/
  text2cypher/
    README.md             machine-readable dataset layout
    v1/                   canonical training corpus and reports
    archive/v0/           archived prototype artifacts

tests/
  test_pipeline_components.py
  test_ontology_validator.py
  test_place_hierarchy.py
```

## Text2Cypher Dataset Assets

The supervised text-to-Cypher corpus is now split by role:

- prose and design docs live in [`docs/text2cypher/`](./docs/text2cypher/README.md)
- canonical machine-readable V1 artifacts live in [`datasets/text2cypher/v1/`](./datasets/text2cypher/README.md)
- archived pre-V1 snapshots live in `datasets/text2cypher/archive/v0/`

The dataset validator in [`src/validate_text2cypher_dataset.py`](./src/validate_text2cypher_dataset.py) now defaults to the canonical V1 artifact set under `datasets/text2cypher/v1/`.

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
- `opencode-go` defaults to `--max-output-tokens 20000`; override it if needed
- every run logs a settings line including pipeline, model, and max output tokens

Load into Neo4j instead:

```bash
docker compose up -d
./venv/bin/python src/main.py data/microsoft_10k.txt
```

Neo4j Browser:
- `http://localhost:7474`
- default credentials: `neo4j / password`

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
WITH company, place,
     CASE
       WHEN place.name = $place THEN 0
       WHEN $place IN coalesce(place.includes_places, []) THEN 1
       WHEN $place IN coalesce(place.within_places, []) THEN 2
       ELSE NULL
     END AS match_rank
WHERE match_rank IS NOT NULL
RETURN company.name AS company,
       CASE match_rank
         WHEN 0 THEN 'exact'
         WHEN 1 THEN 'narrower_place'
         ELSE 'broader_region'
       END AS geography_match
ORDER BY match_rank, company
```

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
./venv/bin/python -m pytest -q tests/test_pipeline_components.py tests/test_ontology_validator.py tests/test_place_hierarchy.py
```
