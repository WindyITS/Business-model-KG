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

When loading multiple companies into Neo4j, `BusinessSegment` and `Offering` nodes are scoped by
`company_name` so same-named nodes from different companies do not collapse into one shared node.
This keeps canonical shared vocabularies such as `Channel`, `CustomerType`, `RevenueModel`, and
`Place` global, while keeping company-owned inventory nodes distinct.

Full ontology spec: [`docs/ontology.md`](./docs/ontology.md)

## Pipeline Philosophy

The canonical runtime follows a few consistent rules:

- scope-first modeling: `Company` is the corporate shell, `BusinessSegment` is the primary semantic anchor, and `Offering` is the inventory layer
- canonical extraction over convenience duplication: the extractor does not materialize inherited or rollup facts just to make the graph denser
- closed semantic vocabularies: `CustomerType`, `Channel`, and `RevenueModel` must map to the approved canonical labels or be omitted
- precision-first semantic extraction: `SERVES`, `SELLS_THROUGH`, `MONETIZES_VIA`, and `PARTNERS_WITH` favor standardization and explicit support over aggressive recall
- broader but still text-grounded company geography capture: `OPERATES_IN` is more recall-friendly than the semantic business-model relations, but still constrained to meaningful company presence
- staged extraction with supervision: the runtime extracts structure first, then relation families, then runs a rule-only reflection pass followed by a filing-aware reconciliation pass

This means the effective behavior of the pipeline comes from three layers together:
- the formal schema in [`configs/ontology.json`](./configs/ontology.json)
- the staged extraction and reflection prompts in [`src/llm_extractor.py`](./src/llm_extractor.py)
- the final normalization and structural enforcement in [`src/ontology_validator.py`](./src/ontology_validator.py)

## Canonical Extraction Pipeline

The default runtime pipeline is the only supported extraction pipeline in the repo.

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
    v2/                   canonical training corpus and reports
    v1/                   superseded dataset release kept for provenance
    archive/v0/           archived prototype artifacts

tests/
  test_pipeline_components.py
  test_ontology_validator.py
  test_place_hierarchy.py
```

## Text2Cypher Dataset Assets

The supervised text-to-Cypher corpus is now split by role:

- prose and design docs live in [`docs/text2cypher/`](./docs/text2cypher/README.md)
- canonical machine-readable V2 artifacts live in [`datasets/text2cypher/v2/`](./datasets/text2cypher/README.md)
- the superseded V1 release remains in-repo for provenance and comparison
- archived pre-V1 snapshots live in `datasets/text2cypher/archive/v0/`

The dataset validator in [`src/validate_text2cypher_dataset.py`](./src/validate_text2cypher_dataset.py) now defaults to the canonical V2 artifact set under `datasets/text2cypher/v2/`.

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

Run only the structural skeleton:

```bash
./venv/bin/python src/main.py data/microsoft_10k.txt --only-pass1 --skip-neo4j
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
- `local` reads `--api-key` first, then `LOCAL_LLM_API_KEY`, then `LM_STUDIO_API_KEY`, and otherwise falls back to `lm-studio`
- `opencode-go` reads `--api-key` first, then `OPENCODE_GO_API_KEY`, then `OPENCODE_API_KEY`
- for `opencode-go`, the runtime rewrites `system` messages to `user` messages for compatibility while keeping the rest of the pipeline flow unchanged
- `opencode-go` defaults to `--max-output-tokens 20000`; override it if needed
- every run writes `run_summary.json`; the console header shows pipeline, provider, and model, and LLM attempt summaries show token counts when available

Useful CLI flags:
- `--only-pass1`: stop after structural extraction, then still resolve/validate and optionally load
- `--max-retries`: change the retry budget per LLM call
- `--base-url`: pass either an API root or a full endpoint URL; the runtime normalizes common suffixes
- `--api-key`: override environment-based key resolution
- `--max-output-tokens`: explicitly cap model output tokens
- `--clear-neo4j`: clear the target database before loading

Load into Neo4j instead:

```bash
docker compose up -d
./venv/bin/python src/main.py data/microsoft_10k.txt
```

Neo4j Browser:
- `http://localhost:7474`
- default credentials: `neo4j / password`

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

Each run writes a timestamped directory under `outputs/` with artifacts such as:
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

## Evaluation Utilities

Compare a predicted extraction to a gold graph:

```bash
./venv/bin/python src/evaluate_graph.py compare path/to/predicted.json path/to/gold.json
```

Inspect the graph currently loaded in Neo4j:

```bash
./venv/bin/python src/evaluate_graph.py dump-neo4j
```

The evaluation utility accepts payloads wrapped as `triples`, `resolved_triples`, or `valid_triples`.

## Tests

Extraction/runtime tests:

```bash
./venv/bin/python -m pytest -q tests/test_pipeline_components.py tests/test_ontology_validator.py tests/test_place_hierarchy.py
```
