# Business Model KG

A local pipeline for turning SEC 10-K business sections into a standardized business-model knowledge graph.

The repo now has:
- one canonical extraction pipeline
- one canonical ontology
- one validator aligned to that ontology

The dataset-building pipeline is intentionally still present and intentionally still separate. It has not been cleaned up in this pass.

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
  ...
```

## Quickstart

Requirements:
- Python 3.10+
- an OpenAI-compatible local endpoint such as LM Studio
- optionally Neo4j

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
./venv/bin/python -m unittest tests.test_pipeline_components tests.test_ontology_validator
```

## Dataset Pipeline Status

The dataset-building code remains in the repo and has not been cleaned up in this pass.

That was intentional:
- the extraction/runtime stack has been simplified first
- the dataset pipeline will be cleaned in a later pass

So if you see files related to:
- FinReflectKG projection
- stage 2 empty sampling
- stage 3 teacher augmentation
- batch repair

they are still present by design.
