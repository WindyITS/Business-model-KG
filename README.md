# SEC 10-K Business Model Knowledge Graph Pipeline

A local-first Python pipeline that ingests SEC 10-K filings, extracts a structured business-model knowledge graph using a local LLM, and loads the result into Neo4j. The objective is getting this done at scale, to make it usable in production.

The base pipeline is functional with a sufficiently capable model (~30B parameters), but extraction quality is still being refined through prompt engineering. Currently, one knowledge graph per company is extracted per run.

**Planned improvements:**

- Token-efficient inference and higher-recall extraction with the strong local model
- Multi-company runs that produce an interconnected, cross-company knowledge graph
- Entity resolution workflow to deduplicate and merge entities across companies

**If resources allow:**

- A labeled NER and relation extraction (RE) dataset suitable for fine-tuning smaller LLMs
- Specialized small LLMs capable of running this extraction pipeline at scale

## How It Works

```text
10-K .txt file
      │
      ▼
 chunker.py (if chunked-mode enabled)
      │  semantic passage chunking — heading/item heuristics + token budgets
      ▼
 llm_extractor.py
      │  chunked (default), zero-shot, or incremental extraction mode
      ▼
 entity_resolver.py
      │  whitespace normalization + canonical deduplication without destroying casing
      ▼
 resolved_triples.json
      │
      ├──▶ evaluate_graph.py  ··· compare against gold set
      │
      └──▶ neo4j_loader.py ──▶ Neo4j
```

### Extraction Modes

| Mode                        | Description                                                                                                                                                     |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **chunked** (default) | Text is split into semantic passages; each chunk is extracted independently with a rolling memory of previously seen entities and triples to avoid duplication. |
| **zero-shot**         | The entire document is sent in a single prompt. Fast but context-limited.                                                                                       |
| **incremental**       | The full document is sent once; the model then walks through it section by section using a cursor maintained in the conversation history.                       |

## Ontology

Here you can find the entity types and the relations that are taken onto consideration by the model. The goal would be to build a strong pipeline for this ontology but, moreover, for every personalized business-based ontology.

### Node Types

| Type                | Description                                                                                                                                               |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Company`         | The primary reporting entity, named partners, or named competitors                                                                                        |
| `BusinessSegment` | Explicitly named formal reporting segments                                                                                                                |
| `Offering`        | Named products, services, brands, or platforms                                                                                                            |
| `CustomerType`    | Granular buyer profiles copied verbatim from the filing                                                                                                   |
| `Channel`         | Distribution or sales methods                                                                                                                             |
| `Place`           | Explicit geographical markers (countries, regions, cities)                                                                                                |
| `RevenueModel`    | One of:`subscription`, `advertising`, `licensing`, `consumption-based`, `hardware sales`, `service fees`, `royalties`, `transaction fees` |

### Relations

| Relation          | Valid Subject → Object                                                            |
| ----------------- | ---------------------------------------------------------------------------------- |
| `HAS_SEGMENT`   | Company → BusinessSegment                                                         |
| `OFFERS`        | Company → Offering, BusinessSegment → Offering                                   |
| `PART_OF`       | Offering → BusinessSegment                                                        |
| `SERVES`        | Company → CustomerType, Offering → CustomerType                                  |
| `OPERATES_IN`   | Company → Place, BusinessSegment → Place                                         |
| `SELLS_THROUGH` | Company → Channel, Offering → Channel                                            |
| `PARTNERS_WITH` | Company → Company                                                                 |
| `SUPPLIED_BY`   | Company → Company, Offering → Company                                            |
| `MONETIZES_VIA` | Company → RevenueModel, BusinessSegment → RevenueModel, Offering → RevenueModel |

## Prerequisites

- Python 3.10+
- Docker (for Neo4j)
- [LM Studio](https://lmstudio.ai) with a model loaded and the local server running at `http://localhost:1234/v1`
  - Current model used: Gemma 4 26B A4B

## Setup

Run the following:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Start Neo4j

```bash
docker compose up -d
```

Neo4j Browser: `http://localhost:7474`

Default credentials are set by `NEO4J_AUTH` in `docker-compose.yml` and fall back to `neo4j / password`.

## Run the Pipeline

### Chunked mode (default)

At the moment is also the one that performs the worse. The big problem I'm encountering is attention breaking between chunks.

```bash
python src/main.py data/palantir_10k.txt
```

### Zero-shot mode

Currently the best solution, but not yet at the needed level.

```bash
python src/main.py data/palantir_10k.txt --zero-shot
```

### Incremental mode

Token-hungry and often bugs due to prompt engineering mistakes.

```bash
python src/main.py data/palantir_10k.txt --incremental
```

### Skip Neo4j (extraction only)

Useful for debugging or evaluation when Neo4j is not running.

```bash
python src/main.py data/microsoft_10k.txt --skip-neo4j
```

## CLI Reference

| Argument             | Default                      | Description                            |
| -------------------- | ---------------------------- | -------------------------------------- |
| `file_path`        | —                           | Path to the input filing `.txt` file |
| `--zero-shot`      | off                          | Single-prompt full-document mode       |
| `--incremental`    | off                          | Cursor-based iterative mode            |
| `--max-retries`    | `3`                        | LLM retries per chunk or iteration     |
| `--max-iterations` | `20`                       | Iteration cap for incremental mode     |
| `--skip-neo4j`     | off                          | Write artifacts only, skip graph load  |
| `--output-dir`     | `outputs`                  | Root directory for run artifacts       |
| `--base-url`       | `http://localhost:1234/v1` | LM Studio API endpoint                 |
| `--model`          | `local-model`              | Model identifier sent to the API       |
| `--neo4j-uri`      | `bolt://localhost:7687`    | Neo4j Bolt URI                         |
| `--neo4j-user`     | `neo4j`                    | Neo4j username                         |
| `--neo4j-password` | `password`                 | Neo4j password                         |

## Run Artifacts

Each run writes a timestamped folder under `outputs/`:

```text
outputs/<stem>_<mode>_<timestamp>/
├── run_summary.json       # status, counts, timing
├── chunks.json            # all chunks with character/word counts
├── extractions.json       # raw per-chunk LLM output
└── resolved_triples.json  # deduplicated, canonical triples
```

Failed runs are fully inspectable from these artifacts.

## Evaluation

A manually curated gold set for Microsoft lives at `data/microsoft_eval_gold.json`.

Compare any predicted triples file against it:

```bash
python src/evaluate_graph.py compare outputs/<run>/resolved_triples.json data/microsoft_eval_gold.json
```

Inspect the live Neo4j graph:

```bash
python src/evaluate_graph.py dump-neo4j
```

The evaluator reports precision, recall, F1, and optionally prints the missing and extra triples.

## A Note on Chunking Noisy 10-K Text

Raw `.txt` 10-K filings are typically noisy: page headers, OCR artifacts, broken line wraps, and filing boilerplate make simple section regexes brittle. The chunker uses heading-aware passage splitting with token budgets and overlap as the default strategy. The more robust long-term approach is to ingest the original EDGAR HTML and parse structure before converting to plain text, keeping this chunker as the fallback for already-normalized text.

## Project Structure

```text
kg-v0/
├── data/
│   ├── dummy_test_10k.txt
│   ├── manual.txt
│   ├── microsoft_10k.txt
│   ├── microsoft_eval_gold.json
│   └── palantir_10k.txt
├── outputs/
├── src/
│   ├── chunker.py
│   ├── entity_resolver.py
│   ├── evaluate_graph.py
│   ├── llm_extractor.py
│   ├── load_custom_graph.py
│   ├── load_manual_graph.py
│   ├── main.py
│   └── neo4j_loader.py
├── tests/
├── docker-compose.yml
└── requirements.txt
```
