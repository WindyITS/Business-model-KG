# Business Model KG

A pipeline for turning SEC 10-K filings into structured business-model knowledge graphs using a custom ontology, local LLMs, and Neo4j.

## Mission

LLMs are capable of extracting structured information from unstructured text. The hard part is not extraction itself but is doing it consistently enough that the output is actually comparable and usable at scale. This project is a proof of concept for that idea, applied to a domain where the gap is concrete and the stakes are real: public company filings.

Bloomberg, FactSet, and others have built extensive structured databases from SEC 10-K filings. They are excellent at the numerical layer: revenue by segment, margins, headcount, capex. This data is machine-readable, comparable across companies, and deeply integrated into analyst workflows. Doing so they leave a gap: Item 1 of every 10-K, the Business section, is where a company describes in prose how it actually makes money: what it sells, who it sells to, how it distributes, how it monetizes, which companies it depends on. This semantic layer is almost entirely absent from structured databases. Analysts who need to answer questions like "which public companies monetize primarily via transaction fees and sell through marketplaces?" have no machine-readable source. They read filings manually or rely on ad-hoc keyword search.

There are roughly 10,000 public US companies filing annually. The business model information is public, standardized in form, and updated every year. A machine-readable graph of it does not exist based on what I know.

A fine-tuned extractor that runs reliably over the full 10-K universe and produces a structured, ontology-aligned, cross-company-comparable knowledge graph of business models is the exact mission of this project. Not a noisy dump of whatever the LLM inferred, but a validated graph constrained to a strict schema, with canonical labels that mean the same thing across filings.

---

This is `v0`: the extraction pipeline works end-to-end once provided with the Item 1 Business section of a company's 10-K, and the first three dataset-building stages are implemented in the repo. The next phase is turning that data pipeline into a balanced fine-tuning set for a dedicated extractor, fine-tune such extractor and check wether it performs better on extracting information.

The next data-building stage is based on:

- the paper [FinReflectKG: Agentic Construction and Evaluation of Financial Knowledge Graphs](https://arxiv.org/abs/2508.17906)
- the dataset [domyn/FinReflectKG on Hugging Face](https://huggingface.co/datasets/domyn/FinReflectKG)

## The Problem

Annual reports describe how companies make money in exhaustive detail, but the information is buried in pages of inconsistent, noisy prose. Comparing business models across companies means reading every filing manually and translating arbitrary language into comparable concepts.

The idea is to automate that. Extracting a structured, standardized representation of a company's business model from its 10-K: what it offers, who it serves, how it sells, how it monetizes, which segments it operates through, and which companies it partners with or depends on.

The key design choice is that the output is not free-form but it is constrained to a strict ontology. Every extracted triple must use an approved node type, an approved relation, and (for customer, channel, and revenue concepts) an approved canonical label. The goal is a graph you can actually compare across filings, not just a noisy extraction of whatever the LLM felt like saying.

## What's Built

A full extraction pipeline that takes a `.txt` 10-K filing and produces a validated, ontology-aligned graph in Neo4j.

**Pipeline stages:**

```
10-K .txt file
      │
      ▼
chunker.py          -> (if chunking-mode is selected) heading-aware passage splitting with token budgets
      │
      ▼
llm_extractor.py    two-pass LLM extraction (default), plus chunked, zero-shot,
      │             incremental, and reflection modes
      ▼
entity_resolver.py  surface-form cleanup and canonical basic deduplication
      │
      ▼
ontology_validator.py   schema checks, canonical label enforcement, dedup
      │
      ▼
resolved_triples.json
      │
      ├──▶ evaluate_graph.py    compare against a gold set
      │
      └──▶ neo4j_loader.py ──▶ Neo4j
```

**The default mode is `two-pass`:**

It is also the only extraction mode whose prompts have been optimized enough to produce relatively good output without any fine-tuning. The other modes are still useful for experimentation, but they should be treated as exploratory baselines rather than equally tuned production paths.

1. Extract the structural skeleton: segments, offerings, geography, and partners
2. Validate and enrich with normalized customers, channels, and revenue models

## The Ontology

Seven node types, eight relations, three closed vocabularies.

| Node type           | What it represents                                    |
| ------------------- | ----------------------------------------------------- |
| `Company`         | The reporting company or any named external company   |
| `BusinessSegment` | A formal internal reporting segment or division       |
| `Offering`        | A named product, platform, service, or subscription   |
| `CustomerType`    | A canonical customer category (closed vocab)          |
| `Channel`         | A canonical sales/distribution channel (closed vocab) |
| `Place`           | A geographic entity                                   |
| `RevenueModel`    | A canonical monetization model (closed vocab)         |

| Relation          | Valid subject → object                                               |
| ----------------- | --------------------------------------------------------------------- |
| `HAS_SEGMENT`   | `Company` → `BusinessSegment`                                    |
| `OFFERS`        | `Company` or `BusinessSegment` → `Offering`                    |
| `PART_OF`       | `Offering` → `BusinessSegment`                                   |
| `SERVES`        | `Company` or `Offering` → `CustomerType`                       |
| `OPERATES_IN`   | `Company` or `BusinessSegment` → `Place`                       |
| `SELLS_THROUGH` | `Company` or `Offering` → `Channel`                            |
| `PARTNERS_WITH` | `Company` → `Company`                                            |
| `MONETIZES_VIA` | `Company`, `BusinessSegment`, or `Offering` → `RevenueModel` |

Canonical `CustomerType` labels: `consumers`, `small businesses`, `mid-market companies`, `large enterprises`, `developers`, `IT professionals`, `government agencies`, `educational institutions`, `healthcare organizations`, `financial services firms`, `manufacturers`, `retailers`.

Canonical `Channel` labels: `direct sales`, `online`, `retail`, `distributors`, `resellers`, `OEMs`, `system integrators`, `managed service providers`, `marketplaces`.

Canonical `RevenueModel` labels: `subscription`, `advertising`, `licensing`, `consumption-based`, `hardware sales`, `service fees`, `royalties`, `transaction fees`.

No labels outside these vocabularies are allowed. If a concept doesn't fit, it gets dropped.

Full ontology spec: [`docs/ontology.md`](./docs/ontology.md)

## What Comes Next

The prompt-based extractor is a baseline, not the end state. The broader goal is a fine-tuned extractor that is cheaper, more reliable, and more standardized.

The data-building pipeline now follows four stages:

1. **Project FinReflectKG** into this ontology with a deterministic mapping script, keep only triples that fit cleanly and discard everything else
2. **Sample candidate empty chunks** so the fine-tuned model can later learn to abstain when nothing relevant is present
3. **Teacher augmentation** done by running a strong base model (Gemma 4 26B for the moment) only on the ontology slices FinReflectKG does not cover well: `SERVES`, `SELLS_THROUGH`, `MONETIZES_VIA`
4. **Finalize the merged dataset** by validating the teacher output, promoting empty chunks that are no longer empty, and rebalancing the final empty ratio

That's why this repo contains both a working extractor and the ontology + validation infrastructure: the same schema and validator that constrain prompts today will constrain the training data tomorrow.

Full roadmap: [`docs/plan.md`](./docs/plan.md)

## Stage 1 Dataset Projection

Stage 1 is now implemented as a deterministic projection pipeline that maps the safe overlap between FinReflectKG and this repo's ontology.

Run it directly against the Hugging Face dataset:

```bash
python scripts/project_finreflectkg_stage1.py --limit-chunks 500
```

This writes:

- `outputs/finreflectkg_stage1/projected_examples.jsonl`
- `outputs/finreflectkg_stage1/projection_report.json`

If you want to run stage 1 against a local shard instead of the remote dataset:

```bash
python scripts/project_finreflectkg_stage1.py \
  --parquet-file data/external/finreflectkg/data/train-00000-of-00103.parquet \
  --no-streaming
```

The ignored `data/external/` area is the intended home for downloaded dataset assets such as:

- `data/external/huggingface/` for Hugging Face cache files
- `data/external/finreflectkg/` for locally downloaded parquet shards or the dataset card

## Stage 2 Candidate Empty Sampling

Stage 2 builds a provisional pool of candidate empties by sampling chunks whose mapped target is empty under the stage-1 deterministic projection rules.

Run it after stage 1:

```bash
python scripts/sample_empty_chunks.py \
  --projected-jsonl outputs/finreflectkg_stage1/projected_examples.jsonl \
  --empty-ratio 0.3
```

This writes:

- `outputs/finreflectkg_stage2/empty_examples.jsonl`
- `outputs/finreflectkg_stage2/training_examples.jsonl`
- `outputs/finreflectkg_stage2/empty_sampling_report.json`

The sampling is deterministic and keeps only chunks that pass basic quality filters such as minimum length.

Important: Stage 2 empties are only candidates. They are checked again in Stage 3, because some of them may become non-empty once the teacher recovers missing `SERVES`, `SELLS_THROUGH`, or `MONETIZES_VIA` triples.

## Stage 3 Teacher Augmentation

Stage 3 runs a strong base model over the Stage 1 positives and the Stage 2 candidate empties, but only for the ontology slices that deterministic projection does not cover well:

- `SERVES`
- `SELLS_THROUGH`
- `MONETIZES_VIA`

The teacher is constrained to the repo ontology and the closed canonical vocabularies. It does not re-extract the whole graph.

Stage 3 also supports an optional third source pool: non-Stage1 chunks selected by deterministic lexical triggers and a narrative-prose gate. This is meant to surface chunks where one of the missing relations may appear on its own, without opening the search over the entire filing universe.

Run it after Stage 1 and Stage 2:

```bash
python scripts/augment_finreflectkg_stage3.py \
  --projected-jsonl outputs/finreflectkg_stage1/projected_examples.jsonl \
  --candidate-empty-jsonl outputs/finreflectkg_stage2/empty_examples.jsonl \
  --relation-trigger-count 50 \
  --empty-ratio 0.3 \
  --base-url http://localhost:1234/v1 \
  --model local-model
```

This writes:

- `outputs/finreflectkg_stage3/augmented_positive_examples.jsonl`
- `outputs/finreflectkg_stage3/augmented_empty_candidates.jsonl`
- `outputs/finreflectkg_stage3/augmented_relation_trigger_candidates.jsonl`
- `outputs/finreflectkg_stage3/final_positive_examples.jsonl`
- `outputs/finreflectkg_stage3/final_empty_examples.jsonl`
- `outputs/finreflectkg_stage3/training_examples.jsonl`
- `outputs/finreflectkg_stage3/stage3_report.json`

Stage 3 is where the final training set is produced. If a candidate empty chunk becomes non-empty after teacher augmentation, it is promoted into the positive set and replacement empties are sampled until the target empty ratio is restored or no additional candidates are available.

Important Stage 3 behavior:

- trigger-selected chunks are filtered in deterministic code before teacher inference
- only chunks that look like narrative business prose are eligible for the trigger pool
- teacher-empty trigger candidates are not reused as training empties
- outputs under `outputs/` are intentionally ignored by git

## Quickstart

**Requirements:**

- Python 3.10+
- Docker
- [LM Studio](https://lmstudio.ai) (or any OpenAI-compatible local endpoint)
- A model served at `http://localhost:1234/v1`, current baseline is Gemma 4 26B

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the test suite:

```bash
/Users/giuliosighieri/Desktop/kg-v0/venv/bin/python -m unittest discover -s tests
```

Start Neo4j:

```bash
docker compose up -d
```

Neo4j Browser at `http://localhost:7474`, default credentials `neo4j / password`.

Run the pipeline on a 10-K:

```bash
python src/main.py data/microsoft_10k.txt
```

Skip Neo4j if you just want the extraction artifacts:

```bash
python src/main.py data/microsoft_10k.txt --skip-neo4j
```

---

## Extraction Modes

| Mode            | How it works                                                                                                      |
| --------------- | ----------------------------------------------------------------------------------------------------------------- |
| `two-pass`    | **Default.** Pass 1 builds the structural skeleton. Pass 2 validates and enriches with normalized concepts. |
| `chunked`     | Processes the document chunk by chunk with rolling memory.                                                        |
| `zero-shot`   | Single prompt over the full document. Simpler, less controlled.                                                   |
| `incremental` | Cursor-based multi-turn extraction. Experimental.                                                                 |
| `reflection`  | Two-pass followed by a graph-completion review pass. Experimental.                                                |

---

## CLI Reference

| Argument             | Default                      | Description                                   |
| -------------------- | ---------------------------- | --------------------------------------------- |
| `file_path`        | required                     | Path to the 10-K `.txt` file                |
| `--chunked`        | off                          | Chunk-by-chunk extraction with rolling memory |
| `--zero-shot`      | off                          | Single-prompt full-document extraction        |
| `--incremental`    | off                          | Cursor-based iterative extraction             |
| `--reflection`     | off                          | Two-pass + final review pass                  |
| `--max-retries`    | `3`                        | LLM retries per call                          |
| `--max-iterations` | `20`                       | Iteration cap for incremental mode            |
| `--skip-neo4j`     | off                          | Write artifacts only, skip graph loading      |
| `--clear-neo4j`    | off                          | Clear the whole Neo4j database before loading |
| `--output-dir`     | `outputs`                  | Root directory for run artifacts              |
| `--base-url`       | `http://localhost:1234/v1` | Local LLM API endpoint                        |
| `--model`          | `local-model`              | Model identifier sent to the API              |
| `--neo4j-uri`      | `bolt://localhost:7687`    | Neo4j Bolt URI                                |
| `--neo4j-user`     | `neo4j`                    | Neo4j username                                |
| `--neo4j-password` | `password`                 | Neo4j password                                |

---

## Validation and Evaluation

Every triple is validated before it enters the graph. The validator checks:

- node types are allowed
- relation names are allowed
- subject/object type pairing is valid for that relation
- canonical labels are respected for `CustomerType`, `Channel`, and `RevenueModel`
- duplicates are removed

Run it directly on any artifact:

```bash
python src/ontology_validator.py outputs/<run>/resolved_triples.json
```

With optional source-text grounding checks:

```bash
python src/ontology_validator.py outputs/<run>/resolved_triples.json \
  --source-text-path data/microsoft_10k.txt \
  --require-text-grounding
```

A manually curated Microsoft gold set lives at [`data/microsoft_eval_gold.json`](./data/microsoft_eval_gold.json). Compare any predicted graph against it:

```bash
python src/evaluate_graph.py compare outputs/<run>/resolved_triples.json data/microsoft_eval_gold.json
python src/evaluate_graph.py dump-neo4j
```

## Repository Structure

```
kg-v0/
├── configs/
│   └── ontology.json           machine-readable ontology used by prompts and validator
├── docs/
│   ├── ontology.md             full ontology specification
│   └── plan.md                 fine-tuning roadmap
├── data/
│   ├── manual/                 ignored local comparison artifacts (kept out of git by default)
│   ├── external/               ignored local dataset cache / parquet area
│   ├── dummy_test_10k.txt
│   ├── microsoft_10k.txt
│   ├── microsoft_eval_gold.json    manually curated gold set
│   └── palantir_10k.txt
├── scripts/
│   ├── load_custom_graph.py
│   ├── load_manual_graph.py
│   ├── augment_finreflectkg_stage3.py
│   ├── project_finreflectkg_stage1.py
│   └── sample_empty_chunks.py
├── src/
│   ├── chunker.py              heading-aware passage chunker
│   ├── entity_resolver.py      surface-form cleanup and deduplication
│   ├── evaluate_graph.py       gold-set comparison and Neo4j dump
│   ├── finreflectkg_projection.py   dataset projection + empty sampling helpers
│   ├── finreflectkg_stage3.py  teacher augmentation + dataset finalization
│   ├── llm_extractor.py        all extraction modes
│   ├── main.py                 pipeline entry point
│   ├── neo4j_loader.py         graph loading
│   ├── ontology_config.py      structured ontology config
│   └── ontology_validator.py   reusable triple validator
├── tests/
│   ├── test_finreflectkg_projection.py
│   ├── test_ontology_validator.py
│   ├── test_stage3_augmentation.py
│   ├── test_stage2_empty_sampling.py
│   └── test_pipeline_components.py
├── docker-compose.yml
└── requirements.txt
```
