# Plan: Ontology-Aligned Fine-Tuning Pipeline

## Goal

Build a strict, standardized training dataset for the repo's business-model ontology, then fine-tune a model that extracts only the knowledge graph elements we care about from SEC 10-K text.

The final model should:

- extract only the allowed node types and relations
- use canonical labels for `CustomerType`, `Channel`, and `RevenueModel`
- return empty outputs when nothing relevant is present

## Target Ontology

The target ontology is the one implemented in this repo:

- Node types: `Company`, `BusinessSegment`, `Offering`, `CustomerType`, `Channel`, `Place`, `RevenueModel`
- Relations: `HAS_SEGMENT`, `OFFERS`, `PART_OF`, `SERVES`, `OPERATES_IN`, `SELLS_THROUGH`, `PARTNERS_WITH`, `MONETIZES_VIA`

This is the only ontology the final training set should reinforce.

## High-Level Pipeline

1. Project FinReflectKG into the repo ontology with a deterministic Python script.
2. Keep only mapped triples we care about; discard all others.
3. Keep a provisional sampled set of candidate empty chunks so the model learns abstention.
4. Run base Gemma 4 26B only on the selected chunks to recover missing ontology slices.
5. Constrain teacher extraction to strict canonical labels only.
6. Merge projected triples and teacher triples at chunk level.
7. Recompute the final empty pool after teacher augmentation.
8. Deduplicate and validate the merged dataset with the reusable ontology validator.
9. Fine-tune on the final merged dataset.

## Prompt Workflow Direction

The prompt-based extractor in this repo is now organized around the same principle as the planned training data:

1. extract the structural graph skeleton first
2. normalize and enrich only after the skeleton is in place

In practical terms, the default `two-pass` extractor now follows this sequence:

1. `PASS 1`: extract `HAS_SEGMENT`, `OFFERS`, `PART_OF`, plus explicit `OPERATES_IN` and `PARTNERS_WITH`
2. `PASS 2`: validate the skeleton, fill obvious structural gaps, and add normalized `SERVES`, `SELLS_THROUGH`, and `MONETIZES_VIA`

## Core Principles

- No open-ended ontology expansion.
- No new labels outside the approved canonical vocabularies.
- No weak heuristic labeling from plain Python for semantic relations like `MONETIZES_VIA`.
- Projection must be deterministic.
- Teacher augmentation must be narrow and heavily constrained.
- Final supervision must be chunk-level, not triple-level.

## Stage 1: Deterministic Projection

### Input

- FinReflectKG dataset from Hugging Face or parquet export

### Unit of Processing

- Group rows by chunk identity, not by triple
- One training example = one chunk + all kept triples for that chunk

### Keep Rules

Keep a chunk if at least one source triple maps cleanly into the target ontology.

Within kept chunks:

- keep only triples whose subject type, object type, and relation map cleanly
- discard all unrelated triples
- do not invent missing triples

Also keep a sampled set of chunks whose mapped target is empty.

### Safe Projection Scope

Expected high-confidence projected areas:

- `ORG` -> `Company`
- `COMP` -> `Company`
- `SEGMENT` -> `BusinessSegment`
- `PRODUCT` -> `Offering`
- `GPE` -> `Place`

Expected high-confidence relation mappings:

- `Operates_In` -> `OPERATES_IN`
- `Partners_With` -> `PARTNERS_WITH`
- `Produces` -> `OFFERS` when target is clearly `Offering`
- `Introduces` -> `OFFERS` only when target is clearly `Offering`

Anything ambiguous should be dropped.

### Output of Stage 1

A chunk-level projected dataset where each example contains:

- strict instruction
- chunk text
- JSON output with only mapped triples
- metadata about source coverage and projection counts

## Stage 2: Empty Chunk Sampling

Purpose:

- teach the model to output empty extractions when no relevant ontology content exists
- create a candidate empty pool, not the final empty set

Rules:

- only sample empties from chunks that passed chunk-quality filters
- keep empties as true empty outputs
- cap empty examples so they do not dominate training
- treat these empties as provisional until teacher augmentation is complete

Initial recommendation:

- empty count between 20% and 40% of non-empty example count

## Stage 3: Teacher Augmentation with Base Gemma 4 26B

### Why

FinReflectKG projection is unlikely to cover the most important missing ontology slices:

- `CustomerType`
- `Channel`
- `RevenueModel`

These should be added by a strong base teacher model, not by Python heuristics.

### Teacher Scope

The teacher must recover only the missing ontology categories we care about.

Do not ask it to re-extract the whole graph.

Default teacher targets:

- `SERVES` with canonical `CustomerType`
- `SELLS_THROUGH` with canonical `Channel`
- `MONETIZES_VIA` with canonical `RevenueModel`

Optional later target if needed:

- missing `HAS_SEGMENT` / `PART_OF` cases

### Teacher Constraints

- output only repo ontology labels
- no labels outside the allowed canonical vocabularies
- no extra finance ontology from FinReflectKG
- return empty list if none of the missing ontology slices are present
- output strict JSON only
- do not require evidence snippets per triple in the default workflow
- do not rely on routing heuristics that skip chunks for a given relation family

## Stage 4: Canonical Vocabularies

These categories must be standardized and closed.

### RevenueModel

Fully closed enum:

- `subscription`
- `advertising`
- `licensing`
- `consumption-based`
- `hardware sales`
- `service fees`
- `royalties`
- `transaction fees`

No new revenue labels allowed.

### Channel

Closed vocabulary.

Initial list to define before teacher runs.

Examples may include:

- `direct sales`
- `resellers`
- `distributors`
- `retail`
- `online marketplace`
- `partners`
- `system integrators`
- `OEMs`

No new channel labels allowed.

### CustomerType

Closed vocabulary.

Initial list to define before teacher runs.

Examples may include:

- `consumers`
- `small businesses`
- `mid-market companies`
- `large enterprises`
- `developers`
- `IT professionals`
- `government agencies`
- `educational institutions`
- `healthcare organizations`
- `financial services firms`

No new customer labels allowed.

## Stage 5: Merge Policy

Merge at chunk level.

For each chunk:

- start from projected triples
- add teacher triples only for missing ontology slices
- deduplicate exact triple matches
- normalize direction and canonical labels
- keep provenance for teacher-added triples

### Empty Rebalancing Policy

After teacher augmentation:

- if a candidate empty chunk becomes non-empty, promote it into the positive set
- recompute the target empty count from the updated positive count
- resample additional candidate empties if needed until the target empty ratio is restored or no new candidates remain

### Conflict Policy

If a teacher triple conflicts with a projected triple:

- keep the projected triple by default
- log the conflict for review

If the teacher emits a triple outside the schema:

- drop it

If the teacher emits a non-canonical `CustomerType`, `Channel`, or `RevenueModel`:

- drop it

## Stage 6: Validation

Every final triple must pass:

- valid node types
- valid relation
- valid subject/object type pair for that relation
- canonical label compliance for `CustomerType`, `Channel`, `RevenueModel`
- non-empty subject/object strings

Recommended additional checks:

- entity string cleanup and whitespace normalization
- exact deduplication at merge time
- optional grounding checks for subject/object presence in chunk text

Implementation note:

- the validator should be reusable both as a Python module and as a CLI tool
- the same validator should be used in pipeline runs, dataset preparation, and teacher-label filtering

## Stage 7: Training Dataset Format

Training examples should be chunk-level instruction tuning records.

Example shape:

```json
{
  "instruction": "Extract the business-model knowledge graph from the following SEC 10-K text using the STRICT ONTOLOGY below. Output ONLY valid JSON.",
  "input": "<chunk_text>",
  "output": {
    "extraction_notes": "Ontology-aligned business-model extraction.",
    "triples": []
  },
  "metadata": {
    "source": "finreflectkg_projected+teacher",
    "empty_target": true,
    "projected_triple_count": 0,
    "teacher_triple_count": 0,
    "final_triple_count": 0
  }
}
```

## Stage 8: Fine-Tuning

Fine-tune on the final merged dataset only after projection, teacher augmentation, merge, and validation are complete.

Recommended training strategy:

- start with SFT
- use chunk-level JSON targets
- keep output formatting strict
- evaluate on a manually checked ontology-aligned validation set

## Deliverables

### Data Preparation

- `scripts/project_finreflectkg.py`
- `scripts/sample_empty_chunks.py`
- `scripts/augment_missing_ontology.py`
- `scripts/merge_training_data.py`
- `scripts/validate_training_data.py`

### Config

- `configs/ontology_mapping.json`
- `configs/customer_type_vocab.json`
- `configs/channel_vocab.json`
- `configs/revenue_model_vocab.json`

### Reusable Validation

- `src/ontology_config.py`
- `src/ontology_validator.py`

### Training

- `train/train_gemma_sft.py`
- `data/processed/train.jsonl`
- `data/processed/valid.jsonl`
- `data/processed/test.jsonl`

## Open Decisions

- Final closed vocabulary for `CustomerType`
- Final closed vocabulary for `Channel`
- Exact chunk identity fields in FinReflectKG exports
- Whether to require grounding checks for all entities or only teacher-added triples
- Empty-chunk sampling ratio

## Summary

The plan is to create one strict, standardized ontology-aligned dataset in two passes:

- deterministic projection for the triples FinReflectKG already supports well
- constrained teacher augmentation for the ontology slices it does not support well

Only after those are merged and validated do we fine-tune the model.
