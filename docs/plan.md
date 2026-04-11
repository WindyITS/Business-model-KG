# Roadmap: Ontology Review -> Benchmark -> Dataset Fix -> Fine-Tuning

## Current State

As of April 11, 2026, the repo has three reflection-based extraction pipelines:

- `chat-two-pass-reflection`
- `two-pass-reflection`
- `incremental-reflection`

The current best prompt-only result is the `chat-two-pass-reflection` Microsoft run with clean final-output formatting and zero ontology rejections at the final-output audit stage. On the data side, the deterministic projection, empty sampling, teacher augmentation, and `OPERATES_IN` cleanup pipeline are already implemented, and production batches `1` and `2` have completed.

That means the next work is no longer "make anything run." The next work is to make the whole system coherent:

1. settle the ontology
2. build a benchmark that can compare pipelines
3. repair the dataset generation flow under the settled ontology
4. fine-tune and evaluate the model

The active workstream is **ontology review**.

## Goal

Build a reliable business-model KG system where:

- the ontology is stable enough to supervise extraction and fine-tuning
- the benchmark tells us which pipeline is actually better
- the dataset reflects the final ontology cleanly
- the fine-tuned model is evaluated against that same benchmark

## Workstream 1: Ontology Review

This is the first priority because it controls everything downstream: prompt design, dataset generation, validation, benchmark scoring, and fine-tuning targets.

### Review Objectives

1. Review the hierarchy between `Company`, `BusinessSegment`, and `Offering`
2. Review which relations are truly needed versus redundant
3. Review which subject-type -> relation -> object-type pairs are too permissive
4. Review whether additional relations are needed before we freeze the benchmark and dataset

### Main Questions To Resolve

#### Hierarchy and redundancy

- When should we keep `Company -> OFFERS -> Offering` if we already have:
  - `Company -> HAS_SEGMENT -> BusinessSegment`
  - `BusinessSegment -> OFFERS -> Offering`
  - `Offering -> PART_OF -> BusinessSegment`
- Should company-level semantic facts such as `SERVES`, `SELLS_THROUGH`, or `MONETIZES_VIA` be kept when the text supports only a specific offering or segment?
- Do we need stricter preference rules for the most specific level supported by the text?

#### Relation coverage

- Is the current set of eight relations sufficient for the business-model questions we care about?
- Are there meaningful business-model relations still missing?
- Are any current relations too broad or semantically overloaded?

#### Geography and partner semantics

- Should macro-regions like `Americas`, `APAC`, and `EMEA` remain valid `Place` nodes, or should geography be constrained more tightly?
- Is `PARTNERS_WITH` too broad for OEM, reseller, platform, and technology partner language?

### Expected Outputs

- updated ontology spec in [`docs/ontology.md`](./ontology.md)
- any needed schema changes in [`configs/ontology.json`](../configs/ontology.json)
- corresponding validator changes in [`src/ontology_validator.py`](../src/ontology_validator.py)
- explicit extraction guidance for specificity and redundancy handling

## Workstream 2: Benchmark

Once the ontology is stable enough, build a benchmark to compare pipelines fairly.

### Benchmark Objectives

1. Compare `chat-two-pass-reflection`, `two-pass-reflection`, and `incremental-reflection`
2. Measure not only graph quality, but also formatting robustness and ontology compliance
3. Provide a stable gate for both prompt iteration and fine-tuning

### First Benchmark Scope

Start with the Microsoft filing, because it is already the main manual reference case in the repo. Then expand to a small set of diverse companies.

### Metrics To Track

- triple precision / recall / F1 against gold data where available
- relation-level precision / recall / F1
- raw triple count
- malformed triple count
- ontology-rejected triple count
- duplicate triple count
- kept triple count
- resolved triple count
- runtime and call count

### Benchmark Outputs

- reproducible run commands per pipeline
- a structured comparison report
- clear pass/fail criteria for future prompt or model changes

## Workstream 3: Dataset Fix Pipeline

Only after the ontology and benchmark are defined should we repair the dataset pipeline.

### Objectives

1. Re-check the projected and teacher-generated data under the final ontology
2. Identify which stages need regeneration
3. Improve data quality before fine-tuning

### Expected Tasks

- validate all current batches against the reviewed ontology
- inspect relation coverage gaps
- inspect over-represented noisy patterns
- regenerate affected stages or batches where needed
- keep audit counts for malformed, rejected, and promoted examples

### Main Principle

Do not fine-tune on a dataset that still reflects an ontology we are about to change.

## Workstream 4: Fine-Tuning

Fine-tuning comes last in this sequence.

### Objectives

1. Train on the cleaned dataset aligned to the final ontology
2. Evaluate on the benchmark, not only on training-style samples
3. Decide whether the fine-tuned model replaces prompt-only extraction for production

### Fine-Tuning Requirements

- stable ontology
- stable benchmark
- cleaned training set
- held-out validation/test slices
- relation-level evaluation

## Order of Execution

The intended order is:

1. **Ontology review**
2. **Benchmark**
3. **Dataset fixes**
4. **Fine-tuning**

That ordering is intentional. The ontology defines the target, the benchmark defines success, the dataset must match both, and only then does fine-tuning make sense.
