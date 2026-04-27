# Extraction Evaluation Contract

Status: draft for review.

This document defines the target used to evaluate extraction outputs against the manually created gold benchmark.

## Goal

Evaluate how well each extraction pipeline reproduces the manually annotated business-model knowledge graph for each company.

The evaluation target is the canonical ontology graph, not the Neo4j physical storage model and not the natural-language query stack.

## Compared Triple Shape

Every gold and predicted record is compared as a five-field triple:

```text
subject; subject_type; relation; object; object_type
```

Structured form:

```json
{
  "subject": "Apple",
  "subject_type": "Company",
  "relation": "HAS_SEGMENT",
  "object": "Americas",
  "object_type": "BusinessSegment"
}
```

The gold benchmark may be authored in the semicolon format:

```text
entity; entity_type; link; entity; entity_type
```

During parsing, these columns map to:

```text
subject; subject_type; relation; object; object_type
```

In the repo, raw gold benchmark CSVs should use this header:

```csv
subject,subject_type,relation,object,object_type
```

Raw CSV files live under:

```text
evaluation/benchmarks/dev/raw/
evaluation/benchmarks/test/raw/
```

Clean benchmark files are generated from the raw CSV files and live under:

```text
evaluation/benchmarks/dev/clean/
evaluation/benchmarks/test/clean/
```

The clean format should be JSONL, with one typed triple per line.

The raw-to-clean conversion script is:

```text
evaluation/scripts/prepare_gold.py
```

Run it with:

```bash
./venv/bin/python -m evaluation.scripts.prepare_gold --split all
```

The converter writes one clean `.jsonl` file per raw `.csv` file and a `manifest.json` for each converted split.

## Final Node Types

Allowed node types are:

- `Company`
- `BusinessSegment`
- `Offering`
- `CustomerType`
- `Channel`
- `Place`
- `RevenueModel`

## Final Relation Types

Allowed relation types are:

- `HAS_SEGMENT`
- `OFFERS`
- `SERVES`
- `OPERATES_IN`
- `SELLS_THROUGH`
- `PARTNERS_WITH`
- `MONETIZES_VIA`

## Valid Relation Schema

The valid relation schema is:

| Relation | Subject type | Object type |
| --- | --- | --- |
| `HAS_SEGMENT` | `Company` | `BusinessSegment` |
| `OFFERS` | `Company` | `Offering` |
| `OFFERS` | `BusinessSegment` | `Offering` |
| `OFFERS` | `Offering` | `Offering` |
| `SERVES` | `BusinessSegment` | `CustomerType` |
| `OPERATES_IN` | `Company` | `Place` |
| `SELLS_THROUGH` | `BusinessSegment` | `Channel` |
| `SELLS_THROUGH` | `Offering` | `Channel` |
| `PARTNERS_WITH` | `Company` | `Company` |
| `MONETIZES_VIA` | `Offering` | `RevenueModel` |

## Closed Canonical Labels

These node types use closed canonical labels:

- `CustomerType`
- `Channel`
- `RevenueModel`

Gold triples and predicted triples must use the canonical label text from `src/ontology/ontology.json`.

Examples:

- use `large enterprises`, not `enterprise customers`
- use `direct sales`, not `sales force`
- use `subscription`, not `subscriptions`

## Gold Authority

Gold triples are authoritative.

The evaluator should not attempt to correct, reject, or reinterpret gold triples. The raw benchmark is manually reviewed before evaluation, and the clean benchmark should preserve those triples with only mechanical format conversion.

## Predicted Triples

Predicted triples should be loaded from:

```text
outputs/<company>/<pipeline>/latest/resolved_triples.json
```

This means evaluation compares the same post-resolution, post-validation graph that the runtime would load into Neo4j.

For failed or incomplete runs, the evaluation should skip the run and report it as missing, not score it as zero unless we intentionally choose that policy later.

The evaluation script is:

```text
evaluation/scripts/evaluate.py
```

Run all companies in a split for one selected pipeline with:

```bash
./venv/bin/python -m evaluation.scripts.evaluate --pipeline zero-shot --split dev
```

Typical full-split runs:

```bash
./venv/bin/python -m evaluation.scripts.evaluate --pipeline zero-shot --split dev
./venv/bin/python -m evaluation.scripts.evaluate --pipeline zero-shot --split test
./venv/bin/python -m evaluation.scripts.evaluate --pipeline memo-only --split dev
./venv/bin/python -m evaluation.scripts.evaluate --pipeline memo-only --split test
./venv/bin/python -m evaluation.scripts.evaluate --pipeline analyst --split dev
./venv/bin/python -m evaluation.scripts.evaluate --pipeline analyst --split test
```

Run one selected company and one selected pipeline with:

```bash
./venv/bin/python -m evaluation.scripts.evaluate --pipeline analyst --company microsoft
```

Example cherry-picked runs:

```bash
./venv/bin/python -m evaluation.scripts.evaluate --pipeline zero-shot --company microsoft
./venv/bin/python -m evaluation.scripts.evaluate --pipeline memo-only --company microsoft
./venv/bin/python -m evaluation.scripts.evaluate --pipeline analyst --company microsoft
```

Split results should be written under:

```text
evaluation/results/<pipeline>/<split>/
```

Cherry-picked results should be written under:

```text
evaluation/results/cherry_picked/<pipeline>/<company>/
```

If the target results folder already contains files, the evaluator should ask before overwriting:

```text
There are already files in the results folder <path>. Proceeding with a new evaluation is going to overwrite them. Do you want to proceed? [Y/n]
```

If the answer is `n` or `no`, no evaluation should be performed and the existing files should be left unchanged.

For intentional reruns that should overwrite existing results without an interactive prompt, use:

```bash
./venv/bin/python -m evaluation.scripts.evaluate --pipeline zero-shot --split dev --yes
```

## Strict Normalization

Strict matching should normalize only mechanical surface differences.

For entity names:

- apply Unicode NFKC normalization
- trim whitespace
- strip surrounding quote characters
- collapse repeated whitespace
- compare with casefolded keys
- normalize curly apostrophes and dash variants in comparison keys

For `Place` values:

- apply the same mechanical entity-name normalization as other entity names.

For `CustomerType`, `Channel`, and `RevenueModel`:

- require exact canonical labels after mechanical cleanup.

For relation names and node types:

- require exact labels after trimming.

## Strict Match Definition

A predicted triple is a strict true positive if, after strict normalization, all five fields match a gold triple:

```text
subject_key
subject_type
relation
object_key
object_type
```

False positives are valid predicted triples that do not match any gold triple.

False negatives are valid gold triples that are not matched by any predicted triple.

Metrics:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

If a denominator is zero, the evaluation script should handle it explicitly and consistently.

## Alias-Assisted Relaxed Evaluation

Strict matching is the primary metric, but it may penalize harmless naming differences.

The relaxed metric should use only manually approved aliases.

Automatic fuzzy matching may be used only to generate candidate aliases for review.

Examples of possible aliases:

```json
{
  "Company": {
    "alphabet": "Alphabet Inc."
  },
  "BusinessSegment": {
    "google services segment": "Google Services"
  },
  "Offering": {
    "google cloud platform": "Google Cloud"
  }
}
```

Alias rules:

- aliases are scoped by node type
- aliases map alternative names to the chosen benchmark name
- aliases must be explicitly saved in a versioned alias map
- aliases should not change relation names or node types
- aliases should not collapse genuinely different entities
- aliases should be reviewed before relaxed metrics are reported

## Relaxed Match Definition

A predicted triple is an alias-normalized true positive if:

1. strict normalization is applied
2. approved aliases are applied to subject and object names by node type
3. all five fields match a gold triple

Relaxed metrics should be reported separately from strict metrics.

Recommended labels:

- `Strict`
- `Alias-normalized`

## Metric Scope For The First Evaluator

The first evaluator should keep the metric surface small:

- overall
- by pipeline
- by company
- by company and pipeline

Recommended pipeline comparison:

- `zero-shot`
- `memo-only`
- `analyst`

Relation-specific and entity-specific metrics can be added later once the core evaluator is stable.

## Required Error Reports

For each evaluated company and pipeline, write:

- matched triples
- false positives
- false negatives
- invalid predicted triples, if any
- duplicate predicted triples, if any
- possible alias candidates

The false positive and false negative reports are the main input for error analysis.

## Presentation Interpretation

Use strict metrics as the cleanest objective score:

```text
Strict F1 measures exact graph agreement.
```

Use alias-normalized metrics as a fairer semantic score:

```text
Alias-normalized F1 measures graph agreement after adjudicated naming alignment.
```

The final presentation should clearly state that alias-normalized metrics depend only on approved aliases, not uncontrolled fuzzy matching.
