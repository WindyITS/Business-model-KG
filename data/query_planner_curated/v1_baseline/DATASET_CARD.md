# Query Planner Curated Baseline

This package is a frozen generator output used as the baseline for question-first curation.
It preserves the training row schema but is not the final training artifact.

## Package Contents

- Split JSONL files generated from the runtime-aligned planner dataset builder.
- Split-specific synthetic graph files.
- `manifest.json` summarizing split composition.
- `workflow/` shards and role assignments for row-by-row review.
- `curation_log.jsonl`, intentionally empty at baseline freeze.

## Identity Model

Rows are treated as immutable curation units keyed by `(split_name, case_id, template_id, variant_id)`.

## Training Policy

Use the curated final package rather than this baseline for fine-tuning.
