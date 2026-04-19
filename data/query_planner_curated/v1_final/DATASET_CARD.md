# Query Planner Final Dataset

## Purpose

This artifact is the retained fine-tuning dataset for the query planner task: `question -> supervision_target`.
It is intended for training a model to emit the runtime-aligned planning target rather than raw Cypher or answer rows.

## Composition

- `train`: 8000 rows
- `validation`: 1200 rows
- `release_eval`: 1800 rows

Each row contains:

- `question`: the natural-language request
- `supervision_target`: the authoritative supervision object for training
- `target`: the runtime-facing plan envelope kept for compatibility
- `route_label` and `family`: routing and task-family metadata
- `gold_cypher`, `gold_params`, `gold_rows`: reference execution artifacts aligned to the runtime contract
- `metadata`: provenance and task-specific annotations

## Build Description

This repository treats `v1_final` as an agents-only dataset artifact.
The retained package was produced through agent-led review, rewriting, adjudication, and packaging workflows, and it should be used as the sole training source in this repo.

## Training Guidance

- Train against `supervision_target`.
- Treat `gold_cypher`, `gold_params`, and `gold_rows` as reference artifacts, not as the primary supervision target.
- Keep validation and release evaluation split boundaries intact.

## Quality Notes

- Rows are organized across `local_safe`, `strong_model_candidate`, and `refuse` routes.
- The package preserves held-out validation and release-eval splits for readiness checks.
- Some ranking prompts remain terser than the rest of the dataset, but the artifact is the approved final package.
