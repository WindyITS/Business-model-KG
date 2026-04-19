# Query Planner Curated Dataset

## Purpose

This artifact is the final fine-tuning dataset for the query planner task: `question -> supervision_target -> deterministic compiler`.
It is intended for training a small local model to emit the runtime-aligned planning target rather than Cypher or answer rows.

## Row Schema

- `case_id`, `template_id`, `variant_id`: immutable identity fields inherited from the baseline freeze.
- `question`: curated natural-language prompt.
- `target`: full runtime-facing plan envelope for the row.
- `supervision_target`: canonical training target; this is the field to supervise against.
- `route_label`, `family`: task routing and family metadata.
- `gold_cypher`, `gold_params`, `gold_rows`: deterministic compiler/execution artifacts derived from the runtime contract.
- `metadata`: source-graph provenance plus task-specific metadata.

## Splits

- `train`: 8000 rows over graphs aurora, redwood, lattice.
- `validation`: 1200 rows over graphs nimbus.
- `release_eval`: 1800 rows over graphs vector.
- Split graph files are shipped alongside the JSONL rows.

## Training Target

`supervision_target` is the only authoritative supervision field for model training.
The compiler and runtime remain the source of truth for how that target is executed.

## Gold Field Derivation

- `gold_cypher` and `gold_params` are produced by compiling `supervision_target.plan` through the runtime planner compiler.
- `gold_rows` are produced by evaluating the compiled plan over the split-specific synthetic graph.
- `metadata.source_graph_ids` is recomputed from runtime-consistent graph attribution logic, except that scoped false booleans retain scoped graph provenance rather than positive-match contributors.

## Curation Policy

- Baseline rows were frozen first, then curated question-first with 683 logged edits.
- Question rewrites were allowed by default; route/target edits require runtime-backed adjudication.
- The generator code remains archived for provenance/debugging, but this curated package is the official training source.

## Verifier Policy

- Every row is schema-checked and recompiled through the runtime planner contract.
- `gold_*` fields and `metadata.source_graph_ids` are recomputed and compared to stored values.
- Split invariants are enforced, including zero `local_safe_target_overlap_count` across splits.
- Held-out boolean negatives are present: validation {'false': 15, 'true': 45}, release_eval {'false': 14, 'true': 62}.
- Held-out `company+place` ranking coverage is present: validation=11, release_eval=10.
- Canonicalized rebuilds are required to be byte-identical and checksum-stable.

## Known Non-Blocking Limitations

- Some ranking prompts remain terse search-style requests rather than full conversational questions.
- A small `United Kingdom` place-phrasing pocket remains in otherwise semantically correct rows.
- Shared non-answerable targets still exist across splits for refusal/strong-candidate supervision, but `local_safe` target overlap is zero.

## Bootstrapping Prompt Family

The baseline rows were bootstrapped from a runtime-aligned synthetic prompt family that can be summarized as:

> Given a target route label and canonical planner envelope, generate one question whose semantics exactly match that target, staying within split scope and without broadening or narrowing any explicit constraints.

This description is a reconstruction of the baseline generation setup, not the sole provenance of the final dataset.
