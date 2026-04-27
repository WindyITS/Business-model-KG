# Final Project Completion Plan

This document tracks the remaining work needed before the NLP course presentation.

The goal is to finish the project as an experiment, not only as a working demo:

- extract ontology-valid knowledge graphs for multiple companies
- compare model outputs against a manually annotated gold benchmark
- evaluate the extraction pipelines with defensible metrics
- add one ablation pipeline between `zero-shot` and `analyst`
- publish the final datasets, models, and benchmark artifacts
- make the docs and presentation story match the actual code behavior

## 1. Freeze The Ontology And Evaluation Target

Before adding evaluation code or running final extractions, decide exactly what counts as a valid target graph.

Actions:

- Confirm that `docs/ontology.md` is the final human-readable ontology.
- Confirm that `src/ontology/ontology.json` is the final machine-readable schema.
- Confirm that `src/ontology/validator.py` represents the final validation rules for both model outputs and gold triples.
- Decide whether evaluation should use the same post-processing as runtime outputs, especially entity resolution and validation.
- Decide how to handle invalid gold triples, if any are found.
- Write down the final evaluation contract:
  - node types
  - relation names
  - canonical labels
  - normalization rules
  - alias policy
  - strict and relaxed metric definitions

Output:

- A short frozen evaluation contract in `docs/evaluation_contract.md`.

Acceptance criteria:

- We can say, without ambiguity, what a valid gold triple is.
- We can say, without ambiguity, what a model prediction is compared against.
- The evaluation target will not change after annotation and final extraction runs begin.

## 2. Standardize The Gold Benchmark Format

The gold triples use the typed manual format:

```text
subject; subject_type; relation; object; object_type
```

Actions:

- Store raw gold CSV files under:
  - `evaluation/benchmarks/dev/raw/`
  - `evaluation/benchmarks/test/raw/`
- Use one raw CSV file per company.
- Parse the CSV format into structured triples:
  - `subject`
  - `subject_type`
  - `relation`
  - `object`
  - `object_type`
- Write clean JSONL files under:
  - `evaluation/benchmarks/dev/clean/`
  - `evaluation/benchmarks/test/clean/`
- Write a manifest for each split with file and triple counts.

Output:

- Gold benchmark files in a stable repo location.
- Parsed JSON or JSONL gold triples.
- Split manifests.

Acceptance criteria:

- Every gold triple can be loaded by code.
- The format is simple enough for presentation and reproducibility.

## 3. Build The Evaluation Pipeline

Build evaluation before finishing all final runs, so mistakes in gold format or matching logic surface early.

Actions:

- Load model output triples from `outputs/<company>/<pipeline>/latest/resolved_triples.json`.
- Load gold triples from the benchmark folder.
- Normalize both sides using the same basic rules:
  - trim whitespace
  - normalize quotes and Unicode forms
  - normalize repeated spaces
  - compare entity names case-insensitively
  - normalize places using the existing place normalization
  - enforce exact node types and relation names
- Compute strict exact-match metrics:
  - true positives
  - false positives
  - false negatives
  - precision
  - recall
  - F1
- Break metrics down by:
  - company
  - pipeline
  - relation
  - optionally node type pair
- Write unmatched reports:
  - predicted triples not found in gold
  - gold triples missed by the model
  - likely alias candidates

Output:

- Evaluation CLI or script.
- Machine-readable metrics files.
- Human-readable summary table.
- Unmatched triple reports for error analysis.

Acceptance criteria:

- We can evaluate one company and one pipeline end to end.
- We can evaluate all available companies and pipelines in one command.
- The output is clear enough to use in the final presentation.

## 4. Add Alias-Assisted Relaxed Evaluation

Strict exact matching is necessary, but it can unfairly penalize harmless naming differences.

The relaxed metric should not use uncontrolled fuzzy matching as the final score. Instead, fuzzy matching should propose aliases, and humans should approve them.

Actions:

- Generate candidate alias matches only when relation and node types are compatible.
- Rank candidates using string similarity or simple normalized containment.
- Save candidates to a review file.
- Manually approve aliases into a stable alias map.
- Re-run evaluation with approved aliases applied.
- Report both strict and alias-normalized metrics.

Output:

- Alias candidate report.
- Approved alias map.
- Relaxed evaluation metrics.

Acceptance criteria:

- Strict metrics remain the primary objective metric.
- Relaxed metrics are reproducible because aliases are explicitly approved and versioned.
- The presentation can explain the difference between exact graph agreement and adjudicated semantic agreement.

## 5. Add The Middle-Way Ablation Pipeline

Add a third extraction pipeline between the current two.

Current pipelines:

- `zero-shot`: direct graph extraction from the filing
- `analyst`: memo foundation, memo augmentation, graph compilation, critique

New ablation:

- build only the first analyst memo
- compile that first memo into a graph
- skip memo augmentation
- skip graph critique
- still run normal entity resolution, validation, artifact writing, and optional Neo4j loading

Possible name:

- `memo-only`

Actions:

- Add a new pipeline runner reusing the analyst prompt helpers where possible.
- Add the new pipeline to the pipeline registry.
- Add runtime artifact handling for the new result type.
- Add tests for stage ordering, output artifacts, and CLI pipeline selection.
- Ensure the pipeline can write outputs under `outputs/<company>/memo-only/latest/`.

Output:

- A working third extraction pipeline.
- Tests covering the new pipeline.

Acceptance criteria:

- The pipeline runs end to end.
- It is comparable with `zero-shot` and `analyst`.
- It introduces minimal duplicated logic.

## 6. Run Final Extractions

Once the ontology, evaluation target, and three pipelines are stable, run the final extraction experiment.

Actions:

- Decide final company list.
- Decide final provider/model/settings.
- Run each company through:
  - `zero-shot`
  - `memo-only`
  - `analyst`
- Save outputs without overwriting important previous runs unless intended.
- Record model, provider, date, and command settings.
- Validate each output.

Output:

- Final model outputs for every company and pipeline.
- Run summaries and validation reports.

Acceptance criteria:

- Every company has all required pipeline outputs.
- Every run has a clear artifact directory.
- Failed or partial runs are not mixed into final metrics.

## 7. Evaluate And Analyze Results

Run the strict and alias-normalized evaluations on the final outputs.

Actions:

- Compute overall metrics.
- Compute per-company metrics.
- Compute per-relation metrics.
- Compare the three pipelines:
  - `zero-shot`
  - `memo-only`
  - `analyst`
- Inspect false positives and false negatives.
- Identify recurring error categories.

Output:

- Final metrics table.
- Error analysis notes.
- Presentation-ready summary of what improved and why.

Acceptance criteria:

- We can explain which pipeline works best.
- We can explain whether memo augmentation and critique actually help.
- We can discuss common failure modes honestly.

## 8. Review Code Behavior Against The Project Story

Review whether the implementation does what the presentation says it does.

Actions:

- Review extraction pipeline control flow.
- Review prompt behavior and artifact outputs.
- Review ontology validation and normalization.
- Review Neo4j loading behavior.
- Review natural-language query behavior.
- Check that docs do not overclaim.

Output:

- Notes on any mismatches between docs, code, and presentation story.
- Optional small fixes if needed.

Acceptance criteria:

- We understand the project well enough to defend it in Q&A.
- The docs, code, and presentation describe the same system.

## 9. Publish Artifacts To Hugging Face

Publish after the benchmark and final outputs are stable.

Actions:

- Publish the fine-tuning dataset.
- Publish the fine-tuned query planner/router artifacts, if appropriate.
- Publish the benchmark dataset.
- Include dataset cards and model cards.
- Include licenses, intended use, limitations, and reproducibility notes.
- Include checksums or manifests where useful.

Output:

- Hugging Face dataset/model repositories.
- Links ready for the report and presentation.

Acceptance criteria:

- External readers can understand what each artifact contains.
- The benchmark can be reused or inspected.
- Published artifacts match the final local experiment.

## 10. Final Documentation And Presentation Polish

Do this after the experiment is frozen and evaluated.

Actions:

- Review `README.md`.
- Review `docs/ontology.md`.
- Review `docs/project_walkthrough.md`.
- Add or update benchmark/evaluation documentation.
- Prepare presentation tables and diagrams.
- Prepare a short explanation of the ablation design.

Output:

- Final docs.
- Presentation-ready experiment narrative.

Acceptance criteria:

- The project can be understood from the repo.
- The presentation has a clean story:
  - task
  - ontology
  - pipelines
  - benchmark
  - metrics
  - ablation
  - results
  - limitations

## Current Next Step

Start with step 1: freeze the ontology and evaluation target.

The first concrete task is to inspect the ontology document, ontology JSON, validator behavior, and current output format, then write the final evaluation contract.
