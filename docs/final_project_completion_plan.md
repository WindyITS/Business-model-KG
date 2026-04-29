# Final Project Completion Plan

This document tracks the remaining work needed before the NLP course presentation.

The goal is to finish the project as an experiment, not only as a working demo.
Steps 1-7 and Step 10 are implemented. Step 8 is paused for now. The remaining
work is to review the project story and prepare the final presentation polish.

Original goals:

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

- Store one clean JSONL file per company under:
  - `evaluation/benchmarks/dev/clean/`
  - `evaluation/benchmarks/test/clean/`
- Use company-slug file names, for example `microsoft.jsonl`.
- Preserve the five typed triple fields:
  - `subject`
  - `subject_type`
  - `relation`
  - `object`
  - `object_type`

Output:

- Gold benchmark files in a stable repo location.
- Clean JSONL gold triples.

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
  - compare place names with the same mechanical entity-name normalization
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
  - company and pipeline
- Write unmatched reports:
  - predicted triples not found in gold
  - gold triples missed by the model

Output:

- Evaluation CLI under `evaluation/scripts/evaluate.py`.
- Machine-readable metrics files.
- Machine-readable aggregate `summary.json` files.
- Unmatched triple reports for error analysis.

Acceptance criteria:

- We can evaluate one company and one pipeline end to end.
- We can evaluate all available companies for one selected pipeline in one command.
- We can build full results by running the command once per pipeline.
- The output is clear enough to use in the final presentation.

## 4. Add Hand-Matched Relaxed Evaluation

Strict exact matching is necessary, but it can unfairly penalize harmless naming differences.

The relaxed metric should not use uncontrolled fuzzy matching as the final score. Instead, humans should review unmatched triples and explicitly tag corresponding rows.

Actions:

- Save all strict false positives and false negatives to a human-editable review CSV.
- Keep gold-side unmatched triples and predicted-side unmatched triples clearly separated.
- Manually tag corresponding rows with a shared `match_id`.
- Run a second command to compute hand-matched second-tier metrics from the reviewed CSV.
- Report both strict and hand-matched metrics.

Output:

- Unmatched review CSV.
- Hand-matched metrics.

Acceptance criteria:

- Strict metrics remain the primary objective metric.
- Relaxed metrics are reproducible because human matches are explicitly recorded in CSV.
- The presentation can explain the difference between exact graph agreement and human-adjudicated semantic agreement.

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

Pipeline name:

- `memo_graph_only`

Actions:

- Add a new pipeline runner using the copied analyst first-memo and graph-compilation prompts under `prompts/memo_graph_only/`.
- Add the new pipeline to the pipeline registry.
- Add runtime artifact handling for the new result type.
- Add tests for stage ordering, output artifacts, and CLI pipeline selection.
- Ensure the pipeline can write outputs under `outputs/<company>/memo_graph_only/latest/`.

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
  - `memo_graph_only`
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

Run the strict and hand-matched evaluations on the final outputs.

Actions:

- Compute overall metrics.
- Compute per-company metrics.
- Compare the three pipelines:
  - `zero-shot`
  - `memo_graph_only`
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

Status:

- Implemented.
- Strict and hand-matched results are saved under `evaluation/results/`.
- The main metric limitation has been identified: exact and hand-matched triple F1 are highly sensitive to hierarchy alignment.
- Berkshire Hathaway is the clearest example: the benchmark uses a fine-grained 17-segment structure, while all predictions use broader 5-segment structures. This makes otherwise related facts mismatch at the triple level.

## 8. Add Qualitative Hierarchy-Aware Evaluation

The quantitative evaluation is reproducible, but it is not sufficient for judging graph usefulness.

The next evaluation layer should compare the gold benchmark and predictions as hierarchical business-model graphs, not only as flat triples.

Actions:

- Build a compact hierarchical profile for each gold benchmark.
- Build a compact hierarchical profile for each company and pipeline prediction.
- Preserve parent-child structure:
  - company to segments
  - segments to offerings
  - offerings to child offerings
  - offerings to revenue models
  - segments or offerings to channels
  - segments to customer types
- Create a fixed qualitative rubric with scores for:
  - segment coverage
  - offering coverage
  - hierarchy alignment
  - roll-up reasonableness
  - revenue logic
  - customer/channel coverage
  - geography handling
  - noise or hallucination
  - overall usefulness
- Run one qualitative review per company and pipeline output.
- Aggregate the qualitative scores by pipeline and company.

Output:

- Hierarchical gold and prediction profiles.
- One qualitative review per company/pipeline pair.
- Aggregate qualitative comparison table.

Acceptance criteria:

- We can explain whether a pipeline is wrong, too broad, too detailed, or simply using a different but defensible hierarchy.
- The presentation does not rely only on F1.
- Berkshire-like hierarchy mismatches are evaluated explicitly instead of hidden inside false positives and false negatives.

## 9. Review Code Behavior Against The Project Story

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

## 10. Publish Artifacts To Hugging Face

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

Status:

- Implemented.
- Public Hugging Face profile: [`WindyITS`](https://huggingface.co/WindyITS).
- Benchmark and output dataset: [`WindyITS/business-model-kg-benchmark-outputs`](https://huggingface.co/datasets/WindyITS/business-model-kg-benchmark-outputs).
- Query planner fine-tuning dataset: [`WindyITS/business-model-kg-query-planner-data`](https://huggingface.co/datasets/WindyITS/business-model-kg-query-planner-data).
- Query stack model bundle: [`WindyITS/business-model-kg-query-stack`](https://huggingface.co/WindyITS/business-model-kg-query-stack).

## 11. Final Documentation And Presentation Polish

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

Run Step 8: create hierarchy-preserving qualitative profiles and use them for qualitative company/pipeline reviews.
