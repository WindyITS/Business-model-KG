# Evaluation

This folder contains the benchmark data and evaluation code for the extraction pipelines.

The benchmark is split into two sections:

- `benchmarks/dev/`: development benchmark used while building and debugging the evaluator
- `benchmarks/test/`: final held-out benchmark used for presentation metrics

Each section has two subfolders:

- `raw/`: manually annotated CSV files
- `clean/`: normalized benchmark files produced from raw CSVs

Raw benchmark CSVs should use this five-column schema:

```csv
subject,subject_type,relation,object,object_type
```

Clean benchmark files will use JSONL, with one triple per line:

```json
{"subject":"Microsoft","subject_type":"Company","relation":"HAS_SEGMENT","object":"Intelligent Cloud","object_type":"BusinessSegment"}
```

Convert raw CSV files into clean JSONL files with:

```bash
./venv/bin/python -m evaluation.scripts.prepare_gold --split all
```

Useful variants:

```bash
./venv/bin/python -m evaluation.scripts.prepare_gold --split dev
./venv/bin/python -m evaluation.scripts.prepare_gold --split test
```

The converter writes:

- one `.jsonl` file in the matching `clean/` folder for each raw CSV
- one `manifest.json` per converted split with file and triple counts

The evaluation scripts should compare clean gold triples against pipeline outputs from:

```text
outputs/<company>/<pipeline>/latest/resolved_triples.json
```

Primary score:

- strict normalized typed-triple precision, recall, and F1

Secondary score:

- hand-matched typed-triple precision, recall, and F1, using only manually tagged unmatched rows

## Run Evaluation

Evaluate all companies in one split for one pipeline:

```bash
./venv/bin/python -m evaluation.scripts.evaluate --pipeline zero-shot --split dev
./venv/bin/python -m evaluation.scripts.evaluate --pipeline zero-shot --split test
```

This writes results under:

```text
evaluation/results/zero-shot/dev/
evaluation/results/zero-shot/test/
```

Evaluate one selected company for one selected pipeline:

```bash
./venv/bin/python -m evaluation.scripts.evaluate --pipeline analyst --company microsoft
```

This writes results under:

```text
evaluation/results/cherry_picked/analyst/microsoft/
```

Each evaluated company writes:

- `metrics.json`
- `matched.jsonl`
- `false_positives.jsonl`
- `false_negatives.jsonl`
- `unmatched_for_review.csv`

For example:

```text
evaluation/results/zero-shot/dev/companies/microsoft/unmatched_for_review.csv
evaluation/results/zero-shot/test/companies/apple/unmatched_for_review.csv
evaluation/results/cherry_picked/analyst/microsoft/unmatched_for_review.csv
```

Each run also writes a `summary.json`.

If the target result folder already contains files, the evaluator asks before overwriting them.
Answer `n` to cancel and leave existing results untouched.
If overwrite is approved, the old results are replaced only after the new evaluation succeeds.

For deliberate reruns, add `--yes`:

```bash
./venv/bin/python -m evaluation.scripts.evaluate --pipeline zero-shot --split dev --yes
```

## Hand-Match Review

Strict metrics are always written. The evaluator writes every unmatched strict triple to `unmatched_for_review.csv` so naming differences can be reviewed manually in a spreadsheet.

The review CSV separates unmatched gold triples from unmatched predicted triples:

- `source=gold`: gold triples not matched by the pipeline
- `source=predicted`: predicted triples not found in the gold benchmark

To hand-match two rows, put the same value in `match_id` for the corresponding gold and predicted rows. For example, use `1` for the first hand match, `2` for the second hand match, and so on.
Each `match_id` must be used on exactly one `source=gold` row and exactly one `source=predicted` row. Reusing a `match_id` for multiple pairs is rejected and does not affect hand-matched metrics.

After editing the review CSV, compute hand-matched second-tier metrics with:

```bash
./venv/bin/python -m evaluation.scripts.apply_hand_matches --results-dir evaluation/results/zero-shot/dev
```

This writes:

- `hand_matched/companies/<company>/metrics.json` for each reviewed company
- `hand_matched/summary.json` for the whole result folder

If `hand_matched/` already contains files, the script asks before overwriting them.
If overwrite is approved, the old hand-matched results are replaced only after the new computation succeeds.
For deliberate reruns, add `--yes`:

```bash
./venv/bin/python -m evaluation.scripts.apply_hand_matches --results-dir evaluation/results/zero-shot/dev --yes
```
