# Evaluation

This folder contains the benchmark data, generated reports, and evaluator for
the extraction pipelines.

## Data Sources

The evaluator reads clean JSONL benchmark files from:

```text
evaluation/benchmarks/dev/clean/
evaluation/benchmarks/test/clean/
```

The clean JSONL files are the canonical benchmark inputs. Source spreadsheets
belong outside the evaluation package after conversion.

The evaluator compares each clean benchmark against post-resolution pipeline
outputs from:

```text
outputs/<company>/<pipeline>/latest/resolved_triples.json
```

## Metrics

Metrics:

- exact precision over normalized 3-field edges
- exact recall over normalized 3-field edges
- exact F1 over normalized 3-field edges
- exact macro-F1 by company
- relaxed graph-aware F1 with one-to-one greedy matching
- exact typed-triple match: `1.00`
- company alias / lexical normalization: `0.90`
- subject/object parent-child hierarchy relation: `0.75`
- segment roll-up relation: `0.50`

Bootstrap confidence intervals live in:

```text
evaluation/results/bootstrap/
```

## Run Evaluation

Evaluate all companies in one split for one pipeline:

```bash
./venv/bin/python -m evaluation.scripts.evaluate --pipeline zero-shot --split dev
./venv/bin/python -m evaluation.scripts.evaluate --pipeline zero-shot --split test
```

Evaluate one selected company for one selected pipeline:

```bash
./venv/bin/python -m evaluation.scripts.evaluate --pipeline analyst --company microsoft
```

For deliberate reruns, add `--yes`:

```bash
./venv/bin/python -m evaluation.scripts.evaluate --pipeline analyst --split test --yes
```

## Output Files

Split results are written under:

```text
evaluation/results/<pipeline>/<split>/
```

Cherry-picked company results are written under:

```text
evaluation/results/cherry_picked/<pipeline>/<company>/
```

Each evaluated company writes:

- `metrics.json`
- `matched.jsonl`
- `false_positives.jsonl`
- `false_negatives.jsonl`
- `relaxed_matches.jsonl`

Each run also writes `summary.json`.

JSON/JSONL files are the canonical generated reporting artifacts and are the
right format for publication.

If the target result folder already contains files, the evaluator asks before
overwriting. If overwrite is approved, existing results are replaced only after
the new evaluation run succeeds.
