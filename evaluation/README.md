# Evaluation Folder

This folder contains the benchmark files, evaluator code, and generated result
reports for extraction evaluation.

For the canonical evaluation guide, use
[`../docs/evaluation.md`](../docs/evaluation.md). For the reviewer reproduction
path, use [`../docs/reproducibility.md`](../docs/reproducibility.md).

## Folder Layout

```text
evaluation/
  benchmarks/
    dev/clean/
    test/clean/
    annotation_reliability/
  scripts/
    evaluate.py
  results/
```

`benchmarks/` holds curated gold triples and annotation-reliability inputs.
`scripts/evaluate.py` runs the evaluator. `results/` stores generated reports
such as summaries, matched triples, false positives, false negatives, relaxed
matches, bootstrap outputs, and annotation-reliability outputs.

The evaluator compares gold triples against saved extraction outputs at:

```text
outputs/<company>/<pipeline>/latest/resolved_triples.json
```

For deliberate reruns, pass `--yes` to overwrite existing result folders after
the new run succeeds.
