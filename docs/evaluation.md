# Evaluation

This document explains how the extraction evaluation works and how to reproduce
the reported experiment from the public Hugging Face artifacts.

## What Gets Evaluated

The evaluator compares clean gold benchmark triples against generated pipeline
outputs after entity resolution.

Gold benchmarks live under:

```text
evaluation/benchmarks/dev/clean/
evaluation/benchmarks/test/clean/
```

Each benchmark file is JSONL, with one file per company. The evaluator
normalizes every row into a 3-field edge:

```text
subject, relation, object
```

Predictions live under:

```text
outputs/<company>/<pipeline>/latest/resolved_triples.json
```

The three evaluated pipelines are:

```text
zero-shot
memo_graph_only
analyst
```

## Reported Metrics

Extraction evaluation reports only these metrics:

- precision
- recall
- F1
- macro-F1
- relaxed F1

Exact metrics use normalized 3-field edge equality. Relaxed F1 uses the
graph-aware relaxed matcher implemented in `evaluation/scripts/evaluate.py`.

The relaxed matcher uses weighted partial credit to capture graph-near matches
that are not exact string-identical triples:

- exact typed-triple match: `1.00`
- company alias or lexical normalization match: `0.90`
- subject/object parent-child hierarchy relation: `0.75`
- segment roll-up relation: `0.50`

Those weights are part of the relaxed matcher. They are not additional reported
metrics.

Bootstrap confidence intervals are generated separately from the same benchmark
and output files. Annotation reliability is also generated separately from the
annotation inputs in:

```text
evaluation/benchmarks/annotation_reliability/
```

## Public Artifacts

The public evaluation artifacts are published at:

```text
https://huggingface.co/datasets/WindyITS/business-model-kg-benchmark-outputs
```

The Hugging Face dataset contains:

```text
benchmarks/
outputs/
```

The benchmark folder can be copied directly into the repo as
`evaluation/benchmarks/`.

The published outputs use this layout:

```text
outputs/<company>/<pipeline>/resolved_triples.json
```

The local evaluator expects:

```text
outputs/<company>/<pipeline>/latest/resolved_triples.json
```

When reproducing the experiment, place each downloaded output file inside the
corresponding `latest/` folder.

## Reproduce From Hugging Face

From the repository root, install the Hugging Face download CLI if it is not
already available:

```bash
./venv/bin/python -m pip install "huggingface_hub[cli]"
```

Download the public dataset:

```bash
./venv/bin/huggingface-cli download WindyITS/business-model-kg-benchmark-outputs \
  --repo-type dataset \
  --local-dir hf_evaluation_artifacts
```

Install the benchmark files:

```bash
mkdir -p evaluation/benchmarks
rsync -a hf_evaluation_artifacts/benchmarks/ evaluation/benchmarks/
```

Install the output files into the layout expected by the evaluator:

```bash
./venv/bin/python - <<'PY'
from pathlib import Path
import shutil

source_root = Path("hf_evaluation_artifacts/outputs")
target_root = Path("outputs")

for source_path in source_root.glob("*/*/resolved_triples.json"):
    company = source_path.parts[-3]
    pipeline = source_path.parts[-2]
    target_path = target_root / company / pipeline / "latest" / "resolved_triples.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
PY
```

Run the extraction evaluation for every pipeline and split:

```bash
for pipeline in zero-shot memo_graph_only analyst; do
  for split in dev test; do
    ./venv/bin/python -m evaluation.scripts.evaluate \
      --pipeline "$pipeline" \
      --split "$split" \
      --yes
  done
done
```

Run bootstrap confidence intervals:

```bash
./venv/bin/python -m evaluation.scripts.evaluate \
  --bootstrap \
  --split test \
  --yes
```

Run annotation reliability:

```bash
./venv/bin/python -m evaluation.scripts.evaluate \
  --annotation-reliability \
  --yes
```

## Generated Results

Extraction results are written to:

```text
evaluation/results/<pipeline>/<split>/
```

Each company folder contains:

```text
metrics.json
matched.jsonl
false_positives.jsonl
false_negatives.jsonl
relaxed_matches.jsonl
```

Each split also writes:

```text
summary.json
```

Bootstrap results are written to:

```text
evaluation/results/bootstrap/
```

Annotation reliability results are written to:

```text
evaluation/results/annotation_reliability/
```

All generated result folders are overwritten only after the new run succeeds.
