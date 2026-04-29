# Business Model KG Evaluation Handoff

This branch is a lightweight handoff for evaluation work.

It includes:

- evaluation scripts, benchmarks, and current evaluation results
- saved extraction outputs under `outputs/`
- source 10-K text files under `data/`
- prompt assets under `prompts/`
- project docs under `docs/`
- evaluation tests under `tests/test_evaluation/`
- the small `runtime.output_layout` helper needed by the evaluator

It intentionally excludes:

- fine-tuning code and artifacts
- query-stack model artifacts
- Neo4j/query runtime code that is not needed for evaluation
- curated query-planner training datasets

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Run Evaluation

Evaluate one split for one pipeline:

```bash
python -m evaluation.scripts.evaluate --pipeline analyst --split dev
python -m evaluation.scripts.evaluate --pipeline analyst --split test
```

Other pipeline names:

- `zero-shot`
- `memo_graph_only`
- `analyst`

Evaluate one company:

```bash
python -m evaluation.scripts.evaluate --pipeline analyst --company microsoft
```

Apply hand matches after editing review CSVs:

```bash
python -m evaluation.scripts.apply_hand_matches --results-dir evaluation/results/analyst/dev
```

Run the evaluation tests:

```bash
python -m pytest -q tests/test_evaluation
```

More detail lives in [`evaluation/README.md`](./evaluation/README.md) and
[`docs/evaluation_contract.md`](./docs/evaluation_contract.md).
