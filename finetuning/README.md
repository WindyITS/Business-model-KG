# Fine-Tuning Island

This directory is intentionally separate from the main KG/runtime codebase.

It contains an isolated fine-tuning pipeline for:

- a `DeBERTa-v3-small` router classifier
- a `Qwen3-4B-Instruct` MLX LoRA planner

The main repo does not import anything from this island.

## Fixed External Locations

- Environment root: `~/projects/.ML-environments/kg-query-planner-ft`
- Artifact root: `~/projects/.ML-artifacts/kg-query-planner`

The repo keeps only code, docs, and static config here. Prepared datasets, checkpoints, logs, and adapters are written outside the repo.

## Bootstrap

```bash
bash finetuning/scripts/bootstrap_env.sh
```

That creates the dedicated environment and installs this island in editable mode.

## Commands

All commands read the single config file at `finetuning/config/default.json` unless `--config` is passed.

```bash
prepare-data
train-router
eval-router
train-planner
eval-planner
run-local-stack "Which companies partner with Dell?"
```

## High-Level Flow

1. `prepare-data`
2. `train-router`
3. `eval-router`
4. `train-planner`
5. `eval-planner`
6. `run-local-stack`

## Output Layout

Artifacts are written under:

```text
~/projects/.ML-artifacts/kg-query-planner/
  prepared/
    router/
    planner/
      raw/
      balanced/
  router/
    model/
    eval/
  planner/
    adapter/
    eval/
```

## Notes

- The planner is trained only on `local_safe` rows.
- The router maps full-dataset route labels into `local`, `api_fallback`, and `refuse`.
- The local harness is conservative: any planner parse/schema/compile failure downgrades to `api_fallback`.
- The CLIs emit progress bars for dataset prep, router scoring, planner evaluation, and the fine-tuning stages themselves.
