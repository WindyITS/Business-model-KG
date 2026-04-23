# Fine-Tuning Island

This directory is intentionally separate from the main KG/runtime codebase.

It contains an isolated fine-tuning pipeline for:

- a `DeBERTa-v3-small` router classifier
- a `Qwen3-4B-Instruct-2507` 4-bit MLX QLoRA planner

The main repo does not import anything from this island.
The only handoff back to the main runtime is a published deployment bundle under `runtime_assets/query_stack/current/`.

## Locations

- Environment root: `finetuning/.venv`
- Artifact root: `finetuning/artifacts/kg-query-planner`

Prepared datasets, checkpoints, logs, and adapters are written under that in-repo artifact directory, which is ignored by Git.

## Bootstrap

```bash
bash finetuning/scripts/bootstrap_env.sh
```

That creates the dedicated project-local environment and installs this island in editable mode.

## Commands

All commands read the single config file at `finetuning/config/default.json` unless `--config` is passed.

```bash
prepare-data
train-router
eval-router
train-planner
eval-planner
eval-planner --base-only
eval-planner --backend lmstudio --lmstudio-model "Qwen3-32B-Instruct"
publish-query-stack
```

## High-Level Flow

1. `prepare-data`
2. `train-router`
3. `eval-router`
4. `train-planner`
5. `eval-planner`
6. `publish-query-stack`

## Output Layout

Artifacts are written under:

```text
finetuning/artifacts/kg-query-planner/
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
      lmstudio/
```

Publishing writes the runtime bundle that the main query stack consumes by default:

```text
runtime_assets/query_stack/current/
  manifest.json
  router/
    model/
    thresholds.json
  planner/
    adapter/
    system_prompt.txt
```

## Notes

- The planner is trained only on `local_safe` rows.
- Optional planner-only train augmentations can live in `data/query_planner_curated/v1_final/planner_only_open_literal_copying_augmentation.jsonl`; `prepare-data` folds them into the planner train split without changing the router dataset.
- The planner default base model is `mlx-community/Qwen3-4B-Instruct-2507-4bit`, so `train-planner` runs QLoRA rather than full-precision LoRA.
- Planner training saves numbered adapter checkpoints every `500` iterations by default, for example `0000500_adapters.safetensors`.
- `planner.grad_checkpoint` defaults to `true` so the planner train path favors lower memory usage on Apple Silicon. Set it to `false` only if you want to prioritize speed and have enough headroom.
- For a warm restart from adapter weights only, set `planner.resume_adapter_file` in `finetuning/config/default.json` to one of those checkpoint files.
- For a true resume that restores optimizer/RNG/iteration state, set `planner.resume_checkpoint_dir` to one of the checkpoint directories under `planner/adapter/checkpoints/`.
- `eval-planner --base-only` evaluates the standard 4-bit base model without loading adapter weights and writes artifacts under `planner/eval/base_model/`.
- `eval-planner --backend lmstudio --lmstudio-model "<name>"` evaluates an LM Studio-served model with the same frozen planner system prompt and writes artifacts under `planner/eval/lmstudio/<model>/`.
- The router maps full-dataset route labels into `local`, `api_fallback`, and `refuse`.
- `publish-query-stack` copies the trained router, router thresholds, planner adapter, and frozen system prompt into the main-runtime deployment bundle.
- The CLIs emit progress bars for dataset prep, router scoring, planner evaluation, and the fine-tuning stages themselves.
