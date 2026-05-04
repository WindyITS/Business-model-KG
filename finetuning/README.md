# Fine-Tuning Island

This directory is intentionally separate from the main KG/runtime codebase.

It contains an isolated fine-tuning pipeline for:

- a `DeBERTa-v3-small` router classifier
- a `Qwen3-4B-Instruct-2507` 4-bit MLX QLoRA planner

The main repo does not import anything from this island.
The only handoff back to the main runtime is a published deployment bundle under `runtime_assets/query_stack/`.

This island is meant to be reproducible on its own: create the dedicated
environment, download the public query-planner dataset, run the preparation and
training commands, then publish the resulting bundle back into the main runtime.

## Practical Requirements

Fine-tuning is heavier than using the main extraction/runtime project.

The router uses standard Hugging Face/PyTorch tooling around
`DeBERTa-v3-small`. The planner uses MLX with a 4-bit Qwen model and is intended
for Apple Silicon. Running the full planner training path requires network
access for model downloads, several gigabytes of disk space for models and
artifacts, and enough local memory for MLX LoRA training.

As a rough expectation, dataset preparation and router evaluation are short
local jobs, router training is a moderate training job, and planner training is
the long step. A reviewer who only wants to use the local query runtime does not
need to rerun this island; they can download the published query-stack bundle
instead.

## Locations

- Environment root: `finetuning/.venv`
- Artifact root: `finetuning/artifacts/kg-query-planner`

Prepared datasets, checkpoints, logs, and adapters are written under that in-repo artifact directory, which is ignored by Git.

## Bootstrap

```bash
bash finetuning/scripts/bootstrap_env.sh
```

That creates the dedicated project-local environment and installs this island
in editable mode with its fine-tuning dependencies.

Download the public fine-tuning dataset:

```bash
bash finetuning/scripts/download_query_planner_data.sh
```

The download command uses the Hugging Face CLI installed in
`finetuning/.venv`, downloads `WindyITS/business-model-kg-query-planner-data`,
and places it under:

```text
data/query_planner_curated/
```

The default config then reads:

```text
data/query_planner_curated/v1_final/
```

Activate the isolated environment before running the CLIs:

```bash
source finetuning/.venv/bin/activate
```

## Commands

All commands read `finetuning/config/default.json` unless `--config` is passed.

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

1. `prepare-data` converts the curated query dataset into router and planner training files.
2. `train-router` trains the local/fallback/refusal classifier.
3. `eval-router` evaluates the router and writes threshold metadata.
4. `train-planner` trains the compact local graph-query planner.
5. `eval-planner` evaluates planner outputs.
6. `publish-query-stack` copies the trained router and planner adapter into the main runtime bundle.

After `publish-query-stack`, the main runtime can use the generated bundle from
`runtime_assets/query_stack/`.

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
runtime_assets/query_stack/
  manifest.json
  router/
    model/
    thresholds.json
  planner/
    adapter/
    system_prompt.txt
```

## Notes

- The default source-checkout config is `finetuning/config/default.json`.
- Installed wheels carry a fallback copy of that default config. Relative paths
  resolve from the current working directory, or from `KG_QUERY_PLANNER_FT_ROOT`
  when that environment variable is set.
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
- `eval-router` fits temperature scaling but does not search for routing thresholds.
  Runtime policy is fixed: use the local planner only when `P(local) >= 0.97`;
  otherwise choose the larger of `P(api_fallback)` and `P(refuse)`.
- `publish-query-stack` copies the trained router, router policy metadata, planner adapter, and frozen system prompt into the main-runtime deployment bundle.
- The CLIs emit progress bars for dataset prep, router scoring, planner evaluation, and the fine-tuning stages themselves.
