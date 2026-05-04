# Runtime Assets

This folder is for local runtime assets that are too large or too
machine-specific to track in Git.

The main asset is the local query-stack bundle:

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

`runtime_assets/query_stack/` is ignored by Git. A fresh clone will not contain
it.

Download the published bundle with:

```bash
./venv/bin/python -m pip install "huggingface_hub[cli]"
./venv/bin/huggingface-cli download WindyITS/business-model-kg-query-stack \
  --local-dir runtime_assets/query_stack
```

You can also regenerate the same kind of bundle from the isolated fine-tuning
workflow. From the repository root:

```bash
source finetuning/.venv/bin/activate
publish-query-stack
```

Using the bundle's local planner path requires the main environment's
`query-stack` extras on Apple Silicon/macOS with working Metal. Hosted query
fallback does not use this bundle, but it requires `OPENCODE_GO_API_KEY`.

For the full reviewer path, see
[`../docs/reproducibility.md`](../docs/reproducibility.md). For query runtime
usage, see [`../docs/runtime_guide.md`](../docs/runtime_guide.md).
