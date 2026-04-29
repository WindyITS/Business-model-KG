# Runtime Assets

This directory holds local runtime assets that are too large or too machine-specific
to track in Git.

The final local query stack is expected at:

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

`runtime_assets/query_stack/` is ignored by Git. The published copy of the same
bundle lives on Hugging Face at:

```text
WindyITS/business-model-kg-query-stack
```
