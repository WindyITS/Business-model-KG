# Prompt Assets

This directory holds human-edited prompt assets for extraction pipelines.

- `canonical/` contains the prompt files that back the `literal` pipeline
- `analyst/` contains the prompt files used by the analyst-style pipeline

Prompt loading order:
- if `KG_PROMPTS_DIR` is set, the runtime uses that directory
- otherwise, if this repo-level `prompts/` directory exists, the runtime uses it
- otherwise, the runtime falls back to the bundled prompt copy shipped inside the installed package

The repo-level `prompts/` directory remains the editable source of truth for development.
When you update prompts and want packaged installs to pick up the same changes, run:

```bash
./venv/bin/python scripts/sync_bundled_prompts.py
```
