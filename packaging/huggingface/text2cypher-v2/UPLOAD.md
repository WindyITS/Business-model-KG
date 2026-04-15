# Upload Checklist

This folder is the packaging area for publishing `Text2Cypher v2` to Hugging Face.

## What to upload

Run the local export helper first:

```bash
./venv/bin/python scripts/export_hf_text2cypher_dataset.py --force
```

That will build a self-contained release tree under:

```text
dist/huggingface/text2cypher-v2/
```

Upload the contents of that directory to the Hugging Face dataset repo root.

## Before upload

- confirm the repository root README says `Text2Cypher v2`
- confirm the stats match the current release:
  - `24` fixtures
  - `421` source examples
  - `4,501` training rows
  - `112` intents
  - `31` families
  - `421/421` validated
- verify no `v1` dataset references remain in the release-facing card
- keep the agent-orchestrated, spec-first workflow description intact

## Suggested HF repo contents

- `README.md`
- `.gitattributes`
- `source/fixture_instances.jsonl`
- `source/bound_seed_examples.jsonl`
- `reports/bound_seed_validation_report.json`
- `reports/training_split_manifest.json`
- `training/training_examples.jsonl`
- `training/train.jsonl`
- `training/dev.jsonl`
- `training/test.jsonl`

## Suggested validation

Use the local project validator before publishing:

```bash
./venv/bin/python src/validate_text2cypher_dataset.py
```

If you want to inspect the release shape first, check the canonical docs in the main repo and compare them to the packaged artifacts under `datasets/text2cypher/v2/`.
