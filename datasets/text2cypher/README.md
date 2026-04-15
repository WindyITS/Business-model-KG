# Text2Cypher Dataset Assets

This directory is the local workspace for generated text-to-Cypher dataset artifacts, including the training corpus and the held-out evaluation set.

The dataset/database itself was built through agent orchestration over authored query families, synthetic graph fixtures, validated gold Cypher, and expanded user-question variants. See [`../../docs/text2cypher/README.md`](../../docs/text2cypher/README.md) for the full build workflow.

The generated version directories under this path are local build outputs, not checked-in release artifacts. The public workflow explanation lives in [`../../docs/text2cypher/README.md`](../../docs/text2cypher/README.md). The export step copies these local files into the Hugging Face release bundle rather than regenerating them ad hoc.

## Layout

- `v3/`
  The default local dataset build with a dedicated training artifact and a separate held-out evaluation set.
- `v3/training/`
  The fine-tuning corpus, centered on `train_messages.jsonl`.
- `v3/evaluation/`
  The brand-new held-out evaluation corpus, centered on `test_messages.jsonl`.
- `v3/reports/`
  Train/eval manifests plus a leakage report.

Build these local artifacts with `./venv/bin/python scripts/build_text2cypher_dataset.py`, then publish them with `./venv/bin/python scripts/export_hf_text2cypher_dataset.py --force`.

Use [`../../docs/text2cypher/README.md`](../../docs/text2cypher/README.md) for the prose design docs and high-level dataset roadmap.
