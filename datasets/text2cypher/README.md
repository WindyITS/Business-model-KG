# Text2Cypher Dataset Assets

This directory contains the machine-readable artifacts for the active text-to-Cypher supervised dataset, including the training corpus and the held-out evaluation set.

The dataset/database itself was built through agent orchestration over authored query families, synthetic graph fixtures, validated gold Cypher, and expanded user-question variants. See [`../../docs/text2cypher/README.md`](../../docs/text2cypher/README.md) for the full build workflow.

This directory is the canonical in-repo copy of the released corpus and its trainer-facing view. The public workflow explanation lives in [`../../docs/text2cypher/README.md`](../../docs/text2cypher/README.md). Local packaging templates and upload notes can stay outside the tracked repo surface, and the packaging step should copy these files into the release bundle rather than regenerate them ad hoc.

## Layout

- `v3/`
  The active dataset release with a dedicated training artifact and a separate held-out evaluation set.
- `v3/training/`
  The fine-tuning corpus, centered on `train_messages.jsonl`.
- `v3/evaluation/`
  The brand-new held-out evaluation corpus, centered on `test_messages.jsonl`.
- `v3/reports/`
  Train/eval manifests plus a leakage report.

Use [`../../docs/text2cypher/README.md`](../../docs/text2cypher/README.md) for the prose design docs and high-level dataset roadmap.
