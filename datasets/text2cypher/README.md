# Text2Cypher Dataset Assets

This directory contains the machine-readable artifacts for the text-to-Cypher supervised dataset.

The dataset/database itself was built through agent orchestration over authored query families, synthetic graph fixtures, validated gold Cypher, and expanded user-question variants. See [`../../docs/text2cypher/README.md`](../../docs/text2cypher/README.md) for the full build workflow.

This directory is the canonical in-repo copy of the released artifacts. The public workflow explanation lives in [`../../docs/text2cypher/README.md`](../../docs/text2cypher/README.md). Local packaging templates and upload notes can stay outside the tracked repo surface.

## Layout

- `v2/`
  The canonical dataset release used for current validation, training, and evaluation.
- `v2/source/`
  Synthetic fixture instances plus bound gold examples.
- `v2/reports/`
  Validation outputs and split metadata.
- `v2/training/`
  The full training corpus plus train, dev, and test split exports.

Use [`../../docs/text2cypher/README.md`](../../docs/text2cypher/README.md) for the prose design docs and high-level dataset roadmap.
