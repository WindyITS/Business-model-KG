# Text2Cypher Dataset Assets

This directory contains the machine-readable artifacts for the text-to-Cypher supervised dataset.

## Layout

- `v1/`
  The canonical dataset release used for current validation, training, and evaluation.
- `v1/source/`
  Synthetic fixture instances plus bound gold examples.
- `v1/reports/`
  Validation outputs and split metadata.
- `v1/training/`
  The full training corpus plus train, dev, and test split exports.
- `archive/v0/`
  Older prototype artifacts kept for provenance only.

Use [`../../docs/text2cypher/README.md`](../../docs/text2cypher/README.md) for the prose design docs and high-level dataset roadmap.
