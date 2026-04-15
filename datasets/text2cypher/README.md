# Text2Cypher Dataset Assets

This directory contains the machine-readable artifacts for the text-to-Cypher supervised dataset.

## Layout

- `v2/`
  The canonical dataset release used for current validation, training, and evaluation.
- `v2/source/`
  Synthetic fixture instances plus bound gold examples.
- `v2/reports/`
  Validation outputs and split metadata.
- `v2/training/`
  The full training corpus plus train, dev, and test split exports.
- `v1/`
  The superseded first full release, kept for provenance and comparison.
- `archive/v0/`
  Older prototype artifacts kept for provenance only.

Use [`../../docs/text2cypher/README.md`](../../docs/text2cypher/README.md) for the prose design docs and high-level dataset roadmap.
