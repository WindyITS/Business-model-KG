# Text2Cypher Dataset Readiness V3

## Verdict

`Text2Cypher v3` is ready for fine-tuning and evaluation.

The train/eval contract is now explicit in the local build output:

- train on `datasets/text2cypher/v3/training/train_messages.jsonl`
- validate during training on `datasets/text2cypher/v3/training/valid_messages.jsonl`
- evaluate on `datasets/text2cypher/v3/evaluation/test_messages.jsonl`

This release removes the previously duplicated trainer rows, adds a hard-query training extension, and introduces a brand-new held-out evaluation set with leakage checks.

## Final Artifacts

- training corpus: `datasets/text2cypher/v3/training/training_examples.jsonl`
- train messages: `datasets/text2cypher/v3/training/train_messages.jsonl`
- valid examples: `datasets/text2cypher/v3/training/valid_examples.jsonl`
- valid messages: `datasets/text2cypher/v3/training/valid_messages.jsonl`
- held-out test examples: `datasets/text2cypher/v3/evaluation/test_examples.jsonl`
- held-out test messages: `datasets/text2cypher/v3/evaluation/test_messages.jsonl`
- validation report: `datasets/text2cypher/v3/reports/bound_seed_validation_report.json`
- train manifest: `datasets/text2cypher/v3/reports/sft_manifest.json`
- held-out manifest: `datasets/text2cypher/v3/reports/heldout_test_manifest.json`
- leakage report: `datasets/text2cypher/v3/reports/leakage_report.json`

## Counts

- `26` fixtures
- `485` source examples
- `485 / 485` execution validation passed
- `5,004` training rows
- `4,904` train-facing SFT rows
- `100` validation SFT rows
- `512` held-out evaluation rows
- `0` duplicate prompt merges

Train composition:

- `4,825` answerable rows
- `179` refusal rows

Held-out composition:

- `512` answerable rows
- `0` refusal rows

## Leakage Check

The builder now emits an explicit leakage report for `v3`.

Current result:

- `0` normalized question overlaps
- `0` fixture overlaps
- `0` graph overlaps

## Why V3 Is Better For Training

- it trains on the full audited base corpus rather than only the old `train` fold
- it now reserves a clean `100`-row validation split for in-training checkpoint selection
- it adds `512` extra hard-query training rows
- it preserves a separate held-out evaluation set of `512` fresh rows
- it no longer relies on duplicate prompt merges in the trainer-facing export

## Recommendation

Use the locally built `v3` dataset for the next fine-tuning run, then export the release bundle to Hugging Face.
