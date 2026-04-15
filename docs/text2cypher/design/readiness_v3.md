# Text2Cypher Dataset Readiness V3

## Verdict

`Text2Cypher v3` is ready for fine-tuning and evaluation.

The train/eval contract is now explicit:

- train on [`train_messages.jsonl`](../../../datasets/text2cypher/v3/training/train_messages.jsonl)
- evaluate on [`test_messages.jsonl`](../../../datasets/text2cypher/v3/evaluation/test_messages.jsonl)

This release keeps the original `v2` corpus as the audited base, removes the two previously duplicated trainer rows, adds a hard-query training extension, and introduces a brand-new held-out evaluation set with leakage checks.

## Final Artifacts

- training corpus: [`training_examples.jsonl`](../../../datasets/text2cypher/v3/training/training_examples.jsonl)
- train messages: [`train_messages.jsonl`](../../../datasets/text2cypher/v3/training/train_messages.jsonl)
- held-out test examples: [`test_examples.jsonl`](../../../datasets/text2cypher/v3/evaluation/test_examples.jsonl)
- held-out test messages: [`test_messages.jsonl`](../../../datasets/text2cypher/v3/evaluation/test_messages.jsonl)
- validation report: [`bound_seed_validation_report.json`](../../../datasets/text2cypher/v3/reports/bound_seed_validation_report.json)
- train manifest: [`sft_manifest.json`](../../../datasets/text2cypher/v3/reports/sft_manifest.json)
- held-out manifest: [`heldout_test_manifest.json`](../../../datasets/text2cypher/v3/reports/heldout_test_manifest.json)
- leakage report: [`leakage_report.json`](../../../datasets/text2cypher/v3/reports/leakage_report.json)

## Counts

- `26` fixtures
- `485` source examples
- `485 / 485` execution validation passed
- `5,004` training rows
- `5,004` train-facing SFT rows
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
- it adds `512` extra hard-query training rows
- it preserves a separate held-out evaluation set of `512` fresh rows
- it no longer relies on duplicate prompt merges in the trainer-facing export

## Recommendation

Use `v3` for the next fine-tuning run. Keep `v2` as the historical baseline and provenance source.
