# Text2Cypher Dataset Readiness V2

## Verdict

The rebuilt canonical corpus is ready for high-quality fine-tuning, and the checked-in `messages.jsonl` export already gives the little model the right trainer-facing contract.

The training plan is to use this dataset only. No external Cypher warm-up corpus is required before the ontology-specific fine-tune.

This `v2` corpus is training-ready because it now combines:

- production-faithful `company_name` scoping for `BusinessSegment` and `Offering`
- hierarchy-aware geography supervision through `within_places` and `includes_places`
- explicit collision coverage for same-surface inventory names across companies
- validated gold Cypher across the full active family inventory, including `QF29`, `QF30`, and `QF31`
- train/dev/test coverage for every active family
- a generated SFT/messages layer with strict JSON-only assistant targets

## Final Artifacts

- canonical plan: [`dataset_plan.md`](./dataset_plan.md)
- coverage grid: [`coverage_grid.md`](./coverage_grid.md)
- intent inventory: [`intent_cases_by_family.md`](./intent_cases_by_family.md)
- fixture library: [`fixture_library.md`](./fixture_library.md)
- fixture instances: [`fixture_instances.jsonl`](../../../datasets/text2cypher/v2/source/fixture_instances.jsonl)
- bound seed examples: [`bound_seed_examples.jsonl`](../../../datasets/text2cypher/v2/source/bound_seed_examples.jsonl)
- bound validation report: [`bound_seed_validation_report.json`](../../../datasets/text2cypher/v2/reports/bound_seed_validation_report.json)
- training corpus: [`training_examples.jsonl`](../../../datasets/text2cypher/v2/training/training_examples.jsonl)
- split manifest: [`training_split_manifest.json`](../../../datasets/text2cypher/v2/reports/training_split_manifest.json)
- SFT manifest: [`sft_manifest.json`](../../../datasets/text2cypher/v2/reports/sft_manifest.json)
- split files:
  - [`train.jsonl`](../../../datasets/text2cypher/v2/training/train.jsonl)
  - [`dev.jsonl`](../../../datasets/text2cypher/v2/training/dev.jsonl)
  - [`test.jsonl`](../../../datasets/text2cypher/v2/training/test.jsonl)
- message files:
  - [`messages.jsonl`](../../../datasets/text2cypher/v2/training/messages.jsonl)
  - [`train_messages.jsonl`](../../../datasets/text2cypher/v2/training/train_messages.jsonl)
  - [`dev_messages.jsonl`](../../../datasets/text2cypher/v2/training/dev_messages.jsonl)
  - [`test_messages.jsonl`](../../../datasets/text2cypher/v2/training/test_messages.jsonl)

## Readiness Checks

### Graph And Query Layer

- `24` synthetic graph instances
- `421` bound seed examples
- `112` distinct intents
- `31` covered query families
- execution validation result: `421 / 421` passed

This is the primary readiness gate, and `v2` clears it completely.

### Canonical Training Layer

- `4,494` total training rows
- `421` source examples
- question-variant distribution per source example:
  - `8` examples with `5` variants
  - `48` examples with `6` variants
  - `186` examples with `7` variants
  - `179` examples with `16` variants

The variant mix is intentionally uneven because the core, rollup, and refusal surfaces now use different phrasing budgets and different levels of messy-user augmentation.

### Complexity Mix

- `548` low-difficulty rows
- `1,987` medium-difficulty rows
- `1,959` high-difficulty rows

This is the right shape for a small model that needs both syntax stability and deeper graph reasoning.

### Answerability Mix

- canonical corpus: `4,313` answerable rows and `181` refusal rows
- `messages.jsonl`: `4,313` answerable rows and `179` refusal rows, with `2` duplicate-prompt rows merged into a single trainer target

Refusal coverage spans:

- `108` `not_in_graph`
- `13` `no_entity_anchor`
- `13` `ambiguous_entity`
- `47` `ambiguous_scope`

### Split Integrity

The dataset remains split at the `intent_id` level.

Canonical split sizes:

- train: `2,828` rows, `77` intents, `283` source examples
- dev: `675` rows, `16` intents, `60` source examples
- test: `991` rows, `19` intents, `78` source examples

Trainer-facing message split sizes:

- train: `2,827` rows, `2,698` answerable, `129` refusal
- dev: `674` rows, `643` answerable, `31` refusal
- test: `991` rows, `972` answerable, `19` refusal

Split interpretation:

- `train.jsonl`, `dev.jsonl`, and `test.jsonl` are the provenance-rich canonical splits
- `train_messages.jsonl`, `dev_messages.jsonl`, and `test_messages.jsonl` are the corresponding trainer-facing splits
- `messages.jsonl` is the deduplicated all-splits export for packaging or downstream reshaping
- `QF31` now has one intent in each of train, dev, and test, so every active family is represented in fine-tuning while evaluation coverage remains intact
- the intended fine-tuning run should train only on `train_messages.jsonl`, with `dev_messages.jsonl` and `test_messages.jsonl` reserved for evaluation

## What V2 Fixes

`v2` fixes the most important contract mismatches in the older dataset build:

- it trains company-scoped inventory lookup against composite identities instead of bare-name-only fixtures
- it teaches hierarchy-aware place matching instead of exact-place-only geography
- it adds explicit collision handling for reused offering and segment names
- it restores the missing `QF17` and `QF18` reverse-join families, bringing the active corpus up to the full `QF01` through `QF31` design surface
- it removes prompt/contract collisions and blocks cross-split prompt leakage in the builder
- it now includes much messier user phrasing and a larger set of multi-constraint compositions, especially around rollups, intersections, geography+inventory filters, and ambiguity handling
- it keeps the source corpus rich while generating a clean `messages.jsonl` contract boundary for fine-tuning

## Remaining Caveats

This is a strong supervised corpus, but there is still room for later iterations:

- add more alias-heavy phrasing if runtime entity resolution grows beyond exact surface matching
- add more graph instances for the largest rollup families if we see graph-shape memorization
- expand adversarial ambiguity cases further if runtime evaluation shows over-answering
- calibrate refusal phrasing, system prompt wording, and JSON emission against the first fine-tuning run rather than by dataset intuition alone

Those are follow-up improvements, not blockers for training on `v2`.
