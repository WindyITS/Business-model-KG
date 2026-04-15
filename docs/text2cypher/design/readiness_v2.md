# Text2Cypher Dataset Readiness V2

## Verdict

The rebuilt dataset is ready for fine-tuning and materially stronger than the superseded `v1` release.

This `v2` corpus is training-ready because it now combines:

- production-faithful `company_name` scoping for `BusinessSegment` and `Offering`
- hierarchy-aware geography supervision through `within_places` and `includes_places`
- explicit collision coverage for same-surface inventory names across companies
- validated gold Cypher across the full active family inventory, including the new `QF29`, `QF30`, and `QF31` surfaces

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
- split files:
  - [`train.jsonl`](../../../datasets/text2cypher/v2/training/train.jsonl)
  - [`dev.jsonl`](../../../datasets/text2cypher/v2/training/dev.jsonl)
  - [`test.jsonl`](../../../datasets/text2cypher/v2/training/test.jsonl)

## Readiness Checks

### Graph And Query Layer

- `24` synthetic graph instances
- `421` bound seed examples
- `112` distinct intents
- `31` covered query families
- execution validation result: `421 / 421` passed

This is the primary readiness gate, and `v2` clears it completely.

### Training Layer

- `4,501` total training rows
- `421` source examples
- question-variant distribution per source example:
  - `8` examples with `5` variants
  - `41` examples with `6` variants
  - `193` examples with `7` variants
  - `179` examples with `16` variants

The variant mix is intentionally uneven because the core, rollup, and refusal surfaces now use different phrasing budgets and different levels of messy-user augmentation.

### Complexity Mix

- `308` low-difficulty rows
- `1,397` medium-difficulty rows
- `1,551` high-difficulty rows

This is the right shape for a small model that needs both syntax stability and deeper graph reasoning.

### Answerability Mix

- `4,244` answerable rows
- `257` refusal rows

Refusal coverage spans:

- `18` `not_in_graph`
- `2` `no_entity_anchor`
- `2` `ambiguous_entity`
- `7` `ambiguous_scope`

### Split Integrity

The dataset remains split at the `intent_id` level.

Split sizes:

- train: `2,693` rows, `76` intents, `274` source examples
- dev: `547` rows, `15` intents, `52` source examples
- test: `1,261` rows, `21` intents, `95` source examples

## Why V2 Is Better Than V1

`v2` fixes the most important contract mismatches from `v1`:

- it trains company-scoped inventory lookup against composite identities instead of bare-name-only fixtures
- it teaches hierarchy-aware place matching instead of exact-place-only geography
- it adds explicit collision handling for reused offering and segment names
- it restores the missing `QF17` and `QF18` reverse-join families, bringing the active corpus up to the full `QF01` through `QF31` design surface
- it now includes much messier user phrasing and a larger set of multi-constraint compositions, especially around rollups, intersections, geography+inventory filters, and ambiguity handling

## Remaining Caveats

This is a strong supervised corpus, but there is still room for later iterations:

- add more alias-heavy phrasing if runtime entity resolution grows beyond exact surface matching
- add more graph instances for the largest rollup families if we see graph-shape memorization
- expand adversarial ambiguity cases further if runtime evaluation shows over-answering
- calibrate refusal phrasing and JSON emission against the first fine-tuning run rather than by dataset intuition alone

Those are follow-up improvements, not blockers for training on `v2`.
