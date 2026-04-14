# Text2Cypher Dataset Readiness V1

## Verdict

The dataset is ready for a first fine-tuning run on a small model.

This is a good V1 training set because it now has:

- validated gold Cypher grounded in ontology-valid synthetic graphs
- two bound graph contexts per intent
- enough natural-language variation to train a small instruction model without being too small
- explicit refusal coverage for unsupported and ambiguous requests
- intent-level train, dev, and test splits with no leakage

It is not a final or perfect corpus, but it is strong enough to support an initial LoRA or QLoRA fine-tuning cycle and meaningful offline evaluation.

## Final Artifacts

- canonical plan: [`dataset_plan.md`](./dataset_plan.md)
- coverage grid: [`coverage_grid.md`](./coverage_grid.md)
- intent inventory: [`intent_cases_by_family.md`](./intent_cases_by_family.md)
- fixture library: [`fixture_library.md`](./fixture_library.md)
- fixture instances: [`fixture_instances.jsonl`](../../../datasets/text2cypher/v1/source/fixture_instances.jsonl)
- bound seed examples: [`bound_seed_examples.jsonl`](../../../datasets/text2cypher/v1/source/bound_seed_examples.jsonl)
- bound validation report: [`bound_seed_validation_report.json`](../../../datasets/text2cypher/v1/reports/bound_seed_validation_report.json)
- training corpus: [`training_examples.jsonl`](../../../datasets/text2cypher/v1/training/training_examples.jsonl)
- split manifest: [`training_split_manifest.json`](../../../datasets/text2cypher/v1/reports/training_split_manifest.json)
- split files:
  - [`train.jsonl`](../../../datasets/text2cypher/v1/training/train.jsonl)
  - [`dev.jsonl`](../../../datasets/text2cypher/v1/training/dev.jsonl)
  - [`test.jsonl`](../../../datasets/text2cypher/v1/training/test.jsonl)

## Readiness Checks

### Graph And Query Layer

- `25` synthetic graph instances
- `198` bound seed examples
- `99` distinct intents
- `2` graph bindings per intent
- execution validation result: `198 / 198` passed

This is the most important readiness signal.

The model is not being trained on guessed queries. It is being trained on a Cypher layer that was actually executed against synthetic Neo4j graphs matching the ontology.

### Training Layer

- `2,574` total training rows
- `198` source examples
- `13` question variants per source example
- `1` canonical plus `12` paraphrases for each source example
- `0` duplicate training IDs
- `0` duplicate paraphrases within a source example
- `0` paraphrases identical to the canonical question after final cleanup

### Complexity Mix

- `663` low-difficulty rows
- `1,339` medium-difficulty rows
- `572` high-difficulty rows

This is a good balance for a small model:

- enough low-complexity supervision to stabilize syntax and basic retrieval
- enough medium-complexity supervision to learn filtering, membership, counts, and reverse lookups
- enough high-complexity supervision to teach rollups, intersections, and ontology-aware traversals

### Answerability Mix

- `2,184` answerable rows
- `390` refusal rows

That refusal coverage is important.

The model is not only being taught how to emit Cypher. It is also being taught when not to emit Cypher because the graph does not contain the requested information or the request is genuinely ambiguous.

### Split Integrity

The dataset is split at the `intent_id` level.

That means:

- all paraphrases of an intent stay together
- both graph bindings of an intent stay together
- no train/dev/test leakage occurs through paraphrase or entity memorization

Split sizes:

- train: `1,820` rows, `70` intents
- dev: `390` rows, `15` intents
- test: `364` rows, `14` intents

## Why This Is A Good V1

The dataset now teaches the model the things that matter most for this ontology:

- direct one-hop retrieval
- segment, company, and offering counts
- channel and customer-type reverse lookups
- offering hierarchy traversal
- monetization rollups through `OFFERS*`
- company-wide and segment-wide aggregation
- unsupported financial, temporal, and pricing requests
- ambiguous requests that should not be answered with guessed Cypher

Just as importantly, it does not overfit to one graph per family anymore.

Most families now see two distinct graph contexts, and the cross-cutting families see more than two where the ontology allows it.

## Remaining Caveats

This is a strong V1, but there are still reasonable future improvements:

- add more casual, messy, and analyst-style user phrasing
- add more entity alias variation if the runtime will support alias resolution
- add another round of paraphrase polishing if later evaluation shows the language is still too templatic
- expand the number of graph contexts for the largest families if we see graph-shape overfitting
- add a small evaluation set of hand-written adversarial examples outside the synthetic generation process

None of those are blockers for the first fine-tuning run.

## Recommended Next Step

Use this corpus for the first training cycle, then evaluate:

- JSON format compliance
- read-only safety
- Cypher validity
- execution success
- exact parameter extraction
- refusal correctness
- generalization from held-out intents

After that first run, the next dataset work should be driven by measured failure modes rather than more blind expansion.
