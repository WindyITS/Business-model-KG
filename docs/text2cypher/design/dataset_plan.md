# Text-to-Cypher Dataset Plan

## Goal

Build a high-quality supervised dataset for fine-tuning a small model to translate user requests into read-only Cypher queries for this repo's ontology and Neo4j graph.

Current status:

- the active dataset build is complete
- the final readiness assessment is documented in [`readiness_v3.md`](./readiness_v3.md)
- the machine-readable artifacts are generated locally under `datasets/text2cypher/v3/`
- the train-facing `messages` export is written to `datasets/text2cypher/v3/training/`
- the held-out evaluation export is written to `datasets/text2cypher/v3/evaluation/`
- the export step publishes that local build output to Hugging Face via `text2cypher-export-hf` or `scripts/text2cypher/export_hf_text2cypher_dataset.py`
- the next engineering step is model training and evaluation rather than more dataset scaffolding

The target runtime behavior is:

1. receive a natural-language user request
2. decide whether the request is answerable from the target ontology-backed graph
3. emit a safe Cypher query plus parameters when it is answerable

## Why This Repo Needs A Custom Plan

This graph is not a generic property graph.

It has a narrow, closed ontology:

- node types: `Company`, `BusinessSegment`, `Offering`, `CustomerType`, `Channel`, `Place`, `RevenueModel`
- relations: `HAS_SEGMENT`, `OFFERS`, `SERVES`, `OPERATES_IN`, `SELLS_THROUGH`, `PARTNERS_WITH`, `MONETIZES_VIA`

It also uses canonical extraction rules that matter at query time:

- `SERVES` is canonical at `BusinessSegment` scope
- `SELLS_THROUGH` is segment-first, with `Offering` only as fallback
- `MONETIZES_VIA` is canonical at `Offering` scope
- convenience rollups are intentionally recovered downstream in Cypher rather than materialized during extraction

That means the model must learn both:

- direct graph lookups
- ontology-aware rollups such as traversing `Company -> HAS_SEGMENT -> OFFERS* -> MONETIZES_VIA`

## Synthetic Fixture Assumptions

The dataset should be built from synthetic graph fixtures that mirror the production ontology and query contract.

Each synthetic fixture should:

- read-only Cypher only
- use the same labels and relationship types as the production ontology
- identify `BusinessSegment` and `Offering` by the production-style composite of `company_name` plus `name` whenever company-scoped inventory is involved
- support hierarchy-aware geography queries through `within_places` and `includes_places` on `Place`
- the model should not assume extra properties such as revenue amounts, dates, filing metadata, descriptions, or embeddings

Important note:

Synthetic fixtures should mimic the production graph shape closely enough that the trained model does not learn imaginary properties or unsupported schema patterns.

## Output Contract

The fine-tuned model should emit compact JSON, not prose. The canonical corpus retains the provenance and validation fields; the generated trainer-facing export should reshape those rows into a chat-style or prompt/completion-style SFT record whose assistant target is the JSON contract below.

Recommended shape:

```json
{
  "answerable": true,
  "cypher": "MATCH ... RETURN ...",
  "params": {
    "company": "Northstar Systems"
  }
}
```

For unsupported requests:

```json
{
  "answerable": false,
  "reason": "not_in_graph"
}
```

Notes:

- prefer parameterized Cypher over hardcoded literals
- never emit write queries
- never emit explanations, markdown, or chain-of-thought
- in the canonical source corpus, refusal rows may keep `gold_cypher: null`; in `training/messages.jsonl`, refusal targets must become explicit JSON refusal objects

## Dataset Construction Workflow

### Phase 1. Freeze The Query Contract

Before writing examples, lock the target behavior:

- output JSON only
- use read-only Cypher only
- use parameters for entity values
- define the allowed clauses

Recommended allowed clauses:

- `MATCH`
- `OPTIONAL MATCH`
- `WHERE`
- `WITH`
- `RETURN`
- `ORDER BY`
- `LIMIT`
- aggregation such as `COUNT`, `COLLECT`, `DISTINCT`

Disallowed by default:

- `CREATE`
- `MERGE`
- `DELETE`
- `SET`
- `REMOVE`
- `CALL` unless explicitly whitelisted later

### Phase 2. Build The Coverage Grid

Create a query-family inventory from the ontology and target graph schema.

This coverage grid is the source of truth for:

- which user intents we care about
- which graph patterns the model must learn
- how many canonical seed questions we want per family

See [`coverage_grid.md`](./coverage_grid.md).

### Phase 2A. Define The Dataset Layers

Before authoring examples, keep these layers distinct:

- `family_id`
  The broad query family from the coverage grid.
  Example: `QF12 = segment revenue rollup`.
- `intent`
  One slot-based semantic task inside that family.
  Example: "return the distinct revenue models for a business segment".
- `intent_id`
  The stable identifier for that semantic task.
  It should identify the reusable query pattern, not a specific company name.
- `binding`
  A concrete filling of the intent slots using invented but ontology-valid synthetic values.
  Example: `company=Northstar Systems`, `segment=Industrial AI`.
- `example`
  One actual natural-language phrasing of that bound case.
  Example: "How does Northstar Systems' Industrial AI segment make money?"

Practical rule:

- if the semantic task stays the same and only the concrete entity values change, keep the same `intent_id`
- if the query logic changes, create a new `intent_id`

### Phase 2B. Build A Synthetic Fixture Library

Before generating training examples, define a reusable library of synthetic graph fixtures.

Each fixture should:

- be fully valid under the ontology
- use realistic but invented company, segment, and offering names
- cover both common and rarer valid graph structures
- be loadable into a temporary Neo4j database for execution validation

Fixture design goals:

- represent simple one-hop structures
- represent segment-offering hierarchies
- represent rollup cases involving `OFFERS*`
- represent multiple customer, channel, and revenue-model combinations
- represent ambiguous names when useful

No fixture should depend on currently loaded production data.

Training-readiness note:

- one focused instance per fixture class is enough for early seed authoring and schema review
- it is not enough as the sole training substrate for the final dataset
- the complex fixture classes should have multiple concrete graph instances with materially different structure
- especially for `FX04`, `FX06`, `FX09`, `FX10`, and `FX11`, include overlap, near-misses, mixed hierarchical and non-hierarchical roots, and more than one valid retrieval path where the ontology allows it
- `FX12` should include multiple ambiguity modes, including graph-backed monetization ambiguity, graph-backed channel-scope ambiguity, and graph-free pronoun-only refusal cases

### Phase 3. Author Canonical Seed Intents

For each query family:

1. write canonical user questions
2. write the intended Cypher
3. assign parameters
4. define expected result shape
5. tag difficulty and family id

Each canonical seed intent should represent one unique graph reasoning pattern.

Recommended authoring order inside each family:

1. write the slot-based intent definition
2. write the canonical parameterized Cypher
3. bind the intent to synthetic values from the fixture library
4. write the first concrete question using that binding

Examples:

- "What business segments does Northstar Systems have?"
- "Which customer types does Industrial AI serve?"
- "How does Northstar Systems monetize Industrial AI?"
- "Which partners does Meridian Nexus have?"

The first-pass seed inventory should aim for roughly 150 to 220 canonical intents across all families, with the expectation that duplicate or low-value seeds may be pruned after review.

### Phase 4. Validate Gold Cypher

Each canonical seed must pass all of the following:

- syntax check
- read-only safety check
- execution against a synthetic validation Neo4j database or fixture-backed test graph
- result-shape check
- ontology sanity check

The large teacher model may help draft Cypher, but the executed and validated query is the gold truth.

### Phase 5. Generate Natural-Language Variants

Once the gold query is validated:

1. generate 10 to 25 paraphrases of the canonical question
2. keep all variants attached to the same `intent_id`
3. preserve the same Cypher and params unless the paraphrase changes intent

Variant generation should diversify:

- wording
- sentence length
- explicit versus implicit entity references
- business phrasing
- analyst phrasing
- short imperative requests
- question form versus command form

The teacher model is most useful here.

### Phase 5A. Use Synthetic Names Intentionally

All bound training examples should come from synthetic fixtures.

Recommended balance:

- 85 to 100 percent synthetic concrete examples with invented names
- 0 to 15 percent optional slotized examples if they improve generalization

Examples:

- synthetic concrete example:
  "How does Northstar Systems' Industrial AI segment make money?"
- slotized example:
  "How does [BusinessSegment] make money?"
- semi-slotized example:
  "How does [Company]'s [BusinessSegment] segment make money?"

Why synthetic concrete names are still important:

- the model must learn realistic surface forms for company, segment, and offering names
- the model must learn to map named entities into parameters
- the model should generalize across many valid graph structures, not memorize a tiny live database

Important distinction:

- training inputs should use synthetic names or slots
- gold outputs should remain parameterized Cypher plus params
- the model should not be trained to hardcode literal values inside the Cypher string

### Phase 6. Build Negative And Near-Miss Examples

A robust text-to-Cypher model must learn when not to answer.

Include:

- unsupported property requests
- temporal requests when no time dimension exists
- financial metric requests not present in the graph
- supplier, competitor, or employee questions not represented in the ontology
- ambiguous questions that should return `answerable=false`

Target at least 20 to 30 percent of the final dataset as negative or near-miss examples.

### Phase 7. Split By Intent, Not By Paraphrase

This is mandatory.

All variants of the same canonical seed intent must stay in the same split.

Use:

- training split by `intent_id`
- validation split by `intent_id`
- test split by `intent_id`

Never allow one paraphrase into train and another paraphrase of the same query family item into validation or test.

### Phase 8. Package Training Examples

Each canonical source row should contain:

- `intent_id`
- optional `binding_id`
- `query_family`
- `question`
- `answerable`
- `cypher`
- `params`
- `result_shape`
- `difficulty`
- optional metadata such as source company, seed author, and validation status

The generated SFT export should then turn those rows into a trainer-facing message or prompt/completion schema, preserve the split assignment, and keep the assistant output as strict JSON only.

If bindings are synthetic only, `binding_id` should refer to a synthetic fixture assignment rather than a production graph entity.

For the model-facing export, the same semantic content should be rendered into a strict SFT envelope such as `messages` or `prompt`/`completion`, without losing the canonical metadata needed for provenance or evaluation.

## Row Semantics

A single concrete training row should usually mean:

- one semantic task identified by `intent_id`
- one concrete entity binding, optionally identified by `binding_id`
- one phrasing of that bound case

Example:

- `family_id`: segment revenue rollup
- `intent_id`: "return the revenue models of a business segment"
- `binding_id`: "northstar_industrial_ai"
- `question`: "How does Northstar Systems' Industrial AI segment make money?"

If another row says:

- `question`: "What are the revenue models for Meridian Nexus' Public Sector segment?"
- `params`: `company=Meridian Nexus`, `segment=Public Sector`

then it may reuse the same `intent_id` if the query logic is unchanged.

## Suggested Record Shape

```json
{
  "intent_id": "qf12_segment_revenue_rollup_basic",
  "binding_id": "northstar_industrial_ai",
  "query_family": "QF12",
  "slot_pattern": "How does [BusinessSegment] make money?",
  "question": "How does Northstar Systems' Industrial AI segment make money?",
  "answerable": true,
  "cypher": "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(s:BusinessSegment {name: $segment}) MATCH (s)-[:OFFERS]->(root:Offering) MATCH (root)-[:OFFERS*0..]->(o:Offering) MATCH (o)-[:MONETIZES_VIA]->(r:RevenueModel) RETURN DISTINCT r.name AS revenue_model ORDER BY revenue_model",
  "params": {
    "company": "Northstar Systems",
    "segment": "Industrial AI"
  },
  "result_shape": [
    "revenue_model"
  ],
  "difficulty": "medium"
}
```

Field meanings:

- `intent_id`
  Stable ID for the semantic query pattern.
- `binding_id`
  Optional ID for the concrete entity assignment used in this row.
- `query_family`
  Family label from the coverage grid.
- `slot_pattern`
  Abstract slot-based wording of the reusable task.
- `question`
  Concrete user-facing phrasing used as model input.
- `answerable`
  Whether the graph can answer the question.
- `cypher`
  The validated gold query.
- `params`
  Parameter values passed to the query.
- `result_shape`
  Expected returned columns and types.
- `difficulty`
  Informal complexity tag for analysis and balancing.

## Quality Gates

No example enters the dataset unless it passes:

1. JSON-format validation
2. read-only Cypher validation
3. execution against Neo4j
4. result-shape validation
5. duplicate detection

Additional recommended checks:

- normalize Cypher formatting
- keep one canonical query style per family
- avoid multiple equally valid query styles in the gold set unless intentionally teaching alternatives

## Training Strategy

The intended training path is:

1. fine-tune directly on this repo's ontology-specific dataset
2. build the local dataset workspace under `datasets/text2cypher/v3/`
3. use `datasets/text2cypher/v3/training/train_messages.jsonl` as the training split
4. use `datasets/text2cypher/v3/training/valid_messages.jsonl` as the in-training validation split
5. evaluate on `datasets/text2cypher/v3/evaluation/test_messages.jsonl`
6. use the Apple Silicon MLX LoRA pipeline as the default local implementation path for `Qwen/Qwen3-4B`

No public warm-up stage is planned. The goal is for the model to learn this KG's query contract directly from the locally built corpus rather than from a broader mixed-schema Cypher dataset.

Implementation note:

- the local training/evaluation workflow is documented in `docs/text2cypher/fine_tuning_mlx.md`
- the held-out set is suitable as a hard-query benchmark, but it is not yet a full refusal benchmark

## Deliverables

The dataset effort should produce:

- coverage grid
- canonical seed intent list
- synthetic fixture library
- validated gold Cypher set
- paraphrase-expanded training set
- negative example set
- generated `messages.jsonl` SFT export suitable for fine-tuning
- dedicated held-out evaluation set
- evaluation harness based on execution accuracy

## Immediate Next Step

Run the first Gemma `E4B` LoRA baseline on the checked-in `v3` dataset, then compare held-out JSON validity, structured match, and execution accuracy before tuning the training hyperparameters further.
