# Text-to-Cypher Dataset Plan

## Goal

Build a high-quality supervised dataset for fine-tuning a small model to translate user requests into read-only Cypher queries for this repo's ontology and Neo4j graph.

The target runtime behavior is:

1. receive a natural-language user request
2. decide whether the request is answerable from the current graph
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

## Current Graph Assumptions

The initial dataset should reflect the graph as it exists today:

- read-only Cypher only
- current loaded nodes are identified primarily by label plus `name`
- the model should not assume extra properties such as revenue amounts, dates, filing metadata, descriptions, or embeddings

Important note:

The current loader merges nodes by `name` within each label. That is workable for a small controlled graph, but if the database grows across many companies, entity identity should be strengthened before scaling the text-to-Cypher dataset.

## Output Contract

The fine-tuned model should emit compact JSON, not prose.

Recommended shape:

```json
{
  "answerable": true,
  "cypher": "MATCH ... RETURN ...",
  "params": {
    "company": "Microsoft"
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

Create a query-family inventory from the ontology and current graph shape.

This coverage grid is the source of truth for:

- which user intents we care about
- which graph patterns the model must learn
- how many canonical seed questions we want per family

See [`text2cypher_coverage_grid.md`](./text2cypher_coverage_grid.md).

### Phase 3. Author Canonical Seed Intents

For each query family:

1. write canonical user questions
2. write the intended Cypher
3. assign parameters
4. define expected result shape
5. tag difficulty and family id

Each canonical seed intent should represent one unique graph reasoning pattern.

Examples:

- "What business segments does Microsoft have?"
- "Which customer types does Intelligent Cloud serve?"
- "How does Microsoft monetize Intelligent Cloud?"
- "Which partners does Palantir have?"

The first-pass seed inventory should aim for roughly 150 to 220 canonical intents across all families, with the expectation that duplicate or low-value seeds may be pruned after review.

### Phase 4. Validate Gold Cypher

Each canonical seed must pass all of the following:

- syntax check
- read-only safety check
- execution against the real Neo4j database
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

Each final record should contain:

- `intent_id`
- `query_family`
- `question`
- `answerable`
- `cypher`
- `params`
- `result_shape`
- `difficulty`
- optional metadata such as source company, seed author, and validation status

## Suggested Record Shape

```json
{
  "intent_id": "segment_revenue_rollup_001",
  "query_family": "segment_revenue_rollup",
  "question": "How does Intelligent Cloud make money?",
  "answerable": true,
  "cypher": "MATCH (:Company {name: $company})-[:HAS_SEGMENT]->(s:BusinessSegment {name: $segment}) MATCH (s)-[:OFFERS]->(root:Offering) MATCH (root)-[:OFFERS*0..]->(o:Offering) MATCH (o)-[:MONETIZES_VIA]->(r:RevenueModel) RETURN DISTINCT r.name AS revenue_model ORDER BY revenue_model",
  "params": {
    "company": "Microsoft",
    "segment": "Intelligent Cloud"
  },
  "result_shape": [
    "revenue_model"
  ],
  "difficulty": "medium"
}
```

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

1. optional short warm-up on public Text-to-Cypher data to reinforce syntax and query structure
2. main fine-tuning pass on this repo's ontology-specific dataset

The ontology-specific pass should dominate the final model behavior.

## Deliverables

The dataset effort should produce:

- coverage grid
- canonical seed intent list
- validated gold Cypher set
- paraphrase-expanded training set
- negative example set
- train/validation/test splits by `intent_id`
- evaluation harness based on execution accuracy

## Immediate Next Step

Start from the coverage grid and define the query families we need to cover.

Once the coverage grid is stable, the next artifact should be a canonical seed-intent file authored directly from that grid.
