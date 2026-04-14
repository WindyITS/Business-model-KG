# Text-to-Cypher Coverage Grid

## Purpose

This document defines the first-pass query families for the text-to-Cypher dataset.

Each family represents a distinct graph reasoning pattern that the fine-tuned model must learn.

The goal is not to list every possible question phrasing. The goal is to enumerate the canonical query shapes that matter for this ontology and current graph.

## Coverage Rules

- split examples by `intent_id`, not by paraphrase
- include both direct lookups and ontology-aware rollups
- include unsupported and ambiguous requests
- prefer one canonical Cypher style per family
- use parameterized Cypher in all gold examples

## Query Families

| Family ID | Query Family | Typical User Ask | Core Graph Pattern | Difficulty | Seed Target |
| --- | --- | --- | --- | --- | --- |
| `QF01` | Company to segments | "What segments does Microsoft have?" | `(:Company)-[:HAS_SEGMENT]->(:BusinessSegment)` | low | 8 |
| `QF02` | Company to company-level offerings | "What offerings does Palantir have at the company level?" | `(:Company)-[:OFFERS]->(:Offering)` | low | 6 |
| `QF03` | Segment to direct offerings | "What does Intelligent Cloud offer?" | `(:BusinessSegment)-[:OFFERS]->(:Offering)` | low | 8 |
| `QF04` | Offering family children | "What is inside LinkedIn?" | `(:Offering)-[:OFFERS]->(:Offering)` | medium | 8 |
| `QF05` | Offering descendant expansion | "What offerings sit under Microsoft 365 Commercial cloud?" | `(:Offering)-[:OFFERS*1..]->(:Offering)` | medium | 8 |
| `QF06` | Company to places | "Where does Microsoft operate?" | `(:Company)-[:OPERATES_IN]->(:Place)` | low | 8 |
| `QF07` | Company to partners | "Who does Palantir partner with?" | `(:Company)-[:PARTNERS_WITH]->(:Company)` | low | 8 |
| `QF08` | Segment to customer types | "Who does the commercial segment serve?" | `(:BusinessSegment)-[:SERVES]->(:CustomerType)` | low | 8 |
| `QF09` | Segment to channels | "How does the commercial segment sell?" | `(:BusinessSegment)-[:SELLS_THROUGH]->(:Channel)` | low | 8 |
| `QF10` | Fallback offering to channels | "How is this offering sold when no segment anchor exists?" | `(:Offering)-[:SELLS_THROUGH]->(:Channel)` | medium | 4 |
| `QF11` | Offering to revenue models | "How does LinkedIn make money?" | `(:Offering)-[:MONETIZES_VIA]->(:RevenueModel)` | low | 8 |
| `QF12` | Segment revenue rollup | "How does Intelligent Cloud make money?" | `(:BusinessSegment)-[:OFFERS]->(root:Offering)` then `(root)-[:OFFERS*0..]->(o:Offering)-[:MONETIZES_VIA]->(:RevenueModel)` | high | 10 |
| `QF13` | Company revenue rollup | "How does Microsoft monetize its business?" | `(:Company)-[:HAS_SEGMENT]->(:BusinessSegment)-[:OFFERS]->(root:Offering)` then `(root)-[:OFFERS*0..]->(o:Offering)-[:MONETIZES_VIA]->(:RevenueModel)` | high | 10 |
| `QF14` | Company channel rollup | "What channels does Microsoft use?" | `(:Company)-[:HAS_SEGMENT]->(:BusinessSegment)-[:SELLS_THROUGH]->(:Channel)` | medium | 6 |
| `QF15` | Company customer rollup | "What customer types does Microsoft serve?" | `(:Company)-[:HAS_SEGMENT]->(:BusinessSegment)-[:SERVES]->(:CustomerType)` | medium | 6 |
| `QF16` | Segment-offering plus monetization join | "Which offerings in More Personal Computing monetize via advertising?" | `(:BusinessSegment)-[:OFFERS]->(root:Offering)` then `(root)-[:OFFERS*0..]->(o:Offering)-[:MONETIZES_VIA]->(:RevenueModel)` with filters | high | 8 |
| `QF17` | Segment-offering plus channel join | "Which segments sell through resellers?" | `(:BusinessSegment)-[:SELLS_THROUGH]->(:Channel)` with reverse lookup or join | medium | 6 |
| `QF18` | Segment-offering plus customer join | "Which segments serve developers?" | `(:BusinessSegment)-[:SERVES]->(:CustomerType)` with reverse lookup | medium | 6 |
| `QF19` | Intersections | "Which segments serve developers and sell through direct sales?" | multiple predicates across one segment | high | 8 |
| `QF20` | Counts | "How many offerings does Microsoft have under Intelligent Cloud?" | any of the above plus `COUNT(DISTINCT ...)` | medium | 8 |
| `QF21` | Ranking and ordering | "List Microsoft's offerings alphabetically." | list query plus `ORDER BY` and optional `LIMIT` | low | 6 |
| `QF22` | Existence checks | "Does Microsoft partner with OpenAI?" | boolean-style existence query | medium | 6 |
| `QF23` | Path anchoring | "Which segment is Azure under?" | reverse traversal from offering to segment through `OFFERS*` | high | 8 |
| `QF24` | Parent lookup | "What parent offering is Microsoft Teams under?" | reverse traversal from child offering to parent offering | medium | 6 |
| `QF25` | Unsupported financial metrics | "What revenue did Microsoft make from Azure?" | not answerable from current graph | medium | 8 |
| `QF26` | Unsupported time-based requests | "Which segment grew fastest last year?" | not answerable from current graph | medium | 8 |
| `QF27` | Unsupported ontology gaps | "Who are Microsoft's suppliers?" | not answerable from current graph | medium | 8 |
| `QF28` | Ambiguous requests | "How does Copilot make money?" when entity scope is unclear | likely `answerable=false` unless disambiguation rules are added | high | 6 |

Total first-pass seed target: 208 canonical intents.

That total is intentionally larger than the minimum. We can trim later, but it is better to begin with broad coverage and collapse duplicates than to discover coverage holes late.

## Family Notes

### Direct Lookups

These are the easiest families and should be written first:

- `QF01`
- `QF03`
- `QF06`
- `QF07`
- `QF08`
- `QF09`
- `QF11`

### Ontology-Aware Rollups

These are the most important families because they reflect the repo's modeling choices:

- `QF12`
- `QF13`
- `QF14`
- `QF15`
- `QF16`
- `QF23`

They should receive extra review because they are the most likely place for subtle gold-query mistakes.

### Negative And Ambiguous Families

These should be present from the start rather than added later:

- `QF25`
- `QF26`
- `QF27`
- `QF28`

The model must learn not to hallucinate unsupported properties or ontology relations.

## Seed Authoring Order

Recommended order:

1. direct lookups
2. reverse lookups
3. rollups
4. counts and ranking
5. intersections
6. negative and ambiguous families

## Next Artifact

The next dataset artifact should be a canonical seed-intent file with one row per seed intent.

Each row should include:

- `intent_id`
- `family_id`
- canonical question
- gold Cypher
- params
- answerability
- result shape
- difficulty
