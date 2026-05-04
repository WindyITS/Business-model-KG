# Extraction Evaluation Contract

Status: implemented.

This document defines the target used to evaluate extraction outputs against the manually created gold benchmark.

## Goal

Evaluate how well each extraction pipeline reproduces the manually annotated business-model knowledge graph for each company.

The evaluation target is the canonical ontology graph, not the Neo4j physical storage model and not the natural-language query stack.

## Compared Triple Shape

Every gold and predicted record is compared as a five-field triple:

```text
subject; subject_type; relation; object; object_type
```

Structured form:

```json
{
  "subject": "Apple",
  "subject_type": "Company",
  "relation": "HAS_SEGMENT",
  "object": "Americas",
  "object_type": "BusinessSegment"
}
```

The gold benchmark may be authored outside the repo in the semicolon format:

```text
entity; entity_type; link; entity; entity_type
```

During parsing, these columns map to:

```text
subject; subject_type; relation; object; object_type
```

In the repo, the benchmark is stored only in clean JSONL files:

```text
evaluation/benchmarks/dev/clean/
evaluation/benchmarks/test/clean/
```

The clean format is JSONL, with one typed triple per line.
Each file is named with the company slug used in `outputs/`, for example `microsoft.jsonl`, `adobe.jsonl`, or `berkshire.jsonl`.

## Final Node Types

Allowed node types are:

- `Company`
- `BusinessSegment`
- `Offering`
- `CustomerType`
- `Channel`
- `Place`
- `RevenueModel`

## Final Relation Types

Allowed relation types are:

- `HAS_SEGMENT`
- `OFFERS`
- `SERVES`
- `OPERATES_IN`
- `SELLS_THROUGH`
- `PARTNERS_WITH`
- `MONETIZES_VIA`

## Valid Relation Schema

The valid relation schema is:

| Relation | Subject type | Object type |
| --- | --- | --- |
| `HAS_SEGMENT` | `Company` | `BusinessSegment` |
| `OFFERS` | `Company` | `Offering` |
| `OFFERS` | `BusinessSegment` | `Offering` |
| `OFFERS` | `Offering` | `Offering` |
| `SERVES` | `BusinessSegment` | `CustomerType` |
| `OPERATES_IN` | `Company` | `Place` |
| `SELLS_THROUGH` | `BusinessSegment` | `Channel` |
| `SELLS_THROUGH` | `Offering` | `Channel` |
| `PARTNERS_WITH` | `Company` | `Company` |
| `MONETIZES_VIA` | `Offering` | `RevenueModel` |

## Closed Canonical Labels

These node types use closed canonical labels:

- `CustomerType`
- `Channel`
- `RevenueModel`

Gold triples and predicted triples must use the canonical label text from `src/ontology/ontology.json`.

Examples:

- use `large enterprises`, not `enterprise customers`
- use `direct sales`, not `sales force`
- use `subscription`, not `subscriptions`

## Gold Authority

Gold triples are authoritative.

The evaluator should not attempt to correct, reject, or reinterpret gold triples. The clean benchmark is manually reviewed before evaluation and is treated as the authoritative source.

## Predicted Triples

Predicted triples should be loaded from:

```text
outputs/<company>/<pipeline>/latest/resolved_triples.json
```

This means evaluation compares the same post-resolution, post-validation graph that the runtime would load into Neo4j.

For failed or incomplete runs, the evaluation should skip the run and report it as missing, not score it as zero unless we intentionally choose that policy later.

The evaluation script is:

```text
evaluation/scripts/evaluate.py
```

Usage examples live in [`../evaluation/README.md`](../evaluation/README.md).

Split results should be written under:

```text
evaluation/results/<pipeline>/<split>/
```

Cherry-picked results should be written under:

```text
evaluation/results/cherry_picked/<pipeline>/<company>/
```

If the target results folder already contains files, the evaluator should ask before overwriting:

```text
There are already files in the results folder <path>. Proceeding with a new evaluation is going to overwrite them. Do you want to proceed? [Y/n]
```

If the answer is `n` or `no`, no evaluation should be performed and the existing files should be left unchanged.

If overwrite is approved, existing results should be replaced only after the new evaluation run succeeds.

The evaluator supports non-interactive overwrites for deliberate reruns.

## Strict Normalization

Strict matching should normalize only mechanical surface differences.

For entity names:

- apply Unicode NFKC normalization
- trim whitespace
- strip surrounding quote characters
- collapse repeated whitespace
- compare with casefolded keys
- normalize curly apostrophes and dash variants in comparison keys

For `Place` values:

- apply the same mechanical entity-name normalization as other entity names.

The clean benchmark may also canonicalize company-level market coverage to `Worldwide` during manual curation. This is used when the gold annotation lists several countries or regions that jointly express a global operating footprint, while the extraction pipelines represent the same footprint with the broader `Worldwide` label.

This geography canonicalization is a benchmark curation policy, not an evaluator-side fuzzy match. It should be applied only to `Company` `OPERATES_IN` `Place` triples when the broader label preserves the intended meaning. It should not be used to rewrite business segments, offerings, customer types, channels, or revenue models.

After geography canonicalization, duplicate triples should be removed from the clean JSONL files. For example, if several country-level `OPERATES_IN` rows become the same `Company OPERATES_IN Worldwide` triple, the clean benchmark should keep one canonical row.

For `CustomerType`, `Channel`, and `RevenueModel`:

- compare canonical labels case-insensitively after mechanical cleanup.

For relation names and node types:

- require exact labels after trimming.

## Strict Match Definition

A predicted triple is a strict true positive if, after strict normalization, all five fields match a gold triple:

```text
subject_key
subject_type
relation
object_key
object_type
```

False positives are valid predicted triples that do not match any gold triple.

False negatives are valid gold triples that are not matched by any predicted triple.

Metrics:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

If a denominator is zero, the evaluation script should handle it explicitly and consistently.

## Hand-Matched Evaluation

Strict matching is the primary metric, but it may penalize harmless naming differences.

The preferred relaxed metric should use manually tagged unmatched triples.

For every evaluated company, the evaluator should write:

```text
unmatched_for_review.csv
```

The file should live inside the company-specific result folder for that exact run:

```text
evaluation/results/<pipeline>/<split>/companies/<company>/unmatched_for_review.csv
evaluation/results/cherry_picked/<pipeline>/<company>/unmatched_for_review.csv
```

This file should contain all strict false positives and strict false negatives, separated by a `source` column:

- `source=gold`: a gold triple missed by the pipeline
- `source=predicted`: a predicted triple not present in the gold benchmark

The review CSV should include these columns:

```text
row_id,match_id,source,subject,subject_type,relation,object,object_type
```

The human reviewer assigns the same `match_id` to gold and predicted rows that correspond to the same real triple despite naming differences. For example, the first hand-matched pair can use `match_id=1`, the second can use `match_id=2`, and so on.

Each non-empty `match_id` must be used on exactly one `source=gold` row and exactly one `source=predicted` row. Reusing the same `match_id` for multiple pairs is treated as invalid and does not affect second-tier metrics.

After the review CSV is edited, run the hand-match script described in
[`../evaluation/README.md`](../evaluation/README.md#hand-match-review).

The hand-match script should compute second-tier metrics by starting from strict TP/FP/FN and converting each accepted human match into one additional true positive, one fewer false positive, and one fewer false negative.

The script should write:

```text
hand_matched/companies/<company>/metrics.json
hand_matched/summary.json
```

If `hand_matched/` already contains files, the hand-match script should ask before overwriting. If the answer is `n` or `no`, no hand-matched metrics should be recomputed and existing files should be left unchanged.

If overwrite is approved, existing hand-matched results should be replaced only after the new hand-matched computation succeeds.

The script supports non-interactive overwrites for deliberate reruns.

## Hand-Matched Metric Definition

A hand-matched correspondence is accepted only when a `match_id` appears on exactly one unmatched gold row and exactly one unmatched predicted row.

Each accepted correspondence converts one strict false positive and one strict false negative into one additional true positive:

```text
hand_matched_tp = strict_tp + accepted_matches
hand_matched_fp = strict_fp - accepted_matches
hand_matched_fn = strict_fn - accepted_matches
```

Hand-matched metrics should be reported separately from strict metrics.

Recommended labels:

- `Strict`
- `Hand-matched`

## Metric Scope For The First Evaluator

The first evaluator should keep the metric surface small:

- overall
- by pipeline
- by company
- by company and pipeline

Recommended pipeline comparison:

- `zero-shot`
- `memo_graph_only`
- `analyst`

Relation-specific and entity-specific metrics can be added later once the core evaluator is stable.

## Required Error Reports

For each evaluated company and pipeline, write:

- matched triples
- false positives
- false negatives
- unmatched review CSV

The false positive and false negative reports are the main input for error analysis.

## Presentation Interpretation

Use strict metrics as the cleanest objective score:

```text
Strict F1 measures exact graph agreement.
```

Use hand-matched metrics as the human-adjudicated semantic score:

```text
Hand-matched F1 measures graph agreement after manually recorded unmatched-triple correspondences.
```

The final presentation should clearly state that hand-matched metrics depend only on explicit `match_id` labels in the review CSV.

## Berkshire Error Analysis

Berkshire Hathaway is the clearest example of why the quantitative evaluation must be interpreted carefully.

After benchmark canonicalization and hand matching, Berkshire remains the main test-set drag. The problem is not primarily naming. The problem is that the benchmark and the predictions use fundamentally different hierarchy levels.

The Berkshire gold benchmark contains:

```text
327 clean triples
17 business segments
127 OFFERS triples
127 MONETIZES_VIA triples
51 SERVES triples
```

The three prediction outputs use much smaller top-level segment structures:

```text
zero-shot:       43 triples, 5 business segments
memo_graph_only: 121 triples, 5 business segments
analyst:         298 triples, 5 business segments
```

The benchmark uses detailed operating and product groups such as:

- `Industrial products manufacturing`
- `Building Products`
- `Transportation Products`
- `Foodservice Technologies`
- `Apparel and Footwear`
- `Retail Solutions`
- `Plumbing & Refrigeration`
- `Electrical Group`

All three predictions instead use broad reportable/top-level groups such as:

- `Insurance`
- `Burlington Northern Santa Fe`
- `Berkshire Hathaway Energy`
- `Manufacturing`
- `Service and Retailing`

This breaks many exact and hand-matched comparisons because the subject node of otherwise related facts is different. For example:

```text
gold:      Apparel and Footwear OFFERS Brooks
predicted: Manufacturing OFFERS Brooks
```

Business-wise, the predicted triple is related to the benchmark fact. Structurally, it places the fact at a different hierarchy level, so the exact and hand-matched one-to-one triple metrics correctly leave it unmatched.

This effect is large. Test-set strict F1 with and without Berkshire:

| Pipeline | With Berkshire | Without Berkshire |
| --- | ---: | ---: |
| `zero-shot` | 0.252 | 0.468 |
| `memo_graph_only` | 0.224 | 0.379 |
| `analyst` | 0.219 | 0.465 |

Test-set hand-matched F1 without Berkshire:

| Pipeline | Hand-matched F1 without Berkshire |
| --- | ---: |
| `zero-shot` | 0.601 |
| `memo_graph_only` | 0.427 |
| `analyst` | 0.513 |

The analyst pipeline performs especially poorly on Berkshire under strict triple metrics because it produces a richer graph with many additional subsidiary, operating-company, and product nodes. These may be useful business-model details, but they become false positives when the benchmark represents the same business through a different hierarchy.

Presentation interpretation:

```text
Berkshire shows that exact and hand-matched triple F1 are sensitive to hierarchy alignment. They measure agreement with the benchmark structure, not only extraction quality.
```

## Planned Qualitative Evaluation Layer

Strict and hand-matched F1 are useful quantitative scores, but they do not fully capture the quality of a business-model knowledge graph.

The core limitation is that the task is partly subjective. The same business fact can be represented with different naming choices, geographic abstraction, or granularity:

- `Palantir Technologies Inc.` vs `Palantir`
- `Amazon Web Services` vs `Amazon Web Services (AWS)`
- `United States`, `Europe`, and `Asia` vs `worldwide`
- one broad segment such as `Manufacturing` vs several detailed operating groups

Strict F1 treats these as wrong unless all five fields match after mechanical normalization. Hand-matched F1 fixes some of this through human-reviewed correspondences, but it still evaluates individual triples rather than the usefulness and faithfulness of the graph as a business-model representation.

For this reason, the final evaluation should include a qualitative layer alongside strict and hand-matched metrics.

The qualitative layer should not ask an evaluator to inspect a full 10-K, the full gold graph, the full predicted graph, and the ontology at once. That would overload the evaluator and make judgments unstable.

The qualitative layer should also not flatten the graph into a simple list of entities. The Berkshire case shows that hierarchy is itself one of the main evaluation problems. A prediction can mention the right offering but attach it to a broader parent segment than the benchmark. A flat summary would hide this issue.

Instead, evaluation should use a compressed hierarchical-profile workflow:

1. Build a hierarchical company reference profile from the gold benchmark.
2. Build a hierarchical prediction profile from one pipeline output.
3. Ask an evaluator to compare the two profiles using a fixed rubric.

The company reference profile should summarize the benchmark without losing parent-child structure:

- company name
- business segments
- offerings attached to each segment
- child offerings attached to parent offerings
- customer types attached to each segment
- sales channels attached to each segment or offering
- revenue models attached to each offering
- geographies
- known acceptable aliases or naming variants

The prediction profile should use the same structure for a single company and pipeline.

The profile should preserve hierarchy in a compact textual form. Example:

```text
Company: Berkshire Hathaway

Segments:
- Manufacturing
  - offers: Precision Castparts, Lubrizol, Fruit of the Loom, Brooks, Clayton Homes
  - serves: consumers, businesses
  - channels: direct sales
- Service and Retailing
  - offers: McLane, FlightSafety, NetJets, Pilot Travel Centers
  - serves: consumers, businesses

Geography:
- Worldwide
```

The evaluator should explicitly check whether mismatches are caused by:

- missing facts
- unsupported extra facts
- alias or naming differences
- broader/narrower hierarchy placement
- different but reasonable segment taxonomy
- incorrect relation choice

For hierarchy mismatches, the evaluator should distinguish between:

- `equivalent placement`: the same fact is attached to the same logical parent.
- `acceptable roll-up`: the prediction attaches the fact to a broader parent, but the roll-up is business-wise reasonable.
- `over-flattening`: the prediction loses important intermediate structure.
- `wrong placement`: the prediction attaches the fact to the wrong segment or offering.

The qualitative evaluator should then score the prediction on stable dimensions:

| Dimension | Question |
| --- | --- |
| Segment coverage | Did the graph capture the important business structure? |
| Offering coverage | Did it capture the important products and services? |
| Hierarchy alignment | Did it attach offerings, customers, channels, and revenue models to the right parent nodes? |
| Roll-up reasonableness | If the prediction is broader than the benchmark, is the broader taxonomy still defensible? |
| Revenue logic | Did it capture how the company makes money? |
| Customer/channel coverage | Did it capture who is served and how offerings are sold or delivered? |
| Geography handling | Did it use a reasonable level of geographic abstraction? |
| Granularity fit | Is the graph too broad, too detailed, or reasonably aligned with the benchmark? |
| Noise / hallucination | Does it add unsupported or misleading entities or relations? |
| Overall usefulness | Would the graph be useful for querying and understanding the business model? |

Each dimension should receive a small numeric score, for example `1-5`, plus a short justification and concrete examples.

The qualitative report should include a hierarchy diagnosis section:

```text
Hierarchy diagnosis:
- Facts correctly placed at the benchmark level:
- Facts captured but attached to broader parents:
- Facts captured but attached to wrong parents:
- Important benchmark substructures missing:
- Useful predicted substructures absent from the benchmark:
```

The intended unit of qualitative review is one company and one pipeline output. With eight companies and three pipelines, the full qualitative review has twenty-four independent evaluations.

Final reporting should present all three layers separately:

- `Strict F1`: exact typed-triple agreement after mechanical normalization.
- `Hand-matched F1`: quantitative agreement after explicit human-reviewed unmatched-triple correspondences.
- `Qualitative graph score`: rubric-based assessment of coverage, granularity, geography, noise, and usefulness.

This framing keeps strict F1 as a reproducible lower-bound metric while acknowledging that exact triple matching alone is too brittle for a subjective knowledge-graph extraction task.
