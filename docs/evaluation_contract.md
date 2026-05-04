# Extraction Evaluation Contract

Status: implemented.

This document defines the target used to evaluate extraction outputs against the manually created gold benchmark.

## Goal

Evaluate how well each extraction pipeline reproduces the manually annotated business-model knowledge graph for each company.

The evaluation target is the canonical ontology graph, not the Neo4j physical storage model and not the natural-language query stack.

## Source Data

The evaluator reads clean JSONL benchmark files from:

```text
evaluation/benchmarks/dev/clean/
evaluation/benchmarks/test/clean/
```

Each JSONL record keeps the five typed-triple fields:

```text
subject; subject_type; relation; object; object_type
```

The primary exact score intentionally ignores the type fields, but they remain necessary for ontology validation and relaxed hierarchy matching.

Predicted triples are loaded from:

```text
outputs/<company>/<pipeline>/latest/resolved_triples.json
```

This means evaluation compares the same post-resolution, post-validation graph that the runtime would load into Neo4j.

For failed or incomplete runs, the evaluation skips the run and reports it as missing.

## Ontology Surface

Allowed node types:

- `Company`
- `BusinessSegment`
- `Offering`
- `CustomerType`
- `Channel`
- `Place`
- `RevenueModel`

Allowed relation types:

- `HAS_SEGMENT`
- `OFFERS`
- `SERVES`
- `OPERATES_IN`
- `SELLS_THROUGH`
- `PARTNERS_WITH`
- `MONETIZES_VIA`

Valid relation schema:

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

Closed canonical labels:

- `CustomerType`
- `Channel`
- `RevenueModel`

Gold and predicted triples should use the canonical label text from `src/ontology/ontology.json`.

## Normalization

Exact matching normalizes only mechanical surface differences:

- apply Unicode NFKC normalization
- trim whitespace
- strip surrounding quote characters
- collapse repeated whitespace
- compare entity names with casefolded keys
- normalize curly apostrophes and dash variants in comparison keys

Relation names and node types require exact labels after trimming.

The clean benchmark may canonicalize company-level market coverage to `Worldwide` during manual curation. This is a benchmark policy, not evaluator-side fuzzy matching.

## Metrics

The evaluator computes only these five scores:

- `precision`
- `recall`
- `f1`
- `macro_f1`
- `relaxed_f1`

Exact precision, recall, F1, and macro-F1 use normalized 3-field edge agreement over:

```text
subject_key
relation
object_key
```

This is the headline exact matching target because the downstream graph edge is operationally the triple, and historical annotation sheets sometimes differed on type fields.

Relaxed F1 is the secondary graph-aware metric. It estimates how much exact F1 penalizes graph-near naming and hierarchy-alignment differences.

Weights:

| Match type | Weight |
| --- | ---: |
| Exact typed-triple match | 1.00 |
| Company alias / lexical normalization | 0.90 |
| Subject/object parent-child hierarchy relation | 0.75 |
| Segment roll-up relation | 0.50 |

Each gold and predicted triple can be matched at most once. Matching is greedy over positive-weight candidate matches sorted by descending weight.

Weighted counts are:

```text
weighted_TP = sum(selected_match_weights)
weighted_FP = unique_predictions - weighted_TP
weighted_FN = unique_gold - weighted_TP
```

## Outputs

Split results are written under:

```text
evaluation/results/<pipeline>/<split>/
```

Cherry-picked results are written under:

```text
evaluation/results/cherry_picked/<pipeline>/<company>/
```

Each evaluated company writes:

- `metrics.json`
- `matched.jsonl`
- `false_positives.jsonl`
- `false_negatives.jsonl`
- `relaxed_matches.jsonl`

Each split or cherry-picked run also writes a `summary.json`.

If the target results folder already contains files, the evaluator asks before overwriting. If overwrite is approved, existing results are replaced only after the new evaluation run succeeds. The `--yes` flag supports deliberate non-interactive reruns.

## Annotation Reliability

Annotation reliability outputs are reporting artifacts, not inputs to the extraction evaluator. They live under:

```text
evaluation/results/annotation_reliability/
```

The inter-annotator Amazon report compares unique normalized five-field triples between Luca and Zhong and reports precision, recall, F1, and Jaccard overall and by relation. The intra-annotator report stores the company-level, combined micro, and macro-average repeatability metrics.

## Presentation Interpretation

Use exact 3-field edge metrics as the objective headline:

```text
Exact 3-field F1 measures edge-level graph agreement.
```

Use relaxed graph-aware F1 as a secondary semantic/hierarchy diagnostic:

```text
Relaxed F1 estimates agreement after documented alias, hierarchy, and roll-up allowances.
```

## Berkshire Error Analysis

Berkshire Hathaway is the clearest example of why the quantitative evaluation must be interpreted carefully.

After benchmark canonicalization, Berkshire remains the main test-set drag. The problem is not primarily naming. The problem is that the benchmark and the predictions use fundamentally different hierarchy levels.

The Berkshire gold benchmark contains:

```text
317 clean triples
17 business segments
124 OFFERS triples
115 MONETIZES_VIA triples
50 SERVES triples
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

This breaks many exact and relaxed comparisons because the subject node of otherwise related facts is different. For example:

```text
gold:      Apparel and Footwear OFFERS Brooks
predicted: Manufacturing OFFERS Brooks
```

Business-wise, the predicted triple is related to the benchmark fact. Structurally, it places the fact at a different hierarchy level, so exact one-to-one edge metrics correctly leave it unmatched, while relaxed graph-aware matching can only recover part of the hierarchy alignment.

This effect is large. Test-set exact 3-field micro-F1 with and without Berkshire:

| Pipeline | With Berkshire | Without Berkshire |
| --- | ---: | ---: |
| `zero-shot` | 0.228 | 0.416 |
| `memo_graph_only` | 0.245 | 0.417 |
| `analyst` | 0.242 | 0.501 |

Test-set relaxed micro-F1 with and without Berkshire:

| Pipeline | With Berkshire | Without Berkshire |
| --- | ---: | ---: |
| `zero-shot` | 0.257 | 0.444 |
| `memo_graph_only` | 0.269 | 0.440 |
| `analyst` | 0.276 | 0.511 |

The analyst pipeline performs especially poorly on Berkshire under exact micro metrics because it produces a richer graph with many additional subsidiary, operating-company, and product nodes. These may be useful business-model details, but they become false positives when the benchmark represents the same business through a different hierarchy.

Presentation interpretation:

```text
Berkshire shows that exact edge F1 is sensitive to hierarchy alignment. It measures agreement with the benchmark structure, not only extraction quality.
```

## Planned Qualitative Evaluation Layer

Exact and relaxed F1 are useful quantitative scores, but they do not fully capture the quality of a business-model knowledge graph.

The core limitation is that the task is partly subjective. The same business fact can be represented with different naming choices, geographic abstraction, or granularity:

- `Palantir Technologies Inc.` vs `Palantir`
- `Amazon Web Services` vs `Amazon Web Services (AWS)`
- `United States`, `Europe`, and `Asia` vs `worldwide`
- one broad segment such as `Manufacturing` vs several detailed operating groups

Exact 3-field F1 treats these as wrong unless the normalized edge fields match. Relaxed graph-aware F1 fixes some of this through documented alias, hierarchy, and roll-up allowances, but it still evaluates individual triples rather than the usefulness and faithfulness of the graph as a business-model representation.

For this reason, the final evaluation can include a qualitative layer alongside exact and relaxed metrics.

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

The intended unit of qualitative review is one company and one pipeline output. On the held-out test set, the full qualitative review has twelve independent evaluations: four companies across three pipelines. Including both dev and test splits would produce twenty-four evaluations.

Final reporting should present all three layers separately:

- `Exact 3-field precision`, `recall`, `F1`, and `macro-F1`: normalized edge agreement over `subject`, `relation`, and `object`.
- `Relaxed F1`: graph-aware agreement after documented alias, hierarchy, and roll-up allowances.
- `Qualitative graph score`: rubric-based assessment of coverage, granularity, geography, noise, and usefulness.

This framing keeps exact F1 as a reproducible lower-bound metric while acknowledging that exact edge matching alone is too brittle for a subjective knowledge-graph extraction task.
