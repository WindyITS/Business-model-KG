# Text-to-Cypher Intent Cases By Family

## Purpose

This document is the first draft of the per-family intent inventory for the synthetic-only text-to-Cypher dataset.

It is not the dataset itself. It is the semantic checklist we will use to generate dataset rows one intent at a time.

## Reading Rules

- Intent cases are distinct semantic tasks, not paraphrases.
- All examples assume synthetic fixture values only.
- Slot patterns are examples of user phrasing, not final questions.
- Expected result shapes describe what the query should return, not the actual values.
- Difficulty notes are relative authoring signals, not dataset labels.

## Family Index

- `QF01` Company to segments
- `QF02` Company to company-level offerings
- `QF03` Segment to direct offerings
- `QF04` Offering family children
- `QF05` Offering descendant expansion
- `QF06` Company to places
- `QF07` Company to partners
- `QF08` Segment to customer types
- `QF09` Segment to channels
- `QF10` Fallback offering to channels
- `QF11` Offering to revenue models
- `QF12` Segment revenue rollup
- `QF13` Company revenue rollup
- `QF14` Company channel rollup
- `QF15` Company customer rollup
- `QF16` Segment-offering plus monetization join
- `QF17` Segment-offering plus channel join
- `QF18` Segment-offering plus customer join
- `QF19` Intersections
- `QF20` Counts
- `QF21` Ranking and ordering
- `QF22` Existence checks
- `QF23` Path anchoring
- `QF24` Parent lookup
- `QF25` Unsupported financial metrics
- `QF26` Unsupported time-based requests
- `QF27` Unsupported ontology gaps
- `QF28` Ambiguous requests

## `QF01` Company to segments

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf01_company_segments_list` | Returns the company’s segment names as a list. | `What business segments does [Company] have?` | `list<string>` | Low difficulty; direct lookup. |
| `qf01_company_segments_count` | Counts distinct segments instead of listing them. | `How many business segments does [Company] have?` | `integer` | Good for teaching aggregation. |
| `qf01_company_segments_membership` | Tests whether a specific segment belongs to the company. | `Does [Company] have [BusinessSegment]?` | `boolean` | Useful yes/no form; avoid paraphrase drift. |

## `QF02` Company to company-level offerings

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf02_company_offerings_list` | Returns offerings attached directly to the company scope. | `What offerings does [Company] have at the company level?` | `list<string>` | Separate from segment-owned offerings. |
| `qf02_company_offerings_count` | Counts direct company-level offerings. | `How many company-level offerings does [Company] have?` | `integer` | Keep fallback company-offers distinct from segment offers. |
| `qf02_company_offerings_membership` | Checks whether a named offering is directly company-scoped. | `Does [Company] directly offer [Offering]?` | `boolean` | Good for fallback `Company-[:OFFERS]->Offering` cases. |

## `QF03` Segment to direct offerings

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf03_segment_direct_offerings_list` | Lists offerings directly owned by a segment. | `What direct offerings does [Company]'s [BusinessSegment] segment have?` | `list<string>` | Primary segment-offering pattern. |
| `qf03_segment_direct_offerings_count` | Counts direct offerings for a segment. | `How many direct offerings does [Company]'s [BusinessSegment] segment have?` | `integer` | Good early aggregation case. |
| `qf03_segment_direct_offerings_membership` | Checks whether a given offering is directly under a segment. | `Does [Company]'s [BusinessSegment] segment offer [Offering]?` | `boolean` | Keep the subject explicitly segment-scoped. |

## `QF04` Offering family children

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf04_offering_children_list` | Returns immediate child offerings only. | `What is inside [Offering]?` | `list<string>` | Must not expand descendants recursively. |
| `qf04_offering_children_count` | Counts direct children of an umbrella offering. | `How many child offerings does [Offering] have?` | `integer` | Distinct from transitive subtree counts. |
| `qf04_offering_children_membership` | Tests a direct parent-child offering relation. | `Is [Child Offering] inside [Offering]?` | `boolean` | Useful for explicit family-structure validation. |

## `QF05` Offering descendant expansion

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf05_offering_descendants_list` | Returns all descendant offerings at any depth. | `What offerings sit under [Offering]?` | `list<string>` | Recursive closure over `OFFERS`. |
| `qf05_offering_descendants_count` | Counts the full descendant set. | `How many offerings sit under [Offering]?` | `integer` | Different from immediate-child counting. |
| `qf05_offering_leaf_descendants_list` | Returns only leaf descendants, not umbrella nodes. | `Which leaf offerings are under [Offering]?` | `list<string>` | Useful for hierarchy-aware filtering. |

## `QF06` Company to places

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf06_company_places_list` | Lists places where the company operates. | `Where does [Company] operate?` | `list<string>` | Geographies only, no office-site granularity. |
| `qf06_company_places_count` | Counts distinct places. | `How many places does [Company] operate in?` | `integer` | Keep places canonical and normalized. |
| `qf06_company_places_membership` | Checks whether the company operates in a given place. | `Does [Company] operate in [Place]?` | `boolean` | Good yes/no format for geography. |

## `QF07` Company to partners

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf07_company_partners_list` | Lists partner companies. | `Who does [Company] partner with?` | `list<string>` | Partnership is company-to-company only. |
| `qf07_company_partners_count` | Counts distinct partners. | `How many partners does [Company] have?` | `integer` | Distinct from supplier or customer language. |
| `qf07_company_partners_membership` | Checks a specific partnership. | `Does [Company] partner with [Company]?` | `boolean` | Keep the role of both companies explicit. |

## `QF08` Segment to customer types

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf08_segment_customer_types_list` | Lists canonical customer categories for a segment. | `Which customer types does [Company]'s [BusinessSegment] segment serve?` | `list<string>` | Closed-label vocabulary matters here. |
| `qf08_segment_customer_types_count` | Counts customer types. | `How many customer types does [Company]'s [BusinessSegment] segment serve?` | `integer` | Good early closed-vocabulary count. |
| `qf08_segment_customer_types_membership` | Tests whether a segment serves one customer type. | `Does [Company]'s [BusinessSegment] segment serve [CustomerType]?` | `boolean` | Useful for yes/no supervision. |

## `QF09` Segment to channels

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf09_segment_channels_list` | Lists channels used by a segment. | `Which channels does [Company]'s [BusinessSegment] segment sell through?` | `list<string>` | Closed-label vocabulary again. |
| `qf09_segment_channels_count` | Counts channels used by a segment. | `How many channels does [Company]'s [BusinessSegment] segment sell through?` | `integer` | Good aggregation variant. |
| `qf09_segment_channels_membership` | Checks whether a channel is used by a segment. | `Does [Company]'s [BusinessSegment] segment sell through [Channel]?` | `boolean` | Keep subject scope at segment level. |

## `QF10` Fallback offering to channels

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf10_offering_channels_list` | Lists channels for offerings that do not have a segment anchor. | `How is [Offering] sold?` | `list<string>` | Fallback-only case, not a segment query. |
| `qf10_offering_channels_count` | Counts channels for a fallback offering. | `How many channels does [Offering] use?` | `integer` | Keep this separate from `QF09`. |
| `qf10_offering_channels_membership` | Tests whether a fallback offering uses a channel. | `Does [Offering] sell through [Channel]?` | `boolean` | Useful for rare but valid fallback fixtures. |

## `QF11` Offering to revenue models

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf11_offering_revenue_models_list` | Lists revenue models for a single offering. | `How does [Offering] make money?` | `list<string>` | Canonical direct monetization lookup. |
| `qf11_offering_revenue_models_count` | Counts distinct revenue models. | `How many revenue models does [Offering] use?` | `integer` | Good aggregation and deduplication case. |
| `qf11_offering_revenue_model_membership` | Checks one offering against one revenue model. | `Does [Offering] monetize via [RevenueModel]?` | `boolean` | Nice yes/no intent for the closed label set. |

## `QF12` Segment revenue rollup

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf12_segment_revenue_rollup_list` | Rolls monetization up from descendant offerings to a segment. | `How does [Company]'s [BusinessSegment] segment make money?` | `list<string>` | High difficulty; recursive closure plus monetization. |
| `qf12_segment_revenue_rollup_count` | Counts distinct revenue models recovered through the segment subtree. | `How many distinct revenue models does [Company]'s [BusinessSegment] segment use?` | `integer` | Good for rollup aggregation. |
| `qf12_segment_revenue_rollup_membership` | Tests whether the segment monetizes via a specific model through descendants. | `Does [Company]'s [BusinessSegment] segment monetize via [RevenueModel]?` | `boolean` | Important yes/no rollup case. |
| `qf12_segment_revenue_contributors_list` | Returns the descendant offerings that actually carry monetization. | `Which offerings under [Company]'s [BusinessSegment] segment monetize via [RevenueModel]?` | `rows: offering, revenue_model` | Distinct because it exposes the monetizing leaves. |

## `QF13` Company revenue rollup

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf13_company_revenue_rollup_list` | Rolls monetization from company to all segment descendants. | `How does [Company] monetize its business?` | `list<string>` | More global than `QF12`. |
| `qf13_company_revenue_rollup_count` | Counts all distinct monetization models at company scope. | `How many revenue models does [Company] use?` | `integer` | Company-wide aggregation. |
| `qf13_company_revenue_rollup_membership` | Checks whether any company subtree monetizes via a model. | `Does [Company] monetize via [RevenueModel]?` | `boolean` | Same model vocabulary, broader scope. |
| `qf13_company_revenue_sources_list` | Returns segments or offerings responsible for monetization. | `Which segments or offerings make money for [Company] via [RevenueModel]?` | `rows: segment, offering, revenue_model` | Useful for provenance-style rollup queries. |

## `QF14` Company channel rollup

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf14_company_channels_list` | Rolls up channels from all segments to the company. | `What channels does [Company] use?` | `list<string>` | Company-level distribution summary. |
| `qf14_company_channels_count` | Counts distinct channels across the company. | `How many channels does [Company] use?` | `integer` | Distinct from segment-scoped `QF09`. |
| `qf14_company_channels_membership` | Checks whether the company uses a given channel through any segment. | `Does [Company] use [Channel]?` | `boolean` | Nice yes/no rollup case. |
| `qf14_company_channel_sources_list` | Shows which segments contribute a given channel. | `Which segments use [Channel] at [Company]?` | `rows: segment, channel` | Useful when generating provenance-heavy examples. |

## `QF15` Company customer rollup

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf15_company_customer_types_list` | Rolls up customer types from all segments to the company. | `What customer types does [Company] serve?` | `list<string>` | Company-wide customer summary. |
| `qf15_company_customer_types_count` | Counts customer types across the company. | `How many customer types does [Company] serve?` | `integer` | Distinct from segment-scoped `QF08`. |
| `qf15_company_customer_types_membership` | Checks whether the company serves a given customer type. | `Does [Company] serve [CustomerType]?` | `boolean` | Good yes/no rollup example. |
| `qf15_company_customer_sources_list` | Shows which segments contribute a customer type. | `Which segments serve [CustomerType] at [Company]?` | `rows: segment, customer_type` | Helps teach provenance over rollups. |

## `QF16` Segment-offering plus monetization join

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf16_segment_offering_monetization_list` | Finds offerings under a segment that monetize through a specific model. | `Which offerings in [BusinessSegment] monetize via [RevenueModel]?` | `list<string>` | Join across segment scope and monetization. |
| `qf16_segment_offering_monetization_count` | Counts offerings matching the segment-plus-model filter. | `How many offerings in [BusinessSegment] monetize via [RevenueModel]?` | `integer` | Good for selective filtering. |
| `qf16_segment_offering_monetization_pairs` | Returns matched offering/model pairs under a segment. | `Which offerings under [BusinessSegment] use which revenue models?` | `rows: offering, revenue_model` | Better when multiple models appear. |
| `qf16_segment_offering_monetization_membership` | Checks whether a specific offering under a segment uses a model. | `Does [BusinessSegment] include [Offering] monetizing via [RevenueModel]?` | `boolean` | High-value join and yes/no combination. |

## `QF17` Segment-offering plus channel join

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf17_channel_segment_reverse_list` | Reverse lookup from channel to the segments using it. | `Which segments sell through [Channel]?` | `list<string>` | Valid because `SELLS_THROUGH` is canonical at segment scope by default. |
| `qf17_channel_segment_reverse_count` | Counts segments using the channel. | `How many segments use [Channel]?` | `integer` | Useful for reverse lookup supervision. |
| `qf17_channel_segment_membership` | Checks whether a segment uses a given channel. | `Does [BusinessSegment] sell through [Channel]?` | `boolean` | Stable yes/no reverse-lookup form. |
| `qf17_company_channel_segment_list` | Returns the segments at a company that use a given channel. | `Which segments at [Company] sell through [Channel]?` | `list<string>` | Keeps channel logic at segment scope while adding company filtering. |

## `QF18` Segment-offering plus customer join

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf18_segment_customer_segment_list` | Finds segments that serve a specific customer type. | `Which segments serve [CustomerType]?` | `list<string>` | Reverse lookup from customer type to segment. |
| `qf18_segment_customer_segment_count` | Counts segments serving the customer type. | `How many segments serve [CustomerType]?` | `integer` | Good for closed-label joins. |
| `qf18_segment_customer_offering_list` | Finds offerings under segments that serve a customer type. | `Which offerings are under segments that serve [CustomerType]?` | `list<string>` | Higher-order join that combines segment and offering scope. |
| `qf18_segment_customer_offering_count` | Counts offerings under those customer-serving segments. | `How many offerings are under segments that serve [CustomerType]?` | `integer` | Good bridge between segment and offering layers. |

## `QF19` Intersections

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf19_segment_customer_channel_list` | Segments must satisfy two predicates at once. | `Which segments serve [CustomerType] and sell through [Channel]?` | `list<string>` | High difficulty because it combines two independent filters. |
| `qf19_segment_customer_offering_list` | Segments must serve a customer type and expose a named offering. | `Which segments serve [CustomerType] and offer [Offering]?` | `list<string>` | Useful for multi-condition matching. |
| `qf19_segment_customer_channel_offering_list` | Segments must satisfy three constraints simultaneously. | `Which segments serve [CustomerType], sell through [Channel], and offer [Offering]?` | `list<string>` | Strongest intersection case; high ambiguity risk. |
| `qf19_segment_intersection_count` | Counts the number of segments meeting the full predicate set. | `How many segments satisfy these conditions?` | `integer` | Use when the natural-language query is vague but the intent is exact. |

## `QF20` Counts

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf20_company_direct_offerings_count` | Counts only offerings attached directly to company scope. | `How many company-level offerings does [Company] have?` | `integer` | Distinct from segment or descendant rollup counts. |
| `qf20_company_descendant_offerings_count` | Counts offerings reachable through the company’s segments and offering hierarchies. | `How many offerings does [Company] have across all segments?` | `integer` | Explicitly a rollup count, not a direct company-scope count. |
| `qf20_segment_offerings_count` | Counts offerings for a segment. | `How many offerings does [BusinessSegment] have?` | `integer` | Separate from company scope and rollup scope. |
| `qf20_rollup_descendant_count` | Counts descendants in a hierarchy or rollup path. | `How many offerings sit under [Offering]?` | `integer` | Useful for closure + aggregation. |
| `qf20_filtered_match_count` | Counts only matches that satisfy a valid filter. | `How many offerings in [BusinessSegment] monetize via [RevenueModel]?` | `integer` | Good practice for filtered aggregations. |

## `QF21` Ranking and ordering

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf21_offerings_alpha_list` | Orders offerings alphabetically. | `List [Company]'s offerings alphabetically.` | `list<string>` | Simple but important ordering behavior. |
| `qf21_offerings_alpha_limited_list` | Orders and truncates the offering list. | `List the first [Limit] offerings for [Company] alphabetically.` | `list<string>` | Good for `LIMIT` supervision. |
| `qf21_descendant_offerings_ordered_list` | Orders descendant offerings in a hierarchy. | `List the offerings under [Offering] in sorted order.` | `list<string>` | Use when hierarchical output needs stable ordering. |
| `qf21_places_alpha_list` | Orders company places alphabetically. | `List the places where [Company] operates in alphabetical order.` | `list<string>` | Keeps ranking behavior tied to real schema outputs rather than invented metadata. |

## `QF22` Existence checks

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf22_partner_existence` | Checks whether a partnership exists. | `Does [Company] partner with [Company]?` | `boolean` | Direct company-to-company existence check. |
| `qf22_segment_customer_existence` | Checks whether a segment serves a customer type. | `Does [BusinessSegment] serve [CustomerType]?` | `boolean` | Good canonical yes/no form. |
| `qf22_offering_revenue_existence` | Checks whether an offering monetizes via a revenue model. | `Does [Offering] monetize via [RevenueModel]?` | `boolean` | Simple but highly useful negative/positive supervision. |
| `qf22_company_place_existence` | Checks whether a company operates in a place. | `Does [Company] operate in [Place]?` | `boolean` | Lets the model learn geography yes/no queries. |

## `QF23` Path anchoring

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf23_offering_segment_anchor` | Finds the unique segment that anchors an offering subtree in fixtures where exactly one segment anchor exists. | `Which segment is [Offering] under?` | `single string` | Reverse traversal from offering to segment; fixtures must enforce a single valid anchor. |
| `qf23_offering_root_anchor` | Finds the top-level offering that anchors a descendant offering. | `What parent offering anchors [Offering]?` | `single string` | Distinct from immediate parent lookup. |
| `qf23_offering_anchor_path` | Returns the path breadcrumb from segment to offering. | `Show the segment-to-offering path for [Offering].` | `rows: segment, offering, depth` | High difficulty because it surfaces path structure. |

## `QF24` Parent lookup

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf24_offering_parent_list` | Finds the immediate parent of a child offering. | `What parent offering is [Offering] under?` | `single string` | Immediate parent only. |
| `qf24_offering_ancestor_list` | Returns the ancestor chain for a child offering. | `Which offerings are above [Offering]?` | `list<string>` | Different from direct parent lookup. |
| `qf24_offering_root_ancestor` | Returns the topmost ancestor in the offering family. | `What is the root offering for [Offering]?` | `single string` | Useful for umbrella family normalization. |

## `QF25` Unsupported financial metrics

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf25_offering_revenue_amount` | Requests revenue amounts for an offering. | `What revenue did [Offering] generate?` | `refusal` | Not represented in the ontology. |
| `qf25_company_revenue_amount` | Requests company-level revenue totals. | `What total revenue does [Company] make?` | `refusal` | The graph stores categories, not revenue figures. |
| `qf25_segment_revenue_amount` | Requests segment-level revenue amounts. | `How much revenue does [BusinessSegment] generate?` | `refusal` | Good negative case for overconfident models. |
| `qf25_revenue_time_series` | Requests time-indexed revenue values. | `What revenue did [Company] make last year?` | `refusal` | Strong guardrail against invented time metrics. |

## `QF26` Unsupported time-based requests

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf26_fastest_growing_segment` | Requests growth-rate ranking over time. | `Which segment grew fastest last year?` | `refusal` | Time dimension is absent from the ontology. |
| `qf26_yoy_change_request` | Requests year-over-year change. | `How did [Company]'s business change year over year?` | `refusal` | The graph does not encode year-over-year measures. |
| `qf26_latest_period_request` | Requests the latest period or quarter. | `What happened in the latest quarter for [Company]?` | `refusal` | Avoids hallucinating filing chronology. |
| `qf26_time_window_comparison` | Requests a comparison between time windows. | `Compare [BusinessSegment] this year versus last year.` | `refusal` | Distinct from pure entity lookup. |

## `QF27` Unsupported ontology gaps

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf27_supplier_request` | Requests suppliers, which are not modeled. | `Who are [Company]'s suppliers?` | `refusal` | Supplier relations are outside the ontology. |
| `qf27_headcount_request` | Requests employee or headcount data. | `How many employees does [Company] have?` | `refusal` | Headcount is not represented here. |
| `qf27_pricing_request` | Requests pricing or discount information. | `What does [Offering] cost?` | `refusal` | Pricing is not part of the current schema. |
| `qf27_customer_list_request` | Requests actual customer entities rather than customer types. | `Who are [Company]'s customers?` | `refusal` | The ontology stores `CustomerType`, not customer lists. |

## `QF28` Ambiguous requests

| Proposed intent_id stem | What makes it distinct | Example slot pattern | Expected result shape | Notes |
| --- | --- | --- | --- | --- |
| `qf28_ambiguous_name_request` | The named entity could refer to more than one synthetic subject. | `How does Atlas make money?` | `refusal` | Keep this as ambiguity handling, not hallucinated disambiguation. |
| `qf28_scope_ambiguous_request` | The query omits whether it refers to company, segment, or offering scope. | `How does [Name] sell?` | `refusal` | Good for teaching scope clarification. |
| `qf28_pronoun_only_request` | The request contains no recoverable entity anchor. | `How does it make money?` | `refusal` | Useful for conversation-level ambiguity, if included. |

## Practical Notes For Dataset Generation

1. Prefer the earliest families for seed authoring: `QF01`, `QF03`, `QF08`, `QF09`, `QF11`, and `QF12`.
2. Treat `QF12`, `QF13`, `QF14`, `QF15`, `QF23`, and `QF24` as the highest-review families because they encode the ontology-specific traversal logic.
3. Keep refusal cases explicit and separate from answerable cases.
4. Avoid near-duplicates unless they change the semantic task, result shape, or traversal scope.
5. When a family includes both a list form and a count form, keep both if they require different Cypher shape or aggregation logic.

## Recommended Next Step

Generate the first synthetic fixture library from the answerable families in this document, then build intent rows one family at a time starting with `QF01`, `QF03`, `QF08`, `QF09`, `QF11`, and `QF12`.
