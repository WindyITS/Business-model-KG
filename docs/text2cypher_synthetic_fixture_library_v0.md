# Synthetic Fixture Library v0

## Purpose

This document defines the reusable synthetic graph structures for the synthetic-only text-to-Cypher dataset.

It is a structure library, not a dataset. The intent is to give us a small number of ontology-valid fixture patterns that can be combined, cloned, and parameterized while we generate intent-level examples.

## Design Rules

- Use only synthetic names and synthetic bindings.
- Stay strictly within the ontology from [`docs/ontology.md`](./ontology.md).
- Model read-only graph shapes only; do not invent write-time or extraction-only concepts.
- Keep `Company`, `BusinessSegment`, and `Offering` as the core anchors.
- Treat `CustomerType`, `Channel`, `Place`, and `RevenueModel` as closed canonical labels.
- Prefer a minimal number of fixture classes that can support multiple families each.
- Do not create fixtures for `QF25`, `QF26`, or `QF27`; those are refusal families.
- Use one ambiguity-oriented fixture for `QF28`.

## Global Fixture Invariants

- Never attach `SERVES` to `Offering` or `Company`.
- Never attach `SELLS_THROUGH` to `Company`.
- Use `Offering-[:SELLS_THROUGH]->Channel` only for offerings that do not have a business-segment anchor.
- Never attach `MONETIZES_VIA` to `BusinessSegment` or `Company`.
- If an offering hierarchy exists, attach `MONETIZES_VIA` to the family parent or to a non-hierarchical offering root, not to a child offering under that family.
- Do not give the same offering both `Company-[:OFFERS]->Offering` and `BusinessSegment-[:OFFERS]->Offering` unless the dataset explicitly wants to model ambiguity, and then keep that case isolated to the ambiguity fixture.
- Use only ontology-valid `Place`, `Channel`, `CustomerType`, and `RevenueModel` values.
- Keep each fixture deterministic enough that a gold query and expected result can be validated by inspection.

## Fixture Overview

| Fixture ID | Structure It Models | Covered Families / Intent Groups |
| --- | --- | --- |
| `FX01_company_segment_core` | A company with a clean internal segment map and no extra structure. | `QF01`, `QF20` |
| `FX02_company_direct_offerings` | A company with fallback company-level offerings. | `QF02`, `QF20`, `QF21` |
| `FX03_segment_direct_offerings` | A segment with direct offerings. | `QF03`, `QF20`, `QF21` |
| `FX04_offering_hierarchy_tree` | An umbrella offering with children and descendants. | `QF04`, `QF05`, `QF23`, `QF24`, `QF21`, `QF20` |
| `FX05_geo_partner_profile` | A company with places and named partners. | `QF06`, `QF07`, `QF22` |
| `FX06_segment_customer_channel_profile` | A segment with customer types and channels. | `QF08`, `QF09`, `QF17`, `QF18`, `QF22` |
| `FX07_offering_fallback_channel` | A fallback offering that sells through channels without a segment anchor. | `QF10`, `QF22` |
| `FX08_offering_revenue_profile` | A monetized offering with one or more revenue models. | `QF11`, `QF22` |
| `FX09_segment_revenue_rollup_tree` | A segment whose monetization is recovered through descendant offerings. | `QF12`, `QF16`, `QF20`, `QF22` |
| `FX10_company_rollup_profile` | A company with multiple segments, each contributing rollups. | `QF13`, `QF14`, `QF15`, `QF20`, `QF21`, `QF22` |
| `FX11_intersection_and_filtering_mesh` | A dense segment-and-offering mesh for joins, intersections, ordering, and reverse lookups. | `QF16`, `QF17`, `QF18`, `QF19`, `QF20`, `QF21`, `QF22` |
| `FX12_ambiguity_collision_set` | A name-collision setup for ambiguous requests. | `QF28` |

## Fixture Details

### `FX01_company_segment_core`

**What it models:** A company with a small set of clearly named business segments and no additional graph complexity.

**Covered families / intent groups:** `QF01`, plus count and membership variants in `QF20`.

**Node blueprint:** 1 `Company`; 2 to 4 `BusinessSegment` nodes.

**Relationship blueprint:** `Company-[:HAS_SEGMENT]->BusinessSegment`.

**Sample synthetic names:** Company `Northstar Systems`; segments `Industrial AI`, `Public Sector`, `Developer Ecosystem`.

**Constraints / authoring rules:** Keep the fixture pure. Do not attach offerings, places, partners, channels, customer types, or revenue models here. Each segment name should be unique within the company so `QF01` examples stay deterministic.

**Why this fixture is needed:** This is the smallest canonical anchor for segment lookup, segment counts, and simple membership queries.

### `FX02_company_direct_offerings`

**What it models:** A company that has direct company-scoped offerings in addition to, or instead of, segment-scoped offerings.

**Covered families / intent groups:** `QF02`, `QF20`, `QF21`.

**Node blueprint:** 1 `Company`; 2 to 4 `Offering` nodes; optional 1 to 2 `BusinessSegment` nodes if the fixture also needs contrast with segment-owned offerings.

**Relationship blueprint:** `Company-[:OFFERS]->Offering`.

**Sample synthetic names:** Company `Meridian Nexus`; offerings `Meridian Atlas`, `Meridian Studio`, `Meridian Vault`.

**Constraints / authoring rules:** Keep company-level offerings clearly separate from segment-level offerings in naming and in the fixture design. Do not assume a segment anchor exists for these offerings. If a segment is included, it should not own the same offerings.

**Why this fixture is needed:** It supports the fallback company-offers family and gives us clean material for list, count, and ordering examples at company scope.

### `FX03_segment_direct_offerings`

**What it models:** A standard company -> segment -> offering structure with direct segment ownership.

**Covered families / intent groups:** `QF03`, `QF20`, `QF21`.

**Node blueprint:** 1 `Company`; 1 to 3 `BusinessSegment`; 2 to 5 `Offering`.

**Relationship blueprint:** `Company-[:HAS_SEGMENT]->BusinessSegment`; `BusinessSegment-[:OFFERS]->Offering`.

**Sample synthetic names:** Company `Asteron Analytics`; segment `Industrial AI`; offerings `Asteron Control Suite`, `Asteron Insight Cloud`, `Asteron Model Forge`.

**Constraints / authoring rules:** The segment should own at least two offerings so ordering and counting intents are meaningful. Do not add child-offering hierarchy inside this fixture; keep it one-hop from segment to offering.

**Why this fixture is needed:** This is the canonical direct-offering shape and the workhorse for `QF03` as well as count and order variants.

### `FX04_offering_hierarchy_tree`

**What it models:** A segment-anchored umbrella offering that has child offerings and, optionally, deeper descendant offerings.

**Covered families / intent groups:** `QF04`, `QF05`, `QF23`, `QF24`, `QF21`, `QF20`.

**Node blueprint:** 1 `Company`; 1 `BusinessSegment`; 1 root `Offering`; 2 to 4 child `Offering`; optional grandchild `Offering` nodes for recursive depth.

**Relationship blueprint:** `Company-[:HAS_SEGMENT]->BusinessSegment`; `BusinessSegment-[:OFFERS]->Offering`; `Offering-[:OFFERS]->Offering` with a single-parent rule for each child offering.

**Sample synthetic names:** Company `Northstar Systems`; segment `Industrial AI`; root `Northstar Platform`; child offerings `Vision Grid`, `Predict Forge`, `Civic Graph`; optional deeper node `Vision Grid Edge`.

**Constraints / authoring rules:** Preserve an explicit tree. A child offering may have at most one parent offering. The hierarchy root should be segment-anchored so reverse path queries can recover the unique segment anchor. Use at least one root with multiple children so list, count, parent lookup, descendant traversal, and anchor-path queries are all testable.

**Why this fixture is needed:** This is the core hierarchy fixture for immediate children, transitive descendants, parent lookup, and path anchoring.

### `FX05_geo_partner_profile`

**What it models:** A company with normalized operating geographies and named strategic partners.

**Covered families / intent groups:** `QF06`, `QF07`, `QF22`.

**Node blueprint:** 1 `Company`; 2 to 4 `Place`; 2 to 4 partner `Company` nodes.

**Relationship blueprint:** `Company-[:OPERATES_IN]->Place`; `Company-[:PARTNERS_WITH]->Company`.

**Sample synthetic names:** Company `Helioforge`; places `United States`, `Germany`, `Japan`; partners `Cobalt Ridge`, `Lumen Grid`, `Verity Works`.

**Constraints / authoring rules:** Places must be valid ontology values only. Keep partners as companies, not products or segments. Do not use cities or office sites. The partner names should be distinct from the subject company to avoid trivial self-loops unless a self-loop is intentionally being tested.

**Why this fixture is needed:** It covers the two company-level relation families and gives us a clean boolean existence surface for geography and partner checks.

### `FX06_segment_customer_channel_profile`

**What it models:** A segment with canonical customer types and channels.

**Covered families / intent groups:** `QF08`, `QF09`, `QF17`, `QF18`, `QF22`.

**Node blueprint:** 1 `Company`; 1 to 2 `BusinessSegment`; 2 to 4 `CustomerType`; 2 to 4 `Channel`; optional 1 to 3 `Offering` nodes for join-heavy variants.

**Relationship blueprint:** `Company-[:HAS_SEGMENT]->BusinessSegment`; `BusinessSegment-[:SERVES]->CustomerType`; `BusinessSegment-[:SELLS_THROUGH]->Channel`; optional `BusinessSegment-[:OFFERS]->Offering`.

**Sample synthetic names:** Company `Cobalt Ridge`; segment `Field Intelligence`; customer types `manufacturers`, `large enterprises`, `government agencies`; channels `direct sales`, `resellers`, `system integrators`.

**Constraints / authoring rules:** Use only canonical labels from the ontology. Keep the segment-to-customer and segment-to-channel links direct. If offerings are added, they should be direct segment-owned offerings used only to support join or reverse-lookup examples. Do not attach offering-level channels inside this fixture.

**Why this fixture is needed:** This fixture powers the closed-vocabulary families and gives us the minimum structure for reverse lookups and join-based segment questions.

### `FX07_offering_fallback_channel`

**What it models:** A standalone offering that has channels but no usable segment anchor, so `SELLS_THROUGH` must be queried at offering scope.

**Covered families / intent groups:** `QF10`, `QF22`.

**Node blueprint:** 1 `Offering`; 2 to 3 `Channel`; optional 1 `Company` node for context only.

**Relationship blueprint:** `Offering-[:SELLS_THROUGH]->Channel`; optional `Company-[:OFFERS]->Offering` when company context is needed.

**Sample synthetic names:** Offering `Studio Edge`; channels `online`, `marketplaces`, `direct sales`.

**Constraints / authoring rules:** Do not add a segment anchor if the purpose is to exercise the fallback path. If a company is present, connect it through `Company-[:OFFERS]->Offering` rather than through a segment. This fixture should make the model learn that some offerings are queried directly for channels.

**Why this fixture is needed:** The fallback offering family must be explicit because it is a different query shape from segment-level channel lookup.

### `FX08_offering_revenue_profile`

**What it models:** A direct offering-to-revenue-model mapping with one or more monetization categories.

**Covered families / intent groups:** `QF11`, `QF22`.

**Node blueprint:** 1 `Offering`; 1 to 3 `RevenueModel` nodes.

**Relationship blueprint:** `Offering-[:MONETIZES_VIA]->RevenueModel`.

**Sample synthetic names:** Offering `Northstar Control Suite`; revenue models `subscription`, `service fees`, `licensing`.

**Constraints / authoring rules:** Use only the canonical revenue-model vocabulary from the ontology. Do not add revenue amounts, dates, margins, or pricing properties. If multiple revenue models are attached, make sure the fixture is still coherent and not a grab bag of unrelated labels.

**Why this fixture is needed:** This is the direct monetization fixture and the simplest valid source of revenue-model queries.

### `FX09_segment_revenue_rollup_tree`

**What it models:** A segment whose monetization is recovered indirectly through one or more segment-owned offerings, including at least one offering family root.

**Covered families / intent groups:** `QF12`, `QF16`, `QF20`, `QF22`.

**Node blueprint:** 1 `Company`; 1 `BusinessSegment`; 1 root `Offering`; 2 to 4 descendant `Offering`; 1 to 3 `RevenueModel`.

**Relationship blueprint:** `Company-[:HAS_SEGMENT]->BusinessSegment`; `BusinessSegment-[:OFFERS]->Offering`; optional `Offering-[:OFFERS]->Offering`; `Offering-[:MONETIZES_VIA]->RevenueModel` on the family parent or on a non-hierarchical offering root.

**Sample synthetic names:** Company `Northstar Systems`; segment `Industrial AI`; root offering `Northstar Platform`; descendants `Vision Grid`, `Predict Forge`, `Civic Graph`; revenue models `subscription`, `consumption-based`.

**Constraints / authoring rules:** The monetizing facts should never live on the segment. When an offering family hierarchy exists, attach `MONETIZES_VIA` to the family parent rather than to the child offerings beneath it. If the segment has multiple direct offerings, at least one can be a non-hierarchical offering root with its own monetization. Keep the tree small enough to reason about by hand, and include at least one branch that does not monetize so filtered rollups have contrast.

**Why this fixture is needed:** This is the defining ontology-aware rollup pattern for the dataset and the key reason the model must learn recursive traversal without inventing segment-level monetization facts.

### `FX10_company_rollup_profile`

**What it models:** A company with multiple segments, where each segment contributes channels, customer types, and monetization evidence.

**Covered families / intent groups:** `QF13`, `QF14`, `QF15`, `QF20`, `QF21`, `QF22`.

**Node blueprint:** 1 `Company`; 2 to 4 `BusinessSegment`; 4 to 8 `Offering`; 2 to 4 `CustomerType`; 2 to 4 `Channel`; 1 to 3 `RevenueModel`.

**Relationship blueprint:** `Company-[:HAS_SEGMENT]->BusinessSegment`; `BusinessSegment-[:OFFERS]->Offering`; `BusinessSegment-[:SERVES]->CustomerType`; `BusinessSegment-[:SELLS_THROUGH]->Channel`; `Offering-[:MONETIZES_VIA]->RevenueModel`; optional `Offering-[:OFFERS]->Offering` if a family hierarchy is needed.

**Sample synthetic names:** Company `Meridian Nexus`; segments `Commercial Platform`, `Public Sector`, `Developer Ecosystem`; offerings `Meridian Atlas`, `Meridian Vault`, `Meridian Studio`; customer types `developers`, `government agencies`; channels `direct sales`, `resellers`, `marketplaces`.

**Constraints / authoring rules:** This fixture should have enough breadth to support company-level rollups and ordering, but it should still be internally consistent. Segment-scoped facts should be the source for company rollups. If an offering hierarchy is present, keep `MONETIZES_VIA` on the hierarchy root rather than on its children. Do not materialize company-level convenience triples unless a downstream query pattern explicitly needs to traverse them.

**Why this fixture is needed:** It is the main company-level rollup fixture and gives us a place to test aggregation, ordering, and provenance-style output across multiple segments.

### `FX11_intersection_and_filtering_mesh`

**What it models:** A dense segment-and-offering mesh that supports joins, intersections, reverse lookups, filtered counts, and ordered results.

**Covered families / intent groups:** `QF16`, `QF17`, `QF18`, `QF19`, `QF20`, `QF21`, `QF22`.

**Node blueprint:** 1 `Company`; 2 to 3 `BusinessSegment`; 4 to 8 `Offering`; 2 to 4 `CustomerType`; 2 to 4 `Channel`; 1 to 3 `RevenueModel`.

**Relationship blueprint:** `Company-[:HAS_SEGMENT]->BusinessSegment`; `BusinessSegment-[:OFFERS]->Offering`; `BusinessSegment-[:SERVES]->CustomerType`; `BusinessSegment-[:SELLS_THROUGH]->Channel`; `Offering-[:MONETIZES_VIA]->RevenueModel`; optional `Offering-[:OFFERS]->Offering` for nested matching.

**Sample synthetic names:** Company `Asteron Analytics`; segments `Industrial AI`, `Retail Operations`, `Commercial Platform`; offerings `Asteron Control Suite`, `Asteron Insight Cloud`, `Asteron Model Forge`; customer types `retailers`, `large enterprises`, `developers`; channels `direct sales`, `system integrators`, `resellers`; revenue models `subscription`, `advertising`, `service fees`.

**Constraints / authoring rules:** This fixture should intentionally overlap labels and relationships so the model must learn filtering and intersections rather than simple one-hop lookup. Keep the structure deterministic enough that gold Cypher can be validated by eye. Ensure at least one segment satisfies multiple predicates and at least one similar segment does not, so boolean and intersection queries have real contrast. If a hierarchy is included, keep monetization on the family parent rather than on its child offerings.

**Why this fixture is needed:** This is the most flexible answerable fixture and the one that powers the harder join-heavy families.

### `FX12_ambiguity_collision_set`

**What it models:** A synthetic ambiguity setup where the same surface name can plausibly refer to more than one graph entity or scope.

**Covered families / intent groups:** `QF28`.

**Node blueprint:** At least 1 `Company`; at least 1 `BusinessSegment`; at least 1 `Offering`; optional partners, channels, or customer types if they help create realistic ambiguity.

**Relationship blueprint:** Any ontology-valid subset that creates competing interpretations, such as `Company-[:HAS_SEGMENT]->BusinessSegment`, `BusinessSegment-[:OFFERS]->Offering`, and optional `Company-[:OFFERS]->Offering`.

**Sample synthetic names:** Company `Atlas Dynamics`; segment `Atlas`; offering `Atlas`; optional partner company `Atlas Works`.

**Constraints / authoring rules:** The point is ambiguity, not invalidity. Do not make the graph impossible to query. Instead, create a surface form that could refer to multiple valid entities or scopes. Keep the ambiguous names synthetic and intentionally reused across labels or nearby entities. Dataset rows built from this fixture should normally resolve to refusal or disambiguation-needed behavior rather than arbitrary best-guess Cypher.

**Why this fixture is needed:** We need explicit ambiguity fixtures so refusal or clarification behavior can be trained instead of hallucinated.

Practical ambiguity modes to instantiate:

- lexical name collision without operational facts
- monetization ambiguity where the same surface form could refer to a company, segment, or offering with different valid revenue answers
- channel-scope ambiguity where the same surface form could refer to a segment or a fallback offering with different valid channel answers
- graph-free pronoun-only ambiguity for cases like `How does it make money?`

## Families That Do Not Need Fixtures

- `QF25` unsupported financial metrics
- `QF26` unsupported time-based requests
- `QF27` unsupported ontology gaps

These are refusal families. They should be represented in the dataset as negative examples, but they do not require special graph structures.

## Fixture Coverage Notes

1. `FX01` through `FX08` give us the one-hop and fallback lookup space.
2. `FX09` and `FX10` cover the ontology-specific rollups that matter most.
3. `FX11` is the main workhorse for join-heavy, intersection-heavy, and reverse-lookup queries.
4. `FX04` is the only hierarchy fixture we need for `QF04`, `QF05`, `QF23`, and `QF24`.
5. `FX12` should stay small, but it should be instantiated in more than one ambiguity mode so the model learns ambiguity handling rather than one repeated collision pattern.

## Recommended Use

Build dataset rows by combining:

- one fixture
- one intent case
- one binding set
- one validated Cypher pattern

If a future intent needs a slightly richer structure, extend one of these fixtures rather than creating a new fixture class first.
