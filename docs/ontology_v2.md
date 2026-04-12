# Business Model Ontology v2

Draft proposal for the next ontology revision.

This document defines the proposed normalized ontology for the business-model knowledge graph. The goal of `v2` is to make the graph more precise, more consistent across companies, easier to benchmark, cleaner for fine-tuning, and robust against the natural sparseness of SEC Form 10-K filings.

It is not just a list of allowed triples. It also defines:

- the hierarchy of business scopes
- the canonical graph shape
- the specificity rules for where facts should live
- how to resolve the sparse-text problem using downstream hierarchical inheritance
- the stricter rules for `OPERATES_IN` and `PARTNERS_WITH`

## Design Goals

The ontology should be:

- exhaustive enough to represent the main business-model structure of a company
- precise enough to avoid noisy, over-broad, or duplicated triples
- organized around clear business scopes with a segment-centric semantic hub
- stable enough to supervise dataset generation and fine-tuning

## Core Principles

### 1. Scope-first modeling

The graph is organized around three business scopes, each with a distinct analytical purpose:

- `Company` (The Corporate Shell): used for universal legal footprints, macro geography, and strategic corporate partnerships
- `BusinessSegment` (The Semantic Anchor): the primary hub for the business model; this is where monetization, sales channels, and customer types should primarily reside
- `Offering` (The Inventory): treated primarily as leaf nodes owned by a segment, but may also act as an explicit umbrella or family offering when the filing states that hierarchy directly

### 2. Most-specific supported fact wins, with a segment-first bias

Facts should be stored at the most specific level explicitly supported by the filing, but with a heavy bias toward the `BusinessSegment` for semantic business logic (`SERVES`, `SELLS_THROUGH`, `MONETIZES_VIA`).

- prefer `BusinessSegment` when the filing describes sales motions, monetization, or customer targeting for a division
- prefer `Offering` only when the fact is explicitly and exclusively tied to a specific product
- use `Company` only when the filing states the fact at the universal corporate level or when the company has no reported segments

### 3. No automatic upward duplication

Do not automatically duplicate a fact upward.

If a filing states:

- `Azure -> SERVES -> developers`

do not also emit:

- `Microsoft -> SERVES -> developers`

unless the filing explicitly supports that company-level fact independently.

### 4. Canonical graph first, convenience graph later

The extracted graph should store canonical facts, not convenience duplicates.

If downstream applications want easier traversal, they can derive convenience edges later in Neo4j or in analytics code.

### 5. Downstream hierarchical inheritance

Do not force the extraction pipeline to hallucinate or guess offering-level business attributes if the 10-K groups them by segment.

Granularity is achieved downstream in the graph database, not during extraction.

- If an `Offering` lacks a `SELLS_THROUGH`, `SERVES`, or `MONETIZES_VIA` edge, downstream analytics may traverse upward to inherit these facts from the parent `BusinessSegment`.
- The extractor must only capture canonical facts, not inherited convenience facts.

### 6. Closed labels stay closed

`CustomerType`, `Channel`, and `RevenueModel` remain closed canonical vocabularies. If a phrase does not map clearly, omit it.

## Node Types

The node set remains:

- `Company`: a legally distinct commercial organization, including the reporting company and named external companies
- `BusinessSegment`: a formally named internal segment, reporting segment, division, or line of business
- `Offering`: a specific named product, service, platform, application, brand, subscription, solution, or explicitly named product family offered commercially
- `CustomerType`: a canonical customer category
- `Channel`: a canonical commercial or distribution channel
- `Place`: a normalized business-relevant geography
- `RevenueModel`: a canonical monetization model

## Canonical Structural Backbone

These are the core structural relations of the graph.

### `HAS_SEGMENT`

- `Company -> BusinessSegment`

Use when the filing explicitly presents a formal internal segment or reporting segment.

### `OFFERS`

Primary canonical use:

- `BusinessSegment -> Offering`

Fallback use:

- `Company -> Offering`

Explicit umbrella-family use:

- `Offering -> Offering`

Rule:

- if the filing clearly ties an offering to a segment, emit `BusinessSegment -> OFFERS -> Offering`
- if no segment anchor exists, or the filing clearly presents the offering directly at company level as a universal company offering, emit `Company -> OFFERS -> Offering`
- if the filing explicitly states that one offering is a suite, family, umbrella, or parent offering for another offering, emit `Offering -> OFFERS -> Offering`

## Semantic Relations

These relations describe the core business-model logic.

### `SERVES`

- `BusinessSegment | Offering | Company -> CustomerType`

Rule:

- primary anchor: attach to `BusinessSegment`
- offering exception: attach directly to an `Offering` only if the filing explicitly isolates that attribute to a specific product
- company fallback: use `Company` only if the company has no reported segments, or if the text explicitly states the strategy applies universally across the entire corporate entity

### `SELLS_THROUGH`

- `BusinessSegment | Offering | Company -> Channel`

Rule:

- primary anchor: attach to `BusinessSegment`
- offering exception: attach to an `Offering` only if the filing explicitly isolates the channel to that product
- company fallback: use `Company` only if the company has no reported segments, or if the filing explicitly states the channel strategy universally at corporate level

### `MONETIZES_VIA`

- `BusinessSegment | Offering | Company -> RevenueModel`

Rule:

- primary anchor: attach to `BusinessSegment`
- offering exception: attach to an `Offering` only if the filing explicitly isolates the monetization model to that product
- company fallback: use `Company` only if the company has no reported segments, or if the filing explicitly states the monetization model universally at corporate level

## Corporate-Level Relations

### `OPERATES_IN`

- `Company -> Place`

Definition:

Use `OPERATES_IN` only for meaningful corporate business geography explicitly tied to operations, commercial presence, sales activity, or meaningful market activity.

Scope rule:

- strictly company-level
- do not attach `OPERATES_IN` to `BusinessSegment` or `Offering`

Allowed `Place` granularity:

- sovereign countries
- U.S. states
- `District of Columbia`
- approved macro-regions

Approved macro-regions:

- `Africa`
- `APAC`
- `Americas`
- `Asia`
- `Asia Pacific`
- `Caribbean`
- `Central America`
- `EMEA`
- `Eastern Europe`
- `Europe`
- `European Union`
- `Latin America`
- `Middle East`
- `North America`
- `South America`
- `Southeast Asia`
- `Western Europe`

Normalization rules:

- `U.S.`, `USA` -> `United States`
- `U.K.`, `UK` -> `United Kingdom`
- `asia-pacific` -> `Asia Pacific`
- `emea` -> `EMEA`

Do not use:

- cities
- office locations
- vague placeholders such as `global`
- synthetic or unclear geography strings

### `PARTNERS_WITH`

- `Company -> Company`

Definition:

Use only when the filing explicitly describes a strategic, commercial, technology, distribution, integration, or go-to-market partnership with a named company.

Scope rationale:

- strictly company-level
- partnerships are legal agreements between corporate entities
- product-level integrations do not qualify as structural business partnerships

Invalid `PARTNERS_WITH` uses:

- ordinary suppliers
- customer relationships
- competitor mentions
- platform availability without partnership framing
- ecosystem or marketplace presence alone

## Relation Validity Matrix

| Relation | Subject types | Object type |
| --- | --- | --- |
| `HAS_SEGMENT` | `Company` | `BusinessSegment` |
| `OFFERS` | `Company`, `BusinessSegment`, `Offering` | `Offering` |
| `SERVES` | `BusinessSegment` (Primary), `Offering`, `Company` | `CustomerType` |
| `OPERATES_IN` | `Company` (Strict) | `Place` |
| `SELLS_THROUGH` | `BusinessSegment` (Primary), `Offering`, `Company` | `Channel` |
| `PARTNERS_WITH` | `Company` (Strict) | `Company` |
| `MONETIZES_VIA` | `BusinessSegment` (Primary), `Offering`, `Company` | `RevenueModel` |

## Canonical Extraction Rules

### Rule 1: Segment-anchored offerings

If an offering is clearly tied to a segment, emit `BusinessSegment -> OFFERS -> Offering`.

Do not automatically emit `Company -> OFFERS -> Offering`.

Use `Offering -> OFFERS -> Offering` only when the filing explicitly states that one offering is an umbrella, suite, family, or parent offering for another offering.

### Rule 2: Resolving textual sparseness

Corporate filings frequently list products as bullet points but describe channels and monetization in bulk paragraphs.

- If a paragraph describes a channel, monetization model, or customer strategy generally within a segment's section, extract it as `BusinessSegment -> Relation -> Node`.
- Do not duplicate that relation for every `Offering` listed in that section.
- Let the segment hold the context.

### Rule 3: No derivation during extraction

Do not derive:

- company-level facts from segment-level or offering-level facts
- offering-level facts from segment-level facts

Any such rollups or inheritances should happen later in Neo4j analysis, not in extraction.

## Examples

### Example 1: Resolving sparseness

If the filing says the `Intelligent Cloud` segment generates revenue via `subscription`, and lists Azure, SQL Server, and GitHub as offerings:

Correct:

- `Intelligent Cloud -> MONETIZES_VIA -> subscription`
- `Intelligent Cloud -> OFFERS -> Azure`
- `Intelligent Cloud -> OFFERS -> SQL Server`
- `Intelligent Cloud -> OFFERS -> GitHub`

Incorrect:

- `Azure -> MONETIZES_VIA -> subscription`
- `SQL Server -> MONETIZES_VIA -> subscription`

Do not force the extractor to push segment-level business logic down to every leaf if the filing grouped it at the segment.

### Example 2: Offering-level exception

If the filing says: "While Intelligent Cloud primarily sells via direct sales, GitHub is sold via web self-serve."

- `Intelligent Cloud -> SELLS_THROUGH -> direct sales`
- `GitHub -> SELLS_THROUGH -> online`

### Example 3: Geography and partnerships

If the filing states: "Our AWS division recently partnered with Anthropic, and we are expanding AWS data centers in Germany."

- `Amazon -> HAS_SEGMENT -> AWS`
- `Amazon -> PARTNERS_WITH -> Anthropic`
- `Amazon -> OPERATES_IN -> Germany`

## Status

This is the draft `v2` proposal. It resolves previous issues with text sparseness by shifting the analytical hub to the `BusinessSegment` and enforcing strict corporate-level boundaries for geography and partnerships.

It should be treated as the target ontology candidate for:

- benchmark design
- dataset migration
- prompt design
- future fine-tuning
