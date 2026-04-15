# Business Model Ontology

Final segment-centered ontology used by the canonical extraction pipeline.

This version prioritizes:
- canonical structure
- higher standardization across companies
- conservative extraction for semantic business-model facts, with broader but still text-grounded company geography capture
- downstream recovery of convenience rollups in Neo4j rather than during extraction

## Design Principles

### 1. Scope-first modeling

The graph is organized around three scopes:
- `Company`: the corporate shell
- `BusinessSegment`: the primary semantic anchor
- `Offering`: the inventory layer

### 2. Canonical extraction over convenience duplication

The extractor should emit only canonical facts that are justified directly by the filing.

Do not:
- duplicate facts upward automatically
- push segment facts down to offerings automatically
- materialize inherited convenience triples during extraction

Those behaviors belong in downstream Neo4j querying.

### 2A. Company-scoped inventory identity in Neo4j

The extraction format stays canonical and does not add extra scope fields to triples.

At Neo4j load time, however, the runtime scopes company-owned inventory nodes so same-named
entities from different companies do not collapse together:
- `BusinessSegment` nodes are keyed by `(company_name, name)`
- `Offering` nodes are keyed by `(company_name, name)`
- `Company`, `Channel`, `CustomerType`, `RevenueModel`, and `Place` remain globally keyed by `name`

This means two different companies can each have an offering named `Advertising` without creating
one shared `Offering` node in the graph.

### 3. Sparse-text honesty with bounded inference

If the filing only states a fact at a broad level, do not force a more granular triple just to make the graph denser.

Exception:
- for `SERVES`, conservative segment-level inference is allowed when a customer type is clearly stated or clearly implied by a segment's offerings and descriptions

### 4. Closed-label discipline

`CustomerType`, `Channel`, and `RevenueModel` are closed canonical vocabularies.

If a phrase does not map clearly to one canonical label, omit it.

### 5. Effective ontology = schema + prompt policy + validator enforcement

The canonical node and relation schema lives in [`configs/ontology.json`](../configs/ontology.json).

In practice, though, the production ontology is enforced through three layers together:
- the formal schema and canonical labels
- the staged extraction pipeline in [`src/llm/`](../src/llm/), the pipeline registry under [`src/llm_extraction/pipelines/`](../src/llm_extraction/pipelines/), and canonical prompt assets in [`prompts/canonical/`](../prompts/canonical/)
- the runtime validator in [`src/ontology/validator.py`](../src/ontology/validator.py)

This document records the full effective behavior of the maintained pipeline, not just the raw schema file.

## Node Types

### `Company`

A legally distinct commercial organization referenced in the filing, including the reporting company itself and named external companies such as partners, distributors, resellers, competitors, subsidiaries, or marketplace operators when they act as companies.

### `BusinessSegment`

A formally named internal business segment, reporting segment, division, or line of business. In this ontology variant, `BusinessSegment` is the primary semantic anchor for business-model logic.

### `Offering`

A specific named product, service, platform, application, subscription, brand, solution, or explicitly named product family offered commercially. Offerings are usually inventory leaves, but an offering may also act as an umbrella for other offerings when the filing explicitly states that hierarchy.

### `CustomerType`

A standardized category representing the kind of customer or user targeted by a business segment.

### `Channel`

A standardized category representing how a business segment or offering reaches customers commercially.

### `Place`

A normalized business-relevant geography in which the company operates, conducts business, or has meaningful market presence.

### `RevenueModel`

A standardized category representing how an offering earns revenue.

## Relations

### `HAS_SEGMENT`

`Company -> BusinessSegment`

Links a company to a formally named internal business segment that is part of the company’s organizational or reporting structure.

### `OFFERS`

`Company -> Offering | BusinessSegment -> Offering | Offering -> Offering`

Links:
- a business segment to its offerings
- a fallback company subject to an offering when no segment anchor exists
- an explicit umbrella offering to a child offering when the filing directly states that family relationship

Rules:
- `BusinessSegment -> OFFERS -> Offering` is primary
- `Company -> OFFERS -> Offering` is fallback only when, after considering the whole filing, no segment anchor is supportable anywhere
- do not use `Company -> OFFERS -> Offering` just because an offering is described at company scope, as universal, or as shared/common infrastructure; if the evidence ties it to multiple segments, attach it to each supported `BusinessSegment`
- `Offering -> OFFERS -> Offering` is allowed only when the filing explicitly states the umbrella / family / suite / parent relationship
- a child `Offering` may have at most one offering parent
- extract explicit named offerings individually and as written
- do not compress explicit offering lists into invented summary labels, but if the filing itself uses a single semantic parent heading for the list, keep that heading as the parent offering
- if the filing explicitly breaks a heading into named subcategories, create those subcategories as child offerings even when they are broad category names
- if a local list contains one bare offering name plus sibling offerings that share the same stem with added modifiers, treat the bare name as the first-order family offering and the variants as child offerings
- if an offering has support for more than one segment, attach it to every supported `BusinessSegment`
- shared-platform evidence can still count as segment support; if an offering is described as backing, enabling, bundling with, integrating with, or providing a common layer for offerings used in multiple segments, attach it to each supported segment
- `BusinessSegment` should link only to first-order offerings; if an offering has an explicit offering parent, keep the child under that parent rather than attaching it directly to the segment as well
- do not invent intermediate umbrella offerings or extra nesting just to organize the graph more neatly

### `SERVES`

`BusinessSegment -> CustomerType`

Links a business segment to a canonical customer category that it targets, supports, or serves commercially.

Rules:
- `SERVES` is canonical only at `BusinessSegment` scope
- if a customer type is stated at company scope, attach it only to the `BusinessSegment` nodes where it is clearly stated or clearly implied by the segment's offerings and descriptions
- do not spread one customer type across multiple segments unless each segment has its own support
- do not attach `SERVES` to `Company` or `Offering`
- precision is more important than recall
- conservative inference is allowed only for `SERVES`, and it must remain segment-specific rather than global
- do not make weak guesses from vague proximity or broad context
- rare or specialized customer types should not be fanned out across many segments unless each segment has its own support
- when several offerings inside the same segment point to the same customer type, attach that customer type to the `BusinessSegment`

### `OPERATES_IN`

`Company -> Place`

Links a company to a normalized, business-relevant geography in which it operates, conducts business, has employees or a local entity, serves customers in a meaningful way, or otherwise has meaningful market presence.

Rules:
- strictly company-level
- use named countries or approved macro-regions where the company has meaningful market presence
- when a named geography is clearly tied to the company's own business presence, prefer recall over unnecessary omission
- meaningful market presence can be shown by signals such as a named subsidiary or local entity, employee presence, labor structure, customer or revenue presence, country-specific availability, or present-tense current use in that geography
- when the filing states geography at an approved macro-region level, prefer that macro-region rather than expanding it into many countries
- when the filing provides a clearly exhaustive country list that unambiguously corresponds to one approved macro-region, and the individual countries do not add distinct business signal, emit the macro-region instead of listing every country
- keep individual countries when the filing gives country-specific business significance such as a named subsidiary, datacenter, employees, customers, revenue, availability, or legal entity in that country
- do not infer unmentioned countries just to complete a region
- only roll up countries into a macro-region when the mapping is clear and unambiguous; if overlap makes the roll-up uncertain, keep the explicit countries
- do not substitute one overlapping macro-region for another unless the filing supports that exact label
- exclude geographies that appear only as incidental context, such as regulatory references or IP jurisdiction without company presence
- do not attach to `BusinessSegment` or `Offering`

### `SELLS_THROUGH`

`BusinessSegment -> Channel | Offering -> Channel`

Links a business segment or, only when no segment anchor exists, an offering to a canonical sales or distribution channel through which it reaches customers.

Rules:
- `BusinessSegment` is the primary anchor
- `Offering` is a fallback only for offerings without a business segment
- `Company` is not a valid subject
- if the filing states a channel universally across a company that has reported segments, attach that channel to each reported `BusinessSegment`
- default to `BusinessSegment` whenever a segment anchor exists
- do not derive offering-level channel facts from segment-level evidence unless the offering truly has no `BusinessSegment` anchor
- if an `Offering` already has a direct `BusinessSegment` anchor, the runtime validator rejects `Offering -> SELLS_THROUGH -> Channel`

### `PARTNERS_WITH`

`Company -> Company`

Links a company to another company with which it has a named strategic, commercial, distribution, technology, integration, or go-to-market partnership.

Rules:
- strictly company-level
- excludes incidental mentions and ordinary supplier or customer relationships
- do not use `PARTNERS_WITH` for suppliers, customers, competitors, ecosystem mentions, or channel relationships

### `MONETIZES_VIA`

`Offering -> RevenueModel`

Links an offering to the canonical revenue model through which it earns money.

Rules:
- `MONETIZES_VIA` is canonical only at `Offering` scope
- do not attach directly to `BusinessSegment` or `Company`
- if an offering family hierarchy exists, attach `MONETIZES_VIA` to the family parent rather than to its child offerings
- if the filing states monetization broadly at segment level, do not force a segment-level triple; recover rollups downstream from offering-level facts
- attach monetization to the first-order family parent rather than to a child offering beneath it
- if a child offering already has an explicit `Offering` parent, the runtime validator rejects `MONETIZES_VIA` on that child

## Relation Validity Matrix

| Relation | Subject types | Object type |
| --- | --- | --- |
| `HAS_SEGMENT` | `Company` | `BusinessSegment` |
| `OFFERS` | `Company`, `BusinessSegment`, `Offering` | `Offering` |
| `SERVES` | `BusinessSegment` | `CustomerType` |
| `OPERATES_IN` | `Company` | `Place` |
| `SELLS_THROUGH` | `BusinessSegment`, `Offering` | `Channel` |
| `PARTNERS_WITH` | `Company` | `Company` |
| `MONETIZES_VIA` | `Offering` | `RevenueModel` |

## Runtime Validation And Normalization

The runtime validator does more than just check the schema:

- entity text is normalized with NFKC, quote cleanup, and whitespace cleanup before validation
- `Place` values are normalized through the canonical place alias map
- `CustomerType`, `Channel`, and `RevenueModel` are enforced as closed canonical labels
- duplicate triples are removed using normalized entity keys
- validator reports preserve invalid triples, duplicate triples, and issue codes as audit output
- optional text grounding can be enabled when validating triples outside the main runtime
- the main CLI runtime validates with text grounding disabled by default and relies on the staged extraction and reflection process for factual support

Additional structural enforcement:
- a child `Offering` may have at most one `Offering` parent
- a child `Offering` with an explicit offering parent cannot carry `MONETIZES_VIA`
- an `Offering` with a direct `BusinessSegment` anchor cannot carry `SELLS_THROUGH`

## Canonical Label Definitions

### `CustomerType`

- `consumers`: Individual end users buying or using a product or service for personal, household, or non-business use.
- `small businesses`: Commercial organizations clearly described as small business, SMB, or an equivalent small-scale business segment.
- `mid-market companies`: Commercial organizations explicitly described as mid-market, midsize, or an equivalent segment between SMB and enterprise scale.
- `large enterprises`: Large organizations with enterprise-scale procurement, operations, compliance, or IT needs.
- `developers`: Software builders who create, test, extend, integrate, or deploy applications, code, or technical solutions.
- `IT professionals`: Technical staff primarily responsible for operating, administering, securing, supporting, or governing information systems and infrastructure.
- `government agencies`: Public-sector bodies with governmental authority or administrative responsibility at federal, state, regional, local, or supranational level.
- `educational institutions`: Organizations whose primary function is formal instruction, training, or education, including schools, colleges, and universities.
- `healthcare organizations`: Organizations whose primary function is delivering, administering, financing, or coordinating health care services.
- `financial services firms`: Organizations primarily engaged in financial transactions, intermediation, insurance, investment, payments, lending, brokerage, or related financial services.
- `manufacturers`: Businesses primarily engaged in transforming materials, components, or substances into physical products.
- `retailers`: Businesses primarily engaged in selling goods in relatively small quantities to end customers through physical or digital retail operations.

### `Channel`

- `direct sales`: Sales where the company sells directly to the end customer through its own sales force, account teams, or direct contracting process.
- `online`: Sales transacted through the company's own website, app, digital storefront, or self-service online purchasing experience.
- `retail`: Sales through physical retail stores or store-like consumer-facing locations that sell directly to end customers.
- `distributors`: Intermediaries that purchase, hold, allocate, or supply products for onward sale through other partners rather than primarily selling to end customers themselves.
- `resellers`: Third parties that resell the company's products or services to end customers without being the original manufacturer or the primary long-term operator of the solution.
- `OEMs`: Original equipment manufacturers that embed, bundle, preinstall, or incorporate the company's offering into their own hardware or integrated systems before sale.
- `system integrators`: Partners whose primary role is combining, customizing, implementing, or integrating multiple technologies into a customer-specific solution.
- `managed service providers`: Third parties that operate, administer, monitor, maintain, or support technology solutions for customers on an ongoing service basis.
- `marketplaces`: Third-party digital platforms through which customers discover, procure, subscribe to, or transact for products or services.

### `RevenueModel`

- `subscription`: Revenue earned from recurring payments for ongoing access to a product or service over time.
- `advertising`: Revenue earned by selling advertising inventory, sponsored visibility, promotional placement, or audience access to third-party advertisers.
- `licensing`: Revenue earned by granting another party the right to use software, technology, content, trademarks, patents, or other protected assets without transferring ownership.
- `consumption-based`: Revenue earned according to measured customer usage, volume, metering, or pay-as-you-go consumption rather than a fixed recurring amount alone.
- `hardware sales`: Revenue earned from selling physical devices, equipment, components, appliances, or other tangible hardware products.
- `service fees`: Revenue earned from performing services for customers, such as implementation, consulting, support, maintenance, training, professional services, or managed services.
- `royalties`: Revenue earned as recurring or usage-linked payments from another party's authorized use of the company's intellectual property, content, brand, or technology.
- `transaction fees`: Revenue earned by charging a fee each time a payment, trade, booking, exchange, or other platform-mediated transaction is processed or facilitated.

## `Place` Constraints

`OPERATES_IN` is limited to:
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

Normalization examples:
- `U.S.`, `USA` -> `United States`
- `U.K.`, `UK` -> `United Kingdom`
- `asia-pacific` -> `Asia Pacific`
- `emea` -> `EMEA`

Do not use:
- cities
- office sites
- vague global placeholders
- synthetic geography strings

### Downstream Place Query Metadata

The extractor still emits only canonical `Company-[:OPERATES_IN]->Place` facts.

In some cases, that canonical `Place` may be an approved macro-region selected from a clearly exhaustive country list when the roll-up is unambiguous and the individual countries do not add distinct business signal.

For downstream querying, Neo4j keeps only canonical `Company-[:OPERATES_IN]->Place`
facts and does not materialize a derived place hierarchy as relationships.

Instead, the loader may attach query helper list properties to each extracted `Place`:
- `within_places`: broader canonical places that contain the place
- `includes_places`: narrower canonical places that the place contains

Examples:
- `Italy` may carry `within_places = ["Europe", "Western Europe", "EMEA", "European Union"]`
- `Europe` may carry `includes_places = ["Western Europe", "Eastern Europe", "Italy", "Germany", ...]`
- `United States` may carry `includes_places = ["Alabama", "Alaska", ..., "Wyoming"]`

Recommended Cypher pattern:

```cypher
MATCH (company:Company)-[:OPERATES_IN]->(place:Place)
WITH company, place.name AS matched_place,
     CASE
       WHEN matched_place = $place THEN 0
       WHEN $place IN coalesce(place.includes_places, []) THEN 1
       WHEN $place IN coalesce(place.within_places, []) THEN 2
       ELSE NULL
     END AS match_rank
WHERE match_rank IS NOT NULL
WITH company, MIN(match_rank) AS best_rank, collect(DISTINCT matched_place) AS matched_places
RETURN company.name AS company,
       CASE best_rank
         WHEN 0 THEN 'exact'
         WHEN 1 THEN 'narrower_place'
         ELSE 'broader_region'
       END AS geography_match,
       matched_places
ORDER BY best_rank, company
```

A company can match through more than one direct place tag. Aggregate by company and use
`MIN(match_rank)` so the final result keeps one row per company with the strongest match
class.

### Downstream Company-Scoped Inventory Queries

For downstream querying, treat `BusinessSegment` and `Offering` as company-scoped even when the
surface name is ambiguous across companies.

Recommended Cypher pattern:

```cypher
MATCH (company:Company {name: $company})-[:HAS_SEGMENT]->(segment:BusinessSegment {company_name: $company})
OPTIONAL MATCH (segment)-[:OFFERS]->(offering:Offering {company_name: $company})
RETURN company.name AS company,
       segment.name AS segment,
       collect(DISTINCT offering.name) AS offerings
ORDER BY segment
```

Collision check example:

```cypher
MATCH (offering:Offering {name: $offering_name})
RETURN offering.name AS offering,
       offering.company_name AS company_name
ORDER BY company_name
```

## Canonical Extraction Rules

### Rule 1: Segment-anchored offerings first

If an offering is clearly tied to a segment, emit `BusinessSegment -> OFFERS -> Offering`.

Do not automatically emit `Company -> OFFERS -> Offering`.

### Rule 2: Family hierarchies are explicit, not inferred

Use `Offering -> OFFERS -> Offering` only when the filing explicitly states that hierarchy.

An umbrella offering does not replace its explicit named child offerings.

If the filing itself uses a single semantic parent heading, explicit composite heading, or explicit named subcategories, keep that structure rather than flattening it away.

### Rule 3: No derivation during extraction

Do not derive:
- company-level facts from segment or offering facts
- offering-level facts from segment facts
- inherited convenience triples for analytics

### Rule 4: Precision first, with explicit exceptions

If a fact is ambiguous, omit it.

For `SERVES`, conservative segment-specific inference is allowed when a customer type is clearly stated or clearly implied by a segment's offerings and descriptions.

For `OPERATES_IN`, prefer recall over unnecessary omission when a named geography is clearly tied to the company's own business presence.

The ontology is designed so downstream Neo4j traversal can recover convenience rollups later.
