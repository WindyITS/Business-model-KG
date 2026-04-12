# Business Model Ontology v2

Final segment-centered ontology used by the current extraction pipeline.

This version prioritizes:
- canonical structure
- higher standardization across companies
- conservative extraction from sparse 10-K text
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

### 3. Sparse-text honesty

If the filing only states a fact at a broad level, do not force a more granular triple just to make the graph denser.

### 4. Closed-label discipline

`CustomerType`, `Channel`, and `RevenueModel` are closed canonical vocabularies.

If a phrase does not map clearly to one canonical label, omit it.

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
- `Company -> OFFERS -> Offering` is fallback only when no segment anchor exists or the offering is presented as universal
- `Offering -> OFFERS -> Offering` is allowed only when the filing explicitly states the umbrella / family / suite / parent relationship
- a child `Offering` may have at most one offering parent

### `SERVES`

`BusinessSegment -> CustomerType`

Links a business segment to a canonical customer category that it targets, supports, or serves commercially.

Rules:
- `SERVES` is canonical only at `BusinessSegment` scope
- if a customer type is universal across the company, attach it to each reported `BusinessSegment` rather than to `Company`
- do not attach `SERVES` to `Offering`

### `OPERATES_IN`

`Company -> Place`

Links a company to a normalized, business-relevant geography in which it operates, conducts business, or has meaningful market presence.

Rules:
- strictly company-level
- do not attach to `BusinessSegment` or `Offering`

### `SELLS_THROUGH`

`BusinessSegment -> Channel | Offering -> Channel`

Links a business segment or, only when no segment anchor exists, an offering to a canonical sales or distribution channel through which it reaches customers.

Rules:
- `BusinessSegment` is the primary anchor
- `Offering` is a fallback only for offerings without a business segment
- `Company` is not a valid subject
- if the filing states a channel universally across a company that has reported segments, attach that channel to each reported `BusinessSegment`

### `PARTNERS_WITH`

`Company -> Company`

Links a company to another company with which it has a named strategic, commercial, distribution, technology, integration, or go-to-market partnership.

Rules:
- strictly company-level
- excludes incidental mentions and ordinary supplier or customer relationships

### `MONETIZES_VIA`

`Offering -> RevenueModel`

Links an offering to the canonical revenue model through which it earns money.

Rules:
- `MONETIZES_VIA` is canonical only at `Offering` scope
- do not attach directly to `BusinessSegment` or `Company`
- if the filing states monetization broadly at segment level, do not force a segment-level triple; recover rollups downstream from offering-level facts

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

## Canonical Extraction Rules

### Rule 1: Segment-anchored offerings first

If an offering is clearly tied to a segment, emit `BusinessSegment -> OFFERS -> Offering`.

Do not automatically emit `Company -> OFFERS -> Offering`.

### Rule 2: Family hierarchies are explicit, not inferred

Use `Offering -> OFFERS -> Offering` only when the filing explicitly states that hierarchy.

An umbrella offering does not replace its explicit named child offerings.

### Rule 3: No derivation during extraction

Do not derive:
- company-level facts from segment or offering facts
- offering-level facts from segment facts
- inherited convenience triples for analytics

### Rule 4: Precision over recall

If a fact is ambiguous, omit it.

The ontology is designed so downstream Neo4j traversal can recover convenience rollups later.

## Examples

### Example 1: Segment-level customers

If multiple offerings inside `Intelligent Cloud` clearly point to developers and IT professionals:

- `Intelligent Cloud -> SERVES -> developers`
- `Intelligent Cloud -> SERVES -> IT professionals`

Do not emit:
- `Microsoft -> SERVES -> developers`
- `Azure -> SERVES -> developers`

### Example 2: Company-wide channel phrasing

If the filing says the company sells through resellers across its business, and the company has three reported segments:

- `Productivity and Business Processes -> SELLS_THROUGH -> resellers`
- `Intelligent Cloud -> SELLS_THROUGH -> resellers`
- `More Personal Computing -> SELLS_THROUGH -> resellers`

Do not emit:
- `Microsoft -> SELLS_THROUGH -> resellers`

### Example 3: Segment-level monetization language

If the filing says a segment generates subscription revenue but does not isolate which offering does so:

- keep the segment structure and offerings
- do not force `BusinessSegment -> MONETIZES_VIA -> subscription`

Only emit `MONETIZES_VIA` when the offering-level monetization is supported clearly enough.
