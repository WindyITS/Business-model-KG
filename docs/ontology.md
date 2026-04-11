# Business Model Ontology

This ontology defines what the knowledge graph can contain. It is intentionally strict: the goal is a graph that is standardized across companies, useful for comparison across filings, compact enough for reliable extraction, and clean enough for supervised fine-tuning. It does not try to capture every financial fact in a 10-K but instead it captures the **business-model structure**: what a company offers, who it serves, how it sells, how it monetizes, how offerings relate to segments, and which companies it partners with.

## Current Review Focus

As of April 2026, ontology review is the active workstream for the project. The current ontology is usable, but it is still being reviewed before benchmark freeze and fine-tuning.

The main review questions are:

1. **Hierarchy and redundancy**
   - When do we want `Company -> OFFERS -> Offering` in addition to `BusinessSegment -> OFFERS -> Offering` and `Offering -> PART_OF -> BusinessSegment`?
   - When should company-level semantic facts be kept versus replaced by more specific segment-level or offering-level facts?
2. **Relation coverage**
   - Are the current eight relations sufficient for the business-model questions we care about?
   - Are there missing relations we should add before freezing the ontology?
3. **Relation strictness**
   - Are some subject/object pairings too permissive?
   - Are some current relations semantically too broad, especially `PARTNERS_WITH` and `OPERATES_IN`?

Until that review is complete, this document should be treated as the current working ontology, not yet the final frozen one.

## Triple Shape

Every fact in the graph is a triple:

```
(subject, subject_type, relation, object, object_type)
```

All five fields must be valid. Subject and object types must be allowed node types. The relation must be allowed for that subject-type → object-type pair. Canonical labels must be used for `CustomerType`, `Channel`, and `RevenueModel`.

## Node Types

### `Company`

A legally distinct commercial organization: the reporting company itself or any named external company (partner, customer, subsidiary, competitor) when it appears as a company in the described relationship.

**Not for:** internal business units (`BusinessSegment`), named products or platforms (`Offering`), government bodies, people, or metrics.

### `BusinessSegment`

A formally named internal segment, division, or reporting unit used by the company to organize operations or external reporting.

**Not for:** the company as a whole, general themes or market categories, informal groupings.

### `Offering`

A specific named product, service, platform, application, brand, subscription, or solution the company provides commercially.

**Not for:** generic categories like "software" or "cloud services", internal business units, markets, or customer groups.

### `CustomerType`

A standardized category for the kind of customer the company or offering serves. **Must use only the canonical labels defined below.**

### `Channel`

A standardized category for how the company or offering reaches customers. **Must use only the canonical labels defined below.**

### `Place`

A named geographic location or market area where the company or a segment operates.

**Not for:** jurisdictions mentioned only in legal boilerplate, customer segments with geographic adjectives, individual facilities.

### `RevenueModel`

A standardized category for how the company, segment, or offering earns revenue. **Must use only the canonical labels defined below.**

---

## Relations

### `HAS_SEGMENT` - `Company` → `BusinessSegment`

The company has, includes, or reports through this segment.

### `OFFERS` - `Company` or `BusinessSegment` → `Offering`

The company or segment provides this offering commercially.

### `PART_OF` - `Offering` → `BusinessSegment`

This offering belongs to, is managed by, or is reported within this segment.

### `SERVES` - `Company` or `Offering` → `CustomerType`

The company or offering targets or serves this canonical customer category. Prefer `Offering → CustomerType` when the customer group is tied to a specific offering.

### `OPERATES_IN` - `Company` or `BusinessSegment` → `Place`

The company or segment has operational presence, sales activity, or meaningful market activity in this place.

### `SELLS_THROUGH` - `Company` or `Offering` → `Channel`

The company or offering reaches customers via this canonical sales or distribution channel. Prefer `Offering → Channel` when the channel is clearly tied to a specific offering.

### `PARTNERS_WITH` - `Company` → `Company`

The company has a named strategic, commercial, distribution, technology, or go-to-market partnership with this company. Not for incidental mentions or relationships that are not framed as partnerships.

### `MONETIZES_VIA` - `Company`, `BusinessSegment`, or `Offering` → `RevenueModel`

The company, segment, or offering earns revenue through this canonical monetization model. Prefer the most specific level supported by the text.

---

## Relation Validity Matrix

| Relation          | Subject types                                  | Object type         |
| ----------------- | ---------------------------------------------- | ------------------- |
| `HAS_SEGMENT`   | `Company`                                    | `BusinessSegment` |
| `OFFERS`        | `Company`, `BusinessSegment`               | `Offering`        |
| `PART_OF`       | `Offering`                                   | `BusinessSegment` |
| `SERVES`        | `Company`, `Offering`                      | `CustomerType`    |
| `OPERATES_IN`   | `Company`, `BusinessSegment`               | `Place`           |
| `SELLS_THROUGH` | `Company`, `Offering`                      | `Channel`         |
| `PARTNERS_WITH` | `Company`                                    | `Company`         |
| `MONETIZES_VIA` | `Company`, `BusinessSegment`, `Offering` | `RevenueModel`    |

### Review Notes On Validity

These pairings are currently allowed, but some of them are under active review:

- whether `Company -> OFFERS -> Offering` should always coexist with `BusinessSegment -> OFFERS -> Offering`
- whether company-level `SERVES`, `SELLS_THROUGH`, and `MONETIZES_VIA` are sometimes too coarse when a more specific offering-level fact is available
- whether broad macro-regions should remain valid `Place` nodes for `OPERATES_IN`

---

## Canonical Vocabularies

These are closed. No new labels are allowed. If a concept doesn't map cleanly, omit it.

### `CustomerType`

| Label                        | Who it covers                                                        |
| ---------------------------- | -------------------------------------------------------------------- |
| `consumers`                | Individual end users buying for personal or household use            |
| `small businesses`         | Clearly described as small business, SMB, or equivalent              |
| `mid-market companies`     | Mid-market or midsize, explicitly signaled — not generic businesses |
| `large enterprises`        | Enterprise-scale, large account, or global enterprise                |
| `developers`               | Software builders creating or deploying applications on a platform   |
| `IT professionals`         | Admins, infrastructure teams, security teams, operations staff       |
| `government agencies`      | Federal, state, local, or supranational governmental bodies          |
| `educational institutions` | Schools, colleges, universities, and comparable educational bodies   |
| `healthcare organizations` | Hospitals, clinics, health systems delivering or administering care  |
| `financial services firms` | Banks, insurers, asset managers, payment processors, and similar     |
| `manufacturers`            | Businesses transforming materials into physical products             |
| `retailers`                | Businesses selling goods in small quantities to end customers        |

### `Channel`

| Label                         | What it covers                                                           |
| ----------------------------- | ------------------------------------------------------------------------ |
| `direct sales`              | Company's own sales force or account teams selling directly to customers |
| `online`                    | Company's own website, app, or self-service digital storefront           |
| `retail`                    | Physical retail stores or consumer-facing store locations                |
| `distributors`              | Intermediaries buying and supplying products for onward resale           |
| `resellers`                 | Third parties reselling the company's products to end customers          |
| `OEMs`                      | Manufacturers embedding or bundling the offering in their own hardware   |
| `system integrators`        | Partners combining or implementing multiple technologies for customers   |
| `managed service providers` | Third parties operating or administering solutions for customers         |
| `marketplaces`              | Third-party digital platforms through which customers transact           |

### `RevenueModel`

| Label                 | What it covers                                                               |
| --------------------- | ---------------------------------------------------------------------------- |
| `subscription`      | Recurring payments for ongoing access — monthly, annual, or term-based      |
| `advertising`       | Selling ad inventory, placements, or audience access to advertisers          |
| `licensing`         | Granting rights to use software, IP, content, or trademarks                  |
| `consumption-based` | Pay-as-you-go or metered usage billing                                       |
| `hardware sales`    | Selling physical devices, equipment, or components                           |
| `service fees`      | Implementation, consulting, support, maintenance, or managed services        |
| `royalties`         | Recurring payments from another party's authorized use of IP                 |
| `transaction fees`  | Per-transaction charges for payments, trades, or platform-mediated exchanges |

---

## Extraction Rules

- Use exact company, segment, offering, and place names from the text.
- Use only canonical labels for `CustomerType`, `Channel`, and `RevenueModel`.
- Do not create a triple unless the relation is clearly supported by the text.
- If the text contains nothing relevant to this ontology, return an empty list.
- If a fact doesn't fit cleanly, omit it -> precision over recall.

**Out of scope** (do not extract unless required as context for a valid triple): people, regulations, risk factors, ESG topics, financial metrics, stock movements, litigation, macroeconomic concepts.

---

## Implementation

The ontology is encoded as structured config in [`configs/ontology.json`](../configs/ontology.json) and enforced by the validator in [`src/ontology_validator.py`](../src/ontology_validator.py).
