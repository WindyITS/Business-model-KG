from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Stage3SmokeCase:
    case_id: str
    relation: str
    text: str
    expected_triples: tuple[dict[str, str], ...]
    company_name: str = "acme"
    allowed_subjects: dict[str, list[str]] | None = None
    existing_triples: tuple[dict[str, str], ...] = ()


SMOKE_CASES = (
    Stage3SmokeCase(
        case_id="serves_explicit_segments",
        relation="SERVES",
        text=(
            "The company's cloud security platform is designed for large enterprises and government agencies "
            "that require centralized policy management, auditability, and compliance controls. Sales efforts "
            "during the year remained focused on these customer groups."
        ),
        expected_triples=(
            {
                "subject": "acme",
                "subject_type": "Company",
                "relation": "SERVES",
                "object": "large enterprises",
                "object_type": "CustomerType",
            },
            {
                "subject": "acme",
                "subject_type": "Company",
                "relation": "SERVES",
                "object": "government agencies",
                "object_type": "CustomerType",
            },
        ),
    ),
    Stage3SmokeCase(
        case_id="sells_through_partner_channels",
        relation="SELLS_THROUGH",
        text=(
            "The company sells its endpoint software through resellers and system integrators in North America "
            "and Europe. Certain enterprise accounts are also served by a direct sales force, but partner channels "
            "remained the primary route to market during the year."
        ),
        expected_triples=(
            {
                "subject": "acme",
                "subject_type": "Company",
                "relation": "SELLS_THROUGH",
                "object": "resellers",
                "object_type": "Channel",
            },
            {
                "subject": "acme",
                "subject_type": "Company",
                "relation": "SELLS_THROUGH",
                "object": "system integrators",
                "object_type": "Channel",
            },
            {
                "subject": "acme",
                "subject_type": "Company",
                "relation": "SELLS_THROUGH",
                "object": "direct sales",
                "object_type": "Channel",
            },
        ),
    ),
    Stage3SmokeCase(
        case_id="monetizes_via_subscriptions_and_transactions",
        relation="MONETIZES_VIA",
        text=(
            "Revenue is derived primarily from subscription fees paid under annual and multi-year software "
            "contracts. The company also earns transaction fees when customers process payments through its platform."
        ),
        expected_triples=(
            {
                "subject": "acme",
                "subject_type": "Company",
                "relation": "MONETIZES_VIA",
                "object": "subscription",
                "object_type": "RevenueModel",
            },
            {
                "subject": "acme",
                "subject_type": "Company",
                "relation": "MONETIZES_VIA",
                "object": "transaction fees",
                "object_type": "RevenueModel",
            },
        ),
    ),
    Stage3SmokeCase(
        case_id="mixed_chunk_sells_through",
        relation="SELLS_THROUGH",
        text=(
            "The company's digital learning platform is sold online through its website and through marketplaces "
            "operated by third parties. The platform is intended for educational institutions and individual "
            "consumers. Revenue is generated from subscription fees for premium content and from advertising shown "
            "on the free tier."
        ),
        expected_triples=(
            {
                "subject": "acme",
                "subject_type": "Company",
                "relation": "SELLS_THROUGH",
                "object": "online",
                "object_type": "Channel",
            },
            {
                "subject": "acme",
                "subject_type": "Company",
                "relation": "SELLS_THROUGH",
                "object": "marketplaces",
                "object_type": "Channel",
            },
        ),
    ),
    Stage3SmokeCase(
        case_id="mixed_chunk_serves",
        relation="SERVES",
        text=(
            "The company's digital learning platform is sold online through its website and through marketplaces "
            "operated by third parties. The platform is intended for educational institutions and individual "
            "consumers. Revenue is generated from subscription fees for premium content and from advertising shown "
            "on the free tier."
        ),
        expected_triples=(
            {
                "subject": "acme",
                "subject_type": "Company",
                "relation": "SERVES",
                "object": "educational institutions",
                "object_type": "CustomerType",
            },
            {
                "subject": "acme",
                "subject_type": "Company",
                "relation": "SERVES",
                "object": "consumers",
                "object_type": "CustomerType",
            },
        ),
    ),
    Stage3SmokeCase(
        case_id="mixed_chunk_monetizes",
        relation="MONETIZES_VIA",
        text=(
            "The company's digital learning platform is sold online through its website and through marketplaces "
            "operated by third parties. The platform is intended for educational institutions and individual "
            "consumers. Revenue is generated from subscription fees for premium content and from advertising shown "
            "on the free tier."
        ),
        expected_triples=(
            {
                "subject": "acme",
                "subject_type": "Company",
                "relation": "MONETIZES_VIA",
                "object": "subscription",
                "object_type": "RevenueModel",
            },
            {
                "subject": "acme",
                "subject_type": "Company",
                "relation": "MONETIZES_VIA",
                "object": "advertising",
                "object_type": "RevenueModel",
            },
        ),
    ),
    Stage3SmokeCase(
        case_id="negative_empty_serves",
        relation="SERVES",
        text=(
            "The company operates 214 facilities in 146 cities and continues to invest in technology, logistics, "
            "and personnel. Management believes these facilities are adequate for current operations, and competition "
            "remains intense across the markets in which the company participates."
        ),
        expected_triples=(),
    ),
    Stage3SmokeCase(
        case_id="negative_empty_sells_through",
        relation="SELLS_THROUGH",
        text=(
            "The company operates 214 facilities in 146 cities and continues to invest in technology, logistics, "
            "and personnel. Management believes these facilities are adequate for current operations, and competition "
            "remains intense across the markets in which the company participates."
        ),
        expected_triples=(),
    ),
    Stage3SmokeCase(
        case_id="negative_empty_monetizes_via",
        relation="MONETIZES_VIA",
        text=(
            "The company operates 214 facilities in 146 cities and continues to invest in technology, logistics, "
            "and personnel. Management believes these facilities are adequate for current operations, and competition "
            "remains intense across the markets in which the company participates."
        ),
        expected_triples=(),
    ),
)


def smoke_case_example(case: Stage3SmokeCase) -> dict[str, Any]:
    allowed_subjects = case.allowed_subjects or {"Company": [case.company_name]}
    return {
        "instruction": "Stage 3 smoke-case benchmark.",
        "input": case.text,
        "output": {
            "extraction_notes": "Synthetic smoke-case benchmark example.",
            "triples": list(case.existing_triples),
        },
        "metadata": {
            "company_name": case.company_name,
            "chunk_key": {
                "ticker": case.company_name.upper(),
                "year": 2026,
                "source_file": "stage3_smoke_cases.txt",
                "page_id": "smoke",
                "chunk_id": case.case_id,
            },
            "smoke_case": {
                "case_id": case.case_id,
                "expected_triples": list(case.expected_triples),
                "allowed_subjects": allowed_subjects,
            },
        },
        "_allowed_subjects_override": allowed_subjects,
    }
