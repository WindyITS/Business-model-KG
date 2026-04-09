import copy
import hashlib
import heapq
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from chunk_quality import chunk_quality_report, is_narrative_business_prose
from finreflectkg_projection import (
    DEFAULT_INSTRUCTION,
    build_empty_example,
    iter_grouped_rows,
    load_finreflectkg_rows,
    sample_empty_examples_by_count,
)
from ontology_config import allowed_subject_types, canonical_labels, load_ontology_config
from ontology_validator import canonical_entity_key, validate_triples

logger = logging.getLogger(__name__)

STAGE3_RELATIONS = ("SERVES", "SELLS_THROUGH", "MONETIZES_VIA")
STAGE3_PROVENANCE = {
    "SERVES": "teacher_serves",
    "SELLS_THROUGH": "teacher_sells_through",
    "MONETIZES_VIA": "teacher_monetizes_via",
}
GENERIC_SUBJECT_KEYS = {
    "company",
    "our business",
    "our company",
    "the business",
    "the company",
    "us",
    "we",
}

RELATION_TASKS = {
    "SERVES": {
        "object_type": "CustomerType",
        "trigger_text": "the text explicitly says that a company or offering serves, targets, is for, is designed for, or sells to a customer group",
        "logical_leap_example": (
            "I am extracting 'small businesses' because the company provides products and services to businesses of all types"
        ),
        "relation_specific_skepticism": [
            '**Audience vs. Description Distinction:** Do not conflate a broad company description with a SERVES relation.',
            '    - _Descriptive Statement (REJECT):_ "The company provides products and services to businesses of all types."',
            '    - _Commercial Targeting Statement (ACCEPT):_ "The offering is designed for small businesses" or "The company sells to government agencies."',
            '**Required Wording:** Require explicit language that names who the company or offering is for, targets, serves, is designed for, supports, or sells to.',
            'Do not map generic mentions of customers, businesses, users, organizations, markets, demand, or convenience into a canonical customer label unless the customer type itself is directly stated.',
            'A general reference to "customers" is not enough to decide between consumers, small businesses, large enterprises, manufacturers, retailers, or any other canonical label.',
        ],
    },
    "SELLS_THROUGH": {
        "object_type": "Channel",
        "trigger_text": "the text explicitly says that a company or offering is sold, distributed, marketed, or delivered through a channel",
        "logical_leap_example": "I am extracting 'direct sales' because they operate 'sales offices'",
        "relation_specific_skepticism": [
            '**Operational vs. Commercial Distinction:** Do not conflate operational infrastructure with commercial channels. The existence of an asset (e.g., "sales offices," "distribution centers," "warehouses," "processing plants") is evidence of **operations**, not necessarily a **SELLS_THROUGH** relation.',
            '    - _Operational Statement (REJECT):_ "The company operates direct sales offices."',
            '    - _Commercial Statement (ACCEPT):_ "The company reaches customers through its direct sales offices" or "Products are sold via direct sales."',
            '**Required Verbs:** Require explicit language about the commercial movement or availability, such as being _sold through, distributed via, available at, marketed via,_ or _reaches customers by_ a channel.',
            'Do not map general go-to-market language, partnerships, or mere commercial presence into a channel unless the **selling path is directly stated.**',
        ],
    },
    "MONETIZES_VIA": {
        "object_type": "RevenueModel",
        "trigger_text": "the text explicitly says how a company, segment, or offering makes money, charges customers, prices usage, licenses access, earns fees, earns royalties, earns advertising revenue, or uses subscriptions",
        "logical_leap_example": (
            "I am extracting 'service fees' because the segment provides services, so it probably charges fees"
        ),
        "relation_specific_skepticism": [
            '**Offering Type vs. Revenue Mechanism Distinction:** Do not confuse what the company provides with how it earns revenue. A product, service, segment, or sales program is not itself a revenue model.',
            '    - _Business Description (REJECT):_ "The segment consists of first aid and safety services" or "The operating segment consists of the direct sale of uniforms and related items."',
            '    - _Revenue Mechanism Statement (ACCEPT):_ "Revenue is derived from subscription fees" or "The company earns service fees from installation and maintenance."',
            '**Required Wording:** Require explicit language about the charging or revenue mechanism, such as _revenue derived from, earns revenue from, charges, subscription fees, licensing revenue, usage-based pricing, transaction fees, service fees, royalties,_ or _advertising revenue_.',
            'Do not map the existence of products, services, sales, rentals, or business activity into a canonical revenue model unless the way money is earned is directly stated.',
            'Do not use the fact that something is physically sold to infer "hardware sales" unless the text explicitly frames the monetization in those terms.',
        ],
    },
}

RELATION_TRIGGER_PHRASES = {
    "SERVES": (
        "small business",
        "small businesses",
        "smb",
        "mid market",
        "mid-market",
        "midsize business",
        "large enterprise",
        "large enterprises",
        "enterprise customers",
        "government agencies",
        "educational institutions",
        "schools and universities",
        "healthcare organizations",
        "healthcare providers",
        "financial services firms",
        "manufacturers",
        "retailers",
        "developers",
        "it professionals",
        "customers include",
    ),
    "SELLS_THROUGH": (
        "sells through",
        "sold through",
        "distributed through",
        "distributed via",
        "available through",
        "available via",
        "through resellers",
        "through distributors",
        "through oems",
        "through system integrators",
        "through managed service providers",
        "through marketplaces",
        "direct sales",
        "sales force",
        "online store",
        "available online",
    ),
    "MONETIZES_VIA": (
        "subscription fees",
        "subscription revenue",
        "subscription-based",
        "licensing revenue",
        "license fees",
        "service fees",
        "maintenance fees",
        "usage-based pricing",
        "usage based pricing",
        "transaction fees",
        "advertising revenue",
        "royalty revenue",
        "royalties",
        "commission revenue",
        "per-use fees",
    ),
}

def example_chunk_key(example: dict[str, Any]) -> dict[str, Any]:
    metadata = example.get("metadata") or {}
    return metadata.get("chunk_key") or {}


def example_chunk_key_text(example: dict[str, Any]) -> str:
    return json.dumps(example_chunk_key(example), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def chunk_key_matches_filter(example: dict[str, Any], filter_text: str | None) -> bool:
    if not filter_text:
        return False

    filter_key = canonical_entity_key(filter_text)
    if not filter_key:
        return False

    chunk_key = example_chunk_key(example)
    if any(filter_key in canonical_entity_key(str(value)) for value in chunk_key.values()):
        return True

    return filter_key in canonical_entity_key(str(example.get("input", "")))


def triple_key(triple: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        canonical_entity_key(str(triple.get("subject", ""))),
        str(triple.get("subject_type", "")),
        str(triple.get("relation", "")),
        canonical_entity_key(str(triple.get("object", ""))),
        str(triple.get("object_type", "")),
    )


def select_examples_deterministically(
    examples: list[dict[str, Any]],
    *,
    target_count: int,
) -> list[dict[str, Any]]:
    ranked = sorted(examples, key=example_chunk_key_text)
    return ranked[: max(0, target_count)]


def _stable_chunk_score_from_text(chunk_key_text: str) -> int:
    return int(hashlib.sha1(chunk_key_text.encode("utf-8")).hexdigest(), 16)


def relation_trigger_match_report(chunk_text: str) -> dict[str, Any]:
    normalized_text = canonical_entity_key(str(chunk_text or ""))
    matched_triggers: dict[str, list[str]] = {}
    trigger_score = 0

    for relation, phrases in RELATION_TRIGGER_PHRASES.items():
        hits = []
        seen_normalized: set[str] = set()
        for phrase in phrases:
            normalized_phrase = canonical_entity_key(phrase)
            if not normalized_phrase or normalized_phrase in seen_normalized:
                continue
            if normalized_phrase in normalized_text:
                hits.append(phrase)
                seen_normalized.add(normalized_phrase)

        if hits:
            matched_triggers[relation] = hits
            trigger_score += len(hits)

    return {
        "matched_triggers": matched_triggers,
        "matched_relation_count": len(matched_triggers),
        "trigger_score": trigger_score,
    }


def relation_trigger_eligibility_report(chunk_text: str) -> dict[str, Any]:
    return chunk_quality_report(chunk_text)


def build_relation_trigger_candidate_example(
    rows: list[dict[str, Any]],
    *,
    instruction: str = DEFAULT_INSTRUCTION,
    min_word_count: int = 80,
    min_char_count: int = 400,
) -> dict[str, Any] | None:
    example = build_empty_example(
        rows,
        instruction=instruction,
        min_word_count=min_word_count,
        min_char_count=min_char_count,
    )
    if example is None:
        return None

    eligibility_report = relation_trigger_eligibility_report(example.get("input", ""))
    if not eligibility_report["is_narrative_business_prose"]:
        return None

    trigger_report = relation_trigger_match_report(example.get("input", ""))
    if trigger_report["matched_relation_count"] <= 0:
        return None

    metadata = example.setdefault("metadata", {})
    metadata["empty_target"] = False
    metadata["stage3_candidate_source"] = "relation_trigger"
    metadata["relation_trigger_hits"] = trigger_report["matched_triggers"]
    metadata["relation_trigger_score"] = trigger_report["trigger_score"]
    metadata["relation_trigger_relation_count"] = trigger_report["matched_relation_count"]
    metadata["relation_trigger_eligibility"] = eligibility_report
    example["output"]["extraction_notes"] = (
        "No ontology-aligned business-model triples projected from this chunk; selected as a Stage 3 relation-trigger candidate."
    )
    return example


def build_relation_trigger_candidate_pool_from_rows(
    rows: Iterable[dict[str, Any]],
    *,
    target_candidate_count: int,
    instruction: str = DEFAULT_INSTRUCTION,
    limit_chunks: int | None = None,
    min_word_count: int = 80,
    min_char_count: int = 400,
    exclude_chunk_keys: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    max_heap: list[tuple[tuple[int, int, str], dict[str, Any]]] = []
    processed_chunks = 0
    eligible_trigger_chunk_count = 0
    excluded_chunk_count = 0
    matched_relation_counts: Counter[str] = Counter()
    excluded_chunk_keys = exclude_chunk_keys or set()

    for chunk_rows in iter_grouped_rows(rows):
        processed_chunks += 1
        example = build_relation_trigger_candidate_example(
            chunk_rows,
            instruction=instruction,
            min_word_count=min_word_count,
            min_char_count=min_char_count,
        )
        if example is not None:
            chunk_key_text = example_chunk_key_text(example)
            if chunk_key_text in excluded_chunk_keys:
                excluded_chunk_count += 1
                if limit_chunks is not None and processed_chunks >= limit_chunks:
                    break
                continue

            eligible_trigger_chunk_count += 1
            for relation in (example.get("metadata", {}).get("relation_trigger_hits") or {}):
                matched_relation_counts[relation] += 1

            trigger_score = int(example.get("metadata", {}).get("relation_trigger_score") or 0)
            stable_score = _stable_chunk_score_from_text(chunk_key_text)
            rank = (trigger_score, -stable_score, chunk_key_text)
            heap_item = (rank, example)
            if len(max_heap) < target_candidate_count:
                heapq.heappush(max_heap, heap_item)
            elif target_candidate_count > 0 and rank > max_heap[0][0]:
                heapq.heapreplace(max_heap, heap_item)

        if limit_chunks is not None and processed_chunks >= limit_chunks:
            break

    selected_items = sorted(max_heap, key=lambda item: (-item[0][0], -item[0][1], item[0][2]))
    selected_examples = [example for _, example in selected_items]
    report = {
        "processed_chunk_count": processed_chunks,
        "eligible_relation_trigger_chunk_count": eligible_trigger_chunk_count,
        "excluded_chunk_count": excluded_chunk_count,
        "sampled_relation_trigger_chunk_count": len(selected_examples),
        "target_relation_trigger_count": target_candidate_count,
        "matched_relation_counts": dict(sorted(matched_relation_counts.items())),
        "min_word_count": min_word_count,
        "min_char_count": min_char_count,
    }
    return selected_examples, report


def build_relation_trigger_candidate_pool(
    *,
    hf_dataset: str | None,
    split: str,
    cache_dir: str | None,
    parquet_files: list[str] | None,
    streaming: bool,
    limit_rows: int | None,
    limit_chunks: int | None,
    target_candidate_count: int,
    exclude_chunk_key_texts: set[str],
    instruction: str = DEFAULT_INSTRUCTION,
    min_word_count: int,
    min_char_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_candidate_count <= 0:
        return [], {
            "processed_chunk_count": 0,
            "eligible_relation_trigger_chunk_count": 0,
            "excluded_chunk_count": 0,
            "sampled_relation_trigger_chunk_count": 0,
            "target_relation_trigger_count": 0,
            "matched_relation_counts": {},
            "min_word_count": min_word_count,
            "min_char_count": min_char_count,
        }

    rows = load_finreflectkg_rows(
        hf_dataset=hf_dataset,
        split=split,
        cache_dir=cache_dir,
        parquet_files=parquet_files,
        streaming=streaming,
        limit_rows=limit_rows,
    )
    return build_relation_trigger_candidate_pool_from_rows(
        rows,
        target_candidate_count=target_candidate_count,
        instruction=instruction,
        limit_chunks=limit_chunks,
        min_word_count=min_word_count,
        min_char_count=min_char_count,
        exclude_chunk_keys=exclude_chunk_key_texts,
    )


def existing_valid_triples(example: dict[str, Any]) -> list[dict[str, str]]:
    existing_triples = example.get("output", {}).get("triples", [])
    validation = validate_triples(existing_triples, dedupe=True)
    return validation["valid_triples"]


def example_company_name(example: dict[str, Any], *, existing_triples: list[dict[str, str]] | None = None) -> str:
    metadata = example.get("metadata") or {}
    company_name = str(metadata.get("company_name") or "").strip()
    if company_name:
        return company_name

    valid_triples = existing_triples if existing_triples is not None else existing_valid_triples(example)
    for triple in valid_triples:
        if triple.get("subject_type") == "Company":
            return str(triple.get("subject", "")).strip()
    return ""


def _append_inventory_entity(
    inventory: dict[str, list[str]],
    seen: set[tuple[str, str]],
    *,
    node_type: str,
    name: str,
) -> None:
    normalized_name = str(name or "").strip()
    if not normalized_name:
        return

    entity_key = (node_type, canonical_entity_key(normalized_name))
    if entity_key in seen:
        return

    inventory.setdefault(node_type, []).append(normalized_name)
    seen.add(entity_key)


def subject_inventory_for_relation(
    example: dict[str, Any],
    relation: str,
    *,
    existing_triples: list[dict[str, str]] | None = None,
) -> dict[str, list[str]]:
    valid_triples = existing_triples if existing_triples is not None else existing_valid_triples(example)
    permitted_types = set(allowed_subject_types(relation))
    inventory: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()

    company_name = example_company_name(example, existing_triples=valid_triples)
    if company_name and "Company" in permitted_types:
        _append_inventory_entity(inventory, seen, node_type="Company", name=company_name)

    for triple in valid_triples:
        subject_type = str(triple.get("subject_type", ""))
        if subject_type in permitted_types:
            _append_inventory_entity(
                inventory,
                seen,
                node_type=subject_type,
                name=str(triple.get("subject", "")),
            )

        object_type = str(triple.get("object_type", ""))
        if object_type in permitted_types and object_type != "Company":
            _append_inventory_entity(
                inventory,
                seen,
                node_type=object_type,
                name=str(triple.get("object", "")),
            )

    return dict(sorted(inventory.items()))


def _is_text_grounded(value: str, source_text: str) -> bool:
    normalized_value = canonical_entity_key(str(value or ""))
    normalized_source = canonical_entity_key(str(source_text or ""))
    return bool(normalized_value) and normalized_value in normalized_source


def filter_relation_triples(
    example: dict[str, Any],
    relation: str,
    triples: list[dict[str, Any]],
) -> dict[str, Any]:
    source_text = str(example.get("input") or "")
    valid_existing_triples = existing_valid_triples(example)
    subject_inventory = subject_inventory_for_relation(
        example,
        relation,
        existing_triples=valid_existing_triples,
    )
    allowed_subject_keys = {
        (node_type, canonical_entity_key(name))
        for node_type, names in subject_inventory.items()
        for name in names
    }

    validation = validate_triples(triples, dedupe=True)
    grounded_triples: list[dict[str, str]] = []
    grounding_rejections: list[dict[str, Any]] = []

    for triple in validation["valid_triples"]:
        subject = str(triple.get("subject", ""))
        subject_type = str(triple.get("subject_type", ""))
        subject_key = canonical_entity_key(subject)

        if subject_key in GENERIC_SUBJECT_KEYS:
            grounding_rejections.append(
                {
                    "triple": triple,
                    "reason": "generic_subject_name",
                }
            )
            continue

        if _is_text_grounded(subject, source_text) or (subject_type, subject_key) in allowed_subject_keys:
            grounded_triples.append(triple)
            continue

        grounding_rejections.append(
            {
                "triple": triple,
                "reason": "subject_not_grounded_in_text_or_known_subject_inventory",
            }
        )

    return {
        "valid_triples": grounded_triples,
        "invalid_triple_count": validation["summary"]["invalid_triple_count"],
        "duplicate_triple_count": validation["summary"]["duplicate_triple_count"],
        "grounding_rejection_count": len(grounding_rejections),
        "grounding_rejections": grounding_rejections,
        "subject_inventory": subject_inventory,
    }


def relation_system_prompt(relation: str) -> str:
    task = RELATION_TASKS[relation]
    object_type = task["object_type"]
    permitted_subject_types = allowed_subject_types(relation)
    ontology = load_ontology_config()
    relation_definition = str(ontology["relations"][relation]["definition"]).strip()
    object_definition = str(ontology["node_types"][object_type]["definition"]).strip()
    canonical_group = canonical_labels(object_type) if object_type in {"CustomerType", "Channel", "RevenueModel"} else []
    labels_json = json.dumps(canonical_group, ensure_ascii=False, separators=(",", ":"))
    canonical_definitions = ontology["canonical_labels"].get(object_type, {})
    canonical_definition_lines = "\n".join(
        f'- "{label}": {canonical_definitions[label]}'
        for label in canonical_group
    )
    skepticism_lines = "\n".join(f"- {line}" for line in task["relation_specific_skepticism"])
    return (
        f"You are a strict knowledge-graph teacher for SEC 10-K extraction. "
        f"Your mission is to help build the cleanest possible training dataset for a future extractor. "
        f"You know this mission succeeds only when **noise is kept extremely low**. "
        f"A missed triple is acceptable; a noisy triple is harmful because it teaches the wrong behavior. "
        f"You are literal, skeptical, and evidence-bound. "
        f"You reward precision over recall and prefer missing a triple over adding a weak one.\n\n"
        f"You augment one SEC 10-K chunk by extracting only {relation} triples.\n\n"
        f"You are not re-extracting the whole graph. You are only adding {relation} facts.\n"
        f"Inputs may include <company_name>, <existing_triples>, <allowed_subjects>, and <text_to_analyze>. Your only focus is {relation} facts.\n"
        f'The default answer is {{"triples":[]}}. If the relation is not explicit, return {{"triples":[]}}.\n'
        f"**Be conservative. Quality is more important than quantity.**\n"
        f"Only change that default when the chunk contains very strong, relation-specific evidence.\n"
        f"Only extract from chunks that read like explanatory business prose.\n"
        f"If the chunk is mostly a table, list, header, address block, legal boilerplate, accounting line items, or other heavily formatted/non-narrative text, return {{\"triples\":[]}}.\n"
        f"If the chunk does not read like a business explanation written in prose sentences, return {{\"triples\":[]}}.\n"
        f"Extract only facts that the chunk clearly and explicitly states.\n"
        f"Do not extract facts that are inferable, deducible, implied, suggested, likely, or merely consistent with the text.\n"
        f"Do not use world knowledge, business intuition, probability, or background expectations to complete the extraction.\n"
        f"Treat every candidate triple like a claim in an audit: if the chunk would not prove it on its own, it must not be extracted.\n"
        f"The chunk must stand alone as proof. The ontology, your background knowledge, and likely business logic are not proof.\n"
        f"Return a triple only if {task['trigger_text']}.\n"
        f"**Do not guess. Do not infer. Do not complete patterns from world knowledge.**\n\n"
        f"ONTOLOGY DEFINITION FOR {relation}:\n"
        f"- Relation: {relation_definition}\n"
        f'- Object node type "{object_type}": {object_definition}\n'
        f"- Allowed subject types: {json.dumps(permitted_subject_types, ensure_ascii=False)}\n"
        f"- Allowed object type: {object_type}\n"
        f"- Allowed {object_type} labels: {labels_json}\n"
        f"- Canonical {object_type} label definitions:\n"
        f"{canonical_definition_lines}\n\n"
        f"DECISION STANDARD:\n"
        f"- First decide whether the chunk is suitable evidence at all: it must be prose-like, business-explanatory, and not dominated by formatting, lists, tables, or ledger-style content.\n"
        f"- If the chunk fails that suitability gate, output {{\"triples\":[]}} immediately.\n"
        f"- First decide whether the chunk **explicitly and surely states the relation.**\n"
        f"- Then decide whether the wording matches **exactly** one canonical {object_type} label.\n"
        f"- The chunk must do both jobs itself: it must explicitly state the relation and explicitly support the chosen canonical label.\n"
        f"- If the relation is stated but the canonical label still requires interpretation, output {{\"triples\":[]}}.\n"
        f"- If the canonical label seems likely but the relation itself is not directly stated, output {{\"triples\":[]}}.\n"
        f'- If either step fails, output {{"triples":[]}} for that candidate.\n\n'
        f"REMEMBER: better one good triple less than one bad triple more.\n\n"
        f"Burden of proof:\n"
        f"- Ask: what exact words in the chunk prove this relation?\n"
        f"- Ask separately: what exact words in the chunk justify this exact canonical label?\n"
        f"- If you cannot answer both questions from the text alone, output no triple.\n"
        f'- **Logical Leap Test:** If you have to use the word "because" to connect the text to the triple (e.g., "{task["logical_leap_example"]}"), then you are inferring. **If you are inferring, you must return an empty list.**\n'
        f"- If you cannot point to concrete wording in the chunk, output no triple.\n"
        f"- If the chunk only makes the relation seem plausible, output no triple.\n"
        f"- If the chunk is generic, high-level, or boilerplate, output no triple.\n"
        f"- If the chunk is heavily formatted, table-like, mostly numeric, list-like, or looks like a caption/header/statement block rather than explanatory prose, output no triple.\n"
        f"- If the text names an office, facility, segment, product, service, or business activity but does not explicitly state the target relation, output no triple.\n"
        f"- If the text names a business segment whose title contains words like sales, rental, service, or subscription, do not treat the title itself as proof of the relation.\n"
        f"- If you are between a triple and an empty list, choose the empty list.\n\n"
        f"Subject policy:\n"
        f"- Prefer subjects already listed in <allowed_subjects>.\n"
        f"- If <allowed_subjects> is empty for a subject type, emit a triple only when that subject is explicitly named in the chunk.\n"
        f"- The subject must either appear in the chunk text or be provided in <allowed_subjects> / <company_name>.\n"
        f"- Never use pronouns like 'we', 'our company', or 'the company' as entity names.\n"
        f"- Never use an external partner, distributor, reseller, marketplace, or customer name as the subject just because it appears in the chunk.\n\n"
        f"**Relation-specific skepticism ({relation}):**\n"
        f"{skepticism_lines}\n\n"
        f"NEGATIVE EXAMPLES:\n"
        f'- "The primary markets served by all operating segments are local in nature and highly fragmented." -> no triple\n'
        f'- "Product, design, price, quality, service and convenience to the customer are competitive elements." -> no triple\n'
        f'- "The company competes with national, regional and local providers." -> no triple\n'
        f'- "Cintas Corporation, Washington, principal executive offices..." -> no triple\n\n'
        f"Never output any relation other than {relation}.\n"
        f"Never invent labels outside the allowed list.\n"
        f"\nOutput only JSON.\n"
        f'Output shape: {{"triples":[{{"subject":"...","subject_type":"...","relation":"{relation}","object":"...","object_type":"{object_type}"}}]}}'
    )


def build_stage3_prompt(example: dict[str, Any], relation: str) -> str:
    valid_existing_triples = existing_valid_triples(example)
    allowed_subjects = subject_inventory_for_relation(
        example,
        relation,
        existing_triples=valid_existing_triples,
    )
    company_name = example_company_name(example, existing_triples=valid_existing_triples)

    prompt_parts = []
    prompt_parts.append(
        "<instruction>\n"
        "First decide whether the chunk is suitable evidence at all.\n"
        "Only extract from text that reads like explanatory business prose.\n"
        "If the chunk is mostly a table, list, heading block, financial statement, legal disclosure, address section, or other heavily formatted/non-narrative text, return {\"triples\":[]}.\n"
        "Return a triple only when the chunk clearly and explicitly states the relation.\n"
        "Do not return a triple when the relation is only inferable, deducible, implied, suggested, probable, or merely consistent with the text.\n"
        "If you would need personal knowledge, business reasoning, or synthesis across clues, return {\"triples\":[]}.\n"
        "The chunk itself must prove both the relation and the exact canonical label.\n"
        "If either of those still requires interpretation, return {\"triples\":[]}.\n"
        "Names of offices, facilities, segments, products, services, or business activities are not enough by themselves.\n"
        "If the text does not directly say it, do not extract it.\n"
        "</instruction>"
    )
    if company_name:
        prompt_parts.append(f"<company_name>\n{company_name}\n</company_name>")
    prompt_parts.append(
        f"<existing_triples>\n{_compact_json({'triples': valid_existing_triples})}\n</existing_triples>"
    )
    prompt_parts.append(f"<allowed_subjects>\n{_compact_json(allowed_subjects)}\n</allowed_subjects>")
    prompt_parts.append(f"<text_to_analyze>\n{example.get('input', '')}\n</text_to_analyze>")
    prompt_parts.append(f"<relation_to_extract>\n{relation}\n</relation_to_extract>")
    return "\n\n".join(prompt_parts)


class Stage3TeacherAugmentor:
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        model: str = "local-model",
        debug_dir: str | None = None,
        debug_chunk_filter: str | None = None,
    ):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.debug_chunk_filter = debug_chunk_filter

    @staticmethod
    def _schema_def(name: str, relation: str, object_type: str) -> dict[str, Any]:
        object_field: dict[str, Any] = {"type": "string"}
        canonical_group = canonical_labels(object_type) if object_type in {"CustomerType", "Channel", "RevenueModel"} else []
        if canonical_group:
            object_field["enum"] = canonical_group

        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": {
                    "type": "object",
                    "properties": {
                        "triples": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "subject": {"type": "string"},
                                    "subject_type": {"type": "string", "enum": allowed_subject_types(relation)},
                                    "relation": {"type": "string", "enum": [relation]},
                                    "object": object_field,
                                    "object_type": {"type": "string", "enum": [object_type]},
                                },
                                "required": ["subject", "subject_type", "relation", "object", "object_type"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["triples"],
                    "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def _load_json_payload(content: str, fallback_payload: str) -> dict[str, Any]:
        content = (content or "").strip()
        if not content:
            return json.loads(fallback_payload)
        if not content.endswith("}"):
            last_object_end = content.rfind("}")
            content = content[: last_object_end + 1] if last_object_end != -1 else fallback_payload
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return json.loads(fallback_payload)

    def _call_relation(
        self,
        *,
        example: dict[str, Any],
        relation: str,
        max_retries: int = 3,
        temperature: float = 0.0,
    ) -> tuple[dict[str, Any], str | None]:
        fallback_payload = '{"triples":[]}'
        object_type = RELATION_TASKS[relation]["object_type"]
        schema_name = "Stage3" + "".join(part.title() for part in relation.lower().split("_")) + "Extraction"
        schema_def = self._schema_def(schema_name, relation, object_type)
        system_prompt = relation_system_prompt(relation)
        user_prompt = build_stage3_prompt(example, relation)
        debug_enabled = self.debug_dir is not None and chunk_key_matches_filter(example, self.debug_chunk_filter)

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    response_format=schema_def,
                )
                content = response.choices[0].message.content
                payload = self._load_json_payload(content or "", fallback_payload)
                triples = payload.get("triples", [])
                filtered = filter_relation_triples(example, relation, triples)
                if debug_enabled:
                    self._write_debug_payload(
                        example=example,
                        relation=relation,
                        attempt=attempt,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        schema_def=schema_def,
                        raw_response=content or "",
                        filtered=filtered,
                    )
                return {
                    "relation": relation,
                    "attempts_used": attempt,
                    "raw_response": content or "",
                    "valid_triples": filtered["valid_triples"],
                    "invalid_triple_count": filtered["invalid_triple_count"],
                    "duplicate_triple_count": filtered["duplicate_triple_count"],
                    "grounding_rejection_count": filtered["grounding_rejection_count"],
                    "grounding_rejections": filtered["grounding_rejections"],
                    "subject_inventory": filtered["subject_inventory"],
                }, content
            except Exception as exc:  # pragma: no cover - exercised only in real API calls.
                last_error = str(exc)
                if debug_enabled:
                    self._write_debug_payload(
                        example=example,
                        relation=relation,
                        attempt=attempt,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        schema_def=schema_def,
                        raw_response="",
                        filtered={"valid_triples": [], "invalid_triple_count": 0, "duplicate_triple_count": 0, "grounding_rejection_count": 0, "grounding_rejections": [], "subject_inventory": {}},
                        error=last_error,
                    )
                logger.warning("Stage 3 teacher call failed for %s on attempt %s/%s: %s", relation, attempt, max_retries, exc)

        return {
            "relation": relation,
            "attempts_used": max_retries,
            "raw_response": "",
            "valid_triples": [],
            "invalid_triple_count": 0,
            "duplicate_triple_count": 0,
            "grounding_rejection_count": 0,
            "grounding_rejections": [],
            "subject_inventory": subject_inventory_for_relation(example, relation),
            "error": last_error or "unknown_error",
        }, None

    def _write_debug_payload(
        self,
        *,
        example: dict[str, Any],
        relation: str,
        attempt: int,
        system_prompt: str,
        user_prompt: str,
        schema_def: dict[str, Any],
        raw_response: str,
        filtered: dict[str, Any],
        error: str | None = None,
    ) -> None:
        if self.debug_dir is None:
            return

        chunk_key = example_chunk_key(example)
        filename = "__".join(
            [
                str(chunk_key.get("ticker") or "unknown"),
                str(chunk_key.get("page_id") or "page"),
                str(chunk_key.get("chunk_id") or "chunk"),
                relation.lower(),
            ]
        ).replace("/", "_")
        path = self.debug_dir / f"{filename}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunk_key": chunk_key,
            "relation": relation,
            "attempt": attempt,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "schema_def": schema_def,
            "raw_response": raw_response,
            "filtered_report": filtered,
            "error": error,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def augment_example(
        self,
        example: dict[str, Any],
        *,
        relations: Iterable[str] = STAGE3_RELATIONS,
        max_retries: int = 3,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        relation_reports: dict[str, dict[str, Any]] = {}

        for relation in relations:
            report, _ = self._call_relation(
                example=example,
                relation=relation,
                max_retries=max_retries,
            )
            relation_reports[relation] = report

        augmented_example, merge_report = merge_teacher_reports_into_example(example, relation_reports)
        call_log = {
            "chunk_key": example_chunk_key(example),
            "relation_reports": relation_reports,
            "merge_report": merge_report,
        }
        return augmented_example, call_log


def merge_teacher_reports_into_example(
    example: dict[str, Any],
    relation_reports: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    existing_triples = example.get("output", {}).get("triples", [])
    existing_validation = validate_triples(existing_triples, dedupe=True)
    existing_valid = existing_validation["valid_triples"]
    existing_keys = {triple_key(triple) for triple in existing_valid}

    added_triples: list[dict[str, Any]] = []
    added_keys: set[tuple[str, str, str, str, str]] = set()
    added_relation_counts: Counter[str] = Counter()
    provenance_records: list[dict[str, Any]] = []
    overlap_with_existing = 0
    teacher_invalid_total = 0
    teacher_duplicate_total = 0
    teacher_grounding_rejection_total = 0
    teacher_error_relations: list[str] = []

    for relation in STAGE3_RELATIONS:
        report = relation_reports.get(relation, {})
        teacher_invalid_total += int(report.get("invalid_triple_count", 0))
        teacher_duplicate_total += int(report.get("duplicate_triple_count", 0))
        teacher_grounding_rejection_total += int(report.get("grounding_rejection_count", 0))
        if report.get("error"):
            teacher_error_relations.append(relation)

        for triple in report.get("valid_triples", []):
            key = triple_key(triple)
            if key in existing_keys:
                overlap_with_existing += 1
                continue
            if key in added_keys:
                teacher_duplicate_total += 1
                continue

            added_keys.add(key)
            added_triples.append(triple)
            added_relation_counts[relation] += 1
            provenance_records.append(
                {
                    "source": STAGE3_PROVENANCE[relation],
                    "triple": triple,
                }
            )

    merged_validation = validate_triples(existing_valid + added_triples, dedupe=True)
    final_triples = merged_validation["valid_triples"]

    augmented_example = copy.deepcopy(example)
    augmented_example["output"]["triples"] = final_triples
    augmented_example["output"]["extraction_notes"] = (
        "Stage 1 deterministic projection merged with Stage 3 teacher augmentation for missing ontology slices."
    )

    metadata = augmented_example.setdefault("metadata", {})
    metadata["empty_target"] = len(final_triples) == 0
    metadata["stage3"] = {
        "teacher_relations_requested": list(STAGE3_RELATIONS),
        "teacher_added_triple_count": len(added_triples),
        "teacher_added_relation_counts": dict(sorted(added_relation_counts.items())),
        "teacher_invalid_triple_count": teacher_invalid_total,
        "teacher_duplicate_triple_count": teacher_duplicate_total,
        "teacher_grounding_rejection_count": teacher_grounding_rejection_total,
        "teacher_overlap_with_existing_count": overlap_with_existing,
        "teacher_error_relations": teacher_error_relations,
        "teacher_added_triples": provenance_records,
    }

    merge_report = {
        "input_existing_triple_count": len(existing_valid),
        "teacher_added_triple_count": len(added_triples),
        "final_triple_count": len(final_triples),
        "teacher_added_relation_counts": dict(sorted(added_relation_counts.items())),
        "teacher_invalid_triple_count": teacher_invalid_total,
        "teacher_duplicate_triple_count": teacher_duplicate_total,
        "teacher_grounding_rejection_count": teacher_grounding_rejection_total,
        "teacher_overlap_with_existing_count": overlap_with_existing,
        "teacher_error_relations": teacher_error_relations,
    }
    return augmented_example, merge_report


def finalize_stage3_dataset(
    augmented_positive_examples: list[dict[str, Any]],
    augmented_empty_candidates: list[dict[str, Any]],
    *,
    empty_ratio: float,
    augmented_relation_trigger_candidates: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    relation_trigger_candidates = augmented_relation_trigger_candidates or []
    final_positive_examples = [example for example in augmented_positive_examples if example.get("output", {}).get("triples")]
    promoted_empty_examples = [example for example in augmented_empty_candidates if example.get("output", {}).get("triples")]
    verified_empty_examples = [example for example in augmented_empty_candidates if not example.get("output", {}).get("triples")]
    promoted_relation_trigger_examples = [example for example in relation_trigger_candidates if example.get("output", {}).get("triples")]
    rejected_relation_trigger_examples = [example for example in relation_trigger_candidates if not example.get("output", {}).get("triples")]

    final_positive_examples.extend(promoted_empty_examples)
    final_positive_examples.extend(promoted_relation_trigger_examples)
    target_empty_count = max(0, int(round(len(final_positive_examples) * empty_ratio)))
    selected_empty_examples = select_examples_deterministically(
        verified_empty_examples,
        target_count=target_empty_count,
    )
    training_examples = final_positive_examples + selected_empty_examples

    relation_counts: Counter[str] = Counter()
    for example in training_examples:
        for triple in example.get("output", {}).get("triples", []):
            relation_counts[str(triple.get("relation", ""))] += 1

    report = {
        "augmented_positive_example_count": len(augmented_positive_examples),
        "augmented_empty_candidate_count": len(augmented_empty_candidates),
        "augmented_relation_trigger_candidate_count": len(relation_trigger_candidates),
        "promoted_empty_example_count": len(promoted_empty_examples),
        "verified_empty_candidate_count": len(verified_empty_examples),
        "promoted_relation_trigger_example_count": len(promoted_relation_trigger_examples),
        "rejected_relation_trigger_example_count": len(rejected_relation_trigger_examples),
        "final_positive_example_count": len(final_positive_examples),
        "target_empty_count": target_empty_count,
        "selected_empty_count": len(selected_empty_examples),
        "final_training_example_count": len(training_examples),
        "target_empty_ratio": empty_ratio,
        "realized_empty_ratio": (len(selected_empty_examples) / len(final_positive_examples)) if final_positive_examples else 0.0,
        "relation_counts": dict(sorted(relation_counts.items())),
    }
    return final_positive_examples, selected_empty_examples, training_examples, report


def build_additional_empty_pool(
    *,
    hf_dataset: str | None,
    split: str,
    cache_dir: str | None,
    parquet_files: list[str] | None,
    streaming: bool,
    limit_rows: int | None,
    limit_chunks: int | None,
    target_empty_count: int,
    exclude_chunk_key_texts: set[str],
    min_word_count: int,
    min_char_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_finreflectkg_rows(
        hf_dataset=hf_dataset,
        split=split,
        cache_dir=cache_dir,
        parquet_files=parquet_files,
        streaming=streaming,
        limit_rows=limit_rows,
    )
    return sample_empty_examples_by_count(
        rows,
        target_empty_count=target_empty_count,
        empty_ratio=None,
        positive_example_count=None,
        limit_chunks=limit_chunks,
        min_word_count=min_word_count,
        min_char_count=min_char_count,
        exclude_chunk_keys=exclude_chunk_key_texts,
    )


def refill_and_finalize_stage3_dataset(
    *,
    augmentor: Stage3TeacherAugmentor,
    augmented_positive_examples: list[dict[str, Any]],
    initial_augmented_empty_candidates: list[dict[str, Any]],
    augmented_relation_trigger_candidates: list[dict[str, Any]] | None,
    empty_ratio: float,
    hf_dataset: str | None,
    split: str,
    cache_dir: str | None,
    parquet_files: list[str] | None,
    streaming: bool,
    limit_rows: int | None,
    limit_chunks: int | None,
    min_word_count: int,
    min_char_count: int,
    max_retries: int,
    refill_pool_multiplier: float,
    max_refill_rounds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    all_augmented_empty_candidates = list(initial_augmented_empty_candidates)
    relation_trigger_candidates = augmented_relation_trigger_candidates or []
    refill_reports: list[dict[str, Any]] = []
    refill_logs: list[dict[str, Any]] = []

    final_positive_examples, selected_empty_examples, training_examples, report = finalize_stage3_dataset(
        augmented_positive_examples,
        all_augmented_empty_candidates,
        empty_ratio=empty_ratio,
        augmented_relation_trigger_candidates=relation_trigger_candidates,
    )

    exclude_chunk_key_texts = {
        example_chunk_key_text(example)
        for example in augmented_positive_examples + all_augmented_empty_candidates + relation_trigger_candidates
    }

    refill_round = 0
    while report["selected_empty_count"] < report["target_empty_count"] and refill_round < max_refill_rounds:
        refill_round += 1
        deficit = report["target_empty_count"] - report["selected_empty_count"]
        candidate_target_count = max(deficit, int(math.ceil(deficit * refill_pool_multiplier)))
        if candidate_target_count <= 0:
            break

        new_candidates, pool_report = build_additional_empty_pool(
            hf_dataset=hf_dataset,
            split=split,
            cache_dir=cache_dir,
            parquet_files=parquet_files,
            streaming=streaming,
            limit_rows=limit_rows,
            limit_chunks=limit_chunks,
            target_empty_count=candidate_target_count,
            exclude_chunk_key_texts=exclude_chunk_key_texts,
            min_word_count=min_word_count,
            min_char_count=min_char_count,
        )
        if not new_candidates:
            refill_reports.append(
                {
                    "round": refill_round,
                    "requested_candidate_count": candidate_target_count,
                    "sampled_candidate_count": 0,
                    "note": "No additional empty candidates available.",
                }
            )
            break

        refill_reports.append(
            {
                "round": refill_round,
                "requested_candidate_count": candidate_target_count,
                **pool_report,
            }
        )
        for example in new_candidates:
            exclude_chunk_key_texts.add(example_chunk_key_text(example))

        new_augmented_candidates: list[dict[str, Any]] = []
        for example in new_candidates:
            augmented_example, call_log = augmentor.augment_example(example, max_retries=max_retries)
            new_augmented_candidates.append(augmented_example)
            refill_logs.append(call_log)

        all_augmented_empty_candidates.extend(new_augmented_candidates)
        final_positive_examples, selected_empty_examples, training_examples, report = finalize_stage3_dataset(
            augmented_positive_examples,
            all_augmented_empty_candidates,
            empty_ratio=empty_ratio,
            augmented_relation_trigger_candidates=relation_trigger_candidates,
        )

    report["refill_round_count"] = refill_round
    report["refill_reports"] = refill_reports
    report["final_candidate_empty_pool_count"] = len(all_augmented_empty_candidates)
    return final_positive_examples, selected_empty_examples, training_examples, all_augmented_empty_candidates, report, refill_logs


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
