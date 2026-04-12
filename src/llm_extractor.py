import json
import logging
import re
from collections import Counter
from typing import Any, List, Literal

from ontology_config import canonical_labels, load_ontology_config
from ontology_validator import validate_triples
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


NodeType = Literal["Company", "BusinessSegment", "Offering", "CustomerType", "Channel", "Place", "RevenueModel"]
RelationType = Literal[
    "HAS_SEGMENT",
    "OFFERS",
    "PART_OF",
    "SERVES",
    "OPERATES_IN",
    "SELLS_THROUGH",
    "PARTNERS_WITH",
    "MONETIZES_VIA",
]

CANONICAL_CUSTOMER_TYPES = canonical_labels("CustomerType")
CANONICAL_CHANNELS = canonical_labels("Channel")
CANONICAL_REVENUE_MODELS = canonical_labels("RevenueModel")
V2_SEGMENT_SERVES_CANONICAL_DEFINITIONS = load_ontology_config("v2_segment_serves")["canonical_labels"]


def _xml_definition_lines(definitions: dict[str, str]) -> str:
    return "\n".join(f'- "{label}": {definition}' for label, definition in definitions.items())


def _v2_source_context_user_prompt(full_text: str, company_name: str | None = None) -> str:
    return (
        "<source_context>\n"
        f"<company_name>\n{company_name or ''}\n</company_name>\n\n"
        f"<source_filing>\n{full_text}\n</source_filing>\n"
        "</source_context>\n\n"
        "<context_instruction>\n"
        "This is context only. Do not answer this message. Wait for the workflow step.\n"
        "</context_instruction>"
    )


class Triple(BaseModel):
    subject: str
    subject_type: NodeType
    relation: RelationType
    object: str
    object_type: NodeType


class KnowledgeGraphExtraction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    extraction_notes: str = Field(
        default="",
        validation_alias=AliasChoices("extraction_notes", "chain_of_thought_reasoning"),
        serialization_alias="extraction_notes",
    )
    triples: List[Triple] = Field(default_factory=list)


class IncrementalKnowledgeGraphExtraction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    current_section_analyzed: str = Field(default="", description="The exact title or header of the section you just analyzed.")
    next_section_to_analyze: str = Field(default="", description="The exact title or header of the next section to analyze in the next turn.")
    extraction_notes: str = Field(
        default="",
        validation_alias=AliasChoices("extraction_notes", "chain_of_thought_reasoning"),
        serialization_alias="extraction_notes",
    )
    triples: List[Triple] = Field(default_factory=list)
    has_reached_end_of_document: bool = False


class TwoPassExtractionResult(BaseModel):
    success: bool
    skeleton_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    final_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    skeleton_audit: dict[str, Any] = Field(default_factory=dict)
    final_audit: dict[str, Any] = Field(default_factory=dict)
    reflection_audit: dict[str, Any] = Field(default_factory=dict)
    raw_skeleton_response: str | None = None
    raw_final_response: str | None = None
    raw_reflection_response: str | None = None
    skeleton_attempts_used: int = 0
    final_attempts_used: int = 0
    reflection_attempts_used: int = 0
    error: str | None = None

    @property
    def re_extraction(self) -> KnowledgeGraphExtraction:
        return self.final_extraction


class IncrementalExtractionResult(BaseModel):
    success: bool
    extractions: List[KnowledgeGraphExtraction] = Field(default_factory=list)
    audits: List[dict[str, Any]] = Field(default_factory=list)
    raw_responses: List[str] = Field(default_factory=list)
    iterations: int = 0
    error: str | None = None
    failed_iteration: int | None = None


class ChatTwoPassReflectionResult(BaseModel):
    success: bool
    skeleton_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    pass2_channels_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    pass2_revenue_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    pass2_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    pass3_serves_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    pass4_corporate_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    pre_reflection_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    reflection1_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    final_extraction: KnowledgeGraphExtraction = Field(default_factory=KnowledgeGraphExtraction)
    skeleton_audit: dict[str, Any] = Field(default_factory=dict)
    pass2_channels_audit: dict[str, Any] = Field(default_factory=dict)
    pass2_revenue_audit: dict[str, Any] = Field(default_factory=dict)
    pass2_audit: dict[str, Any] = Field(default_factory=dict)
    pass3_serves_audit: dict[str, Any] = Field(default_factory=dict)
    pass4_corporate_audit: dict[str, Any] = Field(default_factory=dict)
    pre_reflection_audit: dict[str, Any] = Field(default_factory=dict)
    reflection1_audit: dict[str, Any] = Field(default_factory=dict)
    final_reflection_audit: dict[str, Any] = Field(default_factory=dict)
    raw_skeleton_response: str | None = None
    raw_pass2_channels_response: str | None = None
    raw_pass2_revenue_response: str | None = None
    raw_pass2_response: str | None = None
    raw_pass3_serves_response: str | None = None
    raw_pass4_corporate_response: str | None = None
    raw_reflection1_response: str | None = None
    raw_final_reflection_response: str | None = None
    skeleton_attempts_used: int = 0
    pass2_channels_attempts_used: int = 0
    pass2_revenue_attempts_used: int = 0
    pass2_attempts_used: int = 0
    pass3_serves_attempts_used: int = 0
    pass4_corporate_attempts_used: int = 0
    reflection1_attempts_used: int = 0
    final_reflection_attempts_used: int = 0
    error: str | None = None


class ExtractionError(RuntimeError):
    pass


TRIPLE_REQUIRED_KEYS = ("subject", "subject_type", "relation", "object", "object_type")
FORMAT_ISSUE_CODES = {"empty_subject", "empty_object"}
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def normalize_lenient_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        return {"triples": payload}
    if isinstance(payload, dict):
        if "triples" in payload:
            return payload
        if all(key in payload for key in TRIPLE_REQUIRED_KEYS):
            return {"triples": [payload]}
        return payload
    return {}


def audit_knowledge_graph_payload(
    payload: Any,
    *,
    payload_parse_recovered: bool = False,
    ontology_version: str = "v1",
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    normalized_payload = normalize_lenient_payload(payload)
    raw_triples = normalized_payload.get("triples", [])
    payload_triples_is_list = isinstance(raw_triples, list)
    if not payload_triples_is_list:
        raw_triples = []

    candidate_triples: list[dict[str, Any]] = []
    non_dict_triple_count = 0
    missing_key_triple_count = 0

    for triple in raw_triples:
        if not isinstance(triple, dict):
            non_dict_triple_count += 1
            continue
        if any(key not in triple for key in TRIPLE_REQUIRED_KEYS):
            missing_key_triple_count += 1
            continue
        candidate_triples.append({key: triple.get(key) for key in TRIPLE_REQUIRED_KEYS})

    validation_report = validate_triples(candidate_triples, dedupe=True, ontology_version=ontology_version)
    invalid_issue_counts: Counter[str] = Counter()
    malformed_from_validation = 0
    ontology_rejected_triple_count = 0

    for invalid in validation_report["invalid_triples"]:
        issue_codes = {issue["code"] for issue in invalid["issues"]}
        invalid_issue_counts.update(issue_codes)
        if issue_codes & FORMAT_ISSUE_CODES:
            malformed_from_validation += 1
        else:
            ontology_rejected_triple_count += 1

    malformed_triple_count = non_dict_triple_count + missing_key_triple_count + malformed_from_validation
    audit = {
        "payload_parse_recovered": payload_parse_recovered,
        "payload_triples_is_list": payload_triples_is_list,
        "raw_triple_count": len(raw_triples),
        "non_dict_triple_count": non_dict_triple_count,
        "missing_key_triple_count": missing_key_triple_count,
        "malformed_triple_count": malformed_triple_count,
        "ontology_rejected_triple_count": ontology_rejected_triple_count,
        "invalid_triple_count": validation_report["summary"]["invalid_triple_count"],
        "duplicate_triple_count": validation_report["summary"]["duplicate_triple_count"],
        "kept_triple_count": len(validation_report["valid_triples"]),
        "invalid_issue_counts": dict(sorted(invalid_issue_counts.items())),
        "ontology_version": ontology_version,
    }
    return validation_report["valid_triples"], audit


def aggregate_extraction_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    aggregated = {
        "raw_triple_count": 0,
        "non_dict_triple_count": 0,
        "missing_key_triple_count": 0,
        "malformed_triple_count": 0,
        "ontology_rejected_triple_count": 0,
        "invalid_triple_count": 0,
        "duplicate_triple_count": 0,
        "kept_triple_count": 0,
        "payload_parse_recovered_count": 0,
        "invalid_issue_counts": {},
    }
    issue_counts: Counter[str] = Counter()

    for audit in audits:
        aggregated["raw_triple_count"] += int(audit.get("raw_triple_count", 0))
        aggregated["non_dict_triple_count"] += int(audit.get("non_dict_triple_count", 0))
        aggregated["missing_key_triple_count"] += int(audit.get("missing_key_triple_count", 0))
        aggregated["malformed_triple_count"] += int(audit.get("malformed_triple_count", 0))
        aggregated["ontology_rejected_triple_count"] += int(audit.get("ontology_rejected_triple_count", 0))
        aggregated["invalid_triple_count"] += int(audit.get("invalid_triple_count", 0))
        aggregated["duplicate_triple_count"] += int(audit.get("duplicate_triple_count", 0))
        aggregated["kept_triple_count"] += int(audit.get("kept_triple_count", 0))
        aggregated["payload_parse_recovered_count"] += 1 if audit.get("payload_parse_recovered") else 0
        issue_counts.update(audit.get("invalid_issue_counts", {}))

    aggregated["invalid_issue_counts"] = dict(sorted(issue_counts.items()))
    return aggregated


def _json_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


PROMPT_ONTOLOGY = f"""=== BUSINESS-MODEL ONTOLOGY ===

NODE TYPES:
- Company: reporting company or named external commercial company
- BusinessSegment: formally named internal segment or line of business
- Offering: specific named product, service, platform, subscription, application, or brand
- CustomerType: canonical label only
- Channel: canonical label only
- Place: explicit named geography
- RevenueModel: canonical label only

CANONICAL CustomerType LABELS:
{_json_list(CANONICAL_CUSTOMER_TYPES)}

CANONICAL Channel LABELS:
{_json_list(CANONICAL_CHANNELS)}

CANONICAL RevenueModel LABELS:
{_json_list(CANONICAL_REVENUE_MODELS)}

ALLOWED RELATIONS:
- HAS_SEGMENT: Company -> BusinessSegment
- OFFERS: Company -> Offering | BusinessSegment -> Offering
- PART_OF: Offering -> BusinessSegment
- SERVES: Company -> CustomerType | Offering -> CustomerType
- OPERATES_IN: Company -> Place | BusinessSegment -> Place
- SELLS_THROUGH: Company -> Channel | Offering -> Channel
- PARTNERS_WITH: Company -> Company
- MONETIZES_VIA: Company -> RevenueModel | BusinessSegment -> RevenueModel | Offering -> RevenueModel

GLOBAL RULES:
- Closed ontology: output only these node types and relations.
- Closed labels: CustomerType, Channel, and RevenueModel must use only canonical labels.
- If a customer, channel, or monetization phrase does not map clearly to one canonical label, omit it.
- Precision and standardization are more important than recall.
- Output only explicit, text-grounded facts.
"""


SKELETON_SYSTEM_PROMPT = f"""You are running PASS 1 of a two-pass business-model extraction workflow for a SEC 10-K filing.

Your job in PASS 1 is to extract the structural graph skeleton only.

{PROMPT_ONTOLOGY}

PASS 1 SCOPE:
- Company
- BusinessSegment
- Offering
- Place
- explicit Company partners when they clearly matter for structure

PASS 1 RELATIONS TO EXTRACT:
- HAS_SEGMENT
- OFFERS
- PART_OF
- OPERATES_IN
- PARTNERS_WITH

PASS 1 RELATIONS TO IGNORE COMPLETELY:
- SERVES
- SELLS_THROUGH
- MONETIZES_VIA

RULES:
- Build the graph in this order:
  1. find all BusinessSegments
  2. find all named Offerings
  3. emit HAS_SEGMENT
  4. emit OFFERS
  5. emit PART_OF
  6. emit explicit OPERATES_IN / PARTNERS_WITH
- For every offering explicitly tied to a segment, emit BOTH:
  - BusinessSegment -> OFFERS -> Offering
  - Offering -> PART_OF -> BusinessSegment
- Ignore generic categories and broad capability phrases.
- Do not emit any CustomerType, Channel, or RevenueModel nodes in this pass.
- Do not guess. Omit uncertain structure.

OUTPUT:
Return ONLY valid JSON matching the KnowledgeGraphExtraction schema.
"""


ENRICHMENT_SYSTEM_PROMPT = f"""You are running PASS 2 of a two-pass business-model extraction workflow for a SEC 10-K filing.

Your job in PASS 2 is to review the existing skeleton and return the COMPLETE final graph.

{PROMPT_ONTOLOGY}

INPUTS:
- <existing_triples>: the PASS 1 skeleton extraction
- <text_to_analyze>: the source filing text

PASS 2 OBJECTIVES:
1. Keep correct skeleton triples.
2. Add any clearly missing skeleton triples if the text explicitly supports them.
3. Add normalized enrichment triples:
   - SERVES
   - SELLS_THROUGH
   - MONETIZES_VIA
4. Remove weak, invalid, or non-canonical triples if needed.

PASS 2 WORK ORDER:
1. Validate and complete the skeleton:
   - HAS_SEGMENT
   - OFFERS
   - PART_OF
2. Search for explicit company-level and offering-level channels.
3. Search for explicit customer groups and map them to exactly one canonical CustomerType label.
4. Search for monetization language and map it to exactly one canonical RevenueModel label.
5. Return the final corrected graph.

HARD RULES:
- If a raw phrase does not map clearly to one canonical CustomerType, Channel, or RevenueModel label, omit it.
- Do not invent new labels.
- Prefer Offering-level SERVES, SELLS_THROUGH, and MONETIZES_VIA when the text supports that specificity.
- Search distribution / go-to-market sections carefully for channel facts.
- Search demand / revenue / product description sections carefully for monetization facts.
- It is better to omit a doubtful enrichment triple than to add a noisy one.

OUTPUT:
Return ONLY valid JSON matching the KnowledgeGraphExtraction schema.
"""


REFLECTION_SYSTEM_PROMPT = f"""You are an independent, expert reviewer reconciling a business-model knowledge graph extracted from a SEC 10-K filing.

=== MANDATORY DATA CONTRACT (CRITICAL) ===
You must output ONLY a valid JSON object.
Every single fact in the graph MUST be serialized as a strict 5-field triple.
If you omit `subject_type` or `object_type`, the pipeline will crash.

REQUIRED JSON STRUCTURE:
{{
  "extraction_notes": "Summarize your reconciliation: what you pruned, added, or consolidated.",
  "triples": [
    {{
      "subject": "Name of subject",
      "subject_type": "EXACT_NODE_TYPE",
      "relation": "EXACT_RELATION",
      "object": "Name of object or canonical label",
      "object_type": "EXACT_NODE_TYPE"
    }}
  ]
}}

=== EXAMPLES OF VALID VS. INVALID TRIPLES ===
CORRECT (All 5 fields present):
{{
  "subject": "Microsoft",
  "subject_type": "Company",
  "relation": "HAS_SEGMENT",
  "object": "Intelligent Cloud",
  "object_type": "BusinessSegment"
}}

FATAL ERROR (Shorthand / missing types - NEVER DO THIS):
{{
  "subject": "Microsoft",
  "relation": "HAS_SEGMENT",
  "object": "Intelligent Cloud"
}}

=== BUSINESS-MODEL ONTOLOGY ===

{PROMPT_ONTOLOGY}

=== YOUR TASK ===

INPUTS PROVIDED IN THE USER MESSAGE:
- <current_triples>: the draft extracted graph from previous steps.
- <text_to_analyze>: the source SEC filing text.

RECONCILIATION RULES:
1. Keep correct triples; remove weak or unsupported triples.
2. Add clearly missing triples explicitly supported by the text.
3. Preserve the structural graph skeleton (HAS_SEGMENT, OFFERS, PART_OF) before enrichment.
4. Enforce Canonical Labels ONLY for CustomerType, Channel, and RevenueModel.
5. Reconcile the graph globally across the full document, including links whose evidence is distributed across different sections.
6. Remove duplicates, generic placeholders, and triples that are less specific than a clearly better-supported alternative.
7. Prefer a coherent, highly accurate final graph over preserving every extraction decision from the draft <current_triples>.

OUTPUT RULES:
- Return the ENTIRE reconciled graph. Do not output deltas or diffs.
- Return ONLY raw JSON starting with `{{` and ending with `}}`.
- Do not use markdown code blocks (e.g. ```json).
- Do not output any conversational text outside the JSON.
"""


INCREMENTAL_SYSTEM_PROMPT = f"""You progressively extract a strict business-model knowledge graph from a large SEC 10-K filing.

{PROMPT_ONTOLOGY}

RULES:
- Analyze one section at a time.
- Build the graph cumulatively across sections.
- Extract only NEW triples from the section currently under review.
- Do not repeat triples already emitted in prior turns.
- Keep section-local claims conservative when supporting context is incomplete; the final review pass will reconcile the full graph.
- Prefer structural triples first, then enrich when explicit in the current section.
- Use canonical labels only for CustomerType, Channel, and RevenueModel.
- If a canonical mapping is unclear, omit it.
- Use the anchored reporting-company name when the text refers to the filer as "the company", "we", or "our".
- When a section contains no relevant new facts, return an empty triple list and still identify the next section.

OUTPUT:
Return ONLY valid JSON matching the IncrementalKnowledgeGraphExtraction schema.
"""


CHAT_TWO_PASS_SYSTEM_PROMPT = """You are executing a multi-step business-model knowledge graph workflow for a SEC 10-K filing.

=== MANDATORY DATA CONTRACT (CRITICAL) ===
You must output ONLY a valid JSON object.
Every single fact you extract MUST be serialized as a strict 5-field triple.
If you omit `subject_type` or `object_type`, the pipeline will crash.

REQUIRED JSON STRUCTURE:
{
  "extraction_notes": "Step-by-step reasoning about what you found and why you mapped it.",
  "triples": [
    {
      "subject": "Name of subject",
      "subject_type": "EXACT_NODE_TYPE",
      "relation": "EXACT_RELATION",
      "object": "Name of object or canonical label",
      "object_type": "EXACT_NODE_TYPE"
    }
  ]
}

=== EXAMPLES OF VALID VS. INVALID TRIPLES ===
CORRECT (All 5 fields present):
{
  "subject": "Microsoft",
  "subject_type": "Company",
  "relation": "HAS_SEGMENT",
  "object": "Intelligent Cloud",
  "object_type": "BusinessSegment"
}

FATAL ERROR (Shorthand / missing types - NEVER DO THIS):
{
  "subject": "Microsoft",
  "relation": "HAS_SEGMENT",
  "object": "Intelligent Cloud"
}

=== BUSINESS-MODEL ONTOLOGY ===

NODE TYPES:
- Company: reporting company or named external commercial company
- BusinessSegment: formally named internal segment or line of business
- Offering: specific named product, service, platform, subscription, application, or brand
- CustomerType: canonical label only
- Channel: canonical label only
- Place: explicit named geography
- RevenueModel: canonical label only

CANONICAL CustomerType LABELS:
["consumers", "small businesses", "mid-market companies", "large enterprises", "developers", "IT professionals", "government agencies", "educational institutions", "healthcare organizations", "financial services firms", "manufacturers", "retailers"]

CANONICAL Channel LABELS:
["direct sales", "online", "retail", "distributors", "resellers", "OEMs", "system integrators", "managed service providers", "marketplaces"]

CANONICAL RevenueModel LABELS:
["subscription", "advertising", "licensing", "consumption-based", "hardware sales", "service fees", "royalties", "transaction fees"]

ALLOWED RELATIONS:
- HAS_SEGMENT: Company -> BusinessSegment
- OFFERS: Company -> Offering | BusinessSegment -> Offering
- PART_OF: Offering -> BusinessSegment
- SERVES: Company -> CustomerType | Offering -> CustomerType
- OPERATES_IN: Company -> Place | BusinessSegment -> Place
- SELLS_THROUGH: Company -> Channel | Offering -> Channel
- PARTNERS_WITH: Company -> Company
- MONETIZES_VIA: Company -> RevenueModel | BusinessSegment -> RevenueModel | Offering -> RevenueModel

GLOBAL RULES:
- Closed ontology: output only these node types and relations.
- Closed labels: CustomerType, Channel, and RevenueModel must use only canonical labels.
- If a customer, channel, or monetization phrase does not map clearly to one canonical label, omit it. No free invention of labels.
- Precision and standardization are more important than recall. Extract ONLY text-grounded facts.

WORKFLOW RULES:
- You will receive one workflow step at a time.
- Preserve continuity across turns inside this chat.
- Follow ONLY the scope requested in the current step.
- Reuse the same filing provided earlier unless a new text block is supplied.
- Return ONLY raw JSON starting with `{` and ending with `}`.
- Do not output markdown code blocks (e.g., no ```json ... ```).
- Do not output explanations outside the JSON payload.
"""


V2_APPROVED_MACRO_REGIONS = [
    "Africa",
    "APAC",
    "Americas",
    "Asia",
    "Asia Pacific",
    "Caribbean",
    "Central America",
    "EMEA",
    "Eastern Europe",
    "Europe",
    "European Union",
    "Latin America",
    "Middle East",
    "North America",
    "South America",
    "Southeast Asia",
    "Western Europe",
]


def _v2_same_chat_system_prompt(full_text: str) -> str:
    return f"""<system_role>
You are an expert information extraction engine for SEC 10-K business-model analysis.
Your task is to build a canonical business-model knowledge graph from the filing.
You must follow the ontology exactly and output only valid JSON when a workflow step asks for output.
</system_role>

<data_contract>
You must output ONLY one valid JSON object.

The required output shape is:
{{
  "extraction_notes": "Brief reasoning summary",
  "triples": [
    {{
      "subject": "Name of subject",
      "subject_type": "EXACT_NODE_TYPE",
      "relation": "EXACT_RELATION",
      "object": "Name of object or canonical label",
      "object_type": "EXACT_NODE_TYPE"
    }}
  ]
}}

Every triple MUST contain exactly these 5 fields:
- subject
- subject_type
- relation
- object
- object_type

If subject_type or object_type is missing, the output is invalid.
The response must start with {{ and end with }}.
</data_contract>

<ontology>
<node_types>
- Company: reporting company or named external commercial company
- BusinessSegment: formally named internal segment or line of business
- Offering: specific named product, service, platform, subscription, application, brand, solution, or explicitly named product family
- CustomerType: canonical label only
- Channel: canonical label only
- Place: normalized business-relevant geography
- RevenueModel: canonical label only
</node_types>

<canonical_labels>
<customer_types>
{_json_list(CANONICAL_CUSTOMER_TYPES)}
</customer_types>
<channels>
{_json_list(CANONICAL_CHANNELS)}
</channels>
<revenue_models>
{_json_list(CANONICAL_REVENUE_MODELS)}
</revenue_models>
</canonical_labels>

<allowed_relations>
- HAS_SEGMENT: Company -> BusinessSegment
- OFFERS: Company -> Offering | BusinessSegment -> Offering | Offering -> Offering
- SERVES: Company -> CustomerType | BusinessSegment -> CustomerType | Offering -> CustomerType
- OPERATES_IN: Company -> Place
- SELLS_THROUGH: Company -> Channel | BusinessSegment -> Channel | Offering -> Channel
- PARTNERS_WITH: Company -> Company
- MONETIZES_VIA: Company -> RevenueModel | BusinessSegment -> RevenueModel | Offering -> RevenueModel
</allowed_relations>
</ontology>

<canonical_graph_policy>
<scope_hierarchy>
- Company = corporate shell
- BusinessSegment = primary semantic anchor
- Offering = inventory leaf, except when the filing explicitly states that one offering is an umbrella or family for another offering
</scope_hierarchy>

<scope_rules>
- Prefer BusinessSegment for SERVES, SELLS_THROUGH, and MONETIZES_VIA when the filing describes the business logic at segment level.
- Use Offering for semantic relations only when the filing explicitly and exclusively isolates the fact to that product.
- Use Company for semantic relations only when the company has no reported segments or the filing explicitly states the fact universally across the company.
- Do not automatically duplicate facts upward across scopes.
- Do not derive company-level facts from lower-level facts during extraction.
- Do not derive offering-level facts from segment-level facts during extraction.
</scope_rules>

<structure_rules>
- BusinessSegment -> OFFERS -> Offering is the primary segment-offering edge.
- Company -> OFFERS -> Offering is allowed only as fallback when no segment anchor exists, or when the filing explicitly presents the offering at company level.
- Offering -> OFFERS -> Offering is allowed only when the filing explicitly states that one offering is a suite, family, umbrella, or parent offering for another offering.
</structure_rules>
</canonical_graph_policy>

<entity_normalization_rules>
<offering_rules>
- When the filing enumerates products, services, platforms, brands, applications, or solutions, extract each explicit named offering separately.
- Do not compress a list of named offerings into a broader umbrella label.
- Do not merge similar but distinct offering names.
- If both a family label and specific named offerings are explicit, keep the specific named offerings as separate Offering nodes.
- Use an umbrella Offering -> OFFERS -> Offering relation only when the filing directly states that hierarchy.
</offering_rules>

<label_rules>
- CustomerType, Channel, and RevenueModel must use only canonical labels.
- If a phrase does not map clearly to a canonical label, omit it.
- Do not invent new labels.
</label_rules>

<place_rules>
- OPERATES_IN is strictly company-level.
- Valid Place objects are countries, U.S. states, District of Columbia, or these approved macro-regions: {_json_list(V2_APPROVED_MACRO_REGIONS)}.
- Normalize aliases such as U.S. -> United States, U.K. -> United Kingdom, asia-pacific -> Asia Pacific, emea -> EMEA.
- Do not use cities, office sites, vague global labels, or market placeholders.
</place_rules>

<partnership_rules>
- PARTNERS_WITH is strictly company-level.
- Use PARTNERS_WITH only when the filing explicitly describes a named partnership.
- Do not use it for suppliers, customers, competitors, ecosystem mentions, or channel relationships.
</partnership_rules>
</entity_normalization_rules>

<inference_policy>
- Precision is more important than recall.
- Extract only text-grounded facts.
- Conservative inference is allowed only for SERVES.
- A SERVES inference is valid only when the product or segment description clearly implies the customer type.
- Do not make weak guesses from vague proximity or broad context.
</inference_policy>

<workflow_rules>
- You will receive one workflow step at a time.
- Follow only the instructions of the current step.
- Preserve continuity within the current chat.
- Later steps MUST treat earlier outputs as fixed context unless the current step explicitly says otherwise.
</workflow_rules>

<source_filing>
{full_text}
</source_filing>

<startup_instruction>
Do not output anything now.
Wait for the first user instruction.
</startup_instruction>"""


def _v2_reflection_system_prompt(full_text: str) -> str:
    return f"""<system_role>
You are an independent expert reviewer reconciling a business-model knowledge graph extracted from an SEC 10-K filing.
Your role is Reviewer, not first extractor.
You must start from the draft graph provided by the user, audit it against the filing and the ontology, and return one final canonical graph.
Do not ignore the draft graph and rebuild blindly from zero.
Think primarily about whether the provided triples are correct, missing, malformed, redundant, or wrongly scoped.
</system_role>

<data_contract>
You must output ONLY one valid JSON object.

The required output shape is:
{{
  "extraction_notes": "Summarize what you pruned, added, or consolidated.",
  "triples": [
    {{
      "subject": "Name of subject",
      "subject_type": "EXACT_NODE_TYPE",
      "relation": "EXACT_RELATION",
      "object": "Name of object or canonical label",
      "object_type": "EXACT_NODE_TYPE"
    }}
  ]
}}

Every triple MUST contain exactly these 5 fields:
- subject
- subject_type
- relation
- object
- object_type

If subject_type or object_type is missing, the output is invalid.
Output ONLY JSON object.
</data_contract>

<review_rules>
- Keep the scope hierarchy clear: Company = corporate shell, BusinessSegment = primary semantic anchor, Offering = inventory leaf unless the filing explicitly states an umbrella offering hierarchy.
- Remove unsupported higher-level duplicates created from lower-level facts.
- Preserve BusinessSegment -> OFFERS -> Offering as the primary extracted structure when a segment anchor exists.
- Company -> OFFERS -> Offering is allowed only as a fallback or explicit company-level fact.
- Offering -> OFFERS -> Offering is allowed only when the filing explicitly states that one offering is a suite, family, umbrella, or parent offering for another offering.
- SERVES, SELLS_THROUGH, and MONETIZES_VIA may attach to Company, BusinessSegment, or Offering, but should default to BusinessSegment when the filing states the business logic at segment level.
- Use Offering for semantic relations only when the filing explicitly and exclusively isolates the fact to that product.
- Use Company for semantic relations only when the company has no reported segments or the filing explicitly states the fact universally across the company.
- For SERVES only, conservative text-grounded inference is allowed when the product or segment description clearly implies a primary customer type.
- Remove SERVES triples that depend on vague adjacency, weak guesswork, or unsupported inheritance from a broader scope.
- Audit explicit product and service enumerations carefully. Add every named offering that is stated in the filing.
- Do not replace a list of named offerings with one summary label such as a generic product family.
- If both a family label and distinct named offerings are explicit, preserve the distinct named offerings as separate Offering nodes.
- Do not merge similar but distinct offering names when both appear explicitly in the filing.
- OPERATES_IN is strictly company-level and limited to countries, U.S. states, District of Columbia, and approved macro-regions: {_json_list(V2_APPROVED_MACRO_REGIONS)}.
- Remove cities, office locations, vague global placeholders, and market descriptors from OPERATES_IN.
- PARTNERS_WITH requires explicit named partnership framing and must remain strictly company-level.
</review_rules>

<ontology>
<node_types>
- Company: reporting company or named external commercial company
- BusinessSegment: formally named internal segment or line of business
- Offering: specific named product, service, platform, subscription, application, brand, solution, or explicitly named product family
- CustomerType: canonical label only
- Channel: canonical label only
- Place: normalized business-relevant geography
- RevenueModel: canonical label only
</node_types>

<canonical_labels>
<customer_types>
{_json_list(CANONICAL_CUSTOMER_TYPES)}
</customer_types>
<channels>
{_json_list(CANONICAL_CHANNELS)}
</channels>
<revenue_models>
{_json_list(CANONICAL_REVENUE_MODELS)}
</revenue_models>
</canonical_labels>

<allowed_relations>
- HAS_SEGMENT: Company -> BusinessSegment
- OFFERS: Company -> Offering | BusinessSegment -> Offering | Offering -> Offering
- SERVES: Company -> CustomerType | BusinessSegment -> CustomerType | Offering -> CustomerType
- OPERATES_IN: Company -> Place
- SELLS_THROUGH: Company -> Channel | BusinessSegment -> Channel | Offering -> Channel
- PARTNERS_WITH: Company -> Company
- MONETIZES_VIA: Company -> RevenueModel | BusinessSegment -> RevenueModel | Offering -> RevenueModel
</allowed_relations>
</ontology>

<source_filing>
{full_text}
</source_filing>

<startup_instruction>
Do not output anything now.
Wait for the user instruction.
</startup_instruction>"""


def _v2_segment_serves_same_chat_system_prompt(full_text: str) -> str:
    return f"""<system_role>
You are an expert information extraction engine for SEC 10-K business-model analysis.
Your task is to build a canonical business-model knowledge graph from the filing.
You must follow the ontology exactly and output only valid JSON when a workflow step asks for output.
</system_role>

<data_contract>
You must output ONLY one valid JSON object.

Required shape:
{{
  "extraction_notes": "Brief reasoning summary",
  "triples": [
    {{
      "subject": "Name of subject",
      "subject_type": "EXACT_NODE_TYPE",
      "relation": "EXACT_RELATION",
      "object": "Name of object or canonical label",
      "object_type": "EXACT_NODE_TYPE"
    }}
  ]
}}

Every triple must contain exactly:
- subject
- subject_type
- relation
- object
- object_type

If subject_type or object_type is missing, the output is invalid.
The response must start with {{ and end with }}.
</data_contract>

<ontology>
<node_types>
- Company: reporting company or named external commercial company
- BusinessSegment: formally named internal segment or line of business
- Offering: specific named product, service, platform, subscription, application, brand, solution, or explicitly named product family
- CustomerType: canonical label only
- Channel: canonical label only
- Place: normalized business-relevant geography
- RevenueModel: canonical label only
</node_types>

<canonical_labels>
<customer_types>
{_json_list(CANONICAL_CUSTOMER_TYPES)}
</customer_types>
<channels>
{_json_list(CANONICAL_CHANNELS)}
</channels>
<revenue_models>
{_json_list(CANONICAL_REVENUE_MODELS)}
</revenue_models>
</canonical_labels>

<allowed_relations>
- HAS_SEGMENT: Company -> BusinessSegment
- OFFERS: Company -> Offering | BusinessSegment -> Offering | Offering -> Offering
- SERVES: BusinessSegment -> CustomerType
- OPERATES_IN: Company -> Place
- SELLS_THROUGH: BusinessSegment -> Channel | Offering -> Channel
- PARTNERS_WITH: Company -> Company
- MONETIZES_VIA: Offering -> RevenueModel
</allowed_relations>
</ontology>

<canonical_graph_policy>
<scope_hierarchy>
- Company = corporate shell
- BusinessSegment = primary semantic anchor
- Offering = inventory leaf, except when the filing explicitly states that one offering is an umbrella or family for another offering
</scope_hierarchy>

<scope_rules>
- SELLS_THROUGH should default to BusinessSegment.
- SELLS_THROUGH may attach to Offering only when that offering has no BusinessSegment anchor.
- If an offering family hierarchy exists, attach MONETIZES_VIA to the family parent rather than to its child offerings.
- Do not automatically duplicate facts upward across scopes.
- Do not derive company-level facts from lower-level facts during extraction.
- Do not derive offering-level facts from segment-level facts during extraction.
</scope_rules>

<structure_rules>
- BusinessSegment -> OFFERS -> Offering is the primary segment-offering edge.
- Company -> OFFERS -> Offering is allowed only as fallback when no segment anchor exists, or when the filing explicitly presents the offering at company level.
- Offering -> OFFERS -> Offering is allowed only when the filing explicitly states that one offering is a suite, family, umbrella, or parent offering for another offering.
- A child Offering may have at most one Offering parent in Offering -> OFFERS -> Offering hierarchy.
</structure_rules>
</canonical_graph_policy>

<normalization_rules>
- CustomerType, Channel, and RevenueModel must use only canonical labels.
- If a phrase does not map clearly to a canonical label, omit it.
- Extract explicit named offerings individually and as written.
- Do not compress explicit offering lists into summary labels.
- Do not merge similar but distinct offering names.
</normalization_rules>

<corporate_shell_rules>
- Valid Place objects are countries, U.S. states, District of Columbia, and these macro-regions: {_json_list(V2_APPROVED_MACRO_REGIONS)}.
- Normalize aliases such as U.S. -> United States, U.K. -> United Kingdom, asia-pacific -> Asia Pacific, emea -> EMEA.
- Do not use cities, office sites, vague global labels, or market placeholders.
- Use PARTNERS_WITH only when the filing explicitly describes a named partnership.
- Do not use it for suppliers, customers, competitors, ecosystem mentions, or channel relationships.
</corporate_shell_rules>

<inference_policy>
- Precision is more important than recall.
- Extract only text-grounded facts.
- Conservative inference is allowed only for SERVES.
- SERVES inference must be segment-specific, not global.
- Do not spread one customer type across multiple segments unless each segment has its own supporting evidence.
- Do not make weak guesses from vague proximity or broad context.
</inference_policy>

<workflow_rules>
- You will receive one workflow step at a time.
- Follow only the instructions of the current step.
- Preserve continuity within the current chat.
- Later steps MUST treat earlier outputs as fixed context unless the current step explicitly says otherwise.
</workflow_rules>

<source_filing>
{full_text}
</source_filing>

<startup_instruction>
Do not output anything now.
Wait for the first user instruction.
</startup_instruction>"""


def _v2_segment_serves_reflection_system_prompt(full_text: str) -> str:
    return f"""<system_role>
You are an independent expert reviewer reconciling a business-model knowledge graph extracted from an SEC 10-K filing.
Your role is Reviewer, not first extractor.
You must start from the draft graph provided by the user, audit it against the filing and the ontology, and return one final canonical graph.
Do not ignore the draft graph and rebuild blindly from zero.
Think primarily about whether the provided triples are correct, missing, malformed, redundant, or wrongly scoped.
</system_role>

<data_contract>
You must output ONLY one valid JSON object.

Required shape:
{{
  "extraction_notes": "Summarize what you pruned, added, or consolidated.",
  "triples": [
    {{
      "subject": "Name of subject",
      "subject_type": "EXACT_NODE_TYPE",
      "relation": "EXACT_RELATION",
      "object": "Name of object or canonical label",
      "object_type": "EXACT_NODE_TYPE"
    }}
  ]
}}

Every triple must contain exactly:
- subject
- subject_type
- relation
- object
- object_type

If subject_type or object_type is missing, the output is invalid.
Output ONLY JSON object.
</data_contract>

<review_policy>
- Correct only if facts are text-grounded in the filing and consistent with the ontology.
- Preserve BusinessSegment -> OFFERS -> Offering as the primary extracted structure when a segment anchor exists.
- Preserve explicit offering-family hierarchy only when the filing states it clearly.
- For offering structure specifically, change an offering's parent only when the current parent-child assignment is clearly wrong and not supported by the filing text.
- A child Offering may have at most one Offering parent.
- If an offering family hierarchy exists, keep MONETIZES_VIA on the family parent rather than on its child offerings.
- Add missing named offerings when the filing states them.
- Do not merge similar but distinct offering names.
- Do not fan out a rare or specialized customer type across multiple segments unless each segment has its own support in the filing.
- Remove unsupported duplicates, malformed scope choices, and very weakly supported semantic edges.
</review_policy>

<ontology>
<node_types>
- Company: reporting company or named external commercial company
- BusinessSegment: formally named internal segment or line of business
- Offering: specific named product, service, platform, subscription, application, brand, solution, or explicitly named product family
- CustomerType: canonical label only
- Channel: canonical label only
- Place: normalized business-relevant geography
- RevenueModel: canonical label only
</node_types>

<canonical_labels>
<customer_types>
{_json_list(CANONICAL_CUSTOMER_TYPES)}
</customer_types>
<channels>
{_json_list(CANONICAL_CHANNELS)}
</channels>
<revenue_models>
{_json_list(CANONICAL_REVENUE_MODELS)}
</revenue_models>
</canonical_labels>

<allowed_relations>
- HAS_SEGMENT: Company -> BusinessSegment
- OFFERS: Company -> Offering | BusinessSegment -> Offering | Offering -> Offering
- SERVES: BusinessSegment -> CustomerType
- OPERATES_IN: Company -> Place
- SELLS_THROUGH: BusinessSegment -> Channel | Offering -> Channel
- PARTNERS_WITH: Company -> Company
- MONETIZES_VIA: Offering -> RevenueModel
</allowed_relations>
</ontology>

<corporate_shell_rules>
- Valid Place objects are countries, U.S. states, District of Columbia, and these macro-regions: {_json_list(V2_APPROVED_MACRO_REGIONS)}.
- Normalize aliases such as U.S. -> United States, U.K. -> United Kingdom, asia-pacific -> Asia Pacific, emea -> EMEA.
- Do not use cities, office sites, vague global labels, or market placeholders.
- Use PARTNERS_WITH only when the filing explicitly describes a named partnership.
- Do not use it for suppliers, customers, competitors, ecosystem mentions, or channel relationships.
</corporate_shell_rules>

<source_filing>
{full_text}
</source_filing>

<startup_instruction>
Do not output anything now.
Wait for the user instruction.
</startup_instruction>"""


class LLMExtractor:
    def __init__(self, base_url: str = "http://localhost:1234/v1", api_key: str = "lm-studio", model: str = "local-model"):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    @staticmethod
    def _schema_def(name: str, model_schema: type[BaseModel], *, ontology_version: str = "v1") -> dict:
        schema = model_schema.model_json_schema(by_alias=True)
        if ontology_version in {"v2", "v2_segment_serves"}:
            relation_enum = (
                schema.get("$defs", {})
                .get("Triple", {})
                .get("properties", {})
                .get("relation", {})
                .get("enum")
            )
            if isinstance(relation_enum, list):
                schema["$defs"]["Triple"]["properties"]["relation"]["enum"] = [
                    value for value in relation_enum if value != "PART_OF"
                ]
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": schema,
            },
        }

    @staticmethod
    def _compact_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _merge_serves_into_base(
        base_extraction: KnowledgeGraphExtraction,
        serves_extraction: KnowledgeGraphExtraction,
    ) -> KnowledgeGraphExtraction:
        return LLMExtractor._merge_relation_subset_into_base(
            base_extraction,
            serves_extraction,
            allowed_relations={"SERVES"},
        )

    @staticmethod
    def _merge_relation_subset_into_base(
        base_extraction: KnowledgeGraphExtraction,
        subset_extraction: KnowledgeGraphExtraction,
        *,
        allowed_relations: set[str],
    ) -> KnowledgeGraphExtraction:
        merged_triples: list[Triple] = []
        seen: set[tuple[str, str, str, str, str]] = set()

        for triple in base_extraction.triples:
            if triple.relation in allowed_relations:
                continue
            key = (triple.subject, triple.subject_type, triple.relation, triple.object, triple.object_type)
            if key in seen:
                continue
            seen.add(key)
            merged_triples.append(triple)

        for triple in subset_extraction.triples:
            if triple.relation not in allowed_relations:
                continue
            key = (triple.subject, triple.subject_type, triple.relation, triple.object, triple.object_type)
            if key in seen:
                continue
            seen.add(key)
            merged_triples.append(triple)

        return KnowledgeGraphExtraction(
            extraction_notes=subset_extraction.extraction_notes,
            triples=merged_triples,
        )

    @staticmethod
    def _load_json_payload(content: str, fallback_payload: str) -> tuple[dict, bool]:
        content = content.strip()
        if not content:
            raise ExtractionError("Empty response from model.")

        candidates = [content]
        match = JSON_OBJECT_RE.search(content)
        if match:
            json_object_text = match.group(0)
            if json_object_text != content:
                candidates.append(json_object_text)

        if not content.endswith(("}", "]")):
            logger.warning("Truncated JSON detected. Attempting to salvage...")
            last_object_end = content.rfind("}")
            last_array_end = content.rfind("]")
            last_json_end = max(last_object_end, last_array_end)
            truncated_candidate = content[: last_json_end + 1] if last_json_end != -1 else fallback_payload
            if truncated_candidate not in candidates:
                candidates.append(truncated_candidate)

        for index, candidate in enumerate(candidates):
            try:
                return json.loads(candidate), index > 0
            except json.JSONDecodeError:
                continue

        return json.loads(fallback_payload), True

    @staticmethod
    def _lenient_model_from_payload(
        schema_model: type[BaseModel],
        payload: Any,
        *,
        ontology_version: str = "v1",
    ) -> tuple[BaseModel, dict[str, Any]]:
        normalized_payload = normalize_lenient_payload(payload)
        valid_triples, audit = audit_knowledge_graph_payload(normalized_payload, ontology_version=ontology_version)
        extraction_notes = str(
            normalized_payload.get("extraction_notes", normalized_payload.get("chain_of_thought_reasoning", "")) or ""
        )
        triple_objects = [Triple(**triple) for triple in valid_triples]
        raw_end_flag = normalized_payload.get("has_reached_end_of_document", False)
        if isinstance(raw_end_flag, str):
            end_of_document = raw_end_flag.strip().casefold() in {"true", "1", "yes"}
        else:
            end_of_document = bool(raw_end_flag)

        if schema_model is KnowledgeGraphExtraction:
            model = KnowledgeGraphExtraction(
                extraction_notes=extraction_notes,
                triples=triple_objects,
            )
            return model, audit

        if schema_model is IncrementalKnowledgeGraphExtraction:
            model = IncrementalKnowledgeGraphExtraction(
                current_section_analyzed=str(normalized_payload.get("current_section_analyzed", "") or ""),
                next_section_to_analyze=str(normalized_payload.get("next_section_to_analyze", "") or ""),
                extraction_notes=extraction_notes,
                triples=triple_objects,
                has_reached_end_of_document=end_of_document,
            )
            return model, audit

        raise TypeError(f"Unsupported schema model for lenient payload parsing: {schema_model!r}")

    def _call_structured_messages(
        self,
        *,
        messages: list[dict[str, str]],
        schema_name: str,
        schema_model: type[BaseModel],
        fallback_payload: str,
        max_retries: int,
        temperature: float = 0.0,
        use_schema: bool = True,
        ontology_version: str = "v1",
    ) -> tuple[BaseModel, str | None, int, dict[str, Any]]:
        schema_def = self._schema_def(schema_name, schema_model, ontology_version=ontology_version)

        for attempt in range(1, max_retries + 1):
            try:
                call_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if use_schema:
                    call_kwargs["response_format"] = schema_def
                response = self.client.chat.completions.create(**call_kwargs)
                content = response.choices[0].message.content
                parsed_payload, payload_parse_recovered = self._load_json_payload(content or "", fallback_payload)
                if use_schema:
                    parsed_model = schema_model(**parsed_payload)
                    _, audit = audit_knowledge_graph_payload(
                        parsed_payload,
                        payload_parse_recovered=payload_parse_recovered,
                        ontology_version=ontology_version,
                    )
                else:
                    parsed_model, audit = self._lenient_model_from_payload(
                        schema_model,
                        parsed_payload,
                        ontology_version=ontology_version,
                    )
                    audit["payload_parse_recovered"] = payload_parse_recovered
                return parsed_model, content, attempt, audit
            except (json.JSONDecodeError, ValidationError, ExtractionError) as exc:
                logger.warning("Structured call failed on attempt %s/%s: %s", attempt, max_retries, exc)
            except Exception as exc:
                logger.warning("LLM API error on attempt %s/%s: %s", attempt, max_retries, exc)

        raise ExtractionError(f"Failed after {max_retries} attempts")

    def _call_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema_model: type[BaseModel],
        fallback_payload: str,
        max_retries: int,
        temperature: float = 0.0,
        use_schema: bool = True,
        ontology_version: str = "v1",
    ) -> tuple[BaseModel, str | None, int, dict[str, Any]]:
        return self._call_structured_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            schema_name=schema_name,
            schema_model=schema_model,
            fallback_payload=fallback_payload,
            max_retries=max_retries,
            temperature=temperature,
            use_schema=use_schema,
            ontology_version=ontology_version,
        )

    def extract_two_pass_detailed(
        self,
        full_text: str,
        company_name: str | None = None,
        max_retries: int = 2,
        strict: bool = True,
        use_schema: bool = True,
    ) -> TwoPassExtractionResult:
        prompt_parts: list[str] = []
        if company_name:
            prompt_parts.append(f"<company_name>\n{company_name}\n</company_name>")
        prompt_parts.append(f"<text_to_analyze>\n{full_text}\n</text_to_analyze>")
        base_prompt = "\n\n".join(prompt_parts)

        try:
            skeleton_extraction, raw_skeleton_response, skeleton_attempts_used, skeleton_audit = self._call_structured(
                system_prompt=SKELETON_SYSTEM_PROMPT,
                user_prompt=base_prompt,
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Truncated skeleton extraction.","triples":[]}',
                max_retries=max_retries,
                use_schema=use_schema,
            )
        except ExtractionError as exc:
            if strict:
                raise
            return TwoPassExtractionResult(
                success=False,
                skeleton_audit={},
                error=str(exc),
            )

        enrichment_prompt = (
            f"<company_name>\n{company_name or ''}\n</company_name>\n\n"
            f"<existing_triples>\n{self._compact_json(skeleton_extraction.model_dump())}\n</existing_triples>\n\n"
            f"<text_to_analyze>\n{full_text}\n</text_to_analyze>"
        )

        try:
            final_extraction, raw_final_response, final_attempts_used, final_audit = self._call_structured(
                system_prompt=ENRICHMENT_SYSTEM_PROMPT,
                user_prompt=enrichment_prompt,
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Truncated final extraction.","triples":[]}',
                max_retries=max_retries,
                use_schema=use_schema,
            )
        except ExtractionError as exc:
            if strict:
                raise
            return TwoPassExtractionResult(
                success=False,
                skeleton_extraction=skeleton_extraction,
                skeleton_audit=skeleton_audit,
                raw_skeleton_response=raw_skeleton_response,
                skeleton_attempts_used=skeleton_attempts_used,
                error=str(exc),
            )

        if not final_extraction.triples and skeleton_extraction.triples:
            logger.warning("Pass 2 returned no triples. Falling back to skeleton extraction.")
            final_extraction = skeleton_extraction

        return TwoPassExtractionResult(
            success=True,
            skeleton_extraction=skeleton_extraction,
            final_extraction=final_extraction,
            skeleton_audit=skeleton_audit,
            final_audit=final_audit,
            raw_skeleton_response=raw_skeleton_response,
            raw_final_response=raw_final_response,
            skeleton_attempts_used=skeleton_attempts_used,
            final_attempts_used=final_attempts_used,
        )

    def extract_with_reflection(
        self,
        full_text: str,
        company_name: str | None = None,
        max_retries: int = 2,
        use_schema: bool = True,
    ) -> tuple[TwoPassExtractionResult, KnowledgeGraphExtraction]:
        two_pass_result = self.extract_two_pass_detailed(
            full_text=full_text,
            company_name=company_name,
            max_retries=max_retries,
            strict=True,
            use_schema=use_schema,
        )
        final_extraction, raw_reflection_response, reflection_attempts_used, reflection_audit = self.reflect_extraction(
            full_text=full_text,
            current_extraction=two_pass_result.final_extraction,
            company_name=company_name,
            max_retries=max_retries,
            strict=False,
            use_schema=use_schema,
        )
        two_pass_result.raw_reflection_response = raw_reflection_response
        two_pass_result.reflection_attempts_used = reflection_attempts_used
        two_pass_result.reflection_audit = reflection_audit
        return two_pass_result, final_extraction

    def reflect_extraction(
        self,
        *,
        full_text: str,
        current_extraction: KnowledgeGraphExtraction,
        company_name: str | None = None,
        max_retries: int = 2,
        strict: bool = True,
        use_schema: bool = True,
        system_prompt: str = REFLECTION_SYSTEM_PROMPT,
        user_prompt: str | None = None,
        ontology_version: str = "v1",
    ) -> tuple[KnowledgeGraphExtraction, str | None, int, dict[str, Any]]:
        reflection_prompt = user_prompt or (
            "WORKFLOW STEP: REFLECTION 2 - FINAL RECONCILIATION (INDEPENDENT REVIEW)\n\n"
            f"<company_name>\n{company_name or ''}\n</company_name>\n\n"
            f"<current_triples>\n{self._compact_json(current_extraction.model_dump())}\n</current_triples>\n\n"
            f"<text_to_analyze>\n{full_text}\n</text_to_analyze>\n\n"
            "=== FINAL INSTRUCTIONS ===\n"
            "Review the draft <current_triples> against the original <text_to_analyze> for <company_name>.\n\n"
            "Remember your critical constraints:\n"
            "1. Reconcile, prune, and enrich to create the most accurate final graph.\n"
            "2. Ensure ALL triples have EXACTLY 5 fields (`subject`, `subject_type`, `relation`, `object`, `object_type`). Do NOT revert to 3-field shorthand.\n"
            "3. Output the ENTIRE updated graph, not just deltas.\n\n"
            'Return ONLY a raw JSON object containing "extraction_notes" and the "triples" array. Do not use markdown code blocks (```json).'
        )

        try:
            final_extraction, raw_response, attempts_used, audit = self._call_structured(
                system_prompt=system_prompt,
                user_prompt=reflection_prompt,
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Reflection failed.","triples":[]}',
                max_retries=max_retries,
                use_schema=use_schema,
                ontology_version=ontology_version,
            )
            if not final_extraction.triples and current_extraction.triples:
                logger.warning("Reflection returned no triples. Falling back to pre-reflection extraction.")
                return current_extraction, raw_response, attempts_used, audit
            return final_extraction, raw_response, attempts_used, audit
        except ExtractionError:
            if strict:
                raise
            logger.warning("Reflection failed. Falling back to pre-reflection extraction.")
            return current_extraction, None, max_retries, {}

    def extract_incremental_detailed(
        self,
        full_text: str,
        company_name: str | None = None,
        max_iterations: int = 10,
        max_retries: int = 2,
        strict: bool = True,
        use_schema: bool = True,
    ) -> IncrementalExtractionResult:
        all_extractions: List[KnowledgeGraphExtraction] = []
        audits: List[dict[str, Any]] = []
        raw_responses: List[str] = []
        messages = [{"role": "system", "content": INCREMENTAL_SYSTEM_PROMPT}]
        fallback_payload = (
            '{"extraction_notes":"Truncated before extractions.","triples":[],"has_reached_end_of_document":false}'
        )

        for iteration in range(max_iterations):
            if iteration == 0:
                prompt = (
                    "Here is the complete document. For this first response, extract triples only from the first section.\n\n"
                    f"<company_name>\n{company_name or ''}\n</company_name>\n\n"
                    f"<text_to_analyze>\n{full_text}\n</text_to_analyze>"
                )
            else:
                prompt = (
                    "Continue from the next section named in your previous response. "
                    "Extract only new triples from that section. "
                    "Do not repeat earlier triples."
                )

            messages.append({"role": "user", "content": prompt})

            try:
                raw_extraction, content, _, audit = self._call_structured_messages(
                    messages=messages,
                    schema_name="IncrementalKnowledgeGraphExtraction",
                    schema_model=IncrementalKnowledgeGraphExtraction,
                    fallback_payload=fallback_payload,
                    max_retries=max_retries,
                    temperature=0.0,
                    use_schema=use_schema,
                )
            except ExtractionError:
                error_message = f"Failed iteration {iteration + 1} after {max_retries} attempts"
                if strict:
                    raise ExtractionError(error_message)
                return IncrementalExtractionResult(
                    success=False,
                    extractions=all_extractions,
                    audits=audits,
                    raw_responses=raw_responses,
                    iterations=iteration + 1,
                    error=error_message,
                    failed_iteration=iteration + 1,
                )

            raw_responses.append(content or "")
            audits.append(audit)
            messages.append(
                {
                    "role": "assistant",
                    "content": content or json.dumps(raw_extraction.model_dump(mode="json"), ensure_ascii=False),
                }
            )
            all_extractions.append(
                KnowledgeGraphExtraction(
                    extraction_notes=raw_extraction.extraction_notes,
                    triples=raw_extraction.triples,
                )
            )
            if raw_extraction.has_reached_end_of_document:
                return IncrementalExtractionResult(
                    success=True,
                    extractions=all_extractions,
                    audits=audits,
                    raw_responses=raw_responses,
                    iterations=iteration + 1,
                )

        error_message = f"Hit max iterations ({max_iterations}) without reaching end of document"
        if strict:
            raise ExtractionError(error_message)
        return IncrementalExtractionResult(
            success=False,
            extractions=all_extractions,
            audits=audits,
            raw_responses=raw_responses,
            iterations=max_iterations,
            error=error_message,
            failed_iteration=max_iterations,
        )

    def extract_incremental_with_reflection(
        self,
        *,
        full_text: str,
        company_name: str | None = None,
        max_iterations: int = 10,
        max_retries: int = 2,
        use_schema: bool = True,
    ) -> tuple[IncrementalExtractionResult, KnowledgeGraphExtraction, str | None, int, dict[str, Any]]:
        incremental_result = self.extract_incremental_detailed(
            full_text=full_text,
            company_name=company_name,
            max_iterations=max_iterations,
            max_retries=max_retries,
            strict=True,
            use_schema=use_schema,
        )
        merged_incremental_extraction = KnowledgeGraphExtraction(
            extraction_notes="Merged incremental section extractions before final reflection.",
            triples=[triple for extraction in incremental_result.extractions for triple in extraction.triples],
        )
        final_extraction, raw_response, attempts_used, audit = self.reflect_extraction(
            full_text=full_text,
            current_extraction=merged_incremental_extraction,
            company_name=company_name,
            max_retries=max_retries,
            strict=False,
            use_schema=use_schema,
        )
        return incremental_result, final_extraction, raw_response, attempts_used, audit

    def extract_chat_two_pass_reflection(
        self,
        *,
        full_text: str,
        company_name: str | None = None,
        max_retries: int = 2,
        use_schema: bool = True,
    ) -> ChatTwoPassReflectionResult:
        messages = [{"role": "system", "content": CHAT_TWO_PASS_SYSTEM_PROMPT}]

        pass1_prompt = (
            "WORKFLOW STEP: PASS 1 - STRUCTURAL SKELETON\n\n"
            "OBJECTIVE:\n"
            "Extract the core business structure of the company.\n"
            "Focus ONLY on the following relations:\n"
            "- HAS_SEGMENT\n"
            "- OFFERS\n"
            "- PART_OF\n"
            "- OPERATES_IN\n"
            "- PARTNERS_WITH\n"
            "Ignore SERVES, SELLS_THROUGH, and MONETIZES_VIA in this step.\n\n"
            "RULES FOR THIS PASS:\n"
            "1. Anchor all company-level facts exactly to the <company_name> provided below.\n"
            "2. In the `extraction_notes` string, briefly step-by-step list the segments, offerings, geographies, and partners you found in the text.\n"
            "3. CRITICAL FORMATTING: You must output the full 5-field triple for every fact. Do not drop `subject_type` or `object_type`. Shorthand 3-field triples will crash the system.\n\n"
            f"<company_name>\n{company_name or ''}\n</company_name>\n\n"
            f"<text_to_analyze>\n{full_text}\n</text_to_analyze>\n\n"
            'Remember: Return ONLY a raw JSON object containing "extraction_notes" and the "triples" array. '
            "Do not use markdown code blocks (```json)."
        )
        messages.append({"role": "user", "content": pass1_prompt})
        try:
            skeleton_extraction, raw_skeleton_response, skeleton_attempts_used, skeleton_audit = self._call_structured_messages(
                messages=messages,
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Truncated skeleton extraction.","triples":[]}',
                max_retries=max_retries,
                use_schema=use_schema,
            )
        except ExtractionError as exc:
            return ChatTwoPassReflectionResult(success=False, error=str(exc))
        messages.append(
            {
                "role": "assistant",
                "content": raw_skeleton_response or json.dumps(skeleton_extraction.model_dump(mode="json"), ensure_ascii=False),
            }
        )

        pass2_prompt = (
            "WORKFLOW STEP: PASS 2 - ENRICHMENT AND FULL GRAPH\n\n"
            "OBJECTIVE:\n"
            "Using the <text_to_analyze> provided in PASS 1 and your previous extraction, return the COMPLETE, enriched business-model graph.\n\n"
            "TASKS FOR THIS PASS:\n"
            "1. KEEP & FIX SKELETON: Retain correct HAS_SEGMENT, OFFERS, PART_OF, OPERATES_IN, and PARTNERS_WITH triples from Pass 1. Add any that you missed.\n"
            "2. ADD ENRICHMENT: Extract SERVES, SELLS_THROUGH, and MONETIZES_VIA relations.\n"
            "   - ONLY add these if explicitly stated in the text. DO NOT guess.\n"
            "   - For these three relations, the `object` MUST be one of the EXACT CANONICAL LABELS defined in the system prompt.\n\n"
            "CRITICAL RULES FOR PASS 2:\n"
            '- NO DELTAS: You must output the entire combined graph (Pass 1 skeleton + Pass 2 enrichments). Do not just output the new additions.\n'
            '- ZERO FORMAT DEGRADATION: Every single triple in the "triples" array MUST have all 5 fields (`subject`, `subject_type`, `relation`, `object`, `object_type`). Do not revert to 3-field shorthand.\n'
            '- CHAIN OF THOUGHT: In "extraction_notes", briefly explain which exact sentences justify your mapping to the canonical CustomerType, Channel, and RevenueModel labels.\n\n'
            'Remember: Return ONLY a raw JSON object containing "extraction_notes" and the "triples" array. Do not use markdown code blocks (```json).'
        )
        messages.append({"role": "user", "content": pass2_prompt})
        try:
            pass2_extraction, raw_pass2_response, pass2_attempts_used, pass2_audit = self._call_structured_messages(
                messages=messages,
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Truncated pass-2 extraction.","triples":[]}',
                max_retries=max_retries,
                use_schema=use_schema,
            )
        except ExtractionError as exc:
            return ChatTwoPassReflectionResult(
                success=False,
                skeleton_extraction=skeleton_extraction,
                skeleton_audit=skeleton_audit,
                raw_skeleton_response=raw_skeleton_response,
                skeleton_attempts_used=skeleton_attempts_used,
                error=str(exc),
            )
        if not pass2_extraction.triples and skeleton_extraction.triples:
            logger.warning("Chat PASS 2 returned no triples. Falling back to PASS 1 skeleton.")
            pass2_extraction = skeleton_extraction
        messages.append(
            {
                "role": "assistant",
                "content": raw_pass2_response or json.dumps(pass2_extraction.model_dump(mode="json"), ensure_ascii=False),
            }
        )

        reflection1_prompt = (
            "WORKFLOW STEP: REFLECTION 1 - CRITIQUE AND REPAIR\n\n"
            "OBJECTIVE:\n"
            "Act as a critical auditor. Review your COMPLETE graph from PASS 2 against the original <text_to_analyze> provided in PASS 1.\n\n"
            "TASKS FOR THIS PASS:\n"
            "1. PRUNE WEAK TRIPLES: Remove any triples that are not explicitly supported by the text.\n"
            "2. FIX ONTOLOGY & LABELS: Ensure all CustomerType, Channel, and RevenueModel objects use ONLY the exact canonical labels from the system prompt. Remove or correct invalid labels.\n"
            "3. ADD MISSING FACTS: Add any obvious facts from the text that were missed in earlier passes.\n"
            "4. PRESERVE SKELETON: Ensure the core structure (HAS_SEGMENT, OFFERS, PART_OF) remains coherent.\n\n"
            "CRITICAL RULES FOR REFLECTION 1:\n"
            '- AUDIT TRAIL: In the "extraction_notes" string, document your corrections. Explicitly state: "Removed X because..." or "Added Y because..." or "Corrected label Z to...".\n'
            "- NO DELTAS: You must output the ENTIRE updated graph. Do not output just the changes.\n"
            '- STRICT 5-FIELD FORMAT: Every single triple must retain all 5 fields (`subject`, `subject_type`, `relation`, `object`, `object_type`). Do not truncate to 3 fields under any circumstances.\n\n'
            'Remember: Return ONLY a raw JSON object containing "extraction_notes" and the "triples" array. Do not use markdown code blocks (```json).'
        )
        messages.append({"role": "user", "content": reflection1_prompt})
        try:
            reflection1_extraction, raw_reflection1_response, reflection1_attempts_used, reflection1_audit = self._call_structured_messages(
                messages=messages,
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Truncated reflection extraction.","triples":[]}',
                max_retries=max_retries,
                use_schema=use_schema,
            )
        except ExtractionError as exc:
            return ChatTwoPassReflectionResult(
                success=False,
                skeleton_extraction=skeleton_extraction,
                pass2_extraction=pass2_extraction,
                skeleton_audit=skeleton_audit,
                pass2_audit=pass2_audit,
                raw_skeleton_response=raw_skeleton_response,
                raw_pass2_response=raw_pass2_response,
                skeleton_attempts_used=skeleton_attempts_used,
                pass2_attempts_used=pass2_attempts_used,
                error=str(exc),
            )
        if not reflection1_extraction.triples and pass2_extraction.triples:
            logger.warning("Chat reflection 1 returned no triples. Falling back to PASS 2 extraction.")
            reflection1_extraction = pass2_extraction

        final_extraction, raw_final_reflection_response, final_reflection_attempts_used, final_reflection_audit = self.reflect_extraction(
            full_text=full_text,
            current_extraction=reflection1_extraction,
            company_name=company_name,
            max_retries=max_retries,
            strict=False,
            use_schema=use_schema,
        )

        return ChatTwoPassReflectionResult(
            success=True,
            skeleton_extraction=skeleton_extraction,
            pass2_extraction=pass2_extraction,
            reflection1_extraction=reflection1_extraction,
            final_extraction=final_extraction,
            skeleton_audit=skeleton_audit,
            pass2_audit=pass2_audit,
            reflection1_audit=reflection1_audit,
            final_reflection_audit=final_reflection_audit,
            raw_skeleton_response=raw_skeleton_response,
            raw_pass2_response=raw_pass2_response,
            raw_reflection1_response=raw_reflection1_response,
            raw_final_reflection_response=raw_final_reflection_response,
            skeleton_attempts_used=skeleton_attempts_used,
            pass2_attempts_used=pass2_attempts_used,
            reflection1_attempts_used=reflection1_attempts_used,
            final_reflection_attempts_used=final_reflection_attempts_used,
        )

    def extract_chat_two_pass_reflection_v2(
        self,
        *,
        full_text: str,
        company_name: str | None = None,
        max_retries: int = 2,
        use_schema: bool = True,
    ) -> ChatTwoPassReflectionResult:
        messages = [{"role": "system", "content": _v2_same_chat_system_prompt(full_text)}]

        pass1_prompt = (
            "<workflow_step>\nPASS 1 - STRUCTURAL SKELETON\n</workflow_step>\n\n"
            "<objective>\nBuild the structural inventory of the business.\n</objective>\n\n"
            f"<company_name>\n{company_name or ''}\n</company_name>\n\n"
            "<extract_only>\n- HAS_SEGMENT\n- OFFERS\n</extract_only>\n\n"
            "<pass_specific_focus>\n"
            "- capture all explicit named segments\n"
            "- capture all explicit named offerings\n"
            "- capture explicit offering-family hierarchies when the filing directly states them\n"
            "- ensure each child offering has at most one offering-parent in Offering -> OFFERS -> Offering hierarchy\n"
            "</pass_specific_focus>\n\n"
            "<ontology_reminder>\n"
            "- follow the ontology rules for Company, BusinessSegment, and Offering\n"
            "- remember that OFFERS may be BusinessSegment -> Offering, Company -> Offering, or Offering -> Offering only when the ontology conditions are satisfied\n"
            "- if more than one umbrella offering could fit, keep only the single most explicit offering-parent\n"
            "</ontology_reminder>\n\n"
            "<output_scope>\nReturn only HAS_SEGMENT and OFFERS triples for this pass.\n</output_scope>"
        )
        messages.append({"role": "user", "content": pass1_prompt})
        try:
            skeleton_extraction, raw_skeleton_response, skeleton_attempts_used, skeleton_audit = self._call_structured_messages(
                messages=messages,
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Truncated skeleton extraction.","triples":[]}',
                max_retries=max_retries,
                use_schema=use_schema,
                ontology_version="v2",
            )
        except ExtractionError as exc:
            return ChatTwoPassReflectionResult(success=False, error=str(exc))
        messages.append(
            {
                "role": "assistant",
                "content": raw_skeleton_response or json.dumps(skeleton_extraction.model_dump(mode="json"), ensure_ascii=False),
            }
        )

        pass2_prompt = (
            "<workflow_step>\nPASS 2 - CHANNELS AND REVENUE MODELS\n</workflow_step>\n\n"
            "<objective>\nUsing the current graph as fixed context, extract commercial logic.\n</objective>\n\n"
            "<extract_only>\n- SELLS_THROUGH\n- MONETIZES_VIA\n</extract_only>\n\n"
            "<pass_specific_focus>\n"
            "- add only channel and revenue-model facts\n"
            "- keep structure unchanged\n"
            "- use product-level facts only when the filing isolates them explicitly\n"
            "</pass_specific_focus>\n\n"
            "<ontology_reminder>\n"
            "- follow the ontology rules for BusinessSegment, Offering, Company, Channel, and RevenueModel\n"
            "- remember the allowed subject scopes and canonical labels for SELLS_THROUGH and MONETIZES_VIA\n"
            "</ontology_reminder>\n\n"
            "<output_scope>\nReturn only SELLS_THROUGH and MONETIZES_VIA triples for this pass.\n</output_scope>"
        )
        messages.append({"role": "user", "content": pass2_prompt})
        try:
            pass2_extraction, raw_pass2_response, pass2_attempts_used, pass2_audit = self._call_structured_messages(
                messages=messages,
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Truncated pass-2 extraction.","triples":[]}',
                max_retries=max_retries,
                use_schema=use_schema,
                ontology_version="v2",
            )
        except ExtractionError as exc:
            return ChatTwoPassReflectionResult(
                success=False,
                skeleton_extraction=skeleton_extraction,
                skeleton_audit=skeleton_audit,
                raw_skeleton_response=raw_skeleton_response,
                skeleton_attempts_used=skeleton_attempts_used,
                error=str(exc),
            )
        pass2_effective_extraction = self._merge_relation_subset_into_base(
            skeleton_extraction,
            pass2_extraction,
            allowed_relations={"SELLS_THROUGH", "MONETIZES_VIA"},
        )
        messages.append(
            {
                "role": "assistant",
                "content": raw_pass2_response or json.dumps(pass2_extraction.model_dump(mode="json"), ensure_ascii=False),
            }
        )

        pass3_serves_prompt = (
            "<workflow_step>\nPASS 3 - CUSTOMER TYPES\n</workflow_step>\n\n"
            "<objective>\nUsing the current graph as fixed context, extract customer-type relations.\n</objective>\n\n"
            "<extract_only>\n- SERVES\n</extract_only>\n\n"
            "<pass_specific_focus>\n"
            "- add only SERVES facts\n"
            "- use conservative text-grounded inference when justified\n"
            "- separate explicit SERVES facts from inferred SERVES facts in extraction_notes\n"
            "</pass_specific_focus>\n\n"
            "<ontology_reminder>\n"
            "- follow the ontology rules for BusinessSegment, Offering, Company, and CustomerType\n"
            "- remember the allowed subject scopes and canonical labels for SERVES\n"
            "</ontology_reminder>\n\n"
            "<output_scope>\nReturn only SERVES triples for this pass.\n</output_scope>"
        )
        messages.append({"role": "user", "content": pass3_serves_prompt})
        try:
            pass3_serves_extraction, raw_pass3_serves_response, pass3_serves_attempts_used, pass3_serves_audit = (
                self._call_structured_messages(
                    messages=messages,
                    schema_name="KnowledgeGraphExtraction",
                    schema_model=KnowledgeGraphExtraction,
                    fallback_payload='{"extraction_notes":"Truncated serves extraction.","triples":[]}',
                    max_retries=max_retries,
                    use_schema=use_schema,
                    ontology_version="v2",
                )
            )
        except ExtractionError as exc:
            return ChatTwoPassReflectionResult(
                success=False,
                skeleton_extraction=skeleton_extraction,
                pass2_extraction=pass2_extraction,
                skeleton_audit=skeleton_audit,
                pass2_audit=pass2_audit,
                raw_skeleton_response=raw_skeleton_response,
                raw_pass2_response=raw_pass2_response,
                skeleton_attempts_used=skeleton_attempts_used,
                pass2_attempts_used=pass2_attempts_used,
                error=str(exc),
            )
        pass3_effective_extraction = self._merge_serves_into_base(pass2_effective_extraction, pass3_serves_extraction)
        messages.append(
            {
                "role": "assistant",
                "content": raw_pass3_serves_response
                or json.dumps(pass3_serves_extraction.model_dump(mode="json"), ensure_ascii=False),
            }
        )

        pass4_corporate_prompt = (
            "<workflow_step>\nPASS 4 - CORPORATE SHELL FACTS\n</workflow_step>\n\n"
            "<objective>\nUsing the current graph as fixed context, extract corporate geography and partnerships.\n</objective>\n\n"
            "<extract_only>\n- OPERATES_IN\n- PARTNERS_WITH\n</extract_only>\n\n"
            "<pass_specific_focus>\n"
            "- add only company-level geography\n"
            "- add only explicit named partnerships\n"
            "- keep all non-corporate facts unchanged\n"
            "</pass_specific_focus>\n\n"
            "<ontology_reminder>\n"
            "- follow the ontology rules for Company, Place, and Company-to-Company partnerships\n"
            "- remember that OPERATES_IN and PARTNERS_WITH are company-level only in ontology v2\n"
            "</ontology_reminder>\n\n"
            "<output_scope>\nReturn only OPERATES_IN and PARTNERS_WITH triples for this pass.\n</output_scope>"
        )
        messages.append({"role": "user", "content": pass4_corporate_prompt})
        try:
            pass4_corporate_extraction, raw_pass4_corporate_response, pass4_corporate_attempts_used, pass4_corporate_audit = self._call_structured_messages(
                messages=messages,
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Truncated corporate-shell extraction.","triples":[]}',
                max_retries=max_retries,
                use_schema=use_schema,
                ontology_version="v2",
            )
        except ExtractionError as exc:
            return ChatTwoPassReflectionResult(
                success=False,
                skeleton_extraction=skeleton_extraction,
                pass2_extraction=pass2_extraction,
                pass3_serves_extraction=pass3_serves_extraction,
                pre_reflection_extraction=pass3_effective_extraction,
                skeleton_audit=skeleton_audit,
                pass2_audit=pass2_audit,
                pass3_serves_audit=pass3_serves_audit,
                pre_reflection_audit=pass3_serves_audit,
                raw_skeleton_response=raw_skeleton_response,
                raw_pass2_response=raw_pass2_response,
                raw_pass3_serves_response=raw_pass3_serves_response,
                reflection1_extraction=pass3_effective_extraction,
                skeleton_attempts_used=skeleton_attempts_used,
                pass2_attempts_used=pass2_attempts_used,
                pass3_serves_attempts_used=pass3_serves_attempts_used,
                error=str(exc),
            )
        pass4_effective_extraction = self._merge_relation_subset_into_base(
            pass3_effective_extraction,
            pass4_corporate_extraction,
            allowed_relations={"OPERATES_IN", "PARTNERS_WITH"},
        )

        final_reflection_prompt = (
            "WORKFLOW STEP: REFLECTION - FINAL RECONCILIATION (INDEPENDENT REVIEW, ONTOLOGY V2)\n\n"
            "<workflow_step>\n"
            "REFLECTION - FINAL RECONCILIATION\n"
            "</workflow_step>\n\n"
            "<objective>\n"
            "Review the draft graph and return the final canonical graph.\n"
            "</objective>\n\n"
            f"<company_name>\n{company_name or ''}\n</company_name>\n\n"
            f"<current_graph>\n{self._compact_json(pass4_effective_extraction.model_dump())}\n</current_graph>\n\n"
            "<review_instruction>\n"
            "Act exactly as the system prompt instructs.\n"
            "Audit the draft graph against the filing and the ontology.\n"
            "Correct, remove, keep, and add triples only as needed to produce the final canonical graph.\n"
            "</review_instruction>"
        )
        final_extraction, raw_final_reflection_response, final_reflection_attempts_used, final_reflection_audit = self.reflect_extraction(
            full_text=full_text,
            current_extraction=pass4_effective_extraction,
            company_name=company_name,
            max_retries=max_retries,
            strict=False,
            use_schema=use_schema,
            system_prompt=_v2_reflection_system_prompt(full_text),
            user_prompt=final_reflection_prompt,
            ontology_version="v2",
        )

        return ChatTwoPassReflectionResult(
            success=True,
            skeleton_extraction=skeleton_extraction,
            pass2_extraction=pass2_extraction,
            pass3_serves_extraction=pass3_serves_extraction,
            pass4_corporate_extraction=pass4_corporate_extraction,
            pre_reflection_extraction=pass4_effective_extraction,
            reflection1_extraction=pass4_effective_extraction,
            final_extraction=final_extraction,
            skeleton_audit=skeleton_audit,
            pass2_audit=pass2_audit,
            pass3_serves_audit=pass3_serves_audit,
            pass4_corporate_audit=pass4_corporate_audit,
            pre_reflection_audit=pass4_corporate_audit,
            reflection1_audit=pass4_corporate_audit,
            final_reflection_audit=final_reflection_audit,
            raw_skeleton_response=raw_skeleton_response,
            raw_pass2_response=raw_pass2_response,
            raw_pass3_serves_response=raw_pass3_serves_response,
            raw_pass4_corporate_response=raw_pass4_corporate_response,
            raw_reflection1_response=raw_pass4_corporate_response,
            raw_final_reflection_response=raw_final_reflection_response,
            skeleton_attempts_used=skeleton_attempts_used,
            pass2_attempts_used=pass2_attempts_used,
            pass3_serves_attempts_used=pass3_serves_attempts_used,
            pass4_corporate_attempts_used=pass4_corporate_attempts_used,
            reflection1_attempts_used=pass4_corporate_attempts_used,
            final_reflection_attempts_used=final_reflection_attempts_used,
        )

    def extract_chat_two_pass_reflection_v2_segment_serves(
        self,
        *,
        full_text: str,
        company_name: str | None = None,
        max_retries: int = 2,
        use_schema: bool = True,
    ) -> ChatTwoPassReflectionResult:
        same_chat_messages = [{"role": "system", "content": _v2_segment_serves_same_chat_system_prompt(full_text)}]

        pass1_prompt = (
            "<workflow_step>\nPASS 1 - STRUCTURAL SKELETON\n</workflow_step>\n\n"
            "<objective>\nBuild the structural inventory of the business.\n</objective>\n\n"
            f"<company_name>\n{company_name or ''}\n</company_name>\n\n"
            "<extract_only>\n- HAS_SEGMENT\n- OFFERS\n</extract_only>\n\n"
            "<pass_specific_focus>\n"
            "- capture all explicit named segments\n"
            "- build the offering inventory segment by segment\n"
            "- capture all explicit named offerings\n"
            "- first identify the direct offering children stated under each BusinessSegment\n"
            "- use Offering -> OFFERS -> Offering only when the text explicitly states that an offering is a family, suite, umbrella, parent, or grouped subcategory for another offering\n"
            "- do not invent intermediate umbrella offerings or extra nesting just to organize the graph more neatly\n"
            "- each child offering may have at most one Offering parent\n"
            "- if the filing states both an umbrella offering and its explicit named children, keep both\n"
            "- before returning, check that no explicit named offering has been omitted\n"
            "</pass_specific_focus>\n\n"
            "<ontology_reminder>\n"
            "- follow the ontology rules for Company, BusinessSegment, and Offering\n"
            "- an explicit umbrella offering does not replace its explicitly named child offerings; keep both when the filing states both\n"
            "</ontology_reminder>\n\n"
            "<format_reminder>\n"
            "Return strictly one JSON object in the exact format defined by the system prompt.\n"
            "</format_reminder>\n\n"
            "<output_scope>\nReturn only HAS_SEGMENT and OFFERS triples for this pass.\n</output_scope>"
        )
        same_chat_messages.append({"role": "user", "content": pass1_prompt})
        try:
            skeleton_extraction, raw_skeleton_response, skeleton_attempts_used, skeleton_audit = self._call_structured_messages(
                messages=same_chat_messages,
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Truncated skeleton extraction.","triples":[]}',
                max_retries=max_retries,
                use_schema=use_schema,
                ontology_version="v2_segment_serves",
            )
        except ExtractionError as exc:
            return ChatTwoPassReflectionResult(success=False, error=str(exc))
        same_chat_messages.append(
            {
                "role": "assistant",
                "content": raw_skeleton_response or json.dumps(skeleton_extraction.model_dump(mode="json"), ensure_ascii=False),
            }
        )

        pass2_channels_prompt = (
            "<workflow_step>\nPASS 2A - CHANNELS\n</workflow_step>\n\n"
            "<objective>\nUsing the current graph as fixed context, extract only sales and distribution channels.\n</objective>\n\n"
            "<extract_only>\n- SELLS_THROUGH\n</extract_only>\n\n"
            "<channel_definitions>\n"
            f"{_xml_definition_lines(V2_SEGMENT_SERVES_CANONICAL_DEFINITIONS['Channel'])}\n"
            "</channel_definitions>\n\n"
            "<pass_specific_focus>\n"
            "- reason about channels first, before any monetization logic\n"
            "- evaluate channels segment by segment\n"
            "- if a channel is stated at company-wide scope and the company has reported segments, translate that evidence into segment-level channel triples rather than company-level ones\n"
            "- use Offering for SELLS_THROUGH only when the offering has no BusinessSegment anchor\n"
            "- keep structure unchanged\n"
            "</pass_specific_focus>\n\n"
            "<format_reminder>\n"
            "Return strictly one JSON object in the exact format defined by the system prompt.\n"
            "</format_reminder>\n\n"
            "<output_scope>\nReturn only SELLS_THROUGH triples for this pass.\n</output_scope>"
        )
        same_chat_messages.append({"role": "user", "content": pass2_channels_prompt})
        try:
            pass2_channels_extraction, raw_pass2_channels_response, pass2_channels_attempts_used, pass2_channels_audit = self._call_structured_messages(
                messages=same_chat_messages,
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Truncated pass-2 channels extraction.","triples":[]}',
                max_retries=max_retries,
                use_schema=use_schema,
                ontology_version="v2_segment_serves",
            )
        except ExtractionError as exc:
            return ChatTwoPassReflectionResult(
                success=False,
                skeleton_extraction=skeleton_extraction,
                skeleton_audit=skeleton_audit,
                raw_skeleton_response=raw_skeleton_response,
                skeleton_attempts_used=skeleton_attempts_used,
                error=str(exc),
            )
        channels_effective_extraction = self._merge_relation_subset_into_base(
            skeleton_extraction,
            pass2_channels_extraction,
            allowed_relations={"SELLS_THROUGH"},
        )
        same_chat_messages.append(
            {
                "role": "assistant",
                "content": raw_pass2_channels_response
                or json.dumps(pass2_channels_extraction.model_dump(mode="json"), ensure_ascii=False),
            }
        )

        pass2_revenue_prompt = (
            "<workflow_step>\nPASS 2B - REVENUE MODELS\n</workflow_step>\n\n"
            "<objective>\nUsing the current graph as fixed context, extract only offering-level revenue models.\n</objective>\n\n"
            "<extract_only>\n- MONETIZES_VIA\n</extract_only>\n\n"
            "<revenue_model_definitions>\n"
            f"{_xml_definition_lines(V2_SEGMENT_SERVES_CANONICAL_DEFINITIONS['RevenueModel'])}\n"
            "</revenue_model_definitions>\n\n"
            "<pass_specific_focus>\n"
            "- evaluate monetization offering by offering\n"
            "- if an offering family hierarchy exists, attach MONETIZES_VIA to the family parent rather than to its child offerings\n"
            "- do not attach MONETIZES_VIA to a child offering that already has an explicit Offering parent\n"
            "- only add MONETIZES_VIA when the filing supports that offering-level monetization clearly enough\n"
            "</pass_specific_focus>\n\n"
            "<format_reminder>\n"
            "Return strictly one JSON object in the exact format defined by the system prompt.\n"
            "</format_reminder>\n\n"
            "<output_scope>\nReturn only MONETIZES_VIA triples for this pass.\n</output_scope>"
        )
        same_chat_messages.append({"role": "user", "content": pass2_revenue_prompt})
        try:
            pass2_revenue_extraction, raw_pass2_revenue_response, pass2_revenue_attempts_used, pass2_revenue_audit = self._call_structured_messages(
                messages=same_chat_messages,
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Truncated pass-2 revenue extraction.","triples":[]}',
                max_retries=max_retries,
                use_schema=use_schema,
                ontology_version="v2_segment_serves",
            )
        except ExtractionError as exc:
            return ChatTwoPassReflectionResult(
                success=False,
                skeleton_extraction=skeleton_extraction,
                pass2_channels_extraction=pass2_channels_extraction,
                pass2_extraction=channels_effective_extraction,
                skeleton_audit=skeleton_audit,
                pass2_channels_audit=pass2_channels_audit,
                pass2_audit=aggregate_extraction_audits([pass2_channels_audit]),
                raw_skeleton_response=raw_skeleton_response,
                raw_pass2_channels_response=raw_pass2_channels_response,
                raw_pass2_response=raw_pass2_channels_response,
                skeleton_attempts_used=skeleton_attempts_used,
                pass2_channels_attempts_used=pass2_channels_attempts_used,
                pass2_attempts_used=pass2_channels_attempts_used,
                error=str(exc),
            )
        pass2_effective_extraction = self._merge_relation_subset_into_base(
            channels_effective_extraction,
            pass2_revenue_extraction,
            allowed_relations={"MONETIZES_VIA"},
        )
        same_chat_messages.append(
            {
                "role": "assistant",
                "content": raw_pass2_revenue_response
                or json.dumps(pass2_revenue_extraction.model_dump(mode="json"), ensure_ascii=False),
            }
        )
        pass2_merged_extraction = KnowledgeGraphExtraction(
            extraction_notes="Merged channel and revenue-model passes.",
            triples=pass2_effective_extraction.triples,
        )
        pass2_aggregate_audit = aggregate_extraction_audits([pass2_channels_audit, pass2_revenue_audit])

        pass3_serves_prompt = (
            "<workflow_step>\nPASS 3 - CUSTOMER TYPES\n</workflow_step>\n\n"
            "<objective>\nUsing the structural graph from PASS 1 as fixed context, extract customer-type relations.\n</objective>\n\n"
            f"<company_name>\n{company_name or ''}\n</company_name>\n\n"
            f"<current_structure>\n{self._compact_json(skeleton_extraction.model_dump())}\n</current_structure>\n\n"
            "<extract_only>\n- SERVES\n</extract_only>\n\n"
            "<customer_type_definitions>\n"
            f"{_xml_definition_lines(V2_SEGMENT_SERVES_CANONICAL_DEFINITIONS['CustomerType'])}\n"
            "</customer_type_definitions>\n\n"
            "<pass_specific_focus>\n"
            "- add only SERVES facts\n"
            "- use conservative text-grounded inference when justified\n"
            "- use only the PASS 1 structure as fixed context\n"
            "- reason segment by segment from the offerings and descriptions inside each BusinessSegment\n"
            "- for each BusinessSegment, check the canonical customer types against it and keep every clearly stated or clearly implied one\n"
            "- do not fan out a rare or specialized customer type like government agencies, educational institutions, healthcare organizations, financial services firms, manufacturers, or retailers unless that segment has its own support\n"
            "</pass_specific_focus>\n\n"
            "<ontology_reminder>\n"
            "- follow the ontology rules for BusinessSegment and CustomerType\n"
            "- when several offerings inside the same segment point to the same customer type, attach that customer type to the BusinessSegment\n"
            "</ontology_reminder>\n\n"
            "<format_reminder>\n"
            "Return strictly one JSON object in the exact format defined by the system prompt.\n"
            "</format_reminder>\n\n"
            "<output_scope>\nReturn only SERVES triples for this pass.\n</output_scope>"
        )
        try:
            pass3_serves_extraction, raw_pass3_serves_response, pass3_serves_attempts_used, pass3_serves_audit = self._call_structured_messages(
                messages=[
                    {"role": "system", "content": _v2_segment_serves_same_chat_system_prompt(full_text)},
                    {"role": "user", "content": pass3_serves_prompt},
                ],
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Truncated serves extraction.","triples":[]}',
                max_retries=max_retries,
                use_schema=use_schema,
                ontology_version="v2_segment_serves",
            )
        except ExtractionError as exc:
            return ChatTwoPassReflectionResult(
                success=False,
                skeleton_extraction=skeleton_extraction,
                pass2_channels_extraction=pass2_channels_extraction,
                pass2_revenue_extraction=pass2_revenue_extraction,
                pass2_extraction=pass2_merged_extraction,
                skeleton_audit=skeleton_audit,
                pass2_channels_audit=pass2_channels_audit,
                pass2_revenue_audit=pass2_revenue_audit,
                pass2_audit=pass2_aggregate_audit,
                raw_skeleton_response=raw_skeleton_response,
                raw_pass2_channels_response=raw_pass2_channels_response,
                raw_pass2_revenue_response=raw_pass2_revenue_response,
                raw_pass2_response=raw_pass2_revenue_response,
                skeleton_attempts_used=skeleton_attempts_used,
                pass2_channels_attempts_used=pass2_channels_attempts_used,
                pass2_revenue_attempts_used=pass2_revenue_attempts_used,
                pass2_attempts_used=pass2_channels_attempts_used + pass2_revenue_attempts_used,
                error=str(exc),
            )
        pass3_effective_extraction = self._merge_serves_into_base(pass2_effective_extraction, pass3_serves_extraction)

        pass4_corporate_prompt = (
            "<workflow_step>\nPASS 4 - CORPORATE SHELL FACTS\n</workflow_step>\n\n"
            "<objective>\nUsing the filing directly, extract corporate geography and partnerships.\n</objective>\n\n"
            "<extract_only>\n- OPERATES_IN\n- PARTNERS_WITH\n</extract_only>\n\n"
            "<pass_specific_focus>\n"
            "- add only company-level geography\n"
            "- add only explicit named partnerships\n"
            "</pass_specific_focus>\n\n"
            "<format_reminder>\n"
            "Return strictly one JSON object in the exact format defined by the system prompt.\n"
            "</format_reminder>\n\n"
            "<output_scope>\nReturn only OPERATES_IN and PARTNERS_WITH triples for this pass.\n</output_scope>"
        )
        try:
            pass4_corporate_extraction, raw_pass4_corporate_response, pass4_corporate_attempts_used, pass4_corporate_audit = self._call_structured_messages(
                messages=[
                    {"role": "system", "content": _v2_segment_serves_same_chat_system_prompt(full_text)},
                    {"role": "user", "content": pass4_corporate_prompt},
                ],
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Truncated corporate-shell extraction.","triples":[]}',
                max_retries=max_retries,
                use_schema=use_schema,
                ontology_version="v2_segment_serves",
            )
        except ExtractionError as exc:
            return ChatTwoPassReflectionResult(
                success=False,
                skeleton_extraction=skeleton_extraction,
                pass2_channels_extraction=pass2_channels_extraction,
                pass2_revenue_extraction=pass2_revenue_extraction,
                pass2_extraction=pass2_merged_extraction,
                pass3_serves_extraction=pass3_serves_extraction,
                pre_reflection_extraction=KnowledgeGraphExtraction(
                    extraction_notes="Merged structure, channels, revenue models, and customer types before corporate-shell extraction.",
                    triples=pass3_effective_extraction.triples,
                ),
                skeleton_audit=skeleton_audit,
                pass2_channels_audit=pass2_channels_audit,
                pass2_revenue_audit=pass2_revenue_audit,
                pass2_audit=pass2_aggregate_audit,
                pass3_serves_audit=pass3_serves_audit,
                pre_reflection_audit=pass3_serves_audit,
                raw_skeleton_response=raw_skeleton_response,
                raw_pass2_channels_response=raw_pass2_channels_response,
                raw_pass2_revenue_response=raw_pass2_revenue_response,
                raw_pass2_response=raw_pass2_revenue_response,
                raw_pass3_serves_response=raw_pass3_serves_response,
                reflection1_extraction=KnowledgeGraphExtraction(
                    extraction_notes="Merged structure, channels, revenue models, and customer types before corporate-shell extraction.",
                    triples=pass3_effective_extraction.triples,
                ),
                skeleton_attempts_used=skeleton_attempts_used,
                pass2_channels_attempts_used=pass2_channels_attempts_used,
                pass2_revenue_attempts_used=pass2_revenue_attempts_used,
                pass2_attempts_used=pass2_channels_attempts_used + pass2_revenue_attempts_used,
                pass3_serves_attempts_used=pass3_serves_attempts_used,
                error=str(exc),
            )
        pass4_effective_extraction = self._merge_relation_subset_into_base(
            pass3_effective_extraction,
            pass4_corporate_extraction,
            allowed_relations={"OPERATES_IN", "PARTNERS_WITH"},
        )
        pre_reflection_extraction = KnowledgeGraphExtraction(
            extraction_notes="Merged structure, channels, revenue models, customer types, and corporate-shell facts before final reflection.",
            triples=pass4_effective_extraction.triples,
        )
        _, pre_reflection_audit = audit_knowledge_graph_payload(
            pre_reflection_extraction.model_dump(mode="json"),
            ontology_version="v2_segment_serves",
        )

        final_reflection_prompt = (
            "<workflow_step>\n"
            "REFLECTION - FINAL RECONCILIATION\n"
            "</workflow_step>\n\n"
            "<objective>\n"
            "Review the draft graph and return the final canonical graph.\n"
            "</objective>\n\n"
            f"<company_name>\n{company_name or ''}\n</company_name>\n\n"
            f"<current_graph>\n{self._compact_json(pre_reflection_extraction.model_dump())}\n</current_graph>\n\n"
            "<review_instruction>\n"
            "Act exactly as the system prompt instructs.\n"
            "Audit the draft graph against the filing and the ontology.\n"
            "Correct, remove, keep, and add triples only as needed to produce the final canonical graph.\n"
            "</review_instruction>\n\n"
            "<format_reminder>\n"
            "Return strictly one JSON object in the exact format defined by the system prompt.\n"
            "</format_reminder>"
        )
        final_extraction, raw_final_reflection_response, final_reflection_attempts_used, final_reflection_audit = self.reflect_extraction(
            full_text=full_text,
            current_extraction=pre_reflection_extraction,
            company_name=company_name,
            max_retries=max_retries,
            strict=False,
            use_schema=use_schema,
            system_prompt=_v2_segment_serves_reflection_system_prompt(full_text),
            user_prompt=final_reflection_prompt,
            ontology_version="v2_segment_serves",
        )

        return ChatTwoPassReflectionResult(
            success=True,
            skeleton_extraction=skeleton_extraction,
            pass2_channels_extraction=pass2_channels_extraction,
            pass2_revenue_extraction=pass2_revenue_extraction,
            pass2_extraction=pass2_merged_extraction,
            pass3_serves_extraction=pass3_serves_extraction,
            pass4_corporate_extraction=pass4_corporate_extraction,
            pre_reflection_extraction=pre_reflection_extraction,
            reflection1_extraction=pre_reflection_extraction,
            final_extraction=final_extraction,
            skeleton_audit=skeleton_audit,
            pass2_channels_audit=pass2_channels_audit,
            pass2_revenue_audit=pass2_revenue_audit,
            pass2_audit=pass2_aggregate_audit,
            pass3_serves_audit=pass3_serves_audit,
            pass4_corporate_audit=pass4_corporate_audit,
            pre_reflection_audit=pre_reflection_audit,
            reflection1_audit=pre_reflection_audit,
            final_reflection_audit=final_reflection_audit,
            raw_skeleton_response=raw_skeleton_response,
            raw_pass2_channels_response=raw_pass2_channels_response,
            raw_pass2_revenue_response=raw_pass2_revenue_response,
            raw_pass2_response=raw_pass2_revenue_response,
            raw_pass3_serves_response=raw_pass3_serves_response,
            raw_pass4_corporate_response=raw_pass4_corporate_response,
            raw_reflection1_response=raw_pass4_corporate_response,
            raw_final_reflection_response=raw_final_reflection_response,
            skeleton_attempts_used=skeleton_attempts_used,
            pass2_channels_attempts_used=pass2_channels_attempts_used,
            pass2_revenue_attempts_used=pass2_revenue_attempts_used,
            pass2_attempts_used=pass2_channels_attempts_used + pass2_revenue_attempts_used,
            pass3_serves_attempts_used=pass3_serves_attempts_used,
            pass4_corporate_attempts_used=pass4_corporate_attempts_used,
            reflection1_attempts_used=pass4_corporate_attempts_used,
            final_reflection_attempts_used=final_reflection_attempts_used,
        )
