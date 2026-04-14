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
    "SERVES",
    "OPERATES_IN",
    "SELLS_THROUGH",
    "PARTNERS_WITH",
    "MONETIZES_VIA",
]

CANONICAL_CUSTOMER_TYPES = canonical_labels("CustomerType")
CANONICAL_CHANNELS = canonical_labels("Channel")
CANONICAL_REVENUE_MODELS = canonical_labels("RevenueModel")
CANONICAL_DEFINITIONS = load_ontology_config()["canonical_labels"]


def _xml_definition_lines(definitions: dict[str, str]) -> str:
    return "\n".join(f'- "{label}": {definition}' for label, definition in definitions.items())


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


class CanonicalPipelineResult(BaseModel):
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
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


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
    ontology_version: str = "canonical",
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


def _canonical_pipeline_system_prompt(full_text: str) -> str:
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
Do not wrap the JSON in markdown code fences.
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


def _canonical_reflection_system_prompt(full_text: str) -> str:
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
Do not wrap the JSON in markdown code fences.
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
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        model: str = "local-model",
        provider: str = "local",
        api_mode: str = "chat_completions",
        max_output_tokens: int | None = None,
    ):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.provider = provider
        self.api_mode = api_mode
        self.max_output_tokens = max_output_tokens

    @staticmethod
    def _schema_def(name: str, model_schema: type[BaseModel], *, ontology_version: str = "canonical") -> dict:
        schema = model_schema.model_json_schema(by_alias=True)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": schema,
            },
        }

    @staticmethod
    def _responses_schema_def(name: str, model_schema: type[BaseModel], *, ontology_version: str = "canonical") -> dict:
        schema = model_schema.model_json_schema(by_alias=True)
        return {
            "type": "json_schema",
            "name": name,
            "schema": schema,
            "strict": True,
        }

    @staticmethod
    def _compact_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _strip_code_fence(content: str) -> str:
        match = CODE_FENCE_RE.match(content.strip())
        if match:
            return match.group(1).strip()
        return content.strip()

    @staticmethod
    def _assistant_history_content(extraction: KnowledgeGraphExtraction) -> str:
        return json.dumps(extraction.model_dump(mode="json"), ensure_ascii=False)

    @staticmethod
    def _prepare_messages_for_provider(messages: list[dict[str, str]], provider: str) -> list[dict[str, str]]:
        if provider != "opencode-go":
            return list(messages)

        # OpenCode Go is handled more conservatively: keep the pipeline structure,
        # but send system instructions as user messages for compatibility.
        return [
            {
                **message,
                "role": "user" if message.get("role") == "system" else message.get("role", "user"),
            }
            for message in messages
        ]

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

        try:
            return json.loads(content), False
        except json.JSONDecodeError:
            pass

        fenced_content = LLMExtractor._strip_code_fence(content)
        if fenced_content != content:
            try:
                return json.loads(fenced_content), True
            except json.JSONDecodeError:
                pass

        match = JSON_OBJECT_RE.search(content)
        if match:
            json_object_text = match.group(0)
            if json_object_text != content:
                try:
                    return json.loads(json_object_text), True
                except json.JSONDecodeError:
                    pass

        likely_truncated = (
            content.count("{") > content.count("}")
            or content.count("[") > content.count("]")
            or not content.endswith(("}", "]"))
        )
        if likely_truncated:
            logger.warning("Model response may be truncated. Attempting to salvage JSON prefix...")
            last_object_end = content.rfind("}")
            last_array_end = content.rfind("]")
            last_json_end = max(last_object_end, last_array_end)
            truncated_candidate = content[: last_json_end + 1] if last_json_end != -1 else ""
            if truncated_candidate:
                try:
                    return json.loads(truncated_candidate), True
                except json.JSONDecodeError:
                    pass

        logger.warning("Model response was not exact raw JSON. Falling back to the recovery payload.")
        return json.loads(fallback_payload), True

    @staticmethod
    def _responses_refusal_text(response: Any) -> str | None:
        for output_item in getattr(response, "output", []) or []:
            for content_item in getattr(output_item, "content", []) or []:
                if getattr(content_item, "type", None) == "refusal":
                    refusal = getattr(content_item, "refusal", None)
                    if refusal:
                        return str(refusal)
        return None

    @staticmethod
    def _responses_output_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text)

        text_parts: list[str] = []
        for output_item in getattr(response, "output", []) or []:
            for content_item in getattr(output_item, "content", []) or []:
                if getattr(content_item, "type", None) != "output_text":
                    continue
                text = getattr(content_item, "text", None)
                if text:
                    text_parts.append(str(text))
        return "".join(text_parts)

    def _call_structured_messages(
        self,
        *,
        messages: list[dict[str, str]],
        schema_name: str,
        schema_model: type[BaseModel],
        fallback_payload: str,
        max_retries: int,
        temperature: float = 0.0,
        use_schema: bool = False,
        ontology_version: str = "canonical",
    ) -> tuple[BaseModel, str | None, int, dict[str, Any]]:
        request_messages = self._prepare_messages_for_provider(messages, self.provider)
        schema_def = self._schema_def(schema_name, schema_model, ontology_version=ontology_version)
        responses_schema_def = self._responses_schema_def(schema_name, schema_model, ontology_version=ontology_version)
        call_label = schema_name
        for message in reversed(request_messages):
            if message.get("role") != "user":
                continue
            for line in message.get("content", "").splitlines():
                stripped = line.strip()
                if stripped:
                    call_label = stripped[:120]
                    break
            if call_label != schema_name:
                break

        for attempt in range(1, max_retries + 1):
            try:
                content = ""
                if self.api_mode == "responses":
                    call_kwargs = {
                        "model": self.model,
                        "input": request_messages,
                        "temperature": temperature,
                    }
                    if use_schema:
                        call_kwargs["text"] = {"format": responses_schema_def}
                    response = self.client.responses.create(**call_kwargs)
                    status = getattr(response, "status", None)
                    usage = getattr(response, "usage", None)
                    output_tokens = getattr(usage, "output_tokens", None) if usage is not None else None
                    logger.info(
                        "Structured call %s attempt %s/%s status=%s output_tokens=%s",
                        call_label,
                        attempt,
                        max_retries,
                        status,
                        output_tokens,
                    )
                    refusal_text = self._responses_refusal_text(response)
                    if refusal_text:
                        raise ExtractionError(f"Model refused request: {refusal_text}")
                    content = self._responses_output_text(response)
                else:
                    call_kwargs = {
                        "model": self.model,
                        "messages": request_messages,
                        "temperature": temperature,
                    }
                    if self.max_output_tokens is not None:
                        call_kwargs["max_tokens"] = self.max_output_tokens
                    if use_schema:
                        call_kwargs["response_format"] = schema_def
                    response = self.client.chat.completions.create(**call_kwargs)
                    choice = response.choices[0]
                    finish_reason = getattr(choice, "finish_reason", None)
                    usage = getattr(response, "usage", None)
                    completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
                    logger.info(
                        "Structured call %s attempt %s/%s finish_reason=%s completion_tokens=%s",
                        call_label,
                        attempt,
                        max_retries,
                        finish_reason,
                        completion_tokens,
                    )
                    refusal_text = getattr(choice.message, "refusal", None)
                    if refusal_text:
                        raise ExtractionError(f"Model refused request: {refusal_text}")
                    content = choice.message.content or ""
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
                logger.warning(
                    "Structured call %s failed on attempt %s/%s: %s",
                    call_label,
                    attempt,
                    max_retries,
                    exc,
                )
            except Exception as exc:
                logger.warning("LLM API error for %s on attempt %s/%s: %s", call_label, attempt, max_retries, exc)

        raise ExtractionError(f"Failed after {max_retries} attempts")

    @staticmethod
    def _lenient_model_from_payload(
        schema_model: type[BaseModel],
        payload: Any,
        *,
        ontology_version: str = "canonical",
    ) -> tuple[BaseModel, dict[str, Any]]:
        normalized_payload = normalize_lenient_payload(payload)
        valid_triples, audit = audit_knowledge_graph_payload(normalized_payload, ontology_version=ontology_version)
        extraction_notes = str(
            normalized_payload.get("extraction_notes", normalized_payload.get("chain_of_thought_reasoning", "")) or ""
        )
        triple_objects = [Triple(**triple) for triple in valid_triples]

        if schema_model is KnowledgeGraphExtraction:
            model = KnowledgeGraphExtraction(
                extraction_notes=extraction_notes,
                triples=triple_objects,
            )
            return model, audit

        raise TypeError(f"Unsupported schema model for lenient payload parsing: {schema_model!r}")

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
        use_schema: bool = False,
        ontology_version: str = "canonical",
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

    def reflect_extraction(
        self,
        *,
        full_text: str,
        current_extraction: KnowledgeGraphExtraction,
        company_name: str | None = None,
        max_retries: int = 2,
        strict: bool = True,
        use_schema: bool = False,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        ontology_version: str = "canonical",
    ) -> tuple[KnowledgeGraphExtraction, str | None, int, dict[str, Any]]:
        system_prompt = system_prompt or _canonical_reflection_system_prompt(full_text)
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

    def extract_canonical_pipeline(
        self,
        *,
        full_text: str,
        company_name: str | None = None,
        max_retries: int = 2,
        use_schema: bool = False,
    ) -> CanonicalPipelineResult:
        same_chat_messages = [{"role": "system", "content": _canonical_pipeline_system_prompt(full_text)}]

        pass1_prompt = (
            "<workflow_step>\nPASS 1 - STRUCTURAL SKELETON\n</workflow_step>\n\n"
            "<objective>\nBuild the structural inventory of the business.\n</objective>\n\n"
            f"<company_name>\n{company_name or ''}\n</company_name>\n\n"
            "<extract_only>\n- HAS_SEGMENT\n- OFFERS\n</extract_only>\n\n"
            "<pass_specific_focus>\n"
            "- capture all explicit named segments\n"
            "- build the offering inventory segment by segment\n"
            "- capture all explicit named offerings\n"
            "- use the full filing, not just the nearest sentence or opening overview, to decide each offering's parent\n"
            "- when the filing reports BusinessSegments, assume each named offering should be attached to one or more BusinessSegments unless the filing truly gives no segment anchor anywhere\n"
            "- search broadly across the filing for segment evidence before using Company -> OFFERS -> Offering\n"
            "- if an offering has support for more than one segment, attach it to every supported segment\n"
            "- if an overview describes offerings at company scope but later text gives segment-specific evidence, prefer BusinessSegment -> OFFERS -> Offering\n"
            "- do not let a company-level introductory list override later segment-specific evidence\n"
            "- Company -> OFFERS -> Offering is a last-resort fallback, not a default organizational shortcut\n"
            "- use Company -> OFFERS -> Offering only if, after considering the whole filing, there is still no credible segment anchor, or the filing explicitly presents the offering only at company scope\n"
            "- first identify the direct offering children stated under each BusinessSegment\n"
            "- use Offering -> OFFERS -> Offering only when the text explicitly states that an offering is a family, suite, umbrella, parent, or grouped subcategory for another offering\n"
            "- do not invent intermediate umbrella offerings or extra nesting just to organize the graph more neatly\n"
            "- each child offering may have at most one Offering parent\n"
            "- if the filing states both an umbrella offering and its explicit named children, keep both\n"
            "- before returning, audit every Company -> OFFERS -> Offering triple and keep it only if no segment assignment is supportable from the filing\n"
            "- before returning, check that no explicit named offering has been omitted\n"
            "</pass_specific_focus>\n\n"
            "<ontology_reminder>\n"
            "- follow the ontology rules for Company, BusinessSegment, and Offering\n"
            "- BusinessSegment -> OFFERS -> Offering is the preferred structure whenever segment evidence exists anywhere in the filing\n"
            "- Company -> OFFERS -> Offering is fallback only\n"
            "- an explicit umbrella offering does not replace its explicitly named child offerings; keep both when the filing states both\n"
            "</ontology_reminder>\n\n"
            "<format_reminder>\n"
            'Return ONLY a raw JSON object containing "extraction_notes" and the "triples" array.\n'
            "Do not use markdown code blocks (```json).\n"
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
                ontology_version="canonical",
            )
        except ExtractionError as exc:
            return CanonicalPipelineResult(success=False, error=str(exc))
        same_chat_messages.append(
            {
                "role": "assistant",
                "content": self._assistant_history_content(skeleton_extraction),
            }
        )

        pass2_channels_prompt = (
            "<workflow_step>\nPASS 2A - CHANNELS\n</workflow_step>\n\n"
            "<objective>\nUsing the current graph as fixed context, extract only sales and distribution channels.\n</objective>\n\n"
            "<extract_only>\n- SELLS_THROUGH\n</extract_only>\n\n"
            "<channel_definitions>\n"
            f"{_xml_definition_lines(CANONICAL_DEFINITIONS['Channel'])}\n"
            "</channel_definitions>\n\n"
            "<pass_specific_focus>\n"
            "- reason about channels first, before any monetization logic\n"
            "- evaluate channels segment by segment\n"
            "- if a channel is stated at company-wide scope and the company has reported segments, translate that evidence into segment-level channel triples rather than company-level ones\n"
            "- use Offering for SELLS_THROUGH only when the offering has no BusinessSegment anchor\n"
            "- keep structure unchanged\n"
            "</pass_specific_focus>\n\n"
            "<format_reminder>\n"
            'Return ONLY a raw JSON object containing "extraction_notes" and the "triples" array.\n'
            "Do not use markdown code blocks (```json).\n"
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
                ontology_version="canonical",
            )
        except ExtractionError as exc:
            return CanonicalPipelineResult(
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
                "content": self._assistant_history_content(pass2_channels_extraction),
            }
        )

        pass2_revenue_prompt = (
            "<workflow_step>\nPASS 2B - REVENUE MODELS\n</workflow_step>\n\n"
            "<objective>\nUsing the current graph as fixed context, extract only offering-level revenue models.\n</objective>\n\n"
            "<extract_only>\n- MONETIZES_VIA\n</extract_only>\n\n"
            "<revenue_model_definitions>\n"
            f"{_xml_definition_lines(CANONICAL_DEFINITIONS['RevenueModel'])}\n"
            "</revenue_model_definitions>\n\n"
            "<pass_specific_focus>\n"
            "- evaluate monetization offering by offering\n"
            "- if an offering family hierarchy exists, attach MONETIZES_VIA to the family parent rather than to its child offerings\n"
            "- do not attach MONETIZES_VIA to a child offering that already has an explicit Offering parent\n"
            "- only add MONETIZES_VIA when the filing supports that offering-level monetization clearly enough\n"
            "</pass_specific_focus>\n\n"
            "<format_reminder>\n"
            'Return ONLY a raw JSON object containing "extraction_notes" and the "triples" array.\n'
            "Do not use markdown code blocks (```json).\n"
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
                ontology_version="canonical",
            )
        except ExtractionError as exc:
            return CanonicalPipelineResult(
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
                "content": self._assistant_history_content(pass2_revenue_extraction),
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
            f"{_xml_definition_lines(CANONICAL_DEFINITIONS['CustomerType'])}\n"
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
            'Return ONLY a raw JSON object containing "extraction_notes" and the "triples" array.\n'
            "Do not use markdown code blocks (```json).\n"
            "</format_reminder>\n\n"
            "<output_scope>\nReturn only SERVES triples for this pass.\n</output_scope>"
        )
        try:
            pass3_serves_extraction, raw_pass3_serves_response, pass3_serves_attempts_used, pass3_serves_audit = self._call_structured_messages(
                messages=[
                    {"role": "system", "content": _canonical_pipeline_system_prompt(full_text)},
                    {"role": "user", "content": pass3_serves_prompt},
                ],
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Truncated serves extraction.","triples":[]}',
                max_retries=max_retries,
                use_schema=use_schema,
                ontology_version="canonical",
            )
        except ExtractionError as exc:
            return CanonicalPipelineResult(
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
            "<relation_definitions>\n"
            "- OPERATES_IN: Company -> Place. Use for a normalized, business-relevant geography where the company conducts business, has employees or a local entity, serves customers in a meaningful way, or otherwise has meaningful market presence.\n"
            "- PARTNERS_WITH: Company -> Company. Use only for an explicit named strategic, commercial, distribution, technology, integration, or go-to-market partnership.\n"
            "</relation_definitions>\n\n"
            "<pass_specific_focus>\n"
            "- add only company-level geography and explicit named partnerships\n"
            "- for **OPERATES_IN**, use the **full filing** and include named countries or approved macro-regions where the company conducts business or has **meaningful market presence**\n"
            "- meaningful market presence can be shown by signals such as a named subsidiary or local entity, employee presence, labor structure, customer or revenue presence, country-specific availability, or present-tense current use in that geography\n"
            "- prefer recall over unnecessary omission when a named geography is tied to the company's own business presence\n"
            "- for **PARTNERS_WITH**, add only explicit named partnerships and do not infer them from ordinary supplier, customer, channel, ecosystem, or competitor mentions\n"
            "- exclude vague global labels, cities, office sites, and geographies that appear only as incidental context such as regulatory reference or IP jurisdiction without company presence\n"
            "</pass_specific_focus>\n\n"
            "<format_reminder>\n"
            'Return ONLY a raw JSON object containing "extraction_notes" and the "triples" array.\n'
            "Do not use markdown code blocks (```json).\n"
            "</format_reminder>\n\n"
            "<output_scope>\nReturn only OPERATES_IN and PARTNERS_WITH triples for this pass.\n</output_scope>"
        )
        try:
            pass4_corporate_extraction, raw_pass4_corporate_response, pass4_corporate_attempts_used, pass4_corporate_audit = self._call_structured_messages(
                messages=[
                    {"role": "system", "content": _canonical_pipeline_system_prompt(full_text)},
                    {"role": "user", "content": pass4_corporate_prompt},
                ],
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Truncated corporate-shell extraction.","triples":[]}',
                max_retries=max_retries,
                use_schema=use_schema,
                ontology_version="canonical",
            )
        except ExtractionError as exc:
            return CanonicalPipelineResult(
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
            ontology_version="canonical",
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
            'Return ONLY a raw JSON object containing "extraction_notes" and the "triples" array.\n'
            "Do not use markdown code blocks (```json).\n"
            "</format_reminder>"
        )
        final_extraction, raw_final_reflection_response, final_reflection_attempts_used, final_reflection_audit = self.reflect_extraction(
            full_text=full_text,
            current_extraction=pre_reflection_extraction,
            company_name=company_name,
            max_retries=max_retries,
            strict=False,
            use_schema=use_schema,
            system_prompt=_canonical_reflection_system_prompt(full_text),
            user_prompt=final_reflection_prompt,
            ontology_version="canonical",
        )

        return CanonicalPipelineResult(
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
