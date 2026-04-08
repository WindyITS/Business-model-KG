import json
import logging
from typing import Any, List, Literal

from ontology_config import canonical_labels
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
    raw_skeleton_response: str | None = None
    raw_final_response: str | None = None
    skeleton_attempts_used: int = 0
    final_attempts_used: int = 0
    error: str | None = None

    @property
    def re_extraction(self) -> KnowledgeGraphExtraction:
        return self.final_extraction


class ChunkExtractionResult(BaseModel):
    success: bool
    extraction: KnowledgeGraphExtraction
    raw_response: str | None = None
    error: str | None = None
    attempts_used: int = 0


class IncrementalExtractionResult(BaseModel):
    success: bool
    extractions: List[KnowledgeGraphExtraction] = Field(default_factory=list)
    raw_responses: List[str] = Field(default_factory=list)
    iterations: int = 0
    error: str | None = None
    failed_iteration: int | None = None


class ExtractionError(RuntimeError):
    pass


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


CHUNKED_SYSTEM_PROMPT = f"""You extract a strict business-model knowledge graph from a single SEC 10-K chunk.

{PROMPT_ONTOLOGY}

TASK:
- Extract only explicit triples supported by this chunk.
- Prefer structural business-model facts first: HAS_SEGMENT, OFFERS, PART_OF.
- Then extract explicit OPERATES_IN, PARTNERS_WITH, SERVES, SELLS_THROUGH, and MONETIZES_VIA facts.

RULES:
- Ignore generic offerings like "software", "solutions", or "cloud services".
- Normalize CustomerType, Channel, and RevenueModel to canonical labels only.
- If a canonical mapping is not clear, omit the fact.
- Return an empty triple list when the chunk contains nothing relevant.

OUTPUT:
Return ONLY valid JSON matching the KnowledgeGraphExtraction schema.
"""


ZERO_SHOT_SYSTEM_PROMPT = f"""You extract a strict, standardized business-model knowledge graph from a full SEC 10-K filing.

{PROMPT_ONTOLOGY}

WORK ORDER:
1. Identify all named BusinessSegments.
2. Identify all named Offerings.
3. Build the structural graph skeleton:
   - Company -> HAS_SEGMENT -> BusinessSegment
   - BusinessSegment -> OFFERS -> Offering
   - Offering -> PART_OF -> BusinessSegment
4. Add explicit OPERATES_IN and PARTNERS_WITH facts.
5. Add SERVES, SELLS_THROUGH, and MONETIZES_VIA only when a canonical label can be assigned unambiguously.

RULES:
- Build the business-model skeleton before enrichment.
- For every offering that clearly belongs to a named segment, emit BOTH OFFERS and PART_OF.
- Search go-to-market sections for company-level channels.
- Search demand/revenue descriptions for monetization only when the mapping is clear.
- If a fact is ambiguous or outside the ontology, omit it.

OUTPUT:
Return ONLY valid JSON matching the KnowledgeGraphExtraction schema.
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


REFLECTION_SYSTEM_PROMPT = f"""You are reviewing an existing business-model knowledge graph extraction from a SEC 10-K filing.

{PROMPT_ONTOLOGY}

INPUTS:
- <current_triples>: the current extracted graph
- <text_to_analyze>: the source filing text

TASK:
- Keep correct triples.
- Remove weak or unsupported triples.
- Add clearly missing triples supported by the text.
- Preserve the structural graph skeleton before enrichment.
- Use canonical labels only for CustomerType, Channel, and RevenueModel.

OUTPUT:
Return ONLY valid JSON matching the KnowledgeGraphExtraction schema.
"""


INCREMENTAL_SYSTEM_PROMPT = f"""You progressively extract a strict business-model knowledge graph from a large SEC 10-K filing.

{PROMPT_ONTOLOGY}

RULES:
- Analyze one section at a time.
- Build structural triples first, then enrich when explicit.
- Use canonical labels only for CustomerType, Channel, and RevenueModel.
- If a canonical mapping is unclear, omit it.

OUTPUT:
Return ONLY valid JSON matching the IncrementalKnowledgeGraphExtraction schema.
"""


class LLMExtractor:
    def __init__(self, base_url: str = "http://localhost:1234/v1", api_key: str = "lm-studio", model: str = "local-model"):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    @staticmethod
    def _schema_def(name: str, model_schema: type[BaseModel]) -> dict:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": model_schema.model_json_schema(by_alias=True),
            },
        }

    @staticmethod
    def _compact_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _load_json_payload(content: str, fallback_payload: str) -> dict:
        content = content.strip()
        if not content:
            raise ExtractionError("Empty response from model.")

        if not content.endswith("}"):
            logger.warning("Truncated JSON detected. Attempting to salvage...")
            last_object_end = content.rfind("}")
            content = content[: last_object_end + 1] if last_object_end != -1 else fallback_payload

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return json.loads(fallback_payload)

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
    ) -> tuple[BaseModel, str | None, int]:
        schema_def = self._schema_def(schema_name, schema_model)

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
                parsed = self._load_json_payload(content or "", fallback_payload)
                return schema_model(**parsed), content, attempt
            except (json.JSONDecodeError, ValidationError, ExtractionError) as exc:
                logger.warning("Structured call failed on attempt %s/%s: %s", attempt, max_retries, exc)
            except Exception as exc:
                logger.warning("LLM API error on attempt %s/%s: %s", attempt, max_retries, exc)

        raise ExtractionError(f"Failed after {max_retries} attempts")

    def _get_system_prompt(self, extraction_mode: str) -> str:
        if extraction_mode == "zero_shot":
            return ZERO_SHOT_SYSTEM_PROMPT
        if extraction_mode == "incremental":
            return INCREMENTAL_SYSTEM_PROMPT
        return CHUNKED_SYSTEM_PROMPT

    def extract_from_chunk_detailed(
        self,
        chunk_text: str,
        company_name: str | None = None,
        memory: dict[str, Any] | None = None,
        extraction_mode: str = "chunked",
        max_retries: int = 2,
        strict: bool = True,
    ) -> ChunkExtractionResult:
        prompt_parts: list[str] = []
        if company_name:
            prompt_parts.append(f"<company_name>\n{company_name}\n</company_name>")
        if memory:
            prompt_parts.append(f"<memory>\n{self._compact_json(memory)}\n</memory>")
        prompt_parts.append(f"<text_to_analyze>\n{chunk_text}\n</text_to_analyze>")
        prompt = "\n\n".join(prompt_parts)

        try:
            extraction, raw_response, attempts_used = self._call_structured(
                system_prompt=self._get_system_prompt(extraction_mode),
                user_prompt=prompt,
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Truncated before extractions.","triples":[]}',
                max_retries=max_retries,
            )
            return ChunkExtractionResult(
                success=True,
                extraction=extraction,
                raw_response=raw_response,
                attempts_used=attempts_used,
            )
        except ExtractionError as exc:
            if strict:
                raise
            return ChunkExtractionResult(
                success=False,
                extraction=KnowledgeGraphExtraction(extraction_notes="", triples=[]),
                raw_response=None,
                error=str(exc),
                attempts_used=max_retries,
            )

    def extract_from_chunk(
        self,
        chunk_text: str,
        company_name: str | None = None,
        memory: dict[str, Any] | None = None,
        extraction_mode: str = "chunked",
        max_retries: int = 2,
    ) -> KnowledgeGraphExtraction:
        return self.extract_from_chunk_detailed(
            chunk_text=chunk_text,
            company_name=company_name,
            memory=memory,
            extraction_mode=extraction_mode,
            max_retries=max_retries,
            strict=True,
        ).extraction

    def extract_two_pass_detailed(
        self,
        full_text: str,
        company_name: str | None = None,
        max_retries: int = 2,
        strict: bool = True,
    ) -> TwoPassExtractionResult:
        prompt_parts: list[str] = []
        if company_name:
            prompt_parts.append(f"<company_name>\n{company_name}\n</company_name>")
        prompt_parts.append(f"<text_to_analyze>\n{full_text}\n</text_to_analyze>")
        base_prompt = "\n\n".join(prompt_parts)

        try:
            skeleton_extraction, raw_skeleton_response, skeleton_attempts_used = self._call_structured(
                system_prompt=SKELETON_SYSTEM_PROMPT,
                user_prompt=base_prompt,
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Truncated skeleton extraction.","triples":[]}',
                max_retries=max_retries,
            )
        except ExtractionError as exc:
            if strict:
                raise
            return TwoPassExtractionResult(
                success=False,
                error=str(exc),
            )

        enrichment_prompt = (
            f"<company_name>\n{company_name or ''}\n</company_name>\n\n"
            f"<existing_triples>\n{self._compact_json(skeleton_extraction.model_dump())}\n</existing_triples>\n\n"
            f"<text_to_analyze>\n{full_text}\n</text_to_analyze>"
        )

        try:
            final_extraction, raw_final_response, final_attempts_used = self._call_structured(
                system_prompt=ENRICHMENT_SYSTEM_PROMPT,
                user_prompt=enrichment_prompt,
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Truncated final extraction.","triples":[]}',
                max_retries=max_retries,
            )
        except ExtractionError as exc:
            if strict:
                raise
            return TwoPassExtractionResult(
                success=False,
                skeleton_extraction=skeleton_extraction,
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
            raw_skeleton_response=raw_skeleton_response,
            raw_final_response=raw_final_response,
            skeleton_attempts_used=skeleton_attempts_used,
            final_attempts_used=final_attempts_used,
        )

    def extract_two_pass(
        self,
        full_text: str,
        company_name: str | None = None,
        max_retries: int = 2,
    ) -> KnowledgeGraphExtraction:
        result = self.extract_two_pass_detailed(
            full_text=full_text,
            company_name=company_name,
            max_retries=max_retries,
            strict=True,
        )
        return result.final_extraction

    def extract_with_reflection(
        self,
        full_text: str,
        company_name: str | None = None,
        max_retries: int = 2,
    ) -> tuple[TwoPassExtractionResult, KnowledgeGraphExtraction]:
        two_pass_result = self.extract_two_pass_detailed(
            full_text=full_text,
            company_name=company_name,
            max_retries=max_retries,
            strict=True,
        )

        reflection_prompt = (
            f"<company_name>\n{company_name or ''}\n</company_name>\n\n"
            f"<current_triples>\n{self._compact_json(two_pass_result.final_extraction.model_dump())}\n</current_triples>\n\n"
            f"<text_to_analyze>\n{full_text}\n</text_to_analyze>"
        )

        try:
            final_extraction, _, _ = self._call_structured(
                system_prompt=REFLECTION_SYSTEM_PROMPT,
                user_prompt=reflection_prompt,
                schema_name="KnowledgeGraphExtraction",
                schema_model=KnowledgeGraphExtraction,
                fallback_payload='{"extraction_notes":"Reflection failed.","triples":[]}',
                max_retries=max_retries,
            )
            if not final_extraction.triples:
                logger.warning("Reflection returned no triples. Falling back to two-pass output.")
                return two_pass_result, two_pass_result.final_extraction
            return two_pass_result, final_extraction
        except ExtractionError:
            logger.warning("Reflection failed. Falling back to two-pass output.")
            return two_pass_result, two_pass_result.final_extraction

    def extract_incremental_detailed(
        self,
        full_text: str,
        max_iterations: int = 10,
        max_retries: int = 2,
        strict: bool = True,
    ) -> IncrementalExtractionResult:
        all_extractions: List[KnowledgeGraphExtraction] = []
        raw_responses: List[str] = []
        messages = [{"role": "system", "content": INCREMENTAL_SYSTEM_PROMPT}]
        schema_def = self._schema_def("IncrementalKnowledgeGraphExtraction", IncrementalKnowledgeGraphExtraction)
        fallback_payload = (
            '{"extraction_notes":"Truncated before extractions.","triples":[],"has_reached_end_of_document":false}'
        )

        for iteration in range(max_iterations):
            if iteration == 0:
                prompt = (
                    "Here is the complete document. For this first response, extract triples only from the first section.\n\n"
                    f"<text_to_analyze>\n{full_text}\n</text_to_analyze>"
                )
            else:
                prompt = (
                    "Continue from the next section named in your previous response. "
                    "Extract only new triples from that section."
                )

            messages.append({"role": "user", "content": prompt})

            for attempt in range(1, max_retries + 1):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.0,
                        response_format=schema_def,
                    )
                    content = response.choices[0].message.content
                    parsed_data = self._load_json_payload(content or "", fallback_payload)
                    raw_extraction = IncrementalKnowledgeGraphExtraction(**parsed_data)
                    raw_responses.append(content or "")
                    messages.append(
                        {
                            "role": "assistant",
                            "content": json.dumps(raw_extraction.model_dump(mode="json"), ensure_ascii=False),
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
                            raw_responses=raw_responses,
                            iterations=iteration + 1,
                        )
                    break
                except (json.JSONDecodeError, ValidationError, ExtractionError) as exc:
                    logger.warning("Incremental iteration %s attempt %s failed: %s", iteration + 1, attempt, exc)
                except Exception as exc:
                    logger.warning("LLM API error on iteration %s attempt %s: %s", iteration + 1, attempt, exc)
            else:
                error_message = f"Failed iteration {iteration + 1} after {max_retries} attempts"
                if strict:
                    raise ExtractionError(error_message)
                return IncrementalExtractionResult(
                    success=False,
                    extractions=all_extractions,
                    raw_responses=raw_responses,
                    iterations=iteration + 1,
                    error=error_message,
                    failed_iteration=iteration + 1,
                )

        error_message = f"Hit max iterations ({max_iterations}) without reaching end of document"
        if strict:
            raise ExtractionError(error_message)
        return IncrementalExtractionResult(
            success=False,
            extractions=all_extractions,
            raw_responses=raw_responses,
            iterations=max_iterations,
            error=error_message,
            failed_iteration=max_iterations,
        )

    def extract_incremental(
        self,
        full_text: str,
        max_iterations: int = 10,
        max_retries: int = 2,
    ) -> List[KnowledgeGraphExtraction]:
        return self.extract_incremental_detailed(
            full_text=full_text,
            max_iterations=max_iterations,
            max_retries=max_retries,
            strict=True,
        ).extractions
