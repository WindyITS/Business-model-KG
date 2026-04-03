import json
import logging
from typing import Any, List, Literal

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
    "SUPPLIED_BY",
    "MONETIZES_VIA",
    "PART_OF",
]


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


# =============================================================================
# STRENGTHENED ONTOLOGY (used in every prompt)
# =============================================================================
STRICT_ONTOLOGY = """=== STRICT ONTOLOGY (Exact Strings Only - Zero Deviation) ===

NODE TYPE DEFINITIONS:
- "Company": The primary reporting entity OR any explicitly named partner, supplier, or competitor.
- "BusinessSegment": ONLY an explicitly named formal reporting segment (e.g. "Commercial", "Government") or a clearly labeled portfolio group presented as such in the filing.
- "Offering": ONLY a specific, explicitly named product, service, brand, or platform (e.g. "Dynamics 365", "GitHub", "Azure AI Foundry", "Gotham"). Never broad categories.
- "CustomerType": Granular, hyper-specific buyer profiles exactly as written (e.g. "Utility Operations Analysts", "developers"). Must be split when listed separately.
- "Channel": A distinct method of distribution or sales explicitly named.
- "Place": Explicit geographical markers (countries, regions, cities) named in the filing.
- "RevenueModel": MUST be one of the exact values below.

ALLOWED RELATIONS (subject → object):
- HAS_SEGMENT: Company → BusinessSegment
- OFFERS: Company → Offering | BusinessSegment → Offering
- PART_OF: Offering → BusinessSegment
- SERVES: Company → CustomerType | Offering → CustomerType
- OPERATES_IN: Company → Place | BusinessSegment → Place
- SELLS_THROUGH: Company → Channel | Offering → Channel
- PARTNERS_WITH: Company → Company
- SUPPLIED_BY: Company → Company | Offering → Company
- MONETIZES_VIA: Company → RevenueModel | BusinessSegment → RevenueModel | Offering → RevenueModel

REVENUE MODEL VALUES (exact match only):
"subscription", "advertising", "licensing", "consumption-based", "hardware sales", "service fees", "royalties", "transaction fees".
"""

# =============================================================================
# MODE-SPECIFIC SYSTEM PROMPTS (clean, self-contained, optimized for Gemma 4 31B)
# =============================================================================

CHUNKED_SYSTEM_PROMPT = f"""You are an expert data engineer extracting precise business-model Knowledge Graphs from SEC 10-K chunks.

{STRICT_ONTOLOGY}

=== OUTPUT SCHEMA (MANDATORY) ===
Output ONLY valid JSON — nothing else:
{{
  "extraction_notes": "terse 1-2 sentence summary, max 40 words",
  "triples": [array of triples]
}}

=== INPUT FORMAT ===
You will receive <text_to_analyze> containing a single CHUNK of a document.
<company_name> (optional): main reporting company.
<memory> (optional): context from earlier chunks.

=== CRITICAL RULES ===
- ONLY extract relationships found EXPLICITLY in this specific chunk. Do not hallucinate based on outside knowledge.
- If the chunk contains no clear relationships, return an empty triples array: [].
- NAMED OFFERINGS ONLY: Ignore broad categories (e.g. "cloud services"). Focus on specific nouns (e.g. "Dynamics 365", "Azure").
- COMPOSITE CUSTOMERS: Split lists (e.g. "developers, IT professionals, and enterprises") into separate CustomerType nodes.
- HIERARCHY RULE: Prefer BusinessSegment → OFFERS → Offering.
- Precision over recall. Avoid ambiguous connections.

=== SILENT REASONING (never output):
1. Find exact entities in this chunk.
2. Build candidate triples.
3. Compare with memory to avoid duplicates and resolve canonical names.
4. Filter out generic or weak connections.

Output ONLY the JSON."""

ZERO_SHOT_SYSTEM_PROMPT = f"""You are an expert data engineer extracting comprehensive business-model Knowledge Graphs from FULL SEC 10-K filings.

{STRICT_ONTOLOGY}

=== OUTPUT SCHEMA (MANDATORY) ===
Output ONLY valid JSON — nothing else:
{{
  "extraction_notes": "terse 1-2 sentence summary, max 60 words",
  "triples": [array of triples]
}}

=== INPUT FORMAT ===
You will receive the COMPLETE document in <text_to_analyze>. 

=== CRITICAL RULES ===
- BE SYSTEMATIC: Because the document is long, you must scan thoroughly. Do not stop extracting halfway. Look for all Operating Segments, explicit Products/Offerings, and Customer segments.
- EXHAUSTIVE BUT PRECISE: Extract all specific named Offerings (e.g. "Azure", "GitHub", "Bing"), but ignore generic marketing-speak (e.g. "AI solutions"). 
- CUSTOMER SPLIT: Separate lists of customers into distinct CustomerType nodes.
- MAXIMIZE RECALL: Aim to capture the complete hierarchy: Company -> Segments -> Offerings -> Customers.

=== SILENT REASONING (never output):
1. Document all BusinessSegments mentioned.
2. Scan the entire filing for every specific Offering named and link them to their segments.
3. Identify CustomerTypes served by those Offerings.
4. Construct all triples strictly respecting the ontology.

Output ONLY the JSON."""

INCREMENTAL_SYSTEM_PROMPT = f"""You are an expert data engineer progressively extracting business-model Knowledge Graphs from a large SEC 10-K filing using a cursor-based approach.

{STRICT_ONTOLOGY}

=== OUTPUT SCHEMA (MANDATORY) ===
Output ONLY valid JSON — nothing else:
{{
  "current_section_analyzed": "Header of the section you are extracting from now",
  "next_section_to_analyze": "Header of the immediate next section you will extract from next (or 'NONE')",
  "extraction_notes": "terse summary of what you found. max 40 words.",
  "triples": [array of triples],
  "has_reached_end_of_document": boolean
}}

=== INCREMENTAL MODE RULES ===
- CURSOR APPROACH: The document is huge. You must maintain a precise reading cursor. In your first response, analyze ONLY the very first section. Identify the next section to analyze and put it in "next_section_to_analyze".
- In the following turns, pick up EXACTLY from your previous "next_section_to_analyze".
- DO NOT SKIM OR SKIP: Proceed linearly section by section.
- NO PLACEHOLDERS: Never output "subject_id_1" or "object_id_2". Use exact company or product names.
- NAMED OFFERINGS ONLY: Ignore generic "software", "solutions", or "cloud services". We only want explicitly named products like "Microsoft Azure" or "Dynamics 365" or "Windows". Do not extract broad categories.
- END CONDITION: set "has_reached_end_of_document": true ONLY when you have processed the very last word of the provided text.

=== SILENT REASONING (never output):
1. Recall the "next_section_to_analyze" from your previous response (if any).
2. Locate that section in the original <text_to_analyze> and limit your extraction strictly to it.
3. Identify exactly what section comes after it.
4. Output the JSON.

Output ONLY the JSON."""

# =============================================================================
# LLM EXTRACTOR CLASS
# =============================================================================

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

    def _get_system_prompt(self, extraction_mode: str) -> str:
        if extraction_mode == "chunked":
            return CHUNKED_SYSTEM_PROMPT
        elif extraction_mode == "zero_shot":
            return ZERO_SHOT_SYSTEM_PROMPT
        elif extraction_mode == "incremental":
            return INCREMENTAL_SYSTEM_PROMPT
        else:
            return CHUNKED_SYSTEM_PROMPT  # fallback

    def extract_from_chunk_detailed(
        self,
        chunk_text: str,
        company_name: str | None = None,
        memory: dict[str, Any] | None = None,
        extraction_mode: str = "chunked",
        max_retries: int = 2,
        strict: bool = True,
    ) -> ChunkExtractionResult:
        prompt_parts = []
        if company_name:
            prompt_parts.append(f"<company_name>\n{company_name}\n</company_name>")
        if memory:
            prompt_parts.append(
                "<memory>\n"
                f"{json.dumps(memory, ensure_ascii=False, indent=2)}\n"
                "</memory>"
            )
        prompt_parts.append(f"<text_to_analyze>\n{chunk_text}\n</text_to_analyze>")
        prompt = "\n\n".join(prompt_parts)

        schema_def = self._schema_def("KnowledgeGraphExtraction", KnowledgeGraphExtraction)
        fallback_payload = '{"extraction_notes": "Truncated before extractions.", "triples": []}'

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt(extraction_mode)},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    response_format=schema_def,
                )
                content = response.choices[0].message.content
                parsed_data = self._load_json_payload(content or "", fallback_payload)
                extraction = KnowledgeGraphExtraction(**parsed_data)
                return ChunkExtractionResult(
                    success=True,
                    extraction=extraction,
                    raw_response=content,
                    attempts_used=attempt,
                )
            except (json.JSONDecodeError, ValidationError, ExtractionError) as exc:
                logger.warning("Chunk extraction failed on attempt %s/%s: %s", attempt, max_retries, exc)
            except Exception as exc:
                logger.warning("LLM API error on attempt %s/%s: %s", attempt, max_retries, exc)

        error_message = f"Failed after {max_retries} attempts"
        if strict:
            raise ExtractionError(error_message)
        return ChunkExtractionResult(
            success=False,
            extraction=KnowledgeGraphExtraction(extraction_notes="", triples=[]),
            raw_response=None,
            error=error_message,
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
            '{"extraction_notes": "Truncated before extractions.", "triples": [], '
            '"has_reached_end_of_document": false}'
        )

        for iteration in range(max_iterations):
            if iteration == 0:
                prompt = (
                    "Here is the complete document. You will be guided to extract progressively. "
                    "For this FIRST response, extract triples ONLY from the very first section of the document. "
                    "Set has_reached_end_of_document to false.\n\n"
                    f"<text_to_analyze>\n{full_text}\n</text_to_analyze>"
                )
            else:
                prompt = (
                    "Look at your previous response. Move your reading cursor to the exact section you specified in 'next_section_to_analyze'. "
                    "Extract ONLY new triples from that specific section. Then determine the next section after that. "
                    "If 'next_section_to_analyze' was 'NONE' or you've reached the absolute end of the document, set has_reached_end_of_document to true."
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