import json

from llm_extraction.prompting import pipeline_prompt_dir, render_prompt
from ontology.config import canonical_labels, load_ontology_config


CANONICAL_CUSTOMER_TYPES = canonical_labels("CustomerType")
CANONICAL_CHANNELS = canonical_labels("Channel")
CANONICAL_REVENUE_MODELS = canonical_labels("RevenueModel")
CANONICAL_DEFINITIONS = load_ontology_config()["canonical_labels"]
APPROVED_MACRO_REGIONS = [
    "Africa",
    "APAC",
    "Americas",
    "Asia",
    "Asia Pacific",
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

PROMPT_DIR = pipeline_prompt_dir("zero-shot")


def _json_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


def _definition_lines(definitions: dict[str, str]) -> str:
    return "\n".join(f'- "{label}": {definition}' for label, definition in definitions.items())


def _render(prompt_name: str, **context: object) -> str:
    return render_prompt(PROMPT_DIR / prompt_name, **context)


def zero_shot_extraction_prompt(full_text: str, company_name: str | None) -> str:
    return _render(
        "extract.txt",
        full_text=full_text,
        company_name=company_name or "",
        canonical_customer_types_json=_json_list(CANONICAL_CUSTOMER_TYPES),
        canonical_channels_json=_json_list(CANONICAL_CHANNELS),
        canonical_revenue_models_json=_json_list(CANONICAL_REVENUE_MODELS),
        customer_type_definitions=_definition_lines(CANONICAL_DEFINITIONS["CustomerType"]),
        channel_definitions=_definition_lines(CANONICAL_DEFINITIONS["Channel"]),
        revenue_model_definitions=_definition_lines(CANONICAL_DEFINITIONS["RevenueModel"]),
        approved_macro_regions_json=_json_list(APPROVED_MACRO_REGIONS),
    )
