import json

from llm_extraction.prompting import pipeline_prompt_dir, render_prompt
from ontology.config import canonical_labels, load_ontology_config


CANONICAL_CUSTOMER_TYPES = canonical_labels("CustomerType")
CANONICAL_CHANNELS = canonical_labels("Channel")
CANONICAL_REVENUE_MODELS = canonical_labels("RevenueModel")
CANONICAL_DEFINITIONS = load_ontology_config()["canonical_labels"]

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

PROMPT_DIR = pipeline_prompt_dir("canonical")


def _json_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


def _xml_definition_lines(definitions: dict[str, str]) -> str:
    return "\n".join(f'- "{label}": {definition}' for label, definition in definitions.items())


def _render(prompt_name: str, **context: object) -> str:
    return render_prompt(PROMPT_DIR / prompt_name, **context)


def canonical_pipeline_system_prompt(full_text: str) -> str:
    return _render("system.txt", full_text=full_text)


def canonical_rule_reflection_system_prompt() -> str:
    return _render(
        "rule_reflection_system.txt",
        canonical_customer_types_json=_json_list(CANONICAL_CUSTOMER_TYPES),
        canonical_channels_json=_json_list(CANONICAL_CHANNELS),
        canonical_revenue_models_json=_json_list(CANONICAL_REVENUE_MODELS),
        customer_type_definitions=_xml_definition_lines(CANONICAL_DEFINITIONS["CustomerType"]),
        channel_definitions=_xml_definition_lines(CANONICAL_DEFINITIONS["Channel"]),
        revenue_model_definitions=_xml_definition_lines(CANONICAL_DEFINITIONS["RevenueModel"]),
        approved_macro_regions_json=_json_list(V2_APPROVED_MACRO_REGIONS),
    )


def canonical_reflection_system_prompt(full_text: str) -> str:
    return _render(
        "reflection_system.txt",
        full_text=full_text,
        canonical_customer_types_json=_json_list(CANONICAL_CUSTOMER_TYPES),
        canonical_channels_json=_json_list(CANONICAL_CHANNELS),
        canonical_revenue_models_json=_json_list(CANONICAL_REVENUE_MODELS),
        customer_type_definitions=_xml_definition_lines(CANONICAL_DEFINITIONS["CustomerType"]),
        channel_definitions=_xml_definition_lines(CANONICAL_DEFINITIONS["Channel"]),
        revenue_model_definitions=_xml_definition_lines(CANONICAL_DEFINITIONS["RevenueModel"]),
        approved_macro_regions_json=_json_list(V2_APPROVED_MACRO_REGIONS),
    )


def canonical_pass1_prompt(company_name: str | None) -> str:
    return _render("pass1.txt", company_name=company_name or "")


def canonical_pass2_channels_prompt(current_structure: str) -> str:
    return _render(
        "pass2_channels.txt",
        current_structure=current_structure,
        channel_definitions=_xml_definition_lines(CANONICAL_DEFINITIONS["Channel"]),
    )


def canonical_pass2_revenue_prompt(current_structure: str) -> str:
    return _render(
        "pass2_revenue.txt",
        current_structure=current_structure,
        revenue_model_definitions=_xml_definition_lines(CANONICAL_DEFINITIONS["RevenueModel"]),
    )


def canonical_pass3_serves_prompt(company_name: str | None, current_structure: str) -> str:
    return _render(
        "pass3_serves.txt",
        company_name=company_name or "",
        current_structure=current_structure,
        customer_type_definitions=_xml_definition_lines(CANONICAL_DEFINITIONS["CustomerType"]),
    )


def canonical_pass4_corporate_prompt() -> str:
    return _render(
        "pass4_corporate.txt",
        approved_macro_regions_json=_json_list(V2_APPROVED_MACRO_REGIONS),
    )


def canonical_rule_reflection_prompt(company_name: str | None, current_graph: str) -> str:
    return _render(
        "rule_reflection_user.txt",
        company_name=company_name or "",
        current_graph=current_graph,
    )


def canonical_final_reflection_prompt(company_name: str | None, current_graph: str) -> str:
    return _render(
        "final_reflection_user.txt",
        company_name=company_name or "",
        current_graph=current_graph,
    )
