import json

from llm_extraction.prompting import pipeline_prompt_dir, render_prompt
from ontology.config import canonical_labels, load_ontology_config


CANONICAL_CUSTOMER_TYPES = canonical_labels("CustomerType")
CANONICAL_CHANNELS = canonical_labels("Channel")
CANONICAL_REVENUE_MODELS = canonical_labels("RevenueModel")
CANONICAL_DEFINITIONS = load_ontology_config()["canonical_labels"]

PROMPT_DIR = pipeline_prompt_dir("memo_graph_only")


def _json_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


def _definition_lines(definitions: dict[str, str]) -> str:
    return "\n".join(f'- "{label}": {definition}' for label, definition in definitions.items())


def _render(prompt_name: str, **context: object) -> str:
    return render_prompt(PROMPT_DIR / prompt_name, **context)


def memo_graph_only_pipeline_system_prompt(full_text: str) -> str:
    return _render(
        "system.txt",
        full_text=full_text,
        canonical_customer_types_json=_json_list(CANONICAL_CUSTOMER_TYPES),
        canonical_channels_json=_json_list(CANONICAL_CHANNELS),
        canonical_revenue_models_json=_json_list(CANONICAL_REVENUE_MODELS),
        customer_type_definitions=_definition_lines(CANONICAL_DEFINITIONS["CustomerType"]),
        channel_definitions=_definition_lines(CANONICAL_DEFINITIONS["Channel"]),
        revenue_model_definitions=_definition_lines(CANONICAL_DEFINITIONS["RevenueModel"]),
    )


def memo_graph_only_graph_system_prompt() -> str:
    return _render(
        "graph_system.txt",
        canonical_customer_types_json=_json_list(CANONICAL_CUSTOMER_TYPES),
        canonical_channels_json=_json_list(CANONICAL_CHANNELS),
        canonical_revenue_models_json=_json_list(CANONICAL_REVENUE_MODELS),
        customer_type_definitions=_definition_lines(CANONICAL_DEFINITIONS["CustomerType"]),
        channel_definitions=_definition_lines(CANONICAL_DEFINITIONS["Channel"]),
        revenue_model_definitions=_definition_lines(CANONICAL_DEFINITIONS["RevenueModel"]),
    )


def memo_graph_only_memo_foundation_prompt(company_name: str | None) -> str:
    return _render("memo_foundation.txt", company_name=company_name or "")


def memo_graph_only_graph_compilation_prompt(company_name: str | None, analyst_memo: str) -> str:
    return _render(
        "graph_compilation.txt",
        company_name=company_name or "",
        analyst_memo=analyst_memo,
    )
