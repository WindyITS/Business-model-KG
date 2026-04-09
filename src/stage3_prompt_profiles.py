from dataclasses import dataclass


@dataclass(frozen=True)
class Stage3PromptProfile:
    name: str
    system_identity_block: str
    system_quality_block: str
    system_evidence_block: str
    decision_standard_prefix: tuple[str, ...]
    burden_of_proof_lines: tuple[str, ...]
    user_instruction_lines: tuple[str, ...]


DEFAULT_PROFILE = Stage3PromptProfile(
    name="default",
    system_identity_block=(
        "You are a strict Information Extraction system. "
        "Your task is to extract one requested relation from SEC 10-K text."
    ),
    system_quality_block="Use the allowed label list as your checklist.",
    system_evidence_block=(
        "Extract only when the relation is explicitly supported by the text.\n"
        "The output object label must be an exact match from the allowed list.\n"
        "If multiple labels are explicitly supported, return one triple for each in the same JSON array.\n"
        "Do not extract from descriptions of what a company does, has, or operates. "
        "Extract only from statements about how it sells, who it sells to, or how it earns money."
    ),
    decision_standard_prefix=(
        "Evaluate the text against the allowed object labels.",
    ),
    burden_of_proof_lines=(
        "Use a subject from <allowed_subjects>.",
        "If the reporting company is referred to generically, use <company_name>.",
        "Do not output duplicates.",
        "If nothing is explicitly supported, return {\"triples\":[]}.",
    ),
    user_instruction_lines=(
        "Analyze the text and extract all explicitly supported triples for the requested relation.",
        "Output ONLY raw JSON.",
        "Do not output markdown code blocks.",
        "Do not output explanations.",
    ),
)


PROMPT_PROFILES = {
    DEFAULT_PROFILE.name: DEFAULT_PROFILE,
}

DEFAULT_PROMPT_PROFILE = DEFAULT_PROFILE.name


def get_prompt_profile(name: str) -> Stage3PromptProfile:
    try:
        return PROMPT_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPT_PROFILES))
        raise ValueError(f"Unknown Stage 3 prompt profile '{name}'. Available: {available}") from exc


def list_prompt_profiles() -> list[str]:
    return sorted(PROMPT_PROFILES)
