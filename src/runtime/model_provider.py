import os
from dataclasses import dataclass
from typing import Literal


ApiMode = Literal["chat_completions", "messages"]

LOCAL_PROVIDER = "local"
OPENCODE_GO_PROVIDER = "opencode-go"

DEFAULT_LOCAL_BASE_URL = "http://localhost:1234/v1"
DEFAULT_LOCAL_MODEL = "local-model"
DEFAULT_OPENCODE_GO_MODEL = "kimi-k2.5"
OPENCODE_GO_DEFAULT_MAX_OUTPUT_TOKENS = 20000

OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"

SUPPORTED_PROVIDERS = (
    LOCAL_PROVIDER,
    OPENCODE_GO_PROVIDER,
)
SUPPORTED_API_MODES = ("chat_completions", "messages")
SUPPORTED_ENDPOINT_FAMILIES = {"chat_completions", "messages"}

OPENCODE_GO_MODEL_ALIASES = {
    "kimi-k2.5": "kimi-k2.5",
    "kimi k2.5": "kimi-k2.5",
    "mimo-v2-pro": "mimo-v2-pro",
    "mimo v2 pro": "mimo-v2-pro",
    "minimax-m2.7": "minimax-m2.7",
    "minimax m2.7": "minimax-m2.7",
}

OPENCODE_GO_MODEL_ENDPOINTS = {
    "kimi-k2.5": "chat_completions",
    "mimo-v2-pro": "chat_completions",
    "minimax-m2.7": "messages",
}


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    model: str
    base_url: str
    api_key: str
    api_mode: ApiMode
    max_output_tokens: int | None


def normalize_provider_name(provider: str) -> str:
    return provider.strip().lower()


def normalize_model_name(provider: str, model: str) -> str:
    normalized_model = model.strip()
    if provider != OPENCODE_GO_PROVIDER:
        return normalized_model

    if normalized_model.lower().startswith(f"{OPENCODE_GO_PROVIDER}/"):
        normalized_model = normalized_model[len(OPENCODE_GO_PROVIDER) + 1 :]

    alias_key = " ".join(normalized_model.replace("_", " ").strip().lower().split())
    return OPENCODE_GO_MODEL_ALIASES.get(alias_key, normalized_model.strip().lower())


def normalize_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return base_url

    normalized = base_url.rstrip("/")
    for suffix in ("/chat/completions", "/messages"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _resolve_api_key(explicit_api_key: str | None, env_names: tuple[str, ...], default: str | None = None) -> str:
    if explicit_api_key:
        return explicit_api_key

    for env_name in env_names:
        value = os.getenv(env_name)
        if value:
            return value

    if default is not None:
        return default

    raise ValueError(f"Missing API key. Pass --api-key or set one of: {', '.join(env_names)}")


def _resolve_api_mode(
    *,
    provider: str,
    model: str,
    explicit_api_mode: str | None,
) -> ApiMode:
    if explicit_api_mode is not None and explicit_api_mode not in SUPPORTED_API_MODES:
        raise ValueError(f"Unsupported API mode: {explicit_api_mode}")

    endpoint_family = None
    if provider == OPENCODE_GO_PROVIDER:
        endpoint_family = OPENCODE_GO_MODEL_ENDPOINTS.get(model)

    if endpoint_family is not None and endpoint_family not in SUPPORTED_ENDPOINT_FAMILIES:
        raise ValueError(
            f"Model {model!r} on provider {provider!r} uses the {endpoint_family!r} endpoint family, "
            "but this pipeline currently supports only chat/completions and messages endpoints."
        )

    if endpoint_family is not None:
        if explicit_api_mode is not None and explicit_api_mode != endpoint_family:
            raise ValueError(
                f"Model {model!r} on provider {provider!r} requires api mode {endpoint_family!r}, "
                f"not {explicit_api_mode!r}."
            )
        return endpoint_family

    if explicit_api_mode is not None:
        return explicit_api_mode  # type: ignore[return-value]

    return "chat_completions"


def resolve_model_settings(
    *,
    provider: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    max_output_tokens: int | None = None,
    api_mode: str | None = None,
) -> ModelSettings:
    normalized_provider = normalize_provider_name(provider)
    if normalized_provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")

    if normalized_provider == LOCAL_PROVIDER:
        resolved_model = normalize_model_name(normalized_provider, model or DEFAULT_LOCAL_MODEL)
        resolved_base_url = normalize_base_url(base_url) or DEFAULT_LOCAL_BASE_URL
        resolved_api_key = _resolve_api_key(
            api_key,
            ("LOCAL_LLM_API_KEY", "LM_STUDIO_API_KEY"),
            default="lm-studio",
        )
    else:
        resolved_model = normalize_model_name(normalized_provider, model or DEFAULT_OPENCODE_GO_MODEL)
        if resolved_model not in OPENCODE_GO_MODEL_ENDPOINTS:
            supported_models = ", ".join(sorted(OPENCODE_GO_MODEL_ENDPOINTS))
            raise ValueError(
                f"Unsupported opencode-go model: {resolved_model!r}. Supported models: {supported_models}."
            )
        resolved_base_url = normalize_base_url(base_url) or OPENCODE_GO_BASE_URL
        resolved_api_key = _resolve_api_key(
            api_key,
            ("OPENCODE_GO_API_KEY", "OPENCODE_API_KEY"),
        )

    resolved_api_mode = _resolve_api_mode(
        provider=normalized_provider,
        model=resolved_model,
        explicit_api_mode=api_mode,
    )
    return ModelSettings(
        provider=normalized_provider,
        model=resolved_model,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        api_mode=resolved_api_mode,
        max_output_tokens=max_output_tokens
        if max_output_tokens is not None
        else (OPENCODE_GO_DEFAULT_MAX_OUTPUT_TOKENS if normalized_provider == OPENCODE_GO_PROVIDER else None),
    )
