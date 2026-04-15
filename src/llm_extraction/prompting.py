import re
from functools import lru_cache
from pathlib import Path


PLACEHOLDER_RE = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")
PROMPTS_ROOT = Path(__file__).resolve().parents[2] / "prompts"


@lru_cache(maxsize=None)
def _load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").rstrip("\n")


def prompt_root() -> Path:
    return PROMPTS_ROOT


def pipeline_prompt_dir(pipeline_name: str) -> Path:
    path = prompt_root() / pipeline_name
    if not path.is_dir():
        raise FileNotFoundError(f"Prompt directory not found for pipeline {pipeline_name!r}: {path}")
    return path


def prompt_path(pipeline_name: str, prompt_name: str) -> Path:
    path = pipeline_prompt_dir(pipeline_name) / prompt_name
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found for pipeline {pipeline_name!r}: {path}")
    return path


def render_prompt(path: Path, **context: object) -> str:
    template = _load_prompt(str(path))

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            raise KeyError(f"Missing template value for {key!r} in {path}")
        return str(context[key])

    return PLACEHOLDER_RE.sub(replace, template)


# Backward-compatible alias for older call sites.
render_template = render_prompt
