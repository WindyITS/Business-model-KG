import re
from functools import lru_cache
from pathlib import Path


PLACEHOLDER_RE = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")


@lru_cache(maxsize=None)
def _load_template(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").rstrip("\n")


def render_template(path: Path, **context: object) -> str:
    template = _load_template(str(path))

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            raise KeyError(f"Missing template value for {key!r} in {path}")
        return str(context[key])

    return PLACEHOLDER_RE.sub(replace, template)
