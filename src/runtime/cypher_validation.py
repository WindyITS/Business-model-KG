from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import urlparse


DISALLOWED_CLAUSE_PATTERNS = (
    r"\bCREATE\b",
    r"\bMERGE\b",
    r"\bDELETE\b",
    r"\bDETACH\b",
    r"\bSET\b",
    r"\bREMOVE\b",
    r"\bCALL\b",
)

PARAM_PATTERN = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
DEFAULT_NEO4J_URI = "bolt://localhost:7687"
BROWSER_PORT_TO_BOLT_PORT = {
    7473: 7687,
    7474: 7687,
}


def normalize_neo4j_uri(uri: str | None) -> str:
    raw = (
        (uri or "").strip()
        or os.getenv("NEO4J_URI", "").strip()
        or os.getenv("NEO4J_URL", "").strip()
        or DEFAULT_NEO4J_URI
    )

    if "://" not in raw:
        host, separator, port_text = raw.rpartition(":")
        if separator and host and port_text.isdigit():
            port = int(port_text)
            bolt_port = BROWSER_PORT_TO_BOLT_PORT.get(port, port)
            return f"bolt://{host}:{bolt_port}"
        return f"bolt://{raw}"

    parsed = urlparse(raw)
    if parsed.scheme in {"bolt", "neo4j", "bolt+s", "bolt+ssc", "neo4j+s", "neo4j+ssc"}:
        return raw

    if parsed.scheme in {"http", "https"}:
        if not parsed.hostname:
            raise ValueError(f"Neo4j URI {raw!r} is missing a hostname")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        bolt_port = BROWSER_PORT_TO_BOLT_PORT.get(port, port)
        return f"bolt://{parsed.hostname}:{bolt_port}"

    raise ValueError(
        "Unsupported Neo4j URI scheme. Use bolt/neo4j URIs directly, or pass a browser URL "
        "such as http://localhost:7474 and it will be normalized."
    )


def validate_read_only_cypher(cypher: str) -> list[str]:
    failures = []
    for pattern in DISALLOWED_CLAUSE_PATTERNS:
        if re.search(pattern, cypher, flags=re.IGNORECASE):
            failures.append(f"Query contains disallowed clause matching {pattern}")
    return failures


def validate_params_match(cypher: str, params: dict[str, Any]) -> list[str]:
    failures = []
    referenced = sorted(set(PARAM_PATTERN.findall(cypher)))
    provided = sorted(params.keys())
    if referenced != provided:
        failures.append(f"Parameter mismatch. Referenced={referenced}, provided={provided}")
    return failures


__all__ = ["normalize_neo4j_uri", "validate_params_match", "validate_read_only_cypher"]
