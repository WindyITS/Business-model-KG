from __future__ import annotations

ROUTER_LABELS = ("api_fallback", "local", "refuse")
ROUTE_TO_ROUTER_LABEL = {
    "local_safe": "local",
    "strong_model_candidate": "api_fallback",
    "refuse": "refuse",
}
PLANNER_FAMILY_KEY = "family"
