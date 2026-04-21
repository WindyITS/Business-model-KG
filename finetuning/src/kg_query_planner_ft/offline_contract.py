from __future__ import annotations

from typing import Any


def normalize_query_plan_contract(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    answerable = payload.get("answerable")
    if not isinstance(answerable, bool):
        return None

    if answerable:
        family = payload.get("family")
        plan_payload = payload.get("payload")
        if not isinstance(family, str) or not family.strip():
            return None
        if not isinstance(plan_payload, dict):
            return None
        if payload.get("reason") is not None:
            return None
        return {
            "answerable": True,
            "family": family,
            "payload": plan_payload,
        }

    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return None
    if payload.get("family") is not None or payload.get("payload") is not None:
        return None
    return {
        "answerable": False,
        "reason": reason,
    }


def validate_query_plan_contract(payload: Any) -> bool:
    return normalize_query_plan_contract(payload) is not None


__all__ = ["normalize_query_plan_contract", "validate_query_plan_contract"]
