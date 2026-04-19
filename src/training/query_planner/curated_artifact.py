from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from runtime.query_planner import QueryPlanEnvelope, QueryPlanPayload, compile_query_plan

from .dataset import (
    DatasetExample,
    GRAPH_IDS_BY_SPLIT,
    SURFACE_WRAPPERS_BY_SPLIT,
    _graph_source_id,
    _limit_mentioned,
    _normalized_json_key,
    _ranking_scope_type,
    _split_overlap_stats,
    _summarize_split,
    build_dataset_manifest,
    matching_graph_ids_for_plan,
    write_dataset_splits,
)
from .graphs import SyntheticCompany, build_synthetic_company_graphs, evaluate_query_plan

DEFAULT_BASELINE_DIR = Path("data/query_planner_curated/v1_baseline")
DEFAULT_FINAL_DIR = Path("data/query_planner_curated/v1_final")
DEFAULT_SEED = 7
DEFAULT_TRAIN_SIZE = 8000
DEFAULT_VALIDATION_SIZE = 1200
DEFAULT_RELEASE_EVAL_SIZE = 1800
DEFAULT_SHARD_SIZE = 200

WORKFLOW_ROLES: tuple[dict[str, str], ...] = (
    {
        "role": "Coordinator",
        "responsibility": "Own shard assignment, apply deterministic question-first curation, and merge approved edits.",
    },
    {
        "role": "Verifier",
        "responsibility": "Run runtime-backed validation after every shard merge and before freezing the final package.",
    },
    {
        "role": "Adjudicator",
        "responsibility": "Handle rows that cannot be repaired by rewriting the question alone.",
    },
    {
        "role": "Worker A",
        "responsibility": "Review assigned 200-row shards under the shared curation rubric.",
    },
    {
        "role": "Worker B",
        "responsibility": "Review assigned 200-row shards under the shared curation rubric.",
    },
    {
        "role": "Worker C",
        "responsibility": "Review assigned 200-row shards under the shared curation rubric.",
    },
)

WORKER_ROLES = ("Worker A", "Worker B", "Worker C")

FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "business segments that serves",
    "business segments that sells",
    "satisfy serve ",
    "satisfy sell through ",
    "whose descendants of ",
    "systems's",
    "company portfolio",
    "business-wide request",
    "within the company scope of",
    "when the company scope is",
    "among companies scoped to",
    "be matched to ",
    "looking only at the supplied dataset",
    "operating in canadian operations",
    "operating in german operations",
    "operating in australian operations",
    "which company, limited to ",
    " 1 results",
    " 1 companies",
)

SINGLE_COMPANY_FORBIDDEN_FAMILIES = {
    "companies_by_segment_filters",
    "companies_by_cross_segment_filters",
    "companies_by_descendant_revenue",
    "companies_by_partner",
}

DESCENDANT_WORDING_PATTERN = re.compile(r"\bdescendant offerings?\b|\bdescend(?:s|ing)? from\b", re.IGNORECASE)
TOP_RANKING_PATTERN = re.compile(r"^Top (\d+) (.+?)(?:\.)?$")
REFUSE_DEV_RETAIL_PATTERN = re.compile(
    r"serve developers.+retailers|developers.+exclude retailers|developers but not retailers",
    re.IGNORECASE,
)


def _natural_join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _stable_index(*parts: str, modulo: int) -> int:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _row_identity(split_name: str, row: dict[str, Any]) -> dict[str, str]:
    return {
        "split_name": split_name,
        "case_id": row["case_id"],
        "template_id": row["template_id"],
        "variant_id": row["variant_id"],
    }


def _rows_to_examples(rows: list[dict[str, Any]]) -> list[DatasetExample]:
    return [DatasetExample(**row) for row in rows]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_artifact_files(output_dir: Path) -> list[Path]:
    files = [path for path in output_dir.rglob("*") if path.is_file()]
    return sorted(path for path in files if path.name != "checksums.txt")


def _write_checksums(output_dir: Path) -> Path:
    checksum_path = output_dir / "checksums.txt"
    lines = [
        f"{_sha256_file(path)}  {path.relative_to(output_dir).as_posix()}"
        for path in _tracked_artifact_files(output_dir)
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


def _extract_wrapper_template(split_name: str, question: str) -> tuple[str | None, str]:
    for _, template in SURFACE_WRAPPERS_BY_SPLIT[split_name]:
        if template is None:
            continue
        prefix = template.split("{body}", 1)[0]
        if question.startswith(prefix):
            return template, question[len(prefix) :]
    return None, question


def _apply_wrapper(template: str | None, body: str) -> str:
    if template is None:
        return body
    if body:
        body = body[:1].lower() + body[1:]
    return template.format(body=body)


def _candidate_wrappers(split_name: str, preferred: str | None) -> list[str | None]:
    wrappers = [preferred]
    wrappers.extend(
        template
        for _, template in SURFACE_WRAPPERS_BY_SPLIT[split_name]
        if template != preferred
    )
    seen: set[str | None] = set()
    ordered: list[str | None] = []
    for template in wrappers:
        if template in seen:
            continue
        ordered.append(template)
        seen.add(template)
    return ordered


def _descendant_inventory_bodies(payload: QueryPlanPayload) -> tuple[str, ...]:
    root_name = payload.offerings[0]
    company_phrase = _natural_join(payload.companies) if payload.companies else "the selected companies"
    if payload.limit:
        return (
            f"Which offerings are in the {root_name} family at {company_phrase}, up to {payload.limit} results?",
            f"List up to {payload.limit} offerings in the {root_name} family for {company_phrase}.",
            f"What offerings are part of the {root_name} family at {company_phrase}, limited to {payload.limit} results?",
            f"Name as many as {payload.limit} offerings in the {root_name} family at {company_phrase}.",
            f"Identify up to {payload.limit} offerings in the {root_name} family for {company_phrase}.",
        )
    return (
        f"Which offerings are in the {root_name} family at {company_phrase}?",
        f"List the offerings in the {root_name} family for {company_phrase}.",
        f"What offerings are part of the {root_name} family at {company_phrase}?",
        f"Name the offerings in the {root_name} family at {company_phrase}.",
        f"Identify the offerings in the {root_name} family for {company_phrase}.",
    )


def _build_descendant_inventory_body(payload: QueryPlanPayload, row: dict[str, Any]) -> str:
    key = f"{row['case_id']}|{row['template_id']}|{row['variant_id']}"
    templates = _descendant_inventory_bodies(payload)
    return templates[_stable_index(key, modulo=len(templates))]


def _descendant_boolean_bodies(payload: QueryPlanPayload) -> tuple[str, ...]:
    root_name = payload.offerings[0]
    company_names = _natural_join(payload.companies)
    if payload.companies:
        return (
            f"Does {company_names} have any offerings in the {root_name} family?",
            f"Is there an offering in the {root_name} family at {company_names}?",
            f"Could {company_names} qualify as having offerings in the {root_name} family?",
            f"Would {company_names} count as having anything in the {root_name} offering family?",
        )
    return (
        f"Is there a company with an offering in the {root_name} family?",
        f"Are there companies with offerings in the {root_name} family?",
        f"Could a company qualify as having offerings in the {root_name} family?",
        f"Would any company count as carrying something in the {root_name} family?",
    )


def _build_descendant_boolean_body(payload: QueryPlanPayload, row: dict[str, Any]) -> str:
    key = f"{row['case_id']}|{row['template_id']}|{row['variant_id']}"
    templates = _descendant_boolean_bodies(payload)
    return templates[_stable_index(key, modulo=len(templates))]


def _descendant_count_bodies(payload: QueryPlanPayload) -> tuple[str, ...]:
    root_name = payload.offerings[0]
    company_phrase = _natural_join(payload.companies) if payload.companies else "the selected companies"
    return (
        f"How many offerings are in the {root_name} family at {company_phrase}?",
        f"Count the offerings in the {root_name} family at {company_phrase}.",
        f"What is the number of offerings in the {root_name} family at {company_phrase}?",
        f"How many offerings can be found in the {root_name} family at {company_phrase}?",
    )


def _build_descendant_count_body(payload: QueryPlanPayload, row: dict[str, Any]) -> str:
    key = f"{row['case_id']}|{row['template_id']}|{row['variant_id']}"
    templates = _descendant_count_bodies(payload)
    return templates[_stable_index(key, modulo=len(templates))]


def _extract_reference_company(question: str) -> str:
    patterns = (
        r"like ([^?]+)\??$",
        r"similar to ([^.?]+)[.?]?$",
        r"using ([^.?]+) as a reference(?: point)?[.?]?$",
        r"comparable to ([^?]+)\??$",
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return "the reference company"


def _refuse_beyond_local_bodies(question: str) -> tuple[str, ...]:
    company = _extract_reference_company(question)
    return (
        f"Which companies should we prioritize for developers rather than retailers, using {company} only as a reference point?",
        f"Recommend which companies to focus on for developers instead of retailers, with {company} as context rather than as a hard filter.",
        f"How would you rank companies if we want to favor developers over retailers, using {company} only as a reference?",
        f"Suggest which companies deserve attention for developers rather than retailers, taking {company} only as a comparison point.",
    )


def _build_refuse_beyond_local_body(question: str, row: dict[str, Any]) -> str:
    key = f"{row['case_id']}|{row['template_id']}|{row['variant_id']}"
    templates = _refuse_beyond_local_bodies(question)
    return templates[_stable_index(key, modulo=len(templates))]


def _normalize_question_text(question: str) -> str:
    question = re.sub(r"\bin United Kingdom\b", "in the United Kingdom", question)
    question = re.sub(r"\bactive in United Kingdom\b", "active in the United Kingdom", question)
    question = re.sub(r"\bfor United Kingdom\b", "for the United Kingdom", question)
    return question


def _question_candidates(split_name: str, row: dict[str, Any]) -> list[str]:
    question = row["question"]
    plan = QueryPlanEnvelope.model_validate(row["supervision_target"]["plan"])
    payload = plan.payload or QueryPlanPayload()
    wrapper_template, body = _extract_wrapper_template(split_name, question)
    wrappers = _candidate_wrappers(split_name, wrapper_template)

    if row["route_label"] == "refuse" and plan.reason == "beyond_local_coverage":
        bodies = _refuse_beyond_local_bodies(body)
    elif row["family"] == "descendant_offerings_by_root":
        bodies = _descendant_inventory_bodies(payload)
    elif row["family"] == "count_aggregate" and payload.base_family == "descendant_offerings_by_root":
        bodies = _descendant_count_bodies(payload)
    elif row["family"] == "boolean_exists" and payload.base_family == "descendant_offerings_by_root":
        bodies = _descendant_boolean_bodies(payload)
    else:
        match = TOP_RANKING_PATTERN.match(body)
        if row["family"] == "ranking_topk" and match is not None:
            bodies = (f"What are the top {match.group(1)} {match.group(2).rstrip('.')}?",)
        else:
            bodies = (body,)

    candidates: list[str] = []
    seen: set[str] = set()
    for template in wrappers:
        for candidate_body in bodies:
            candidate = _normalize_question_text(_apply_wrapper(template, candidate_body))
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def _curate_question(split_name: str, row: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    question = row["question"]
    curated_question = question
    changes: list[dict[str, Any]] = []
    plan = QueryPlanEnvelope.model_validate(row["supervision_target"]["plan"])
    payload = plan.payload or QueryPlanPayload()
    wrapper_template, body = _extract_wrapper_template(split_name, question)
    curated_body = body

    if row["route_label"] == "refuse" and plan.reason == "beyond_local_coverage" and REFUSE_DEV_RETAIL_PATTERN.search(body):
        curated_body = _build_refuse_beyond_local_body(body, row)

    if row["family"] == "descendant_offerings_by_root":
        curated_body = _build_descendant_inventory_body(payload, row)
    elif row["family"] == "count_aggregate" and payload.base_family == "descendant_offerings_by_root":
        curated_body = _build_descendant_count_body(payload, row)
    elif row["family"] == "boolean_exists" and payload.base_family == "descendant_offerings_by_root":
        curated_body = _build_descendant_boolean_body(payload, row)
    else:
        match = TOP_RANKING_PATTERN.match(curated_body)
        if row["family"] == "ranking_topk" and match is not None:
            curated_body = f"What are the top {match.group(1)} {match.group(2).rstrip('.')}?"

    curated_question = _apply_wrapper(wrapper_template, curated_body)
    curated_question = _normalize_question_text(curated_question)
    if curated_question != question:
        changes.append(
            {
                "field": "question",
                "old": question,
                "new": curated_question,
            }
        )
    return curated_question, changes


def _curation_log_entry(split_name: str, row: dict[str, Any], change: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "row_identity": _row_identity(split_name, row),
        "field": change["field"],
        "old": change["old"],
        "new": change["new"],
        "reason": reason,
        "curated_by": "Coordinator",
        "evidence": {
            "family": row["family"],
            "route_label": row["route_label"],
            "supervision_target": row["supervision_target"],
        },
    }


def _duplicate_pair_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        row["question"],
        _normalized_json_key(row["supervision_target"]),
    )


def _curate_rows(split_name: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curated_rows: list[dict[str, Any]] = []
    curation_log: list[dict[str, Any]] = []
    for row in rows:
        updated = json.loads(json.dumps(row))
        question, changes = _curate_question(split_name, updated)
        updated["question"] = question
        curated_rows.append(updated)
        for change in changes:
            reason = "question_rewrite"
            if updated["family"] == "descendant_offerings_by_root":
                reason = "inclusive_hierarchy_wording"
            elif updated["family"] == "count_aggregate" and updated["target"].get("payload", {}).get("base_family") == "descendant_offerings_by_root":
                reason = "inclusive_hierarchy_count_wording"
            elif updated["family"] == "boolean_exists" and updated["target"].get("payload", {}).get("base_family") == "descendant_offerings_by_root":
                reason = "inclusive_hierarchy_boolean_wording"
            elif updated["family"] == "ranking_topk":
                reason = "ranking_prompt_naturalization"
            elif updated["route_label"] == "refuse":
                reason = "refuse_boundary_clarification"
            curation_log.append(_curation_log_entry(split_name, updated, change, reason))
    return curated_rows, curation_log


def _dedupe_question_target_rows(
    split_name: str,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_groups: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        pair_groups.setdefault(_duplicate_pair_key(row), []).append(index)

    used_questions = {row["question"] for row in rows}
    curation_log: list[dict[str, Any]] = []

    for indices in list(pair_groups.values()):
        if len(indices) <= 1:
            continue
        for duplicate_index in indices[1:]:
            row = rows[duplicate_index]
            original_question = row["question"]
            used_questions.discard(original_question)
            replacement = None
            for candidate in _question_candidates(split_name, row):
                pair_key = (
                    candidate,
                    _normalized_json_key(row["supervision_target"]),
                )
                if candidate in used_questions:
                    continue
                if pair_key in pair_groups:
                    continue
                replacement = candidate
                break
            if replacement is None:
                used_questions.add(original_question)
                continue
            row["question"] = replacement
            used_questions.add(replacement)
            pair_groups[(
                replacement,
                _normalized_json_key(row["supervision_target"]),
            )] = [duplicate_index]
            curation_log.append(
                _curation_log_entry(
                    split_name,
                    row,
                    {"field": "question", "old": original_question, "new": replacement},
                    "duplicate_pair_rewrite",
                )
            )
    return rows, curation_log


def _write_review_workflow(output_dir: Path, rows_by_split: dict[str, list[dict[str, Any]]], shard_size: int) -> None:
    workflow_dir = output_dir / "workflow"
    shard_dir = workflow_dir / "review_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_manifest: list[dict[str, Any]] = []
    shard_index = 0
    for split_name, rows in rows_by_split.items():
        split_shard_dir = shard_dir / split_name
        split_shard_dir.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(rows), shard_size):
            chunk = rows[start : start + shard_size]
            shard_name = f"{split_name}_shard_{start // shard_size:03d}.jsonl"
            path = split_shard_dir / shard_name
            review_records = [
                {
                    "split_name": split_name,
                    "row_identity": _row_identity(split_name, row),
                    "example": row,
                }
                for row in chunk
            ]
            _write_jsonl(path, review_records)
            assigned_role = WORKER_ROLES[shard_index % len(WORKER_ROLES)]
            shard_manifest.append(
                {
                    "shard": path.relative_to(output_dir).as_posix(),
                    "split_name": split_name,
                    "row_count": len(chunk),
                    "assigned_role": assigned_role,
                    "start_index": start,
                    "end_index": start + len(chunk) - 1,
                }
            )
            shard_index += 1

    (workflow_dir / "assignments.json").write_text(
        json.dumps(
            {
                "roles": WORKFLOW_ROLES,
                "shards": shard_manifest,
                "rubric": [
                    "question matches the exact semantics of supervision_target",
                    "no hidden scope, limit, company, place, or inclusive/exclusive mismatch",
                    "local_safe wording is faithful to runtime semantics",
                    "strong_model_candidate vs refuse boundary is clean",
                    "prompt is grammatical and natural enough for a small local model",
                    "no synthetic scaffold patterns that weaken supervision",
                    "no contradiction between question, route_label, family, and gold_rows",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (workflow_dir / "RUBRIC.md").write_text(
        "\n".join(
            [
                "# Query Planner Curation Rubric",
                "",
                "1. `question` matches the exact semantics of `supervision_target`.",
                "2. No hidden scope, hidden limit, hidden company/place constraint, or inclusive/exclusive mismatch.",
                "3. `local_safe` wording is faithful to runtime semantics.",
                "4. `strong_model_candidate` vs `refuse` boundary is clean.",
                "5. Prompt is grammatical and natural enough for a small local model.",
                "6. No synthetic scaffold patterns that weaken supervision.",
                "7. No contradiction between `question`, `route_label`, `family`, and `gold_rows`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _baseline_dataset_card() -> str:
    return "\n".join(
        [
            "# Query Planner Curated Baseline",
            "",
            "This package is a frozen generator output used as the baseline for question-first curation.",
            "It preserves the training row schema but is not the final training artifact.",
            "",
            "## Package Contents",
            "",
            "- Split JSONL files generated from the runtime-aligned planner dataset builder.",
            "- Split-specific synthetic graph files.",
            "- `manifest.json` summarizing split composition.",
            "- `workflow/` shards and role assignments for row-by-row review.",
            "- `curation_log.jsonl`, intentionally empty at baseline freeze.",
            "",
            "## Identity Model",
            "",
            "Rows are treated as immutable curation units keyed by `(split_name, case_id, template_id, variant_id)`.",
            "",
            "## Training Policy",
            "",
            "Use the curated final package rather than this baseline for fine-tuning.",
        ]
    ) + "\n"


def _final_dataset_card(manifest: dict[str, Any], curation_log_count: int) -> str:
    validation_boolean = manifest["split_stats"]["validation"]["boolean_answer_counts"]
    release_boolean = manifest["split_stats"]["release_eval"]["boolean_answer_counts"]
    validation_ranking = manifest["split_stats"]["validation"]["ranking_scope_counts"].get("company+place", 0)
    release_ranking = manifest["split_stats"]["release_eval"]["ranking_scope_counts"].get("company+place", 0)
    return "\n".join(
        [
            "# Query Planner Curated Dataset",
            "",
            "## Purpose",
            "",
            "This artifact is the final fine-tuning dataset for the query planner task: `question -> supervision_target -> deterministic compiler`.",
            "It is intended for training a small local model to emit the runtime-aligned planning target rather than Cypher or answer rows.",
            "",
            "## Row Schema",
            "",
            "- `case_id`, `template_id`, `variant_id`: immutable identity fields inherited from the baseline freeze.",
            "- `question`: curated natural-language prompt.",
            "- `target`: full runtime-facing plan envelope for the row.",
            "- `supervision_target`: canonical training target; this is the field to supervise against.",
            "- `route_label`, `family`: task routing and family metadata.",
            "- `gold_cypher`, `gold_params`, `gold_rows`: deterministic compiler/execution artifacts derived from the runtime contract.",
            "- `metadata`: source-graph provenance plus task-specific metadata.",
            "",
            "## Splits",
            "",
            f"- `train`: {manifest['split_sizes']['train']} rows over graphs {', '.join(manifest['graph_assignments']['train'])}.",
            f"- `validation`: {manifest['split_sizes']['validation']} rows over graphs {', '.join(manifest['graph_assignments']['validation'])}.",
            f"- `release_eval`: {manifest['split_sizes']['release_eval']} rows over graphs {', '.join(manifest['graph_assignments']['release_eval'])}.",
            "- Split graph files are shipped alongside the JSONL rows.",
            "",
            "## Training Target",
            "",
            "`supervision_target` is the only authoritative supervision field for model training.",
            "The compiler and runtime remain the source of truth for how that target is executed.",
            "",
            "## Gold Field Derivation",
            "",
            "- `gold_cypher` and `gold_params` are produced by compiling `supervision_target.plan` through the runtime planner compiler.",
            "- `gold_rows` are produced by evaluating the compiled plan over the split-specific synthetic graph.",
            "- `metadata.source_graph_ids` is recomputed from runtime-consistent graph attribution logic, except that scoped false booleans retain scoped graph provenance rather than positive-match contributors.",
            "",
            "## Curation Policy",
            "",
            f"- Baseline rows were frozen first, then curated question-first with {curation_log_count} logged edits.",
            "- Question rewrites were allowed by default; route/target edits require runtime-backed adjudication.",
            "- The generator code remains archived for provenance/debugging, but this curated package is the official training source.",
            "",
            "## Verifier Policy",
            "",
            "- Every row is schema-checked and recompiled through the runtime planner contract.",
            "- `gold_*` fields and `metadata.source_graph_ids` are recomputed and compared to stored values.",
            "- Split invariants are enforced, including zero `local_safe_target_overlap_count` across splits.",
            f"- Held-out boolean negatives are present: validation {validation_boolean}, release_eval {release_boolean}.",
            f"- Held-out `company+place` ranking coverage is present: validation={validation_ranking}, release_eval={release_ranking}.",
            "- Canonicalized rebuilds are required to be byte-identical and checksum-stable.",
            "",
            "## Known Non-Blocking Limitations",
            "",
            "- Some ranking prompts remain terse search-style requests rather than full conversational questions.",
            "- A small `United Kingdom` place-phrasing pocket remains in otherwise semantically correct rows.",
            "- Shared non-answerable targets still exist across splits for refusal/strong-candidate supervision, but `local_safe` target overlap is zero.",
            "",
            "## Bootstrapping Prompt Family",
            "",
            "The baseline rows were bootstrapped from a runtime-aligned synthetic prompt family that can be summarized as:",
            "",
            "> Given a target route label and canonical planner envelope, generate one question whose semantics exactly match that target, staying within split scope and without broadening or narrowing any explicit constraints.",
            "",
            "This description is a reconstruction of the baseline generation setup, not the sole provenance of the final dataset.",
        ]
    ) + "\n"


def _build_curated_manifest(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    curation_log_count: int,
) -> dict[str, Any]:
    examples_by_split = {split: _rows_to_examples(rows) for split, rows in rows_by_split.items()}
    split_stats = {split: _summarize_split(examples) for split, examples in examples_by_split.items()}
    local_safe_family_targets = {
        split: dict(
            sorted(
                Counter(
                    example.family
                    for example in examples_by_split[split]
                    if example.route_label == "local_safe"
                ).items()
            )
        )
        for split in rows_by_split
    }
    strong_family_targets = {
        split: dict(
            sorted(
                Counter(
                    example.family
                    for example in examples_by_split[split]
                    if example.route_label == "strong_model_candidate"
                ).items()
            )
        )
        for split in rows_by_split
    }
    overlap_pairs = (
        ("train", "validation", "train__validation"),
        ("train", "release_eval", "train__release_eval"),
        ("validation", "release_eval", "validation__release_eval"),
    )
    return {
        "artifact_version": "v1_final",
        "seed": seed,
        "split_sizes": {split: len(rows) for split, rows in rows_by_split.items()},
        "route_targets": {split: split_stats[split]["route_counts"] for split in rows_by_split},
        "local_safe_bucket_targets": {
            split: split_stats[split]["local_safe_bucket_counts"] for split in rows_by_split
        },
        "local_safe_family_targets": local_safe_family_targets,
        "strong_model_candidate_targets": strong_family_targets,
        "refusal_reason_targets": {
            split: split_stats[split]["refusal_reason_counts"] for split in rows_by_split
        },
        "graph_assignments": {split: list(GRAPH_IDS_BY_SPLIT[split]) for split in rows_by_split},
        "split_overlap_stats": {
            key: _split_overlap_stats(examples_by_split[left], examples_by_split[right])
            for left, right, key in overlap_pairs
        },
        "split_stats": split_stats,
        "curation": {
            "policy": "question-first",
            "curation_log_count": curation_log_count,
            "roles": list(WORKFLOW_ROLES),
        },
    }


def freeze_curated_baseline(
    output_dir: Path = DEFAULT_BASELINE_DIR,
    *,
    train_size: int = DEFAULT_TRAIN_SIZE,
    validation_size: int = DEFAULT_VALIDATION_SIZE,
    release_eval_size: int = DEFAULT_RELEASE_EVAL_SIZE,
    seed: int = DEFAULT_SEED,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> dict[str, Path]:
    written = write_dataset_splits(
        output_dir,
        train_size=train_size,
        validation_size=validation_size,
        release_eval_size=release_eval_size,
        seed=seed,
    )
    rows_by_split = {
        split: _load_jsonl(output_dir / f"{split}.jsonl")
        for split in ("train", "validation", "release_eval")
    }
    _write_review_workflow(output_dir, rows_by_split, shard_size)
    (output_dir / "curation_log.jsonl").write_text("", encoding="utf-8")
    (output_dir / "DATASET_CARD.md").write_text(_baseline_dataset_card(), encoding="utf-8")
    written["curation_log"] = output_dir / "curation_log.jsonl"
    written["DATASET_CARD"] = output_dir / "DATASET_CARD.md"
    written["checksums"] = _write_checksums(output_dir)
    return written


def _copy_split_graphs(baseline_dir: Path, output_dir: Path) -> None:
    for split_name in ("train", "validation", "release_eval"):
        shutil.copy2(
            baseline_dir / f"{split_name}_synthetic_graphs.json",
            output_dir / f"{split_name}_synthetic_graphs.json",
        )


def build_curated_artifact(
    baseline_dir: Path = DEFAULT_BASELINE_DIR,
    output_dir: Path = DEFAULT_FINAL_DIR,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split = {
        split: _load_jsonl(baseline_dir / f"{split}.jsonl")
        for split in ("train", "validation", "release_eval")
    }
    curated_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    curation_log: list[dict[str, Any]] = []
    for split_name, rows in rows_by_split.items():
        curated_rows, split_log = _curate_rows(split_name, rows)
        curated_rows, dedupe_log = _dedupe_question_target_rows(split_name, curated_rows)
        curated_rows_by_split[split_name] = curated_rows
        curation_log.extend(split_log)
        curation_log.extend(dedupe_log)

    for split_name, rows in curated_rows_by_split.items():
        _write_jsonl(output_dir / f"{split_name}.jsonl", rows)
    _copy_split_graphs(baseline_dir, output_dir)

    manifest = _build_curated_manifest(curated_rows_by_split, seed=seed, curation_log_count=len(curation_log))
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(output_dir / "curation_log.jsonl", curation_log)
    (output_dir / "DATASET_CARD.md").write_text(
        _final_dataset_card(manifest, len(curation_log)),
        encoding="utf-8",
    )
    checksums = _write_checksums(output_dir)
    return {
        "train": output_dir / "train.jsonl",
        "validation": output_dir / "validation.jsonl",
        "release_eval": output_dir / "release_eval.jsonl",
        "train_synthetic_graphs": output_dir / "train_synthetic_graphs.json",
        "validation_synthetic_graphs": output_dir / "validation_synthetic_graphs.json",
        "release_eval_synthetic_graphs": output_dir / "release_eval_synthetic_graphs.json",
        "manifest": output_dir / "manifest.json",
        "curation_log": output_dir / "curation_log.jsonl",
        "DATASET_CARD": output_dir / "DATASET_CARD.md",
        "checksums": checksums,
    }


def _copy_support_files(source_dir: Path, output_dir: Path) -> None:
    for name in (
        "train_synthetic_graphs.json",
        "validation_synthetic_graphs.json",
        "release_eval_synthetic_graphs.json",
        "curation_log.jsonl",
    ):
        shutil.copy2(source_dir / name, output_dir / name)


def rewrite_curated_artifact(source_dir: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split = {
        split: _load_jsonl(source_dir / f"{split}.jsonl")
        for split in ("train", "validation", "release_eval")
    }
    curation_log = _load_jsonl(source_dir / "curation_log.jsonl") if (source_dir / "curation_log.jsonl").exists() else []
    for split_name, rows in rows_by_split.items():
        _write_jsonl(output_dir / f"{split_name}.jsonl", rows)
    _copy_support_files(source_dir, output_dir)
    manifest = _build_curated_manifest(rows_by_split, seed=DEFAULT_SEED, curation_log_count=len(curation_log))
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "DATASET_CARD.md").write_text(
        _final_dataset_card(manifest, len(curation_log)),
        encoding="utf-8",
    )
    checksums = _write_checksums(output_dir)
    return {
        "manifest": output_dir / "manifest.json",
        "checksums": checksums,
    }


def _expected_false_boolean_graph_ids(
    companies: tuple[SyntheticCompany, ...],
    plan: QueryPlanEnvelope,
) -> tuple[str, ...]:
    payload = plan.payload or QueryPlanPayload()
    if payload.companies:
        company_by_name = {company.name: company for company in companies}
        return tuple(
            sorted(
                company_by_name[name].graph_id
                for name in payload.companies
                if name in company_by_name
            )
        )
    return ()


def verify_curated_artifact(package_dir: Path) -> dict[str, Any]:
    issues: list[str] = []
    required_files = (
        "train.jsonl",
        "validation.jsonl",
        "release_eval.jsonl",
        "train_synthetic_graphs.json",
        "validation_synthetic_graphs.json",
        "release_eval_synthetic_graphs.json",
        "manifest.json",
        "curation_log.jsonl",
        "checksums.txt",
        "DATASET_CARD.md",
    )
    for name in required_files:
        if not (package_dir / name).exists():
            issues.append(f"missing required file: {name}")
    if issues:
        return {"ok": False, "issues": issues}

    rows_by_split = {
        split: _load_jsonl(package_dir / f"{split}.jsonl")
        for split in ("train", "validation", "release_eval")
    }
    examples_by_split = {split: _rows_to_examples(rows) for split, rows in rows_by_split.items()}
    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    companies = build_synthetic_company_graphs()
    company_map = {company.graph_id: company for company in companies}
    companies_by_split = {
        split: tuple(company_map[graph_id] for graph_id in GRAPH_IDS_BY_SPLIT[split])
        for split in ("train", "validation", "release_eval")
    }

    for split_name, rows in rows_by_split.items():
        split_companies = companies_by_split[split_name]
        for row in rows:
            required_fields = {
                "case_id",
                "template_id",
                "variant_id",
                "question",
                "target",
                "supervision_target",
                "route_label",
                "family",
                "gold_params",
                "gold_rows",
                "metadata",
            }
            missing_fields = sorted(required_fields.difference(row))
            if missing_fields:
                issues.append(f"{split_name}:{row.get('case_id', 'unknown')} missing fields {missing_fields}")
                continue
            if row["supervision_target"]["route_label"] != row["route_label"]:
                issues.append(f"{split_name}:{row['case_id']} supervision_target route mismatch")
            if row["supervision_target"]["plan"] != row["target"]:
                issues.append(f"{split_name}:{row['case_id']} supervision_target plan mismatch")

            plan = QueryPlanEnvelope.model_validate(row["supervision_target"]["plan"])
            compiled = compile_query_plan(plan)
            stored_cypher = row["gold_cypher"]
            if compiled.answerable:
                if stored_cypher != compiled.cypher:
                    issues.append(f"{split_name}:{row['case_id']} gold_cypher mismatch")
                if row["gold_params"] != compiled.params:
                    issues.append(f"{split_name}:{row['case_id']} gold_params mismatch")
                actual_rows = evaluate_query_plan(split_companies, plan)
                if row["gold_rows"] != actual_rows:
                    issues.append(f"{split_name}:{row['case_id']} gold_rows mismatch")
                if plan.family == "boolean_exists" and actual_rows and actual_rows[0].get("is_match") is False:
                    actual_graph_ids = _expected_false_boolean_graph_ids(split_companies, plan)
                else:
                    actual_graph_ids = matching_graph_ids_for_plan(split_companies, plan, actual_rows)
                expected_source_graph_id = _graph_source_id(actual_graph_ids)
                if row["metadata"].get("source_graph_ids") != list(actual_graph_ids):
                    issues.append(f"{split_name}:{row['case_id']} source_graph_ids mismatch")
                if row["metadata"].get("source_graph_id") != expected_source_graph_id:
                    issues.append(f"{split_name}:{row['case_id']} source_graph_id mismatch")
            else:
                if stored_cypher is not None:
                    issues.append(f"{split_name}:{row['case_id']} non-answerable row has gold_cypher")
                if row["gold_rows"] != []:
                    issues.append(f"{split_name}:{row['case_id']} non-answerable row has gold_rows")

            question = row["question"]
            lowered = question.casefold()
            for substring in FORBIDDEN_SUBSTRINGS:
                if substring in lowered:
                    issues.append(f"{split_name}:{row['case_id']} contains forbidden phrase {substring!r}")
            payload = plan.payload or QueryPlanPayload()
            if payload.limit is not None and not _limit_mentioned(question, payload.limit):
                issues.append(f"{split_name}:{row['case_id']} omits visible limit wording")
            if row["family"] in SINGLE_COMPANY_FORBIDDEN_FAMILIES and len(payload.companies) == 1:
                issues.append(f"{split_name}:{row['case_id']} uses forbidden single-company family scope")
            if row["route_label"] == "refuse" and plan.reason == "beyond_local_coverage" and REFUSE_DEV_RETAIL_PATTERN.search(lowered):
                issues.append(f"{split_name}:{row['case_id']} refusal row still reads as closed-world developers vs retailers")
            if row["family"] == "descendant_offerings_by_root" and DESCENDANT_WORDING_PATTERN.search(question):
                issues.append(f"{split_name}:{row['case_id']} still uses exclusive descendant wording")
            if row["family"] == "count_aggregate" and payload.base_family == "descendant_offerings_by_root" and DESCENDANT_WORDING_PATTERN.search(question):
                issues.append(f"{split_name}:{row['case_id']} count row still uses exclusive descendant wording")
            if row["family"] == "boolean_exists" and payload.base_family == "descendant_offerings_by_root" and DESCENDANT_WORDING_PATTERN.search(question):
                issues.append(f"{split_name}:{row['case_id']} boolean row still uses exclusive descendant wording")
            if row["family"] == "ranking_topk" and TOP_RANKING_PATTERN.match(question):
                issues.append(f"{split_name}:{row['case_id']} ranking row still uses fragment prompt")

    overlap_pairs = (
        ("train", "validation", "train__validation"),
        ("train", "release_eval", "train__release_eval"),
        ("validation", "release_eval", "validation__release_eval"),
    )
    for left, right, key in overlap_pairs:
        stats = _split_overlap_stats(examples_by_split[left], examples_by_split[right])
        if stats["question_overlap_count"] != 0:
            issues.append(f"{key} question overlap is not zero")
        if stats["question_target_overlap_count"] != 0:
            issues.append(f"{key} question_target overlap is not zero")
        if stats["local_safe_target_overlap_count"] != 0:
            issues.append(f"{key} local_safe target overlap is not zero")

    for split_name in ("validation", "release_eval"):
        split_stats = _summarize_split(examples_by_split[split_name])
        boolean_counts = split_stats["boolean_answer_counts"]
        if not boolean_counts.get("false") or not boolean_counts.get("true"):
            issues.append(f"{split_name} is missing held-out boolean polarity coverage")
        if split_stats["ranking_scope_counts"].get("company+place", 0) <= 0:
            issues.append(f"{split_name} is missing company+place ranking coverage")
        if split_stats["duplicate_question_target_count"] != 0:
            issues.append(f"{split_name} contains duplicate question+target rows")

    expected_manifest = _build_curated_manifest(
        rows_by_split,
        seed=manifest.get("seed", DEFAULT_SEED),
        curation_log_count=len(_load_jsonl(package_dir / "curation_log.jsonl")),
    )
    if manifest != expected_manifest:
        issues.append("manifest.json does not match recomputed curated manifest")

    expected_checksums = "\n".join(
        f"{_sha256_file(path)}  {path.relative_to(package_dir).as_posix()}"
        for path in _tracked_artifact_files(package_dir)
    ) + "\n"
    actual_checksums = (package_dir / "checksums.txt").read_text(encoding="utf-8")
    if actual_checksums != expected_checksums:
        issues.append("checksums.txt does not match current package contents")

    with TemporaryDirectory() as first_tmp, TemporaryDirectory() as second_tmp:
        rewrite_curated_artifact(package_dir, Path(first_tmp))
        rewrite_curated_artifact(package_dir, Path(second_tmp))
        first_hashes = {
            path.relative_to(first_tmp).as_posix(): _sha256_file(path)
            for path in _tracked_artifact_files(Path(first_tmp)) + [Path(first_tmp) / "checksums.txt"]
        }
        second_hashes = {
            path.relative_to(second_tmp).as_posix(): _sha256_file(path)
            for path in _tracked_artifact_files(Path(second_tmp)) + [Path(second_tmp) / "checksums.txt"]
        }
        if first_hashes != second_hashes:
            issues.append("canonical rewrites are not byte-identical across reruns")

    return {
        "ok": not issues,
        "issues": issues,
        "summary": {
            "split_sizes": manifest["split_sizes"],
            "local_safe_target_overlap": {
                key: manifest["split_overlap_stats"][key]["local_safe_target_overlap_count"]
                for key in manifest["split_overlap_stats"]
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Curated query planner dataset tooling.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser("freeze-baseline", help="Freeze the baseline generator output.")
    freeze_parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    freeze_parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    freeze_parser.add_argument("--validation-size", type=int, default=DEFAULT_VALIDATION_SIZE)
    freeze_parser.add_argument("--release-eval-size", type=int, default=DEFAULT_RELEASE_EVAL_SIZE)
    freeze_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    freeze_parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)

    build_parser = subparsers.add_parser("build-final", help="Build the curated final artifact from a frozen baseline.")
    build_parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    build_parser.add_argument("--output-dir", type=Path, default=DEFAULT_FINAL_DIR)
    build_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    verify_parser = subparsers.add_parser("verify", help="Verify a curated artifact package.")
    verify_parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_FINAL_DIR)

    args = parser.parse_args(argv)
    if args.command == "freeze-baseline":
        freeze_curated_baseline(
            args.output_dir,
            train_size=args.train_size,
            validation_size=args.validation_size,
            release_eval_size=args.release_eval_size,
            seed=args.seed,
            shard_size=args.shard_size,
        )
        return 0
    if args.command == "build-final":
        build_curated_artifact(args.baseline_dir, args.output_dir, seed=args.seed)
        return 0
    report = verify_curated_artifact(args.artifact_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


__all__ = [
    "DEFAULT_BASELINE_DIR",
    "DEFAULT_FINAL_DIR",
    "build_curated_artifact",
    "freeze_curated_baseline",
    "main",
    "rewrite_curated_artifact",
    "verify_curated_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
