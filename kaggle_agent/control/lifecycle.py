from __future__ import annotations

from typing import Iterable


CANONICAL_STAGE_GRAPH = [
    "execute",
    "evidence",
    "report",
    "research",
    "decision",
    "plan",
    "codegen",
    "critic",
    "validate",
    "submission",
]

LIFECYCLE_TEMPLATES: dict[str, list[str]] = {
    "recursive_experiment": list(CANONICAL_STAGE_GRAPH),
    "terminal_experiment": ["execute", "evidence", "report", "validate"],
    "submission_from_target_run": ["submission"],
    "analysis_only": ["research", "decision", "plan"],
}

WORK_TYPE_TO_TEMPLATE = {
    "experiment_iteration": "recursive_experiment",
    "preflight_check": "recursive_experiment",
    "ablation_terminal": "terminal_experiment",
    "submission": "submission_from_target_run",
    "analysis_only": "analysis_only",
}


def _looks_like_submission_branch(payload: dict[str, object]) -> bool:
    branch_role = str(payload.get("branch_role", "") or "").strip().lower()
    idea_class = str(payload.get("idea_class", "") or "").strip().lower()
    title = str(payload.get("title", "") or "").strip().lower()
    tags = [str(item or "").strip().lower() for item in payload.get("tags", []) if str(item or "").strip()]
    blob = " ".join([title, " ".join(tags)])
    if branch_role == "submission":
        return True
    if "submission" in idea_class:
        return True
    if "submit" in blob or "submission bundle" in blob or "leaderboard" in blob:
        return True
    return False


def _normalize_stage_plan(stage_plan: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for stage_name in stage_plan:
        name = str(stage_name or "").strip()
        if not name:
            continue
        normalized.append(name)
    return normalized


def _is_subsequence(candidate: list[str], canonical: list[str]) -> bool:
    position = 0
    for stage_name in canonical:
        if position >= len(candidate):
            break
        if candidate[position] == stage_name:
            position += 1
    return position == len(candidate)


def validate_stage_plan(stage_plan: Iterable[str], *, strict: bool = True) -> list[str]:
    normalized = _normalize_stage_plan(stage_plan)
    if not normalized:
        raise ValueError("Stage plan cannot be empty.")
    unknown = [item for item in normalized if item not in CANONICAL_STAGE_GRAPH]
    if unknown:
        raise ValueError(f"Unknown stage names in stage plan: {', '.join(unknown)}")
    if strict and not _is_subsequence(normalized, CANONICAL_STAGE_GRAPH):
        raise ValueError(
            "Stage plan must be a subsequence of the canonical graph: "
            + " -> ".join(CANONICAL_STAGE_GRAPH)
        )
    return normalized


def resolve_stage_plan(template: str, *, strict: bool = True) -> list[str]:
    if template not in LIFECYCLE_TEMPLATES:
        raise ValueError(f"Unknown lifecycle template: {template}")
    return validate_stage_plan(LIFECYCLE_TEMPLATES[template], strict=strict)


def resolve_lifecycle_template(payload: dict[str, object] | None, *, default: str = "recursive_experiment") -> str:
    if payload is None:
        return default
    explicit = str(payload.get("lifecycle_template", "") or "").strip()
    if explicit:
        if explicit not in LIFECYCLE_TEMPLATES:
            raise ValueError(f"Unknown lifecycle template: {explicit}")
        return explicit
    if _looks_like_submission_branch(payload):
        return "submission_from_target_run"
    work_type = str(payload.get("work_type", "") or "").strip()
    if work_type and work_type in WORK_TYPE_TO_TEMPLATE:
        return WORK_TYPE_TO_TEMPLATE[work_type]
    return default


def infer_lifecycle_template(stage_plan: Iterable[str]) -> str:
    normalized = _normalize_stage_plan(stage_plan)
    for template, candidate in LIFECYCLE_TEMPLATES.items():
        if normalized == candidate:
            return template
    return "recursive_experiment"


def entry_stage_for_plan(stage_plan: Iterable[str]) -> str:
    normalized = validate_stage_plan(stage_plan, strict=False)
    return normalized[0]


def next_stage(stage_plan: Iterable[str], current_stage: str) -> str | None:
    normalized = validate_stage_plan(stage_plan, strict=False)
    if current_stage not in normalized:
        raise ValueError(f"Stage `{current_stage}` is not present in plan: {normalized}")
    index = normalized.index(current_stage)
    if index + 1 >= len(normalized):
        return None
    return normalized[index + 1]


def resolve_target_run_id(
    payload: dict[str, object] | None,
    *,
    lifecycle_template: str,
    default_run_id: str = "",
) -> str:
    if payload is None:
        return default_run_id if lifecycle_template == "submission_from_target_run" else ""
    explicit = str(payload.get("target_run_id", "") or "").strip()
    if explicit:
        return explicit
    if lifecycle_template == "submission_from_target_run":
        return default_run_id
    return ""
