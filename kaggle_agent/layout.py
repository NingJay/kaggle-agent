from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from kaggle_agent.utils import slugify


LEGACY_DEFAULT_ATTEMPT_SLUG = "simplerun-perch-v2embedprobe-bayesian-0-912"
DEFAULT_ATTEMPT_SLUG = "birdclef-2026-attempt"
DEBUG_WORK_TYPES = frozenset({"preflight_check"})
ROOT_SURFACE_DOC_NAMES = (
    "CHECKLIST.md",
    "JOURNAL.md",
    "FINDINGS.md",
    "ISSUES.md",
    "SUBMISSIONS.md",
)
STAGE_SEQUENCE = [
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
STAGE_ORDER = {name: index + 1 for index, name in enumerate(STAGE_SEQUENCE)}


def derive_attempt_slug(seed_notebook_path: str = "") -> str:
    candidate = seed_notebook_path.strip()
    if candidate:
        slug = slugify(Path(candidate).stem)
        if slug:
            return slug
    return DEFAULT_ATTEMPT_SLUG


def current_attempt_slug(runtime, *, fallback: str = DEFAULT_ATTEMPT_SLUG) -> str:
    return getattr(runtime, "current_attempt_slug", "") or fallback


def is_debug_work_item(work_item: Any) -> bool:
    return getattr(work_item, "work_type", "") in DEBUG_WORK_TYPES


def debug_work_item_ids(state: Any) -> set[str]:
    return {getattr(item, "id", "") for item in getattr(state, "work_items", []) if is_debug_work_item(item)}


def visible_work_items(state: Any, *, include_debug: bool = False) -> list[Any]:
    work_items = list(getattr(state, "work_items", []))
    if include_debug:
        return work_items
    return [item for item in work_items if not is_debug_work_item(item)]


def visible_run_ids(state: Any, *, include_debug: bool = False) -> set[str]:
    return {getattr(item, "run_id", "") for item in visible_runs(state, include_debug=include_debug)}


def visible_runs(state: Any, *, include_debug: bool = False) -> list[Any]:
    runs = list(getattr(state, "runs", []))
    if include_debug:
        return runs
    hidden_work_item_ids = debug_work_item_ids(state)
    return [item for item in runs if getattr(item, "work_item_id", "") not in hidden_work_item_ids]


def visible_stage_runs(state: Any, *, include_debug: bool = False) -> list[Any]:
    stage_runs = list(getattr(state, "stage_runs", []))
    if include_debug:
        return stage_runs
    allowed_run_ids = visible_run_ids(state)
    return [item for item in stage_runs if getattr(item, "run_id", "") in allowed_run_ids]


def run_label(run_id: str, title: str) -> str:
    return f"{run_id}__{slugify(title)}"


def run_label_from_path(run_dir: str) -> str:
    if not run_dir:
        return ""
    path = Path(run_dir)
    if path.name == "runtime":
        return path.parent.name
    return path.name


def stage_outcome_slug(
    payload: Mapping[str, Any] | None = None,
    *,
    stage_status: str = "",
    validator_status: str = "",
) -> str:
    if validator_status.strip():
        return slugify(validator_status)
    if payload:
        for key in ("status", "plan_status"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return slugify(value)
    return slugify(stage_status or "pending")


def stage_label(
    stage_name: str,
    payload: Mapping[str, Any] | None = None,
    *,
    stage_status: str = "",
    validator_status: str = "",
) -> str:
    order = STAGE_ORDER.get(stage_name, 0)
    return f"{order:02d}-{stage_name}__{stage_outcome_slug(payload, stage_status=stage_status, validator_status=validator_status)}"


def stage_label_from_path(output_dir: str, stage_name: str, *, stage_status: str = "", validator_status: str = "") -> str:
    if output_dir:
        return Path(output_dir).name
    return stage_label(stage_name, stage_status=stage_status, validator_status=validator_status)


def artifact_relative_path(path: str, root: Path) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return str(candidate.relative_to(root))
    except ValueError:
        return str(candidate)
