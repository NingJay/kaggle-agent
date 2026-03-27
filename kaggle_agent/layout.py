from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from kaggle_agent.utils import slugify


DEFAULT_ATTEMPT_SLUG = "simplerun-perch-v2embedprobe-bayesian-0-912"
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


def current_attempt_slug(runtime) -> str:
    return getattr(runtime, "current_attempt_slug", "") or DEFAULT_ATTEMPT_SLUG


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
