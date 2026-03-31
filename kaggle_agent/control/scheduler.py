from __future__ import annotations

from pathlib import Path

from kaggle_agent.control.store import find_work_item
from kaggle_agent.schema import ExperimentSpec, SpecRecord, WorkItem, WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import now_utc_iso, slugify


RUNNABLE_STATUSES = {"queued"}
DONE_STATUSES = {"complete", "validated", "submitted", "failed", "blocked"}


def active_runs(state: WorkspaceState) -> list[str]:
    return [run.run_id for run in state.runs if run.status == "running"]


def dependencies_satisfied(state: WorkspaceState, work_item: WorkItem) -> bool:
    status_by_id = {item.id: item.status for item in state.work_items}
    return all(status_by_id.get(dependency) in DONE_STATUSES for dependency in work_item.depends_on)


def runnable_work_items(config: WorkspaceConfig, state: WorkspaceState) -> list[WorkItem]:
    candidates: list[WorkItem] = []
    for work_item in state.work_items:
        if work_item.status not in RUNNABLE_STATUSES:
            continue
        if not dependencies_satisfied(state, work_item):
            continue
        config_path = config.root / work_item.config_path
        if not config_path.exists():
            continue
        candidates.append(work_item)
    return sorted(candidates, key=lambda item: (item.priority, item.created_at, item.id))


def choose_next_work_item(config: WorkspaceConfig, state: WorkspaceState) -> WorkItem | None:
    ready = choose_next_work_items(config, state, limit=1)
    return ready[0] if ready else None


def choose_next_work_items(config: WorkspaceConfig, state: WorkspaceState, *, limit: int | None = None) -> list[WorkItem]:
    remaining_capacity = max(0, config.automation.max_active_runs - len(active_runs(state)))
    if remaining_capacity <= 0:
        return []
    ready = runnable_work_items(config, state)
    if not ready:
        return []
    if limit is None:
        limit = remaining_capacity
    return ready[: min(limit, remaining_capacity)]


def _next_work_item_id(state: WorkspaceState, title: str) -> str:
    slug = slugify(title)
    work_item_id = f"workitem-{state.runtime.next_work_item_number:04d}-{slug}"
    state.runtime.next_work_item_number += 1
    return work_item_id


def register_work_item(
    state: WorkspaceState,
    *,
    title: str,
    work_type: str,
    family: str,
    config_path: str,
    priority: int,
    pipeline: list[str],
    depends_on: list[str] | None = None,
    dedupe_key: str = "",
    source_run_id: str = "",
    source_stage_run_id: str = "",
    source_decision_id: str = "",
    portfolio_id: str = "",
    parent_work_item_id: str = "",
    idea_class: str = "",
    branch_role: str = "",
    branch_rank: int = 0,
    knowledge_card_ids: list[str] | None = None,
    notes: list[str] | None = None,
) -> WorkItem:
    existing = next((item for item in state.work_items if dedupe_key and item.dedupe_key == dedupe_key), None)
    if existing is not None:
        existing.config_path = config_path
        existing.priority = priority
        existing.pipeline = list(pipeline)
        existing.depends_on = list(depends_on or existing.depends_on)
        existing.portfolio_id = portfolio_id or existing.portfolio_id
        existing.parent_work_item_id = parent_work_item_id or existing.parent_work_item_id
        existing.idea_class = idea_class or existing.idea_class
        existing.branch_role = branch_role or existing.branch_role
        existing.branch_rank = branch_rank
        existing.knowledge_card_ids = list(knowledge_card_ids or existing.knowledge_card_ids)
        existing.updated_at = now_utc_iso()
        return existing
    work_item = WorkItem(
        id=_next_work_item_id(state, title),
        title=title,
        work_type=work_type,
        family=family,
        priority=priority,
        status="queued",
        config_path=config_path,
        pipeline=list(pipeline),
        depends_on=list(depends_on or []),
        source_run_id=source_run_id,
        source_stage_run_id=source_stage_run_id,
        source_decision_id=source_decision_id,
        dedupe_key=dedupe_key or f"work_item:{slugify(title)}:{slugify(config_path)}",
        portfolio_id=portfolio_id,
        parent_work_item_id=parent_work_item_id,
        idea_class=idea_class,
        branch_role=branch_role,
        branch_rank=branch_rank,
        knowledge_card_ids=list(knowledge_card_ids or []),
        notes=list(notes or []),
        created_at=now_utc_iso(),
        updated_at=now_utc_iso(),
    )
    state.work_items.append(work_item)
    return work_item


def register_experiment_for_work_item(state: WorkspaceState, work_item: WorkItem, spec: SpecRecord | None = None) -> ExperimentSpec:
    existing = next((item for item in state.experiments if item.work_item_id == work_item.id), None)
    if existing is not None:
        if spec is not None:
            existing.spec_id = spec.spec_id
            existing.config_path = spec.config_path
            existing.code_state_ref = spec.code_state_ref
            existing.portfolio_id = spec.portfolio_id or existing.portfolio_id
            existing.idea_class = spec.idea_class or existing.idea_class
            existing.branch_role = spec.branch_role or existing.branch_role
            existing.branch_rank = spec.branch_rank or existing.branch_rank
            existing.knowledge_card_ids = list(spec.knowledge_card_ids or existing.knowledge_card_ids)
        return existing
    experiment = ExperimentSpec(
        id=f"exp-{work_item.id.replace('workitem-', '')}",
        title=work_item.title,
        hypothesis=f"Work item {work_item.id}: {work_item.title}",
        family=work_item.family,
        config_path=spec.config_path if spec is not None else work_item.config_path,
        priority=work_item.priority,
        work_item_id=work_item.id,
        spec_id=spec.spec_id if spec is not None else "",
        depends_on=[],
        tags=[work_item.work_type, "v2"],
        launch_mode="background",
        code_state_ref=spec.code_state_ref if spec is not None else "",
        created_at=now_utc_iso(),
        updated_at=now_utc_iso(),
        dedupe_key=f"workitem:{work_item.id}:experiment",
        portfolio_id=(spec.portfolio_id if spec is not None else work_item.portfolio_id),
        idea_class=(spec.idea_class if spec is not None else work_item.idea_class),
        branch_role=(spec.branch_role if spec is not None else work_item.branch_role),
        branch_rank=(spec.branch_rank if spec is not None else work_item.branch_rank),
        knowledge_card_ids=list(spec.knowledge_card_ids if spec is not None else work_item.knowledge_card_ids),
    )
    state.experiments.append(experiment)
    work_item.updated_at = now_utc_iso()
    return experiment


def ensure_seed_spec(state: WorkspaceState, work_item: WorkItem) -> SpecRecord:
    existing = next((item for item in state.specs if item.work_item_id == work_item.id and item.status == "validated"), None)
    if existing is not None:
        work_item.latest_spec_id = existing.spec_id
        return existing
    spec = SpecRecord(
        spec_id=f"spec-{state.runtime.next_spec_number:04d}",
        work_item_id=work_item.id,
        source_stage_run_id="",
        spec_type="experiment",
        title=work_item.title,
        family=work_item.family,
        config_path=work_item.config_path,
        payload_path=work_item.config_path,
        launch_mode="background",
        code_state_ref="",
        status="validated",
        dedupe_key=f"seed:{work_item.id}:spec",
        portfolio_id=work_item.portfolio_id,
        idea_class=work_item.idea_class,
        branch_role=work_item.branch_role,
        branch_rank=work_item.branch_rank,
        knowledge_card_ids=list(work_item.knowledge_card_ids),
        created_at=now_utc_iso(),
        updated_at=now_utc_iso(),
    )
    state.runtime.next_spec_number += 1
    state.specs.append(spec)
    work_item.latest_spec_id = spec.spec_id
    work_item.updated_at = now_utc_iso()
    return spec


def queue_config_experiment(
    config: WorkspaceConfig,
    state: WorkspaceState,
    config_path: str,
    title: str | None = None,
    family: str = "ad_hoc",
    priority: int = 50,
) -> WorkItem:
    resolved = Path(config_path).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Config not found: {resolved}")
    relative_path = str(resolved.relative_to(config.root)) if resolved.is_relative_to(config.root) else str(resolved)
    return register_work_item(
        state,
        title=title or resolved.stem.replace("_", " "),
        work_type="experiment_iteration",
        family=family,
        config_path=relative_path,
        priority=priority,
        pipeline=["execute", "evidence", "report", "research", "decision", "plan", "codegen", "critic", "validate", "submission"],
        notes=["Queued from direct config request."],
    )


def mark_work_item_status(state: WorkspaceState, work_item_id: str, status: str) -> WorkItem:
    work_item = find_work_item(state, work_item_id)
    work_item.status = status
    work_item.updated_at = now_utc_iso()
    return work_item
