from __future__ import annotations

from pathlib import Path

from kaggle_agent.schema import ExperimentSpec, WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import now_utc_iso, slugify


RUNNABLE_STATUSES = {"queued"}
SUCCESS_STATUSES = {"succeeded"}


def active_runs(state: WorkspaceState) -> list[str]:
    return [run.run_id for run in state.runs if run.status == "running"]


def dependencies_satisfied(state: WorkspaceState, experiment: ExperimentSpec) -> bool:
    completed = {item.id for item in state.experiments if item.status in SUCCESS_STATUSES}
    return all(dependency in completed for dependency in experiment.depends_on)


def runnable_experiments(config: WorkspaceConfig, state: WorkspaceState) -> list[ExperimentSpec]:
    candidates: list[ExperimentSpec] = []
    for experiment in state.experiments:
        if experiment.status not in RUNNABLE_STATUSES:
            continue
        if not dependencies_satisfied(state, experiment):
            continue
        config_path = config.root / experiment.config_path
        if not config_path.exists():
            continue
        candidates.append(experiment)
    return sorted(candidates, key=lambda item: (item.priority, item.created_at, item.id))


def choose_next_experiment(config: WorkspaceConfig, state: WorkspaceState) -> ExperimentSpec | None:
    if len(active_runs(state)) >= config.automation.max_active_runs:
        return None
    ready = runnable_experiments(config, state)
    return ready[0] if ready else None


def register_experiment(
    state: WorkspaceState,
    title: str,
    hypothesis: str,
    family: str,
    config_path: str,
    priority: int,
    depends_on: list[str] | None = None,
    tags: list[str] | None = None,
    launch_mode: str = "background",
    dedupe_key: str = "",
    source_decision_id: str = "",
    requeue_existing: bool = True,
) -> ExperimentSpec:
    experiment = ExperimentSpec(
        id=f"exp-{slugify(title)}",
        title=title,
        hypothesis=hypothesis,
        family=family,
        config_path=config_path,
        priority=priority,
        depends_on=list(depends_on or []),
        tags=list(tags or []),
        launch_mode=launch_mode,
        created_at=now_utc_iso(),
        updated_at=now_utc_iso(),
        dedupe_key=dedupe_key,
        source_decision_id=source_decision_id,
    )
    existing = next((item for item in state.experiments if dedupe_key and item.dedupe_key == dedupe_key), None)
    if existing is None:
        existing = next((item for item in state.experiments if item.id == experiment.id), None)
    if existing is not None:
        existing.config_path = config_path
        existing.hypothesis = hypothesis
        existing.family = family
        existing.priority = priority
        existing.depends_on = list(depends_on or [])
        existing.tags = list(tags or [])
        existing.launch_mode = launch_mode
        if dedupe_key:
            existing.dedupe_key = dedupe_key
        if source_decision_id:
            existing.source_decision_id = source_decision_id
        if requeue_existing:
            existing.status = "queued"
        existing.updated_at = now_utc_iso()
        return existing
    state.experiments.append(experiment)
    return experiment


def queue_config_experiment(
    config: WorkspaceConfig,
    state: WorkspaceState,
    config_path: str,
    title: str | None = None,
    family: str = "ad_hoc",
    priority: int = 50,
) -> ExperimentSpec:
    resolved = Path(config_path).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Config not found: {resolved}")
    relative_path = str(resolved.relative_to(config.root)) if resolved.is_relative_to(config.root) else str(resolved)
    return register_experiment(
        state,
        title=title or resolved.stem.replace("_", " "),
        hypothesis="Queued from direct config request.",
        family=family,
        config_path=relative_path,
        priority=priority,
        tags=["ad-hoc", "config-request"],
    )
