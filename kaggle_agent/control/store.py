from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from kaggle_agent.schema import (
    AgentRun,
    ExperimentSpec,
    FindingRecord,
    IssueRecord,
    MetricObservation,
    ResearchNoteRecord,
    RunRecord,
    RuntimeState,
    SpecRecord,
    StageRun,
    SubmissionCandidate,
    SubmissionResult,
    ValidationRecord,
    WorkItem,
    WorkspaceConfig,
    WorkspaceState,
)
from kaggle_agent.utils import atomic_write_json, ensure_directory, now_utc_iso

TABLE_TO_MODEL = {
    "work_items": WorkItem,
    "experiments": ExperimentSpec,
    "runs": RunRecord,
    "stage_runs": StageRun,
    "agent_runs": AgentRun,
    "specs": SpecRecord,
    "validations": ValidationRecord,
    "metrics": MetricObservation,
    "findings": FindingRecord,
    "issues": IssueRecord,
    "research_notes": ResearchNoteRecord,
    "submissions": SubmissionCandidate,
    "submission_results": SubmissionResult,
}
STATE_TABLES = list(TABLE_TO_MODEL.keys())


def _default_runtime_state() -> RuntimeState:
    timestamp = now_utc_iso()
    return RuntimeState(initialized_at=timestamp, last_tick_at=timestamp, last_report_at=timestamp)


def _open_ledger(config: WorkspaceConfig) -> sqlite3.Connection:
    ensure_directory(config.state_root())
    connection = sqlite3.connect(config.ledger_path())
    connection.row_factory = sqlite3.Row
    return connection


def _ensure_ledger(config: WorkspaceConfig) -> None:
    with _open_ledger(config) as conn:
        for table in STATE_TABLES:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_state (
                id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )


def _table_rows(config: WorkspaceConfig, table: str, model_cls):
    with _open_ledger(config) as conn:
        rows = conn.execute(f"SELECT payload FROM {table} ORDER BY id").fetchall()
    return [model_cls.from_dict(__import__("json").loads(row["payload"])) for row in rows]


def _save_table(config: WorkspaceConfig, table: str, rows) -> None:
    import json

    with _open_ledger(config) as conn:
        conn.execute(f"DELETE FROM {table}")
        conn.executemany(
            f"INSERT INTO {table} (id, payload) VALUES (?, ?)",
            [(getattr(item, next(field for field in item.__dataclass_fields__ if field.endswith('_id') or field == 'id')), json.dumps(item.to_dict(), ensure_ascii=False)) for item in rows],
        )


def _entity_id(item) -> str:
    if hasattr(item, "id"):
        return getattr(item, "id")
    for key in [
        "stage_run_id",
        "agent_run_id",
        "spec_id",
        "validation_id",
        "metric_id",
        "finding_id",
        "issue_id",
        "note_id",
        "result_id",
        "run_id",
    ]:
        if hasattr(item, key):
            return getattr(item, key)
    raise ValueError(f"Cannot infer entity id for {type(item).__name__}")


def _save_rows(config: WorkspaceConfig, table: str, rows) -> None:
    import json

    with _open_ledger(config) as conn:
        conn.execute(f"DELETE FROM {table}")
        conn.executemany(
            f"INSERT INTO {table} (id, payload) VALUES (?, ?)",
            [(_entity_id(item), json.dumps(item.to_dict(), ensure_ascii=False)) for item in rows],
        )


def ensure_layout(config: WorkspaceConfig) -> None:
    ensure_directory(config.state_root())
    ensure_directory(config.export_root())
    ensure_directory(config.snapshot_root())
    ensure_directory(config.artifact_root())
    ensure_directory(config.report_root())
    ensure_directory(config.legacy_root())
    ensure_directory(config.knowledge_root())
    ensure_directory(config.prompt_root())
    for category in [
        "runs",
        "evidence",
        "reports",
        "research",
        "decisions",
        "plans",
        "codegen",
        "validations",
        "submissions",
        "logs",
    ]:
        ensure_directory(config.artifact_path(category))
    for category in ["research", "papers"]:
        ensure_directory(config.knowledge_path(category))
    ensure_directory(config.generated_config_root())


def archive_legacy_artifacts(config: WorkspaceConfig) -> Path | None:
    legacy_candidates = [
        "reports",
        "state",
        "artifacts",
    ]
    existing = [config.root / name for name in legacy_candidates if (config.root / name).exists()]
    if not existing:
        return None
    timestamp = now_utc_iso().replace(":", "").replace("+00:00", "Z")
    archive_root = ensure_directory(config.legacy_root() / timestamp)
    for path in existing:
        shutil.move(str(path), str(archive_root / path.name))
    return archive_root


def _default_work_items(config: WorkspaceConfig) -> list[WorkItem]:
    timestamp = now_utc_iso()
    return [
        WorkItem(
            id="workitem-perch-debug-smoke",
            title="Perch-head debug smoke",
            work_type="experiment_iteration",
            family="perch_head_debug",
            priority=10,
            config_path=str((config.runtime_root() / "configs" / "debug.yaml").relative_to(config.root)),
            pipeline=["execute", "evidence", "report", "research", "decision", "plan", "codegen", "critic", "validate", "submission"],
            dedupe_key="seed:perch-debug-smoke",
            created_at=timestamp,
            updated_at=timestamp,
        ),
        WorkItem(
            id="workitem-perch-baseline",
            title="Perch cached-probe baseline",
            work_type="experiment_iteration",
            family="perch_cached_probe",
            priority=20,
            config_path=str((config.runtime_root() / "configs" / "default.yaml").relative_to(config.root)),
            pipeline=["execute", "evidence", "report", "research", "decision", "plan", "codegen", "critic", "validate", "submission"],
            depends_on=["workitem-perch-debug-smoke"],
            dedupe_key="seed:perch-baseline",
            created_at=timestamp,
            updated_at=timestamp,
        ),
    ]


def _empty_state(config: WorkspaceConfig) -> WorkspaceState:
    return WorkspaceState(
        work_items=_default_work_items(config),
        experiments=[],
        runs=[],
        stage_runs=[],
        agent_runs=[],
        specs=[],
        validations=[],
        metrics=[],
        findings=[],
        issues=[],
        research_notes=[],
        submissions=[],
        submission_results=[],
        runtime=_default_runtime_state(),
    )


def _snapshot_state(config: WorkspaceConfig, state: WorkspaceState) -> None:
    atomic_write_json(config.export_root() / "work_items.json", [item.to_dict() for item in state.work_items])
    atomic_write_json(config.export_root() / "experiments.json", [item.to_dict() for item in state.experiments])
    atomic_write_json(config.export_root() / "runs.json", [item.to_dict() for item in state.runs])
    atomic_write_json(config.export_root() / "stage_runs.json", [item.to_dict() for item in state.stage_runs])
    atomic_write_json(config.export_root() / "agent_runs.json", [item.to_dict() for item in state.agent_runs])
    atomic_write_json(config.export_root() / "specs.json", [item.to_dict() for item in state.specs])
    atomic_write_json(config.export_root() / "validations.json", [item.to_dict() for item in state.validations])
    atomic_write_json(config.export_root() / "metrics.json", [item.to_dict() for item in state.metrics])
    atomic_write_json(config.export_root() / "findings.json", [item.to_dict() for item in state.findings])
    atomic_write_json(config.export_root() / "issues.json", [item.to_dict() for item in state.issues])
    atomic_write_json(config.export_root() / "research_notes.json", [item.to_dict() for item in state.research_notes])
    atomic_write_json(config.export_root() / "submissions.json", [item.to_dict() for item in state.submissions])


def load_state(config: WorkspaceConfig) -> WorkspaceState:
    _ensure_ledger(config)
    tables = {table: _table_rows(config, table, model) for table, model in TABLE_TO_MODEL.items()}
    with _open_ledger(config) as conn:
        row = conn.execute("SELECT payload FROM runtime_state WHERE id = 'singleton'").fetchone()
    runtime = RuntimeState.from_dict(__import__("json").loads(row["payload"])) if row else _default_runtime_state()
    if not any(tables.values()) and row is None:
        state = _empty_state(config)
        save_state(config, state)
        return state
    return WorkspaceState(
        work_items=tables["work_items"],
        experiments=tables["experiments"],
        runs=tables["runs"],
        stage_runs=tables["stage_runs"],
        agent_runs=tables["agent_runs"],
        specs=tables["specs"],
        validations=tables["validations"],
        metrics=tables["metrics"],
        findings=tables["findings"],
        issues=tables["issues"],
        research_notes=tables["research_notes"],
        submissions=tables["submissions"],
        submission_results=tables["submission_results"],
        runtime=runtime,
    )


def save_state(config: WorkspaceConfig, state: WorkspaceState) -> None:
    import json

    _ensure_ledger(config)
    for table in STATE_TABLES:
        _save_rows(config, table, getattr(state, table))
    with _open_ledger(config) as conn:
        conn.execute("DELETE FROM runtime_state")
        conn.execute(
            "INSERT INTO runtime_state (id, payload) VALUES ('singleton', ?)",
            (json.dumps(state.runtime.to_dict(), ensure_ascii=False),),
        )
    _snapshot_state(config, state)


def initialize_workspace(config: WorkspaceConfig, archive_legacy: bool = True, force: bool = False) -> WorkspaceState:
    if archive_legacy:
        archive_legacy_artifacts(config)
    ensure_layout(config)
    if force and config.ledger_path().exists():
        config.ledger_path().unlink()
    _ensure_ledger(config)
    state = load_state(config)
    if force:
        state = _empty_state(config)
        save_state(config, state)
        return state
    if not state.work_items:
        state = _empty_state(config)
        save_state(config, state)
    return state


def find_work_item(state: WorkspaceState, work_item_id: str) -> WorkItem:
    for item in state.work_items:
        if item.id == work_item_id:
            return item
    raise KeyError(f"Unknown work item: {work_item_id}")


def find_experiment(state: WorkspaceState, experiment_id: str) -> ExperimentSpec:
    for item in state.experiments:
        if item.id == experiment_id:
            return item
    raise KeyError(f"Unknown experiment: {experiment_id}")


def find_run(state: WorkspaceState, run_id: str) -> RunRecord:
    for item in state.runs:
        if item.run_id == run_id:
            return item
    raise KeyError(f"Unknown run: {run_id}")


def find_stage_run(state: WorkspaceState, stage_run_id: str) -> StageRun:
    for item in state.stage_runs:
        if item.stage_run_id == stage_run_id:
            return item
    raise KeyError(f"Unknown stage run: {stage_run_id}")
