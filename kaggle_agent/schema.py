from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class _Serializable:
    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompetitionConfig:
    name: str
    slug: str
    url: str
    track: str
    description: str
    contract: str = "birdclef_2026"


@dataclass(frozen=True)
class MetricsConfig:
    primary: str
    secondary: list[str]


@dataclass(frozen=True)
class DataConfig:
    root: str
    train_csv: str
    taxonomy_csv: str
    sample_submission_csv: str
    train_audio_dir: str
    train_soundscapes_dir: str
    train_soundscapes_labels_csv: str
    test_soundscapes_dir: str = "test_soundscapes"
    perch_cache_dir: str = ""
    perch_model_dir: str = ""


@dataclass(frozen=True)
class PathsConfig:
    state_dir: str
    artifact_dir: str
    legacy_dir: str
    runtime_dir: str
    knowledge_dir: str = "knowledge"
    prompt_dir: str = "prompts"
    report_dir: str = "reports"


@dataclass(frozen=True)
class AutomationConfig:
    monitor_interval_seconds: int
    report_interval_seconds: int
    submission_interval_hours: int
    default_timeout_minutes: int
    max_active_runs: int
    auto_execute_plans: bool
    auto_start_planned_runs: bool
    strict_stage_graph: bool = True


@dataclass(frozen=True)
class AdapterConfig:
    evidence_command: str = ""
    report_command: str = ""
    research_command: str = ""
    decision_command: str = ""
    planner_command: str = ""
    codegen_command: str = ""
    critic_command: str = ""
    submission_command: str = ""


@dataclass(frozen=True)
class RuntimeConfig:
    conda_env: str
    shell_init: str
    train_workdir: str
    train_entrypoint: str
    generated_config_dir: str


@dataclass(frozen=True)
class KaggleConfig:
    username: str
    model_dataset_id: str
    kernel_slug: str
    enable_gpu: bool = False
    enable_internet: bool = False
    is_private: bool = True
    cpu_submission_only: bool = True
    scored_max_runtime_minutes: int = 90
    max_daily_submissions: int = 5
    max_final_submissions: int = 2
    dataset_sources: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkspaceConfig:
    root: Path
    competition: CompetitionConfig
    metrics: MetricsConfig
    data: DataConfig
    paths: PathsConfig
    automation: AutomationConfig
    adapters: AdapterConfig
    runtime: RuntimeConfig
    kaggle: KaggleConfig
    notes: list[str]

    def state_root(self) -> Path:
        return self.root / self.paths.state_dir

    def state_path(self, filename: str) -> Path:
        return self.state_root() / filename

    def ledger_path(self) -> Path:
        return self.state_root() / "ledger.db"

    def export_root(self) -> Path:
        return self.state_root() / "exports"

    def snapshot_root(self) -> Path:
        return self.state_root() / "snapshots"

    def artifact_root(self) -> Path:
        return self.root / self.paths.artifact_dir

    def artifact_path(self, category: str, name: str = "") -> Path:
        base = self.artifact_root() / category
        return base / name if name else base

    def stage_dir(self, stage_name: str, token: str) -> Path:
        return self.artifact_path(stage_name, token)

    def run_dir(self, run_id: str) -> Path:
        return self.artifact_path("runs", run_id)

    def report_root(self) -> Path:
        return self.root / self.paths.report_dir

    def report_path(self, name: str) -> Path:
        return self.report_root() / name

    def prompt_root(self) -> Path:
        return self.root / self.paths.prompt_dir

    def prompt_path(self, name: str) -> Path:
        return self.prompt_root() / name

    def legacy_root(self) -> Path:
        return self.root / self.paths.legacy_dir

    def runtime_root(self) -> Path:
        return self.root / self.paths.runtime_dir

    def knowledge_root(self) -> Path:
        return self.root / self.paths.knowledge_dir

    def knowledge_path(self, category: str = "", name: str = "") -> Path:
        base = self.knowledge_root()
        if category:
            base = base / category
        return base / name if name else base

    def generated_config_root(self) -> Path:
        return self.root / self.runtime.generated_config_dir

    def lock_path(self) -> Path:
        return self.root / ".kaggle_agent.lock"

    def root_doc_path(self, name: str) -> Path:
        return self.root / name


@dataclass
class WorkItem(_Serializable):
    id: str
    title: str
    work_type: str
    family: str
    priority: int
    status: str = "queued"
    config_path: str = ""
    pipeline: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    latest_run_id: str = ""
    latest_stage_run_id: str = ""
    latest_spec_id: str = ""
    source_run_id: str = ""
    source_stage_run_id: str = ""
    source_decision_id: str = ""
    dedupe_key: str = ""
    notes: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class ExperimentSpec(_Serializable):
    id: str
    title: str
    hypothesis: str
    family: str
    config_path: str
    priority: int
    work_item_id: str = ""
    spec_id: str = ""
    depends_on: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    status: str = "queued"
    launch_mode: str = "background"
    latest_run_id: str = ""
    latest_decision_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    notes: list[str] = field(default_factory=list)
    dedupe_key: str = ""
    source_decision_id: str = ""


@dataclass
class RunRecord(_Serializable):
    run_id: str
    experiment_id: str
    work_item_id: str
    spec_id: str
    status: str
    command: str
    cwd: str
    run_dir: str
    log_path: str
    started_at: str = ""
    completed_at: str = ""
    pid: int | None = None
    gpu_id: str = ""
    primary_metric_name: str = ""
    primary_metric_value: float | None = None
    secondary_metrics: dict[str, float] = field(default_factory=dict)
    error: str = ""
    artifact_paths: dict[str, str] = field(default_factory=dict)
    root_cause: str = ""
    verdict: str = ""
    stage_cursor: str = ""
    stage_error: str = ""
    stage_updated_at: str = ""
    latest_stage_run_id: str = ""


@dataclass
class StageRun(_Serializable):
    stage_run_id: str
    run_id: str
    work_item_id: str
    stage_name: str
    status: str
    input_ref: str
    output_dir: str
    output_json_path: str
    output_md_path: str
    spec_path: str = ""
    adapter_name: str = ""
    prompt_path: str = ""
    schema_path: str = ""
    provider_meta_path: str = ""
    validator_status: str = ""
    error: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass
class AgentRun(_Serializable):
    agent_run_id: str
    stage_run_id: str
    agent_role: str
    adapter_command: str
    prompt_path: str
    status: str
    provider: str = ""
    model: str = ""
    schema_path: str = ""
    output_json_path: str = ""
    output_md_path: str = ""
    raw_stdout_path: str = ""
    raw_stderr_path: str = ""
    raw_event_log_path: str = ""
    provider_meta_path: str = ""
    session_id: str = ""
    thread_id: str = ""
    exit_code: int | None = None
    log_path: str = ""
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""


@dataclass
class SpecRecord(_Serializable):
    spec_id: str
    work_item_id: str
    source_stage_run_id: str
    spec_type: str
    title: str
    family: str
    config_path: str
    payload_path: str
    launch_mode: str = "background"
    status: str = "draft"
    dedupe_key: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass
class ValidationRecord(_Serializable):
    validation_id: str
    work_item_id: str
    source_stage_run_id: str
    spec_id: str
    status: str
    summary: str
    output_json_path: str
    output_md_path: str
    created_at: str = ""


@dataclass
class MetricObservation(_Serializable):
    metric_id: str
    run_id: str
    metric_name: str
    split: str
    domain_scope: str
    trust_level: str
    postproc_variant: str
    evaluator_version: str
    is_primary: bool
    value: float
    notes: str = ""
    created_at: str = ""


@dataclass
class FindingRecord(_Serializable):
    finding_id: str
    run_id: str
    title: str
    summary: str
    severity: str
    status: str
    dedupe_key: str = ""
    created_at: str = ""


@dataclass
class IssueRecord(_Serializable):
    issue_id: str
    run_id: str
    title: str
    summary: str
    severity: str
    status: str
    dedupe_key: str = ""
    created_at: str = ""


@dataclass
class ResearchNoteRecord(_Serializable):
    note_id: str
    run_id: str
    title: str
    summary: str
    stance: str
    source_type: str
    created_at: str = ""


@dataclass
class SubmissionCandidate(_Serializable):
    id: str
    source_run_id: str
    experiment_id: str
    status: str
    primary_metric_name: str
    primary_metric_value: float | None
    secondary_metrics: dict[str, float]
    predicted_public_lb: float | None = None
    predicted_public_lb_std: float | None = None
    rationale: str = ""
    notebook_dir: str = ""
    candidate_json_path: str = ""
    candidate_md_path: str = ""
    dry_run_json_path: str = ""
    calibration_json_path: str = ""
    cpu_ready: bool = False
    created_at: str = ""
    updated_at: str = ""
    dedupe_key: str = ""


@dataclass
class SubmissionResult(_Serializable):
    result_id: str
    candidate_id: str
    status: str
    public_lb: float | None = None
    private_lb: float | None = None
    kaggle_submission_id: str = ""
    notes: str = ""
    created_at: str = ""


@dataclass
class RuntimeState(_Serializable):
    initialized_at: str
    last_tick_at: str = ""
    last_report_at: str = ""
    active_run_ids: list[str] = field(default_factory=list)
    next_run_number: int = 1
    next_decision_number: int = 1
    next_submission_number: int = 1
    next_work_item_number: int = 1
    next_stage_number: int = 1
    next_agent_run_number: int = 1
    next_spec_number: int = 1
    next_validation_number: int = 1
    next_metric_number: int = 1
    next_finding_number: int = 1
    next_issue_number: int = 1
    next_note_number: int = 1
    notes: list[str] = field(default_factory=list)


@dataclass
class WorkspaceState(_Serializable):
    work_items: list[WorkItem]
    experiments: list[ExperimentSpec]
    runs: list[RunRecord]
    stage_runs: list[StageRun]
    agent_runs: list[AgentRun]
    specs: list[SpecRecord]
    validations: list[ValidationRecord]
    metrics: list[MetricObservation]
    findings: list[FindingRecord]
    issues: list[IssueRecord]
    research_notes: list[ResearchNoteRecord]
    submissions: list[SubmissionCandidate]
    submission_results: list[SubmissionResult]
    runtime: RuntimeState
