from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CompetitionConfig:
    name: str
    slug: str
    url: str
    track: str
    description: str


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


@dataclass(frozen=True)
class AutomationConfig:
    monitor_interval_seconds: int
    report_interval_seconds: int
    submission_interval_hours: int
    default_timeout_minutes: int
    max_active_runs: int
    auto_execute_plans: bool
    auto_start_planned_runs: bool


@dataclass(frozen=True)
class AdapterConfig:
    research_command: str
    decision_command: str
    planner_command: str
    submission_command: str


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
    enable_gpu: bool
    is_private: bool


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

    def state_path(self, filename: str) -> Path:
        return self.root / self.paths.state_dir / filename

    def artifact_root(self) -> Path:
        return self.root / self.paths.artifact_dir

    def artifact_path(self, category: str, name: str = "") -> Path:
        base = self.artifact_root() / category
        return base / name if name else base

    def run_dir(self, run_id: str) -> Path:
        return self.artifact_path("runs", run_id)

    def report_path(self, name: str) -> Path:
        return self.artifact_path("reports", name)

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


@dataclass
class ExperimentSpec:
    id: str
    title: str
    hypothesis: str
    family: str
    config_path: str
    priority: int
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentSpec":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunRecord:
    run_id: str
    experiment_id: str
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
    decision_brief_path: str = ""
    research_summary_path: str = ""
    decision_record_path: str = ""
    plan_path: str = ""
    post_run_stage: str = ""
    post_run_error: str = ""
    post_run_updated_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionRecord:
    decision_id: str
    source_run_id: str
    experiment_id: str
    decision_type: str
    next_action: str
    evidence_strength: str
    root_cause: str
    why: str
    next_experiment_title: str = ""
    next_experiment_family: str = ""
    next_experiment_config: str = ""
    launch_policy: str = "auto"
    submission_recommendation: str = "no"
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionRecord":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SubmissionCandidate:
    id: str
    source_run_id: str
    experiment_id: str
    status: str
    primary_metric_name: str
    primary_metric_value: float | None
    secondary_metrics: dict[str, float]
    rationale: str
    notebook_dir: str
    created_at: str = ""
    updated_at: str = ""
    dedupe_key: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubmissionCandidate":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeState:
    initialized_at: str
    last_tick_at: str = ""
    last_report_at: str = ""
    active_run_ids: list[str] = field(default_factory=list)
    next_run_number: int = 1
    next_decision_number: int = 1
    next_submission_number: int = 1
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeState":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkspaceState:
    experiments: list[ExperimentSpec]
    runs: list[RunRecord]
    decisions: list[DecisionRecord]
    submissions: list[SubmissionCandidate]
    runtime: RuntimeState
