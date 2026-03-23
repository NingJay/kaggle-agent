from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from kaggle_agent.schema import (
    AdapterConfig,
    AutomationConfig,
    CompetitionConfig,
    DataConfig,
    KaggleConfig,
    MetricsConfig,
    PathsConfig,
    RuntimeConfig,
    WorkspaceConfig,
)


WORKSPACE_FILE = "workspace.toml"


def load_workspace_config(root: Path | None = None) -> WorkspaceConfig:
    workspace_root = (root or Path.cwd()).resolve()
    config_path = workspace_root / WORKSPACE_FILE
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    competition = CompetitionConfig(**raw["competition"])
    metrics = MetricsConfig(
        primary=raw["metrics"]["primary"],
        secondary=list(raw["metrics"].get("secondary", [])),
    )
    data = DataConfig(**raw["data"])
    paths = PathsConfig(**raw["paths"])
    automation = AutomationConfig(**raw["automation"])
    adapters = AdapterConfig(**raw["adapters"])
    runtime = RuntimeConfig(**raw["runtime"])
    kaggle = KaggleConfig(**raw["kaggle"])
    notes = list(raw.get("notes", []))
    return WorkspaceConfig(
        root=workspace_root,
        competition=competition,
        metrics=metrics,
        data=data,
        paths=paths,
        automation=automation,
        adapters=adapters,
        runtime=runtime,
        kaggle=kaggle,
        notes=notes,
    )
