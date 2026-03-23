from __future__ import annotations

from pathlib import Path

from kaggle_agent.decision.helpers import load_run_result
from kaggle_agent.schema import WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import atomic_write_text


def build_decision_brief(config: WorkspaceConfig, state: WorkspaceState, run_id: str) -> Path:
    run = next(item for item in state.runs if item.run_id == run_id)
    experiment = next(item for item in state.experiments if item.id == run.experiment_id)
    result = load_run_result(run)
    recent_family_runs = [
        item for item in state.runs if item.experiment_id != run.experiment_id and item.status == "succeeded"
    ]
    lines = [
        f"# Decision Brief for {run.run_id}",
        "",
        f"- Experiment: {experiment.id}",
        f"- Title: {experiment.title}",
        f"- Family: {experiment.family}",
        f"- Config: `{experiment.config_path}`",
        f"- Run status: {run.status}",
        f"- Primary metric: {run.primary_metric_name} = {run.primary_metric_value}",
        f"- Secondary metrics: {run.secondary_metrics}",
        f"- Root cause: {result.get('root_cause', run.error or 'not captured')}",
        "",
        "## Runtime Summary",
        result.get("summary_markdown", "_No runtime summary provided._"),
        "",
        "## Recent Successful Runs",
    ]
    if recent_family_runs:
        for item in recent_family_runs[-5:]:
            lines.append(
                f"- `{item.run_id}` | exp `{item.experiment_id}` | {item.primary_metric_name}={item.primary_metric_value}"
            )
    else:
        lines.append("- No prior successful runs.")
    path = config.artifact_path("decision_briefs", f"{run.run_id}.md")
    atomic_write_text(path, "\n".join(lines) + "\n")
    run.decision_brief_path = str(path)
    return path
