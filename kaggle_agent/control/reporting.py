from __future__ import annotations

import csv
import html
from pathlib import Path

from kaggle_agent.schema import RunRecord, WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import atomic_write_text, ensure_directory


def best_run(state: WorkspaceState) -> RunRecord | None:
    succeeded = [run for run in state.runs if run.status == "succeeded" and run.primary_metric_value is not None]
    if not succeeded:
        return None
    return max(succeeded, key=lambda item: (item.primary_metric_value or float("-inf"), item.completed_at))


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _overview_markdown(config: WorkspaceConfig, state: WorkspaceState) -> str:
    leader = best_run(state)
    lines = [
        f"# {config.competition.name} Overview",
        "",
        f"- Competition: {config.competition.url}",
        f"- Primary metric: {config.metrics.primary}",
        f"- Secondary metrics: {', '.join(config.metrics.secondary) if config.metrics.secondary else 'none'}",
        f"- Active runs: {', '.join(state.runtime.active_run_ids) if state.runtime.active_run_ids else 'none'}",
        f"- Last tick: {state.runtime.last_tick_at or 'n/a'}",
        f"- Last report: {state.runtime.last_report_at or 'n/a'}",
        "",
        "## Best Run",
    ]
    if leader is None:
        lines.append("- No successful run yet.")
    else:
        lines.extend(
            [
                f"- Run id: `{leader.run_id}`",
                f"- Experiment: `{leader.experiment_id}`",
                f"- Primary metric: {leader.primary_metric_name}={leader.primary_metric_value:.6f}",
                f"- Secondary metrics: {leader.secondary_metrics}",
            ]
        )
    lines.extend(["", "## Experiments"])
    for experiment in sorted(state.experiments, key=lambda item: (item.priority, item.id)):
        lines.append(
            f"- `{experiment.id}` | {experiment.status} | p{experiment.priority} | family={experiment.family} | config=`{experiment.config_path}`"
        )
    lines.extend(["", "## Latest Decisions"])
    for decision in sorted(state.decisions, key=lambda item: item.created_at, reverse=True)[:5]:
        lines.append(
            f"- `{decision.decision_id}` | run `{decision.source_run_id}` | {decision.decision_type} | next={decision.next_action}"
        )
    if not state.decisions:
        lines.append("- No decision records yet.")
    lines.append("")
    return "\n".join(lines)


def _dashboard_html(config: WorkspaceConfig, state: WorkspaceState) -> str:
    leader = best_run(state)
    experiment_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(experiment.id)}</td>"
        f"<td>{html.escape(experiment.status)}</td>"
        f"<td>{experiment.priority}</td>"
        f"<td>{html.escape(experiment.family)}</td>"
        f"<td><code>{html.escape(experiment.config_path)}</code></td>"
        "</tr>"
        for experiment in sorted(state.experiments, key=lambda item: (item.priority, item.id))
    )
    decision_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(decision.decision_id)}</td>"
        f"<td>{html.escape(decision.source_run_id)}</td>"
        f"<td>{html.escape(decision.decision_type)}</td>"
        f"<td>{html.escape(decision.next_action)}</td>"
        f"<td>{html.escape(decision.root_cause)}</td>"
        "</tr>"
        for decision in sorted(state.decisions, key=lambda item: item.created_at, reverse=True)[:10]
    )
    best_block = (
        f"<strong>{html.escape(leader.run_id)}</strong><br>{html.escape(leader.experiment_id)}<br>"
        f"{html.escape(leader.primary_metric_name)}={leader.primary_metric_value:.6f}"
        if leader and leader.primary_metric_value is not None
        else "No successful run yet."
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(config.competition.name)} Dashboard</title>
  <style>
    :root {{
      --bg: #f5f3ef;
      --card: #fffdf9;
      --ink: #1f2937;
      --line: #d6d3d1;
      --accent: #0f766e;
    }}
    body {{ margin: 0; font-family: Georgia, serif; background: linear-gradient(135deg, #f8f5ef, #ece6da); color: var(--ink); }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 32px 20px 56px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 18px; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px 12px; text-align: left; }}
    th {{ background: #faf7f0; }}
    code {{ font-family: Consolas, monospace; }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(config.competition.name)} Dashboard</h1>
  <div class="grid">
    <div class="card"><strong>Primary Metric</strong><br>{html.escape(config.metrics.primary)}</div>
    <div class="card"><strong>Active Runs</strong><br>{html.escape(', '.join(state.runtime.active_run_ids) if state.runtime.active_run_ids else 'none')}</div>
    <div class="card"><strong>Best Run</strong><br>{best_block}</div>
  </div>
  <div class="card">
    <h2>Experiments</h2>
    <table>
      <thead><tr><th>ID</th><th>Status</th><th>Priority</th><th>Family</th><th>Config</th></tr></thead>
      <tbody>{experiment_rows or '<tr><td colspan="5">No experiments.</td></tr>'}</tbody>
    </table>
  </div>
  <div class="card">
    <h2>Latest Decisions</h2>
    <table>
      <thead><tr><th>ID</th><th>Run</th><th>Type</th><th>Next</th><th>Root Cause</th></tr></thead>
      <tbody>{decision_rows or '<tr><td colspan="5">No decision records.</td></tr>'}</tbody>
    </table>
  </div>
</main>
</body>
</html>
"""


def write_reports(config: WorkspaceConfig, state: WorkspaceState) -> None:
    state.runtime.last_report_at = state.runtime.last_tick_at or state.runtime.initialized_at
    atomic_write_text(config.report_path("overview.md"), _overview_markdown(config, state))
    atomic_write_text(config.report_path("dashboard.html"), _dashboard_html(config, state))
    _write_csv(
        config.report_path("experiments.csv"),
        ["id", "status", "priority", "family", "config_path", "latest_run_id"],
        [
            [
                item.id,
                item.status,
                str(item.priority),
                item.family,
                item.config_path,
                item.latest_run_id,
            ]
            for item in sorted(state.experiments, key=lambda value: (value.priority, value.id))
        ],
    )
    _write_csv(
        config.report_path("runs.csv"),
        ["run_id", "experiment_id", "status", "primary_metric_name", "primary_metric_value", "completed_at"],
        [
            [
                item.run_id,
                item.experiment_id,
                item.status,
                item.primary_metric_name,
                "" if item.primary_metric_value is None else f"{item.primary_metric_value:.6f}",
                item.completed_at,
            ]
            for item in state.runs
        ],
    )
