from __future__ import annotations

import csv
import html
from pathlib import Path

from kaggle_agent.control.submission import plan_submission_slots
from kaggle_agent.knowledge import write_experiment_conclusions
from kaggle_agent.layout import artifact_relative_path, current_attempt_slug, run_label_from_path, stage_label_from_path
from kaggle_agent.schema import RunRecord, WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import atomic_write_text, ensure_directory, replace_between_markers


AUTO_START = "<!-- AUTO-GENERATED:START -->"
AUTO_END = "<!-- AUTO-GENERATED:END -->"

ROOT_DOC_TEMPLATES = {
    "AGENTS.md": """# Kaggle Research OS Contract

Operate this repository as an agent-primary, artifact-driven, spec-enforced Kaggle research system.

## Hard Rules

- Keep `ledger.db` as machine truth and `artifacts/` as provenance.
- Keep `train_sed.py` as the stable BirdCLEF runtime bridge.
- Every autonomous stage must emit both `*.json` and `*.md`.
- Use CPU-only, internet-off defaults for scored BirdCLEF submission bundles.
- Ask first before any real Kaggle submission or destructive cleanup.
""",
    "COMPETITION.md": """# BirdCLEF 2026 Contract

- Competition: https://www.kaggle.com/competitions/birdclef-2026
- Primary metric: macro ROC-AUC over non-empty classes.
- Scored path: CPU notebook, internet off, 90 minutes maximum runtime.
- Daily submission limit: 5.
- Final selection limit: 2.
""",
    "PLAYBOOK.md": """# Playbook

- `init`: create the ledger, prompts, reports, and root working surface.
- `tick`: finalize runs, advance the strict stage chain, and refresh reports.
- `start-next`: launch the highest-priority runnable work item.
- `watch`: keep the research OS alive on a fixed cadence.
- `build-submission`: package a CPU-first candidate bundle and dry-run it locally.
""",
    "CHECKLIST.md": f"""# Checklist

Use this file for durable task framing. The auto-generated queue view stays below.

{AUTO_START}
{AUTO_END}
""",
    "JOURNAL.md": f"""# Journal

Append narrative notes above the auto-generated session ledger if needed.

{AUTO_START}
{AUTO_END}
""",
    "FINDINGS.md": f"""# Findings

Keep durable wins and losses here.

{AUTO_START}
{AUTO_END}
""",
    "ISSUES.md": f"""# Issues

Track persistent blockers, leakage risks, and validation drift here.

{AUTO_START}
{AUTO_END}
""",
    "SUBMISSIONS.md": f"""# Submissions

Track candidate history, anchors, and slot strategy here.

{AUTO_START}
{AUTO_END}
""",
}

PROMPT_TEMPLATES = {
    "evidence.md": "# Evidence Program\nCompress runtime outputs into evidence bundles with explicit metrics, artifacts, and root cause.\n",
    "report.md": "# Report Program\nWrite a run report that is useful for the next decision, not just for presentation.\n",
    "research.md": "# Research Program\nTurn the current failure or success mode into adopt-now, consider, and reject guidance.\n",
    "decision.md": "# Decision Program\nChoose the next action with priority on fixing the root cause first.\n",
    "plan.md": "# Plan Program\nProduce an executable experiment or submission plan with explicit config paths and dedupe keys.\n",
    "codegen.md": """# Codegen Program

Operate only inside the isolated codegen workspace.

Goals:
- Leave a runnable config inside the isolated workspace.
- Keep source edits inside the explicit allowlist.
- Keep verify artifacts out of the source tree.
- Let the harness own the final deterministic verify run and manifest export.

Rules:
- Edit only `train_sed.py`, `BirdCLEF-2026-Codebase/configs/**`, `BirdCLEF-2026-Codebase/src/**`, `BirdCLEF-2026-Codebase/train.py`, `BirdCLEF-2026-Codebase/inference.py`, and `BirdCLEF-2026-Codebase/scripts/**`.
- Never edit `BirdCLEF-2026-Codebase/outputs/**`, `BirdCLEF-2026-Codebase/models/**`, `BirdCLEF-2026-Codebase/birdclef-2026/**`, `state/**`, or `artifacts/**`.
- Never create notebooks or binary artifact files such as `.ipynb`, `.npz`, `.pkl`, `.pt`, or `.ckpt`.
- Do not return patch text, YAML blobs, or JSON manifests in the final message.
- Finish with a short plain-text summary of source edits only.
""",
    "critic.md": "# Critic Program\nReview the proposed bundle for correctness, safety, and obvious regressions.\n",
    "submission.md": "# Submission Program\nPrepare CPU-first Kaggle submission bundles and reason about scarce submission slots.\n",
}


def best_run(state: WorkspaceState) -> RunRecord | None:
    candidates = [run for run in state.runs if run.status == "succeeded" and run.primary_metric_value is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.primary_metric_value or float("-inf"), item.completed_at, item.run_id))


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def ensure_surface_files(config: WorkspaceConfig) -> None:
    for name, template in ROOT_DOC_TEMPLATES.items():
        path = config.root_doc_path(name)
        if not path.exists():
            atomic_write_text(path, template)
    for name, template in PROMPT_TEMPLATES.items():
        path = config.prompt_path(name)
        if not path.exists():
            atomic_write_text(path, template)


def _replace_auto_block(config: WorkspaceConfig, name: str, content: str) -> None:
    path = config.root_doc_path(name)
    original = path.read_text(encoding="utf-8") if path.exists() else ROOT_DOC_TEMPLATES.get(name, "")
    updated = replace_between_markers(original, AUTO_START, AUTO_END, content)
    atomic_write_text(path, updated)


def _latest_stage_label(state: WorkspaceState, stage_run_id: str) -> str:
    if not stage_run_id:
        return "n/a"
    stage_run = next((item for item in state.stage_runs if item.stage_run_id == stage_run_id), None)
    if stage_run is None:
        return "n/a"
    return stage_label_from_path(
        stage_run.output_dir,
        stage_run.stage_name,
        stage_status=stage_run.status,
        validator_status=stage_run.validator_status,
    )


def _surface_updates(config: WorkspaceConfig, state: WorkspaceState) -> None:
    attempt_slug = current_attempt_slug(state.runtime)
    checklist = [f"- Current attempt: `{attempt_slug}`", ""]
    for work_item in sorted(state.work_items, key=lambda item: (item.priority, item.created_at, item.id)):
        run_display = next((run_label_from_path(run.run_dir) for run in state.runs if run.run_id == work_item.latest_run_id), "") or "n/a"
        checklist.append(
            f"- [{'x' if work_item.status in {'complete', 'submitted'} else ' '}] `{work_item.id}` | {work_item.status} | p{work_item.priority} | {work_item.title} | run={run_display} | stage={_latest_stage_label(state, work_item.latest_stage_run_id)}"
        )
    _replace_auto_block(config, "CHECKLIST.md", "\n".join(checklist) or "- No queued work items.")

    journal = [f"- Current attempt: `{attempt_slug}`", ""]
    for run in state.runs[-12:]:
        latest_stage_label = _latest_stage_label(state, run.latest_stage_run_id)
        journal.append(
            f"- `{run_label_from_path(run.run_dir) or run.run_id}` | {run.status} | cursor={run.stage_cursor or 'n/a'} | latest_stage={latest_stage_label} | metric={run.primary_metric_name}={run.primary_metric_value}"
        )
    _replace_auto_block(config, "JOURNAL.md", "\n".join(journal) or "- No runs yet.")

    findings = [f"- Current attempt: `{attempt_slug}`", ""]
    findings.extend(f"- `{item.run_id}` | {item.title} | {item.summary}" for item in state.findings[-20:])
    _replace_auto_block(config, "FINDINGS.md", "\n".join(findings) or "- No findings yet.")

    issues = [f"- Current attempt: `{attempt_slug}`", ""]
    issues.extend(f"- `{item.run_id}` | {item.title} | {item.summary}" for item in state.issues[-20:])
    _replace_auto_block(config, "ISSUES.md", "\n".join(issues) or "- No issues yet.")

    slot_plan = plan_submission_slots(config, state)
    submissions = [f"- Current attempt: `{attempt_slug}`", ""]
    submissions.extend(
        [
        f"- `{item.id}` | {item.status} | cpu_ready={item.cpu_ready} | pred_lb={item.predicted_public_lb}"
        for item in state.submissions[-20:]
        ]
    )
    submissions.extend(
        [
            "",
            f"- Remaining daily slots: {slot_plan['remaining_daily_slots']}",
            f"- Remaining final slots: {slot_plan['remaining_final_slots']}",
        ]
    )
    _replace_auto_block(config, "SUBMISSIONS.md", "\n".join(submissions) or "- No submission candidates yet.")


def _overview_markdown(config: WorkspaceConfig, state: WorkspaceState) -> str:
    leader = best_run(state)
    slot_plan = plan_submission_slots(config, state)
    attempt_slug = current_attempt_slug(state.runtime)
    lines = [
        f"# {config.competition.name} Research OS",
        "",
        f"- Current attempt: `{attempt_slug}`",
        f"- Competition: {config.competition.url}",
        f"- Active runs: {', '.join(state.runtime.active_run_ids) if state.runtime.active_run_ids else 'none'}",
        f"- Work items: {len(state.work_items)}",
        f"- Stage runs: {len(state.stage_runs)}",
        f"- Last tick: {state.runtime.last_tick_at or 'n/a'}",
        f"- Last report: {state.runtime.last_report_at or 'n/a'}",
        f"- Remaining daily submission slots: {slot_plan['remaining_daily_slots']}",
        "",
        "## Best Run",
    ]
    if leader is None:
        lines.append("- No successful run yet.")
    else:
        lines.extend(
            [
                f"- Run: `{run_label_from_path(leader.run_dir) or leader.run_id}`",
                f"- Experiment: `{leader.experiment_id}`",
                f"- Metric: `{leader.primary_metric_name}={leader.primary_metric_value:.6f}`",
                f"- Verdict: `{leader.verdict}`",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_html_page(title: str, sections: list[tuple[str, str]]) -> str:
    cards = "\n".join(
        f"<section class='card'><h2>{html.escape(heading)}</h2>{body}</section>"
        for heading, body in sections
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --card: #fffdf9;
      --ink: #1f2937;
      --line: #d6cec2;
      --accent: #0f766e;
    }}
    body {{ margin: 0; background: linear-gradient(135deg, #f7f2e9, #ece3d4); color: var(--ink); font-family: Georgia, serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 18px 48px; }}
    h1 {{ margin: 0 0 20px; }}
    .grid {{ display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 18px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ background: #faf6ee; }}
    code {{ font-family: Consolas, monospace; }}
    ul {{ margin: 0; padding-left: 18px; }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
  <div class="grid">{cards}</div>
</main>
</body>
</html>
"""


def write_reports(config: WorkspaceConfig, state: WorkspaceState) -> None:
    ensure_surface_files(config)
    _surface_updates(config, state)
    write_experiment_conclusions(config, state)

    leader = best_run(state)
    slot_plan = plan_submission_slots(config, state)
    atomic_write_text(config.report_path("overview.md"), _overview_markdown(config, state))

    master_sections = [
        (
            "Situation",
            "<ul>"
            f"<li>Current attempt: {html.escape(current_attempt_slug(state.runtime))}</li>"
            f"<li>Active runs: {html.escape(', '.join(state.runtime.active_run_ids) if state.runtime.active_run_ids else 'none')}</li>"
            f"<li>Queued work items: {len([item for item in state.work_items if item.status == 'queued'])}</li>"
            f"<li>Completed work items: {len([item for item in state.work_items if item.status in {'complete', 'submitted'}])}</li>"
            f"<li>Remaining daily slots: {slot_plan['remaining_daily_slots']}</li>"
            "</ul>",
        ),
            (
                "Best Run",
                (
                    "<p>No successful run yet.</p>"
                    if leader is None
                    else f"<p><strong>{html.escape(run_label_from_path(leader.run_dir) or leader.run_id)}</strong><br>{html.escape(leader.experiment_id)}"
                    f"<br>{html.escape(leader.primary_metric_name)}={leader.primary_metric_value:.6f}</p>"
                ),
            ),
        (
            "Next Priorities",
            "<ul>" + "".join(
                f"<li>{html.escape(item.title)} ({item.status}, p{item.priority})</li>"
                for item in sorted(state.work_items, key=lambda value: (value.priority, value.created_at, value.id))[:8]
            ) + "</ul>",
        ),
    ]
    atomic_write_text(config.report_path("master_report.html"), _render_html_page("Master Report", master_sections))

    experiment_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item.id)}</td>"
        f"<td>{html.escape(item.status)}</td>"
        f"<td>{item.priority}</td>"
        f"<td>{html.escape(item.family)}</td>"
        f"<td><code>{html.escape(item.config_path)}</code></td>"
        "</tr>"
        for item in state.experiments
    ) or "<tr><td colspan='5'>No experiments</td></tr>"
    run_rows = "".join(
        "<tr>"
        f"<td>{html.escape(run_label_from_path(item.run_dir) or item.run_id)}</td>"
        f"<td>{html.escape(item.status)}</td>"
        f"<td>{html.escape(item.stage_cursor or 'complete')}</td>"
        f"<td>{html.escape(item.primary_metric_name)}</td>"
        f"<td>{'' if item.primary_metric_value is None else f'{item.primary_metric_value:.6f}'}</td>"
        "</tr>"
        for item in state.runs[-20:]
    ) or "<tr><td colspan='5'>No runs</td></tr>"
    atomic_write_text(
        config.report_path("experiment_report.html"),
        _render_html_page(
            "Experiment Report",
            [
                ("Experiments", f"<table><thead><tr><th>ID</th><th>Status</th><th>Priority</th><th>Family</th><th>Config</th></tr></thead><tbody>{experiment_rows}</tbody></table>"),
                ("Recent Runs", f"<table><thead><tr><th>Run</th><th>Status</th><th>Cursor</th><th>Metric</th><th>Value</th></tr></thead><tbody>{run_rows}</tbody></table>"),
            ],
        ),
    )

    finding_items = "".join(f"<li>{html.escape(item.title)}: {html.escape(item.summary)}</li>" for item in state.findings[-15:]) or "<li>No findings yet.</li>"
    issue_items = "".join(f"<li>{html.escape(item.title)}: {html.escape(item.summary)}</li>" for item in state.issues[-15:]) or "<li>No issues yet.</li>"
    queued_items = "".join(
        f"<li>{html.escape(item.title)} ({html.escape(item.status)})</li>"
        for item in sorted(state.work_items, key=lambda value: (value.priority, value.created_at, value.id))[:10]
    ) or "<li>No work items.</li>"
    atomic_write_text(
        config.report_path("discovery_report.html"),
        _render_html_page(
            "Discovery Report",
            [("Findings", f"<ul>{finding_items}</ul>"), ("Issues", f"<ul>{issue_items}</ul>"), ("Queue", f"<ul>{queued_items}</ul>")],
        ),
    )

    submission_items = "".join(
        f"<li>{html.escape(item.id)} | {html.escape(item.status)} | cpu_ready={item.cpu_ready} | pred_lb={item.predicted_public_lb}</li>"
        for item in state.submissions[-15:]
    ) or "<li>No submission candidates.</li>"
    atomic_write_text(
        config.report_path("submission_report.html"),
        _render_html_page(
            "Submission Report",
            [
                ("Candidates", f"<ul>{submission_items}</ul>"),
                (
                    "Slot Plan",
                    "<ul>"
                    f"<li>Remaining daily slots: {slot_plan['remaining_daily_slots']}</li>"
                    f"<li>Remaining final slots: {slot_plan['remaining_final_slots']}</li>"
                    "</ul>",
                ),
            ],
        ),
    )

    research_items = "".join(
        f"<li>{html.escape(item.title)} | {html.escape(item.stance)} | {html.escape(item.summary)}</li>"
        for item in state.research_notes[-15:]
    ) or "<li>No research notes yet.</li>"
    atomic_write_text(
        config.report_path("research_report.html"),
        _render_html_page("Research Report", [("Research Notes", f"<ul>{research_items}</ul>")]),
    )

    _write_csv(
        config.report_path("work_items.csv"),
        ["id", "title", "status", "priority", "family", "latest_run_id"],
        [[item.id, item.title, item.status, str(item.priority), item.family, item.latest_run_id] for item in state.work_items],
    )
    _write_csv(
        config.report_path("runs.csv"),
        ["run_id", "experiment_id", "work_item_id", "status", "stage_cursor", "metric_name", "metric_value"],
        [
            [
                item.run_id,
                item.experiment_id,
                item.work_item_id,
                item.status,
                item.stage_cursor,
                item.primary_metric_name,
                "" if item.primary_metric_value is None else f"{item.primary_metric_value:.6f}",
            ]
            for item in state.runs
        ],
    )
    _write_csv(
        config.report_path("metrics.csv"),
        ["metric_id", "run_id", "metric_name", "value", "split", "trust_level", "is_primary"],
        [
            [
                item.metric_id,
                item.run_id,
                item.metric_name,
                f"{item.value:.6f}",
                item.split,
                item.trust_level,
                str(item.is_primary),
            ]
            for item in state.metrics
        ],
    )
    _write_csv(
        config.report_path("submissions.csv"),
        ["id", "source_run_id", "status", "cpu_ready", "predicted_public_lb"],
        [
            [
                item.id,
                item.source_run_id,
                item.status,
                str(item.cpu_ready),
                "" if item.predicted_public_lb is None else f"{item.predicted_public_lb:.6f}",
            ]
            for item in state.submissions
        ],
    )
    state.runtime.last_report_at = state.runtime.last_tick_at or state.runtime.initialized_at
