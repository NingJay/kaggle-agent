from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from kaggle_agent.control.submission import plan_submission_slots
from kaggle_agent.knowledge import write_experiment_conclusions
from kaggle_agent.layout import (
    STAGE_ORDER,
    artifact_relative_path,
    current_attempt_slug,
    run_label_from_path,
    stage_label_from_path,
    visible_run_ids,
    visible_runs,
    visible_stage_runs,
    visible_work_items,
)
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
    "decision.md": "# Decision Program\nChoose the next action with priority on fixing the root cause first while preserving room for branch search when high-value axes remain.\n",
    "plan.md": "# Plan Program\nProduce an executable experiment or submission plan with explicit config paths, dedupe keys, and a small branch portfolio when justified.\n",
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
    candidates = [run for run in visible_runs(state) if run.status == "succeeded" and run.primary_metric_value is not None]
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
    work_items = visible_work_items(state)
    runs = visible_runs(state)
    run_ids = visible_run_ids(state)
    checklist = [f"- Current attempt: `{attempt_slug}`", ""]
    for work_item in sorted(work_items, key=lambda item: (item.priority, item.created_at, item.id)):
        run_display = next((run_label_from_path(run.run_dir) for run in runs if run.run_id == work_item.latest_run_id), "") or "n/a"
        branch_bits = []
        if work_item.portfolio_id:
            branch_bits.append(f"portfolio={work_item.portfolio_id}")
        if work_item.branch_role:
            branch_bits.append(f"branch={work_item.branch_role}")
        if work_item.idea_class:
            branch_bits.append(f"idea={work_item.idea_class}")
        if work_item.policy_trace:
            branch_bits.append(f"policy={work_item.policy_trace[0]}")
        if work_item.lifecycle_template and work_item.lifecycle_template != "recursive_experiment":
            branch_bits.append(f"lifecycle={work_item.lifecycle_template}")
        if work_item.target_run_id:
            branch_bits.append(f"target_run={work_item.target_run_id}")
        branch_summary = f" | {' '.join(branch_bits)}" if branch_bits else ""
        checklist.append(
            f"- [{'x' if work_item.status in {'complete', 'submitted'} else ' '}] `{work_item.id}` | {work_item.status} | p{work_item.priority} | {work_item.title}{branch_summary} | run={run_display} | stage={_latest_stage_label(state, work_item.latest_stage_run_id)}"
        )
    _replace_auto_block(config, "CHECKLIST.md", "\n".join(checklist) or "- No queued work items.")

    journal = [f"- Current attempt: `{attempt_slug}`", ""]
    for run in runs[-12:]:
        latest_stage_label = _latest_stage_label(state, run.latest_stage_run_id)
        work_item = next((item for item in work_items if item.id == run.work_item_id), None)
        branch_bits = []
        if work_item is not None and work_item.branch_role:
            branch_bits.append(f"branch={work_item.branch_role}")
        if work_item is not None and work_item.idea_class:
            branch_bits.append(f"idea={work_item.idea_class}")
        if work_item is not None and work_item.portfolio_id:
            branch_bits.append(f"portfolio={work_item.portfolio_id}")
        if work_item is not None and work_item.policy_trace:
            branch_bits.append(f"policy={work_item.policy_trace[0]}")
        if run.lifecycle_template and run.lifecycle_template != "recursive_experiment":
            branch_bits.append(f"lifecycle={run.lifecycle_template}")
        branch_suffix = f" | {' '.join(branch_bits)}" if branch_bits else ""
        journal.append(
            f"- `{run_label_from_path(run.run_dir) or run.run_id}` | {run.status} | cursor={run.stage_cursor or 'n/a'} | latest_stage={latest_stage_label} | metric={run.primary_metric_name}={run.primary_metric_value}{branch_suffix}"
        )
    _replace_auto_block(config, "JOURNAL.md", "\n".join(journal) or "- No runs yet.")

    findings = [f"- Current attempt: `{attempt_slug}`", ""]
    findings.extend(f"- `{item.run_id}` | {item.title} | {item.summary}" for item in state.findings[-20:] if item.run_id in run_ids)
    _replace_auto_block(config, "FINDINGS.md", "\n".join(findings) or "- No findings yet.")

    issues = [f"- Current attempt: `{attempt_slug}`", ""]
    issues.extend(f"- `{item.run_id}` | {item.title} | {item.summary}" for item in state.issues[-20:] if item.run_id in run_ids)
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
    work_items = visible_work_items(state)
    stage_runs = visible_stage_runs(state)
    lines = [
        f"# {config.competition.name} Research OS",
        "",
        f"- Current attempt: `{attempt_slug}`",
        f"- Competition: {config.competition.url}",
        f"- Active runs: {', '.join(state.runtime.active_run_ids) if state.runtime.active_run_ids else 'none'}",
        f"- Work items: {len(work_items)}",
        f"- Stage runs: {len(stage_runs)}",
        f"- Branch memories: {len(state.branch_memories)}",
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


def _latest_stage_run_for_name(state: WorkspaceState, run_id: str, stage_name: str):
    matches = [item for item in state.stage_runs if item.run_id == run_id and item.stage_name == stage_name]
    if not matches:
        return None
    matches.sort(key=lambda item: (item.updated_at, item.created_at, item.stage_run_id))
    return matches[-1]


def _execute_step_status(run: RunRecord, step_index: int) -> str:
    if not run.started_at and run.stage_cursor != "execute":
        return "pending"
    if run.stage_cursor == "execute":
        return "ready"
    if run.status == "running" and not run.stage_cursor:
        return "running"
    later_cursor = run.stage_cursor in set(run.stage_plan[step_index + 1 :]) or run.stage_cursor == "complete"
    if run.completed_at or later_cursor or any(item for item in run.stage_plan[step_index + 1 :] if item):
        return "completed" if run.error == "" else "failed"
    if run.error:
        return "failed"
    return "pending"


def _stage_step_status(run: RunRecord, stage_name: str, stage_run, step_index: int) -> str:
    if stage_name == "execute":
        return _execute_step_status(run, step_index)
    if stage_run is not None:
        return stage_run.validator_status or stage_run.status
    if run.stage_cursor == stage_name:
        return "ready"
    if run.stage_cursor == "complete" and stage_name in run.stage_plan[: step_index + 1]:
        return "completed"
    return "pending"


def _stage_dir_hint(stage_name: str) -> str:
    order = STAGE_ORDER.get(stage_name, 0)
    if order <= 0:
        return stage_name
    return f"{order:02d}-{stage_name}__*"


def _stage_artifact_from_disk(config: WorkspaceConfig, run_root: Path, stage_name: str) -> dict[str, str]:
    stages_root = run_root / "stages"
    if not stages_root.exists():
        return {}
    matches = sorted(
        (path for path in stages_root.glob(_stage_dir_hint(stage_name)) if path.is_dir()),
        key=lambda item: (item.stat().st_mtime, item.name),
    )
    if not matches:
        return {}
    chosen = matches[-1]
    return {
        "artifact_label": chosen.name,
        "artifact_dir": artifact_relative_path(str(chosen), config.root),
        "artifact_json": artifact_relative_path(str(chosen / f"{stage_name}.json"), config.root),
        "artifact_md": artifact_relative_path(str(chosen / f"{stage_name}.md"), config.root),
        "status": chosen.name.split("__", 1)[1] if "__" in chosen.name else "",
    }


def _run_lifecycle_payload(config: WorkspaceConfig, state: WorkspaceState, run: RunRecord, run_root: Path) -> dict[str, object]:
    steps: list[dict[str, object]] = []
    for index, stage_name in enumerate(run.stage_plan, start=1):
        stage_run = _latest_stage_run_for_name(state, run.run_id, stage_name)
        artifact_label = ""
        artifact_dir = ""
        artifact_json = ""
        artifact_md = ""
        artifact_status = ""
        if stage_run is not None:
            artifact_label = stage_label_from_path(
                stage_run.output_dir,
                stage_run.stage_name,
                stage_status=stage_run.status,
                validator_status=stage_run.validator_status,
            )
            artifact_dir = artifact_relative_path(stage_run.output_dir, config.root)
            artifact_json = artifact_relative_path(stage_run.output_json_path, config.root)
            artifact_md = artifact_relative_path(stage_run.output_md_path, config.root)
            artifact_status = stage_run.validator_status or stage_run.status
        disk_artifact = _stage_artifact_from_disk(config, run_root, stage_name)
        if disk_artifact:
            artifact_label = disk_artifact.get("artifact_label", artifact_label)
            artifact_dir = disk_artifact.get("artifact_dir", artifact_dir)
            artifact_json = disk_artifact.get("artifact_json", artifact_json)
            artifact_md = disk_artifact.get("artifact_md", artifact_md)
            artifact_status = disk_artifact.get("status", artifact_status)
        step_status = artifact_status or _stage_step_status(run, stage_name, stage_run, index - 1)
        steps.append(
            {
                "sequence": index,
                "stage_name": stage_name,
                "canonical_dir_hint": _stage_dir_hint(stage_name),
                "status": step_status,
                "artifact_label": artifact_label,
                "artifact_dir": artifact_dir,
                "artifact_json": artifact_json,
                "artifact_md": artifact_md,
            }
        )
    latest_stage_label = next((step["artifact_label"] for step in reversed(steps) if step["artifact_label"]), "")
    return {
        "run_id": run.run_id,
        "run_label": run_label_from_path(run.run_dir) or run.run_id,
        "lifecycle_template": run.lifecycle_template or "recursive_experiment",
        "status": run.status,
        "current_cursor": run.stage_cursor or "complete",
        "note": "Stage directory prefixes keep canonical stage ids. Actual execution order follows `stage_plan` below.",
        "steps": steps,
        "latest_stage_label": latest_stage_label or "n/a",
    }


def _write_run_lifecycle_views(config: WorkspaceConfig, state: WorkspaceState) -> None:
    for run in visible_runs(state):
        run_root = Path(run.run_dir)
        if run_root.name == "runtime":
            run_root = run_root.parent
        stages_root = ensure_directory(run_root / "stages")
        payload = _run_lifecycle_payload(config, state, run, run_root)
        lines = [
            f"# Lifecycle {payload['run_label']}",
            "",
            f"- Run id: `{payload['run_id']}`",
            f"- Lifecycle template: `{payload['lifecycle_template']}`",
            f"- Run status: `{payload['status']}`",
            f"- Current cursor: `{payload['current_cursor']}`",
            f"- Latest realized stage: `{payload['latest_stage_label']}`",
            f"- Note: {payload['note']}",
            "",
            "## Planned Execution Order",
        ]
        for step in payload["steps"]:
            lines.append(
                f"{step['sequence']}. `{step['stage_name']}` | status=`{step['status']}` | canonical_dir=`{step['canonical_dir_hint']}`"
                + (f" | artifact=`{step['artifact_label']}`" if step["artifact_label"] else "")
            )
        markdown = "\n".join(lines).rstrip() + "\n"
        atomic_write_text(run_root / "lifecycle.md", markdown)
        atomic_write_text(stages_root / "ORDER.md", markdown)
        atomic_write_text(run_root / "lifecycle.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


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
    _write_run_lifecycle_views(config, state)
    write_experiment_conclusions(config, state)

    leader = best_run(state)
    slot_plan = plan_submission_slots(config, state)
    work_items = visible_work_items(state)
    runs = visible_runs(state)
    run_ids = visible_run_ids(state)
    research_notes = [item for item in state.research_notes if item.run_id in run_ids]
    atomic_write_text(config.report_path("overview.md"), _overview_markdown(config, state))

    master_sections = [
        (
            "Situation",
            "<ul>"
            f"<li>Current attempt: {html.escape(current_attempt_slug(state.runtime))}</li>"
            f"<li>Active runs: {html.escape(', '.join(state.runtime.active_run_ids) if state.runtime.active_run_ids else 'none')}</li>"
                f"<li>Queued work items: {len([item for item in work_items if item.status == 'queued'])}</li>"
                f"<li>Completed work items: {len([item for item in work_items if item.status in {'complete', 'submitted'}])}</li>"
                f"<li>Branch memories: {len(state.branch_memories)}</li>"
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
                for item in sorted(work_items, key=lambda value: (value.priority, value.created_at, value.id))[:8]
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
        for item in runs[-20:]
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

    finding_items = "".join(f"<li>{html.escape(item.title)}: {html.escape(item.summary)}</li>" for item in state.findings[-15:] if item.run_id in run_ids) or "<li>No findings yet.</li>"
    issue_items = "".join(f"<li>{html.escape(item.title)}: {html.escape(item.summary)}</li>" for item in state.issues[-15:] if item.run_id in run_ids) or "<li>No issues yet.</li>"
    queued_items = "".join(
        f"<li>{html.escape(item.title)} ({html.escape(item.status)})</li>"
        for item in sorted(work_items, key=lambda value: (value.priority, value.created_at, value.id))[:10]
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
        for item in research_notes[-15:]
    ) or "<li>No research notes yet.</li>"
    branch_memory_items = "".join(
        f"<li>{html.escape(item.run_id)} | {html.escape(item.outcome)} | {html.escape(item.summary)}</li>"
        for item in state.branch_memories[-15:]
    ) or "<li>No branch memories yet.</li>"
    atomic_write_text(
        config.report_path("research_report.html"),
        _render_html_page(
            "Research Report",
            [("Research Notes", f"<ul>{research_items}</ul>"), ("Branch Memories", f"<ul>{branch_memory_items}</ul>")],
        ),
    )

    _write_csv(
        config.report_path("work_items.csv"),
        ["id", "title", "status", "priority", "family", "latest_run_id"],
        [[item.id, item.title, item.status, str(item.priority), item.family, item.latest_run_id] for item in work_items],
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
            for item in runs
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
            if item.run_id in run_ids
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
