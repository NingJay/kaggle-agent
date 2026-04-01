from __future__ import annotations

import html
from pathlib import Path

from kaggle_agent.decision.helpers import (
    begin_stage_run,
    complete_stage_run,
    latest_stage_payload,
    run_configured_stage_adapter,
    stage_markdown,
    write_input_manifest,
)
from kaggle_agent.schema import WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import atomic_write_text


def build_report(config: WorkspaceConfig, state: WorkspaceState, run_id: str):
    run = next(item for item in state.runs if item.run_id == run_id)
    experiment = next(item for item in state.experiments if item.id == run.experiment_id)
    work_item = next(item for item in state.work_items if item.id == run.work_item_id)
    evidence = latest_stage_payload(state, run_id, "evidence")
    stage_run, input_manifest_path = begin_stage_run(
        config,
        state,
        run,
        stage_name="report",
        input_ref=run.latest_stage_run_id or run.run_id,
    )
    best_run = None
    for candidate in state.runs:
        if candidate.status != "succeeded" or candidate.primary_metric_value is None:
            continue
        if best_run is None or candidate.primary_metric_value > best_run.primary_metric_value:
            best_run = candidate
    findings = [item for item in state.findings if item.run_id == run_id][-5:]
    issues = [item for item in state.issues if item.run_id == run_id][-5:]
    write_input_manifest(
        input_manifest_path,
        {
            "run": run.to_dict(),
            "experiment": experiment.to_dict(),
            "work_item": work_item.to_dict(),
            "evidence": evidence,
            "recent_findings": [item.to_dict() for item in findings],
            "recent_issues": [item.to_dict() for item in issues],
            "leader_run": best_run.to_dict() if best_run is not None else {},
        },
    )
    adapted = run_configured_stage_adapter(
        config,
        state,
        stage_run,
        input_manifest_path=input_manifest_path,
        extra_env={"KAGGLE_AGENT_RUN_ID": run_id},
    )
    if adapted is not None:
        payload, markdown = adapted
        complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)
    else:
        promotion_candidates = [str(item) for item in evidence.get("promotion_candidates", []) if str(item)]
        demotion_candidates = [str(item) for item in evidence.get("demotion_candidates", []) if str(item)]
        open_questions = [str(item) for item in evidence.get("open_questions", []) if str(item)]
        headline = (
            "Promote for submission intelligence"
            if run.primary_metric_value is not None and run.primary_metric_value >= 0.85
            else ("Stabilize runtime first" if run.status == "failed" else "Keep iterating on the probe stack")
        )
        focus = (
            "resolve runtime blocker"
            if run.status == "failed"
            else (
                "submission calibration and CPU bundle"
                if run.primary_metric_value is not None and run.primary_metric_value >= 0.85
                else "root-cause repair and next config selection"
            )
        )
        payload = {
            "stage": "report",
            "run_id": run_id,
            "headline": headline,
            "focus": focus,
            "best_run_id": best_run.run_id if best_run is not None else "",
            "best_run_metric": best_run.primary_metric_value if best_run is not None else None,
            "primary_metric_value": run.primary_metric_value,
            "root_cause": evidence.get("root_cause", run.root_cause or run.error),
            "verdict": evidence.get("verdict", run.verdict),
            "finding_titles": [item.title for item in findings],
            "issue_titles": [item.title for item in issues],
            "promotion_candidates": promotion_candidates,
            "demotion_candidates": demotion_candidates,
            "open_questions": open_questions,
        }
        lines = [
            f"- Headline: {headline}",
            f"- Current focus: {focus}",
            f"- Run status: `{run.status}`",
            f"- Primary metric: `{run.primary_metric_name}={run.primary_metric_value}`",
            f"- Root cause: {payload['root_cause']}",
        ]
        if best_run is not None:
            lines.append(
                f"- Current ledger leader: `{best_run.run_id}` with {best_run.primary_metric_name}={best_run.primary_metric_value:.6f}"
            )
        if findings:
            lines.extend(["", "## Findings", *(f"- {item.title}: {item.summary}" for item in findings)])
        if issues:
            lines.extend(["", "## Issues", *(f"- {item.title}: {item.summary}" for item in issues)])
        if promotion_candidates:
            lines.extend(["", "## Promotion Candidates", *(f"- {item}" for item in promotion_candidates)])
        if demotion_candidates:
            lines.extend(["", "## Demotion Candidates", *(f"- {item}" for item in demotion_candidates)])
        if open_questions:
            lines.extend(["", "## Next Questions", *(f"- {item}" for item in open_questions)])
        markdown = stage_markdown(f"Run Report {run_id}", lines)
        complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)

    html_path = Path(stage_run.output_dir) / "report.html"
    markdown_text = Path(stage_run.output_md_path).read_text(encoding="utf-8")
    html_body = html.escape(markdown_text)
    atomic_write_text(
        html_path,
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Run Report {html.escape(run_id)}</title>
  <style>
    body {{ margin: 0; font-family: Georgia, serif; background: #f5f1e8; color: #1f2937; }}
    main {{ max-width: 900px; margin: 0 auto; padding: 32px 20px 48px; }}
    article {{ background: #fffdfa; border: 1px solid #d9d1c5; border-radius: 16px; padding: 24px; }}
    pre {{ white-space: pre-wrap; font: 14px/1.6 "SFMono-Regular", Consolas, monospace; }}
  </style>
</head>
<body>
<main>
  <article>
    <pre>{html_body}</pre>
  </article>
</main>
</body>
</html>
""",
    )
    return stage_run
