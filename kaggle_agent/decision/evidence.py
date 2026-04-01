from __future__ import annotations

from pathlib import Path

from kaggle_agent.decision.helpers import (
    begin_stage_run,
    complete_stage_run,
    load_run_result,
    run_configured_stage_adapter,
    stage_markdown,
    write_input_manifest,
)
from kaggle_agent.knowledge import apply_knowledge_stage_outputs, load_session_memory
from kaggle_agent.knowledge_reducer import upsert_observation_atom
from kaggle_agent.schema import FindingRecord, IssueRecord, MetricObservation, WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import now_utc_iso


def _record_metric(
    state: WorkspaceState,
    *,
    run_id: str,
    metric_name: str,
    value: float,
    is_primary: bool,
    notes: str = "",
) -> None:
    state.metrics.append(
        MetricObservation(
            metric_id=f"metric-{state.runtime.next_metric_number:04d}",
            run_id=run_id,
            metric_name=metric_name,
            split="local_proxy",
            domain_scope="source",
            trust_level="clean",
            postproc_variant="none",
            evaluator_version="runtime_result_json",
            is_primary=is_primary,
            value=float(value),
            notes=notes,
            created_at=now_utc_iso(),
        )
    )
    state.runtime.next_metric_number += 1


def _record_finding(
    state: WorkspaceState,
    *,
    run_id: str,
    title: str,
    summary: str,
    severity: str = "medium",
    status: str = "open",
    dedupe_key: str = "",
) -> None:
    state.findings.append(
        FindingRecord(
            finding_id=f"finding-{state.runtime.next_finding_number:04d}",
            run_id=run_id,
            title=title,
            summary=summary,
            severity=severity,
            status=status,
            dedupe_key=dedupe_key,
            created_at=now_utc_iso(),
        )
    )
    state.runtime.next_finding_number += 1


def _record_issue(
    state: WorkspaceState,
    *,
    run_id: str,
    title: str,
    summary: str,
    severity: str = "medium",
    status: str = "open",
    dedupe_key: str = "",
) -> None:
    state.issues.append(
        IssueRecord(
            issue_id=f"issue-{state.runtime.next_issue_number:04d}",
            run_id=run_id,
            title=title,
            summary=summary,
            severity=severity,
            status=status,
            dedupe_key=dedupe_key,
            created_at=now_utc_iso(),
        )
    )
    state.runtime.next_issue_number += 1


def build_evidence(config: WorkspaceConfig, state: WorkspaceState, run_id: str):
    run = next(item for item in state.runs if item.run_id == run_id)
    experiment = next(item for item in state.experiments if item.id == run.experiment_id)
    work_item = next(item for item in state.work_items if item.id == run.work_item_id)
    result = load_run_result(run)
    stage_run, input_manifest_path = begin_stage_run(
        config,
        state,
        run,
        stage_name="evidence",
        input_ref=run.run_id,
    )
    write_input_manifest(
        input_manifest_path,
        {
            "run": run.to_dict(),
            "experiment": experiment.to_dict(),
            "work_item": work_item.to_dict(),
            "result": result,
        },
    )
    adapted = run_configured_stage_adapter(
        config,
        state,
        stage_run,
        input_manifest_path=input_manifest_path,
        extra_env={"KAGGLE_AGENT_RUN_ID": run.run_id},
    )
    if adapted is not None:
        payload, markdown = adapted
        complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)
        return stage_run

    primary_metric = run.primary_metric_name or str(result.get("primary_metric_name", config.metrics.primary))
    root_cause = run.root_cause or str(result.get("root_cause", run.error or "missing runtime summary"))
    verdict = run.verdict or str(result.get("verdict", "unknown"))
    dataset_summary = result.get("dataset_summary", {})
    all_metrics = result.get("all_metrics", {})
    leader = None
    parent_run = next((candidate for candidate in state.runs if candidate.run_id == work_item.source_run_id), None)
    for candidate in state.runs:
        if candidate.status != "succeeded" or candidate.primary_metric_value is None:
            continue
        if leader is None or candidate.primary_metric_value > leader.primary_metric_value:
            leader = candidate
    metric_deltas = {
        "vs_leader": (
            None
            if leader is None or leader.primary_metric_value is None or run.primary_metric_value is None
            else run.primary_metric_value - leader.primary_metric_value
        ),
        "vs_parent": (
            None
            if parent_run is None or parent_run.primary_metric_value is None or run.primary_metric_value is None
            else run.primary_metric_value - parent_run.primary_metric_value
        ),
    }
    anomalies: list[str] = []
    if run.status == "failed":
        anomalies.append(f"Run failed before producing a clean {primary_metric} improvement.")
    if run.primary_metric_value is not None and primary_metric.startswith("soundscape_") and run.primary_metric_value >= 0.98:
        anomalies.append("Primary metric is near-perfect; verify that this is not a misleading resubstitution metric.")
    promotion_candidates = []
    if run.primary_metric_value is not None and metric_deltas["vs_leader"] is not None and metric_deltas["vs_leader"] >= 0:
        promotion_candidates.append(f"Promote {run.run_id} if validation remains trustworthy against the current leader.")
    demotion_candidates = []
    if run.primary_metric_value is not None and run.primary_metric_value < 0.6:
        demotion_candidates.append(f"Demote {run.run_id} from the active frontier until the bottleneck is repaired.")
    open_questions = [f"What exact change caused the current root cause: {root_cause}?"]
    observation_atoms = [
        upsert_observation_atom(
            state,
            run_id=run.run_id,
            stage_name="evidence",
            family=experiment.family,
            component=work_item.idea_class or "primary_metric",
            summary=f"{primary_metric} recorded for {run.run_id}",
            source_type="primary_metric",
            source_ref=run.run_id,
            metric_name=primary_metric,
            comparator="leader",
            value=run.primary_metric_value,
            direction=(
                "up"
                if metric_deltas["vs_leader"] is not None and metric_deltas["vs_leader"] > 0
                else ("down" if metric_deltas["vs_leader"] is not None and metric_deltas["vs_leader"] < 0 else "flat")
            ),
            axis_tags=[work_item.idea_class or "general"],
        ).to_dict()
    ]
    if root_cause:
        observation_atoms.append(
            upsert_observation_atom(
                state,
                run_id=run.run_id,
                stage_name="evidence",
                family=experiment.family,
                component=work_item.idea_class or "runtime",
                summary=root_cause,
                source_type="root_cause",
                source_ref=run.run_id,
                axis_tags=[work_item.idea_class or "general"],
            ).to_dict()
        )
    for anomaly in anomalies[:2]:
        observation_atoms.append(
            upsert_observation_atom(
                state,
                run_id=run.run_id,
                stage_name="evidence",
                family=experiment.family,
                component=work_item.idea_class or "runtime",
                summary=anomaly,
                source_type="anomaly",
                source_ref=run.run_id,
                axis_tags=[work_item.idea_class or "general"],
            ).to_dict()
        )
    session_memory = load_session_memory(config)
    payload = {
        "stage": "evidence",
        "run_id": run.run_id,
        "experiment_id": experiment.id,
        "work_item_id": work_item.id,
        "status": run.status,
        "primary_metric_name": primary_metric,
        "primary_metric_value": run.primary_metric_value,
        "secondary_metrics": run.secondary_metrics,
        "all_metrics": all_metrics,
        "root_cause": root_cause,
        "verdict": verdict,
        "dataset_summary": dataset_summary,
        "artifacts": run.artifact_paths,
        "metric_deltas": metric_deltas,
        "anomalies": anomalies,
        "promotion_candidates": promotion_candidates,
        "demotion_candidates": demotion_candidates,
        "open_questions": open_questions,
        "observation_atoms": observation_atoms,
        "session_memory": session_memory,
    }
    lines = [
        f"- Work item: `{work_item.id}`",
        f"- Experiment: `{experiment.id}`",
        f"- Run status: `{run.status}`",
        f"- Primary metric: `{primary_metric}={run.primary_metric_value}`",
        f"- Verdict: `{verdict}`",
        f"- Root cause: {root_cause}",
    ]
    if metric_deltas["vs_leader"] is not None:
        lines.append(f"- Delta vs leader: {metric_deltas['vs_leader']:+.6f}")
    if metric_deltas["vs_parent"] is not None:
        lines.append(f"- Delta vs parent: {metric_deltas['vs_parent']:+.6f}")
    if dataset_summary:
        lines.extend(
            [
                "",
                "## Dataset Summary",
                *(f"- {key}: {value}" for key, value in dataset_summary.items()),
            ]
        )
    if result.get("summary_markdown"):
        lines.extend(["", "## Runtime Summary", str(result["summary_markdown"])])
    if anomalies:
        lines.extend(["", "## Anomalies", *(f"- {item}" for item in anomalies)])
    if observation_atoms:
        lines.extend(
            [
                "",
                "## Observation Atoms",
                *(
                    f"- `{item.get('source_type', '')}` | `{item.get('component', '')}` | {item.get('summary', '')}"
                    for item in observation_atoms
                    if isinstance(item, dict)
                ),
            ]
        )
    if open_questions:
        lines.extend(["", "## Open Questions", *(f"- {item}" for item in open_questions)])
    markdown = stage_markdown(f"Evidence Bundle {run.run_id}", lines)
    complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)
    apply_knowledge_stage_outputs(config, run_id=run_id, stage="evidence", payload=payload)

    if run.primary_metric_value is not None:
        _record_metric(
            state,
            run_id=run.run_id,
            metric_name=primary_metric,
            value=run.primary_metric_value,
            is_primary=True,
            notes=f"Recorded from {run.run_id}.",
        )
    for metric_name, value in sorted(run.secondary_metrics.items()):
        _record_metric(
            state,
            run_id=run.run_id,
            metric_name=metric_name,
            value=value,
            is_primary=False,
        )
    if run.status == "failed":
        _record_issue(
            state,
            run_id=run.run_id,
            title=f"{experiment.title} failed",
            summary=root_cause,
            severity="high",
            dedupe_key=f"run:{run.run_id}:failure",
        )
    elif run.primary_metric_value is not None and run.primary_metric_value >= 0.85:
        _record_finding(
            state,
            run_id=run.run_id,
            title=f"{experiment.title} is promotion-ready",
            summary=f"{primary_metric}={run.primary_metric_value:.6f} with verdict {verdict}.",
            severity="high",
            dedupe_key=f"run:{run.run_id}:promotion",
        )
    else:
        _record_finding(
            state,
            run_id=run.run_id,
            title=f"{experiment.title} completed",
            summary=f"{primary_metric}={run.primary_metric_value} and verdict {verdict}.",
            severity="medium",
            dedupe_key=f"run:{run.run_id}:completed",
        )
        if run.primary_metric_value is not None and run.primary_metric_value < 0.6:
            _record_issue(
                state,
                run_id=run.run_id,
                title=f"{experiment.title} under target",
                summary=f"{primary_metric}={run.primary_metric_value:.6f} is below the current keep threshold.",
                severity="medium",
                dedupe_key=f"run:{run.run_id}:under-target",
            )
    return stage_run
