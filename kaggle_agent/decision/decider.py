from __future__ import annotations

from pathlib import Path

import yaml

from kaggle_agent.decision.helpers import (
    begin_stage_run,
    complete_stage_run,
    latest_stage_payload,
    run_configured_stage_adapter,
    stage_markdown,
    write_input_manifest,
)
from kaggle_agent.schema import WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import slugify


ROLE_ORDER = ("primary", "hedge", "explore", "support")
COMPONENT_TEMPLATES = {
    "class_coverage": {
        "title": "coverage-first branch",
        "hypothesis": "Improve holdout ROC-AUC by fixing class coverage and long-tail support before calibration-only tuning.",
        "rationale": "Knowledge and recent failures both say coverage is the limiting factor.",
        "target_component": "class_coverage",
    },
    "probe_head": {
        "title": "probe-head capacity branch",
        "hypothesis": "Improve holdout ROC-AUC by changing the probe representation surface after protecting class coverage.",
        "rationale": "Representation-level adjustments are still open and structurally distinct from calibration.",
        "target_component": "probe_head",
    },
    "pseudo_label": {
        "title": "pseudo-label expansion branch",
        "hypothesis": "Improve holdout ROC-AUC by increasing teacher-signal coverage rather than retuning post-processing only.",
        "rationale": "Pseudo-label coverage remains a high-value axis.",
        "target_component": "pseudo_label",
    },
    "optimization": {
        "title": "schedule-recovery branch",
        "hypothesis": "Improve holdout ROC-AUC by changing schedule or LR only when the current branch still looks underfit.",
        "rationale": "Training dynamics remain a secondary hedge branch.",
        "target_component": "optimization",
    },
    "prior_calibration": {
        "title": "calibration hedge branch",
        "hypothesis": "Test calibration only as a hedge after higher-value training and coverage axes are represented.",
        "rationale": "Calibration should not dominate the portfolio unless stronger axes are exhausted.",
        "target_component": "prior_calibration",
    },
}


def _submission_threshold(config: WorkspaceConfig, config_path: str) -> float:
    path = Path(config_path)
    if not path.is_absolute():
        path = config.root / path
    if not path.exists():
        return 0.75
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    metrics = payload.get("metrics", {})
    return float(metrics.get("submission_candidate_threshold", 0.75))


def _component_priority(research: dict[str, object]) -> list[str]:
    cards = [item for item in research.get("policy_cards", []) if isinstance(item, dict)]
    negative_axes = {
        str(item.get("component", ""))
        for item in cards
        if str(item.get("policy_type", "")) in {"veto", "avoid"}
    }
    order: list[str] = []
    for policy_type in ("require", "prefer", "conditional"):
        for card in cards:
            component = str(card.get("component", ""))
            if not component or component == "general" or component in negative_axes:
                continue
            if str(card.get("policy_type", "")) != policy_type:
                continue
            if component not in order:
                order.append(component)
    for memory in research.get("branch_memories", []):
        if not isinstance(memory, dict):
            continue
        component = str(memory.get("idea_class", ""))
        if not component or component in negative_axes or component in order:
            continue
        if str(memory.get("outcome", "")) in {"leader", "improved", "submission_candidate"}:
            order.append(component)
    return order or ["class_coverage", "probe_head", "optimization"]


def _cooldown_idea_classes(research: dict[str, object]) -> list[str]:
    weak_counts: dict[str, int] = {}
    for memory in research.get("branch_memories", []):
        if not isinstance(memory, dict):
            continue
        idea_class = str(memory.get("idea_class", ""))
        if not idea_class:
            continue
        if str(memory.get("outcome", "")) in {"regressed", "critic_rejected", "run_failed", "validate_failed"}:
            weak_counts[idea_class] = weak_counts.get(idea_class, 0) + 1
    return [idea for idea, count in weak_counts.items() if count >= 2]


def _portfolio_policy(config: WorkspaceConfig, run, experiment, research: dict[str, object]) -> dict[str, object]:
    components = _component_priority(research)
    cards = [item for item in research.get("policy_cards", []) if isinstance(item, dict)]
    deprioritized_axes = [
        str(item.get("component", ""))
        for item in cards
        if str(item.get("policy_type", "")) in {"veto", "avoid"} and str(item.get("component", ""))
    ]
    cooldown_axes = _cooldown_idea_classes(research)
    target_branch_count = min(3, max(2, len(components[:3])))
    branch_mix = [
        {"branch_role": role, "target_component": components[index] if index < len(components) else components[-1]}
        for index, role in enumerate(ROLE_ORDER[:target_branch_count])
    ]
    return {
        "portfolio_id": f"portfolio-{run.run_id}-{slugify(experiment.family)}",
        "target_branch_count": target_branch_count,
        "branch_mix": branch_mix,
        "per_portfolio_cap": 1 if config.automation.max_active_runs <= 3 else 2,
        "per_idea_class_cap": 1,
        "dispatch_strategy": "prefer-diverse-frontier-before-follow-on-support",
        "selected_policy_cards": [str(item.get("card_id", "")) for item in cards[:6]],
        "branch_memory_ids": [str(item) for item in research.get("branch_memory_ids", [])][:6],
        "deprioritized_axes": list(dict.fromkeys([item for item in deprioritized_axes if item] + cooldown_axes)),
        "cooldown_idea_classes": cooldown_axes,
    }


def _portfolio_intent(experiment, research: dict[str, object], policy: dict[str, object]) -> str:
    top_component = ""
    branch_mix = policy.get("branch_mix", [])
    if isinstance(branch_mix, list) and branch_mix:
        top_component = str(branch_mix[0].get("target_component", ""))
    if top_component:
        return f"Search a compact branch portfolio around {top_component} first, while keeping one structurally distinct hedge."
    adopt_now = [str(item) for item in research.get("adopt_now", []) if str(item).strip()]
    return adopt_now[0] if adopt_now else f"Keep a compact branch portfolio open for {experiment.family}."


def _default_branch_portfolio(experiment, research: dict[str, object], policy: dict[str, object]) -> list[dict[str, object]]:
    deprioritized = {str(item) for item in policy.get("deprioritized_axes", []) if str(item)}
    cards = [item for item in research.get("policy_cards", []) if isinstance(item, dict)]
    card_ids_by_component: dict[str, list[str]] = {}
    for card in cards:
        component = str(card.get("component", ""))
        if not component:
            continue
        card_ids_by_component.setdefault(component, []).append(str(card.get("card_id", "")))
    portfolio: list[dict[str, object]] = []
    for branch in policy.get("branch_mix", []):
        if not isinstance(branch, dict):
            continue
        role = str(branch.get("branch_role", "explore"))
        component = str(branch.get("target_component", "optimization"))
        if component in deprioritized:
            continue
        template = COMPONENT_TEMPLATES.get(component, COMPONENT_TEMPLATES["optimization"])
        portfolio.append(
            {
                "title": f"{experiment.title} {template['title']}",
                "family": experiment.family,
                "hypothesis": str(template["hypothesis"]),
                "rationale": str(template["rationale"]),
                "branch_role": role,
                "idea_class": component,
                "target_component": str(template["target_component"]),
                "priority_delta": {"primary": 10, "hedge": 12, "explore": 14, "support": 16}.get(role, 14),
                "launch_mode": "background",
                "knowledge_card_ids": card_ids_by_component.get(component, [str(item) for item in research.get("knowledge_card_ids", [])][:3]),
                "policy_trace": [
                    f"{component}:{role}",
                    *(f"card:{card_id}" for card_id in card_ids_by_component.get(component, [])[:2]),
                ],
                "branch_memory_ids": [str(item) for item in research.get("branch_memory_ids", [])][:3],
            }
        )
    return portfolio


def build_decision(config: WorkspaceConfig, state: WorkspaceState, run_id: str):
    run = next(item for item in state.runs if item.run_id == run_id)
    experiment = next(item for item in state.experiments if item.id == run.experiment_id)
    research = latest_stage_payload(state, run_id, "research")
    report = latest_stage_payload(state, run_id, "report")
    stage_run, input_manifest_path = begin_stage_run(
        config,
        state,
        run,
        stage_name="decision",
        input_ref=run.latest_stage_run_id or run.run_id,
    )
    write_input_manifest(
        input_manifest_path,
        {
            "run": run.to_dict(),
            "experiment": experiment.to_dict(),
            "report": report,
            "research": research,
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
        return stage_run

    threshold = _submission_threshold(config, experiment.config_path)
    root_cause = str(report.get("root_cause", run.root_cause or run.error or "unknown"))
    problem_frame = research.get("problem_frame", {}) if isinstance(research.get("problem_frame"), dict) else {}
    knowledge_card_ids = [str(item) for item in research.get("knowledge_card_ids", [])]
    policy = _portfolio_policy(config, run, experiment, research)
    portfolio_intent = _portfolio_intent(experiment, research, policy)
    branch_portfolio = _default_branch_portfolio(experiment, research, policy)
    rejected_axes = list(dict.fromkeys([str(item) for item in research.get("negative_priors", [])][:5] + [str(item) for item in policy.get("deprioritized_axes", []) if str(item)]))
    common_fields = {
        "problem_frame": problem_frame,
        "knowledge_card_ids": knowledge_card_ids,
        "rejected_axes": rejected_axes,
        "portfolio_intent": portfolio_intent,
        "portfolio_policy": policy,
        "selected_policy_cards": [str(item) for item in policy.get("selected_policy_cards", [])],
        "branch_memory_ids": [str(item) for item in policy.get("branch_memory_ids", [])],
        "deprioritized_axes": [str(item) for item in policy.get("deprioritized_axes", [])],
        "branch_mix": [item for item in policy.get("branch_mix", []) if isinstance(item, dict)],
        "branch_portfolio": branch_portfolio,
        "requires_human": False,
    }
    if run.status == "failed":
        decision_type = "blocked" if any(token in root_cause.lower() for token in ["missing", "module", "dependency"]) else "fix"
        payload = {
            "stage": "decision",
            "decision_type": decision_type,
            "next_action": "hold" if decision_type == "blocked" else "run_new_experiment",
            "submission_recommendation": "no",
            "root_cause": root_cause,
            "why": "The runtime failed and needs a direct repair before more exploration.",
            "next_title": f"{experiment.title} runtime repair",
            "next_family": experiment.family,
            "next_config_path": experiment.config_path,
            "priority_delta": -5 if decision_type == "blocked" else 5,
            "launch_mode": "sync" if decision_type == "blocked" else "background",
            **common_fields,
        }
    elif experiment.family == "perch_head_debug":
        payload = {
            "stage": "decision",
            "decision_type": "promote_baseline",
            "next_action": "run_new_experiment",
            "submission_recommendation": "no",
            "root_cause": root_cause,
            "why": "The debug smoke is healthy enough to promote the cached-probe baseline.",
            "next_title": "Perch cached-probe baseline",
            "next_family": "perch_cached_probe",
            "next_config_path": str(config.runtime_root() / "configs" / "default.yaml"),
            "priority_delta": 10,
            "launch_mode": "background",
            **common_fields,
        }
    elif run.primary_metric_value is not None and run.primary_metric_value >= threshold:
        payload = {
            "stage": "decision",
            "decision_type": "submit_candidate",
            "next_action": "submit_candidate",
            "submission_recommendation": "candidate",
            "root_cause": root_cause,
            "why": "This run cleared the local submission threshold and should enter the submission intelligence loop.",
            "next_title": "",
            "next_family": experiment.family,
            "next_config_path": experiment.config_path,
            "priority_delta": 0,
            "launch_mode": "background",
            **common_fields,
        }
    else:
        payload = {
            "stage": "decision",
            "decision_type": "tune",
            "next_action": "run_new_experiment",
            "submission_recommendation": "no",
            "root_cause": root_cause,
            "why": "The run is useful but still below the keep threshold, so the next config should directly target the root cause.",
            "next_title": f"{experiment.title} follow-up",
            "next_family": experiment.family,
            "next_config_path": experiment.config_path,
            "priority_delta": 10,
            "launch_mode": "background",
            **common_fields,
        }
    markdown = stage_markdown(
        f"Decision {run_id}",
        [
            f"- Decision type: `{payload['decision_type']}`",
            f"- Next action: `{payload['next_action']}`",
            f"- Submission recommendation: `{payload['submission_recommendation']}`",
            f"- Root cause: {payload['root_cause']}",
            f"- Why: {payload['why']}",
            f"- Portfolio intent: {payload['portfolio_intent']}",
            "",
            "## Branch Mix",
            *(
                f"- `{item.get('branch_role', '')}` -> `{item.get('target_component', '')}`"
                for item in payload.get("branch_mix", [])
                if isinstance(item, dict)
            ),
            "",
            "## Deprioritized Axes",
            *(
                f"- {item}"
                for item in payload.get("deprioritized_axes", [])
            ),
        ],
    )
    complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)
    return stage_run
