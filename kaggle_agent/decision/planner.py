from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from kaggle_agent.decision.helpers import (
    begin_stage_run,
    complete_stage_run,
    latest_stage_payload,
    run_configured_stage_adapter,
    stage_markdown,
    write_input_manifest,
)
from kaggle_agent.knowledge import compact_knowledge_bundle, retrieve_knowledge_bundle
from kaggle_agent.schema import WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import atomic_write_text, now_utc_iso, slugify


DEFAULT_BRANCH_PLAN_LIMIT = 3


def _branch_plan_limit() -> int:
    raw = os.environ.get("KAGGLE_AGENT_BRANCH_PLAN_LIMIT", "").strip()
    if not raw:
        return DEFAULT_BRANCH_PLAN_LIMIT
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_BRANCH_PLAN_LIMIT
    return max(1, min(value, 8))


def _config_payload(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _probe_signature(path: Path) -> tuple[int, float, int] | None:
    payload = _config_payload(path)
    training = payload.get("training", {})
    if training.get("backend") != "sklearn_cached_probe":
        return None
    return (
        int(training.get("probe_pca_dim", 32)),
        float(training.get("probe_c", 0.25)),
        int(training.get("probe_min_pos", 8)),
    )


def _used_probe_signatures(config: WorkspaceConfig, state: WorkspaceState, family: str) -> set[tuple[int, float, int]]:
    signatures: set[tuple[int, float, int]] = set()
    for experiment in state.experiments:
        if experiment.family != family:
            continue
        config_path = Path(experiment.config_path)
        if not config_path.is_absolute():
            config_path = config.root / config_path
        if not config_path.exists():
            continue
        signature = _probe_signature(config_path)
        if signature is not None:
            signatures.add(signature)
    return signatures


def _relative_config_path(config: WorkspaceConfig, path: Path) -> str:
    return str(path.relative_to(config.root)) if path.is_relative_to(config.root) else str(path)


def _normalize_override_ops(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    overrides: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "")).strip()
        if not path:
            continue
        overrides.append({"path": path, "value": item.get("value")})
    return overrides


def _apply_override_ops(config_payload: dict[str, Any], overrides: list[dict[str, Any]]) -> dict[str, Any]:
    updated = copy.deepcopy(config_payload)
    for item in overrides:
        path = str(item.get("path", "")).strip()
        if not path:
            continue
        parts = [part for part in path.split(".") if part]
        if not parts:
            continue
        target = updated
        for part in parts[:-1]:
            child = target.get(part)
            if not isinstance(child, dict):
                child = {}
                target[part] = child
            target = child
        target[parts[-1]] = item.get("value")
    return updated


def _next_unused_probe_signature(
    candidates: list[tuple[int, float, int]],
    *,
    used: set[tuple[int, float, int]],
    already_selected: set[tuple[int, float, int]],
) -> tuple[int, float, int]:
    for candidate in candidates:
        if candidate in already_selected:
            continue
        if candidate not in used:
            return candidate
    for candidate in candidates:
        if candidate not in already_selected:
            return candidate
    return candidates[-1]


def _summaries_with_component(cards: list[dict[str, Any]], component: str, *, stances: set[str] | None = None) -> list[str]:
    summaries: list[str] = []
    for card in cards:
        if str(card.get("component", "")) != component:
            continue
        stance = str(card.get("stance", ""))
        if stances is not None and stance not in stances:
            continue
        summaries.append(f"{card.get('title', '')}: {card.get('summary', '')}")
    return summaries


def _bundle_cards(knowledge_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    cards = knowledge_bundle.get("cards", [])
    return [item for item in cards if isinstance(item, dict)] if isinstance(cards, list) else []


def _fallback_probe_branch_candidates(
    *,
    config: WorkspaceConfig,
    state: WorkspaceState,
    run_id: str,
    experiment,
    source_config: dict[str, Any],
    research: dict[str, Any],
    knowledge_bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    used = _used_probe_signatures(config, state, experiment.family)
    selected: set[tuple[int, float, int]] = set()
    cards = _bundle_cards(knowledge_bundle)
    coverage_reason = _summaries_with_component(cards, "class_coverage", stances={"positive", "conditional"})
    calibration_veto = _summaries_with_component(cards, "prior_calibration", stances={"negative"})

    coverage_sig = _next_unused_probe_signature(
        [(64, 0.25, 4), (96, 0.25, 4), (96, 0.35, 4), (128, 0.35, 4)],
        used=used,
        already_selected=selected,
    )
    selected.add(coverage_sig)
    capacity_sig = _next_unused_probe_signature(
        [(96, 0.25, 8), (128, 0.25, 8), (128, 0.50, 8), (160, 0.50, 8)],
        used=used,
        already_selected=selected,
    )
    selected.add(capacity_sig)
    regularization_sig = _next_unused_probe_signature(
        [(64, 0.15, 4), (96, 0.15, 4), (128, 0.15, 4), (96, 0.25, 2)],
        used=used,
        already_selected=selected,
    )
    selected.add(regularization_sig)

    candidates = [
        {
            "title": f"{experiment.title} coverage-first pca{coverage_sig[0]}-c{coverage_sig[1]:.2f}-min{coverage_sig[2]}",
            "family": experiment.family,
            "hypothesis": "Improve holdout ROC-AUC by expanding minority-class coverage before calibration-only tuning.",
            "reason": "; ".join(coverage_reason[:2]) or "Coverage-related knowledge suggests long-tail fixes should outrank calibration-only tuning.",
            "branch_role": "primary",
            "idea_class": "class_coverage",
            "target_component": "probe_head",
            "priority_delta": 10,
            "launch_mode": "background",
            "config_overrides": [
                {"path": "training.probe_pca_dim", "value": coverage_sig[0]},
                {"path": "training.probe_c", "value": coverage_sig[1]},
                {"path": "training.probe_min_pos", "value": coverage_sig[2]},
            ],
            "knowledge_card_ids": [
                str(card.get("card_id", ""))
                for card in cards
                if str(card.get("component", "")) in {"class_coverage", "probe_head"}
            ][:4],
        },
        {
            "title": f"{experiment.title} representation-step pca{capacity_sig[0]}-c{capacity_sig[1]:.2f}-min{capacity_sig[2]}",
            "family": experiment.family,
            "hypothesis": "Increase probe representation capacity after a coverage-first branch to test whether hidden structure is underfit.",
            "reason": "Probe-head priors suggest representation capacity is still worth exploring once coverage is protected.",
            "branch_role": "hedge",
            "idea_class": "probe_head",
            "target_component": "probe_head",
            "priority_delta": 12,
            "launch_mode": "background",
            "config_overrides": [
                {"path": "training.probe_pca_dim", "value": capacity_sig[0]},
                {"path": "training.probe_c", "value": capacity_sig[1]},
                {"path": "training.probe_min_pos", "value": capacity_sig[2]},
            ],
            "knowledge_card_ids": [
                str(card.get("card_id", ""))
                for card in cards
                if str(card.get("component", "")) == "probe_head"
            ][:4],
        },
        {
            "title": f"{experiment.title} robust-regularization pca{regularization_sig[0]}-c{regularization_sig[1]:.2f}-min{regularization_sig[2]}",
            "family": experiment.family,
            "hypothesis": "Test whether a more conservative probe regularization setting improves holdout stability without collapsing rare classes.",
            "reason": (
                "Negative calibration priors were found, so this branch stays inside the probe training surface instead of doing post-hoc calibration."
                if calibration_veto
                else "Keep a low-risk structural hedge inside the probe training surface."
            ),
            "branch_role": "explore",
            "idea_class": "probe_regularization",
            "target_component": "probe_head",
            "priority_delta": 14,
            "launch_mode": "background",
            "config_overrides": [
                {"path": "training.probe_pca_dim", "value": regularization_sig[0]},
                {"path": "training.probe_c", "value": regularization_sig[1]},
                {"path": "training.probe_min_pos", "value": regularization_sig[2]},
            ],
            "knowledge_card_ids": [str(item) for item in knowledge_bundle.get("knowledge_card_ids", [])][:4],
        },
    ]
    return candidates[: _branch_plan_limit()]


def _fallback_generic_branch_candidates(
    *,
    experiment,
    source_config: dict[str, Any],
    knowledge_bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    training = source_config.get("training", {})
    epochs = int(training.get("epochs", 1))
    learning_rate = training.get("learning_rate")
    cards = _bundle_cards(knowledge_bundle)
    lr_value = float(learning_rate) if isinstance(learning_rate, (int, float)) else 0.001
    candidates = [
        {
            "title": f"{experiment.title} longer-schedule",
            "family": experiment.family,
            "hypothesis": "Extend training schedule before changing downstream calibration so the model has a chance to realize existing priors.",
            "reason": "General fallback branch that increases schedule depth rather than doing post-hoc tuning only.",
            "branch_role": "primary",
            "idea_class": "optimization",
            "target_component": "training",
            "priority_delta": 10,
            "launch_mode": "background",
            "config_overrides": [{"path": "training.epochs", "value": max(epochs + 2, 3)}],
            "knowledge_card_ids": [str(item.get("card_id", "")) for item in cards[:3]],
        },
        {
            "title": f"{experiment.title} lower-lr-stability",
            "family": experiment.family,
            "hypothesis": "Lower the learning rate to improve holdout stability if the current branch looks noisy.",
            "reason": "Fallback hedge branch for training instability.",
            "branch_role": "hedge",
            "idea_class": "optimization",
            "target_component": "training",
            "priority_delta": 12,
            "launch_mode": "background",
            "config_overrides": [{"path": "training.learning_rate", "value": lr_value * 0.5}],
            "knowledge_card_ids": [str(item.get("card_id", "")) for item in cards[:3]],
        },
        {
            "title": f"{experiment.title} higher-lr-recovery",
            "family": experiment.family,
            "hypothesis": "Raise the learning rate moderately to test whether the branch is under-updating.",
            "reason": "Fallback exploration branch for underfitting.",
            "branch_role": "explore",
            "idea_class": "optimization",
            "target_component": "training",
            "priority_delta": 14,
            "launch_mode": "background",
            "config_overrides": [{"path": "training.learning_rate", "value": lr_value * 1.25}],
            "knowledge_card_ids": [str(item.get("card_id", "")) for item in cards[:3]],
        },
    ]
    return candidates[: _branch_plan_limit()]


def _fallback_branch_candidates(
    *,
    config: WorkspaceConfig,
    state: WorkspaceState,
    run_id: str,
    experiment,
    source_config: dict[str, Any],
    research: dict[str, Any],
    knowledge_bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    backend = str(source_config.get("training", {}).get("backend", ""))
    if backend == "sklearn_cached_probe":
        return _fallback_probe_branch_candidates(
            config=config,
            state=state,
            run_id=run_id,
            experiment=experiment,
            source_config=source_config,
            research=research,
            knowledge_bundle=knowledge_bundle,
        )
    return _fallback_generic_branch_candidates(
        experiment=experiment,
        source_config=source_config,
        knowledge_bundle=knowledge_bundle,
    )


def _resolve_source_config_path(config: WorkspaceConfig, experiment, decision: dict[str, Any]) -> Path:
    candidates: list[Path] = []
    next_config_path_value = str(decision.get("next_config_path") or "").strip()
    if next_config_path_value:
        next_config_path = Path(next_config_path_value)
        if not next_config_path.is_absolute():
            next_config_path = config.root / next_config_path
        candidates.append(next_config_path)

    experiment_config_path = Path(str(experiment.config_path))
    if not experiment_config_path.is_absolute():
        experiment_config_path = config.root / experiment_config_path
    candidates.append(experiment_config_path)

    default_config_path = config.runtime_root() / "configs" / "default.yaml"
    if default_config_path not in candidates:
        candidates.append(default_config_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _canonical_branch_plan(
    *,
    config: WorkspaceConfig,
    run,
    experiment,
    decision: dict[str, Any],
    source_config_path: Path,
    source_config: dict[str, Any],
    portfolio_id: str,
    branch_input: dict[str, Any],
    branch_rank: int,
    default_title: str,
    default_family: str,
    default_hypothesis: str,
    default_reason: str,
    default_launch_mode: str,
    default_priority: int,
    default_knowledge_ids: list[str],
) -> dict[str, Any]:
    title = str(branch_input.get("title") or default_title).strip() or default_title
    family = str(branch_input.get("family") or default_family).strip() or default_family
    hypothesis = str(branch_input.get("hypothesis") or default_hypothesis).strip() or default_hypothesis
    reason = str(branch_input.get("reason") or branch_input.get("rationale") or default_reason).strip() or default_reason
    branch_role = str(branch_input.get("branch_role") or ("primary" if branch_rank == 0 else "explore")).strip() or "explore"
    idea_class = str(branch_input.get("idea_class") or branch_input.get("target_component") or family).strip() or family
    launch_mode = str(branch_input.get("launch_mode") or default_launch_mode or "background").strip() or "background"
    priority_delta = int(branch_input.get("priority_delta", 0) or 0)
    priority = int(branch_input.get("priority") or (default_priority + priority_delta + branch_rank * 2))
    knowledge_card_ids = [str(item) for item in branch_input.get("knowledge_card_ids", [])] or list(default_knowledge_ids)
    config_overrides = _normalize_override_ops(branch_input.get("config_overrides"))
    config_path_value = str(branch_input.get("config_path") or "").strip()

    config_path = Path(config_path_value) if config_path_value else source_config_path
    if not config_path.is_absolute():
        config_path = config.root / config_path
    config_payload = copy.deepcopy(source_config)
    if config_path.exists():
        config_payload = _config_payload(config_path)
    if config_overrides or not config_path.exists() or not config_path_value:
        config_payload = _apply_override_ops(config_payload, config_overrides)
        config_payload.setdefault("experiment", {})
        config_payload["experiment"]["name"] = slugify(title)
        generated_name = f"{slugify(title)}.yaml"
        generated_path = config.generated_config_root() / generated_name
        atomic_write_text(generated_path, yaml.safe_dump(config_payload, sort_keys=False))
        config_path = generated_path

    tags = [str(item) for item in branch_input.get("tags", []) if str(item).strip()]
    if not tags:
        tags = [experiment.family, "planned", "branch-search", branch_role, slugify(idea_class)]

    dedupe_key = str(branch_input.get("dedupe_key") or f"plan:{run.run_id}:{portfolio_id}:{branch_rank:02d}:{slugify(title)}")
    depends_on = [str(item) for item in branch_input.get("depends_on", [run.work_item_id])]
    work_type = str(branch_input.get("work_type") or "experiment_iteration")

    return {
        "title": title,
        "family": family,
        "hypothesis": hypothesis,
        "reason": reason,
        "config_path": _relative_config_path(config, config_path),
        "priority": priority,
        "depends_on": depends_on,
        "tags": tags,
        "launch_mode": launch_mode,
        "dedupe_key": dedupe_key,
        "work_type": work_type,
        "portfolio_id": portfolio_id,
        "idea_class": idea_class,
        "branch_role": branch_role,
        "branch_rank": branch_rank,
        "knowledge_card_ids": knowledge_card_ids,
        "config_overrides": config_overrides,
    }


def _canonicalize_plan_payload(
    *,
    config: WorkspaceConfig,
    state: WorkspaceState,
    run,
    experiment,
    decision: dict[str, Any],
    research: dict[str, Any],
    knowledge_bundle: dict[str, Any],
    source_config_path: Path,
    source_config: dict[str, Any],
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    next_action = str(decision.get("next_action", "hold"))
    if payload is None:
        payload = {}
    plan_status = str(payload.get("plan_status") or ("submission_candidate" if next_action == "submit_candidate" else ("planned" if next_action != "hold" else "hold")))
    reason = str(payload.get("reason") or decision.get("why") or f"Follow-up from {run.run_id}")
    if plan_status == "hold":
        return {
            "stage": "plan",
            "plan_status": "hold",
            "source_run_id": run.run_id,
            "reason": reason,
            "title": str(payload.get("title") or decision.get("next_title") or experiment.title),
            "family": str(payload.get("family") or decision.get("next_family") or experiment.family),
            "hypothesis": str(payload.get("hypothesis") or reason),
            "config_path": str(payload.get("config_path") or experiment.config_path),
            "priority": int(payload.get("priority") or experiment.priority),
            "depends_on": [run.work_item_id],
            "tags": [experiment.family, "hold"],
            "launch_mode": str(payload.get("launch_mode") or decision.get("launch_mode") or "background"),
            "dedupe_key": str(payload.get("dedupe_key") or f"plan:hold:{run.run_id}"),
            "work_type": str(payload.get("work_type") or "experiment_iteration"),
            "portfolio_id": "",
            "knowledge_card_ids": [str(item) for item in knowledge_bundle.get("knowledge_card_ids", [])],
            "problem_frame": knowledge_bundle.get("problem_frame", {}),
            "branch_plans": [],
        }
    if plan_status == "submission_candidate":
        return {
            "stage": "plan",
            "plan_status": "submission_candidate",
            "source_run_id": run.run_id,
            "reason": reason,
            "title": str(payload.get("title") or decision.get("next_title") or experiment.title),
            "family": str(payload.get("family") or decision.get("next_family") or experiment.family),
            "hypothesis": str(payload.get("hypothesis") or reason),
            "config_path": str(payload.get("config_path") or experiment.config_path),
            "priority": int(payload.get("priority") or experiment.priority),
            "depends_on": [run.work_item_id],
            "tags": [experiment.family, "submission_candidate"],
            "launch_mode": str(payload.get("launch_mode") or decision.get("launch_mode") or "background"),
            "dedupe_key": str(payload.get("dedupe_key") or f"plan:submission:{run.run_id}"),
            "work_type": str(payload.get("work_type") or "experiment_iteration"),
            "portfolio_id": "",
            "knowledge_card_ids": [str(item) for item in knowledge_bundle.get("knowledge_card_ids", [])],
            "problem_frame": knowledge_bundle.get("problem_frame", {}),
            "branch_plans": [],
        }

    branch_inputs = payload.get("branch_plans")
    if not isinstance(branch_inputs, list) or not branch_inputs:
        branch_inputs = []
        decision_portfolio = decision.get("branch_portfolio")
        if isinstance(decision_portfolio, list) and decision_portfolio:
            branch_inputs.extend(item for item in decision_portfolio if isinstance(item, dict))
        if not branch_inputs and payload:
            branch_inputs.append(payload)
        if not branch_inputs:
            branch_inputs.extend(
                _fallback_branch_candidates(
                    config=config,
                    state=state,
                    run_id=run.run_id,
                    experiment=experiment,
                    source_config=source_config,
                    research=research,
                    knowledge_bundle=knowledge_bundle,
                )
            )

    branch_inputs = [item for item in branch_inputs if isinstance(item, dict)][: _branch_plan_limit()]
    portfolio_id = str(payload.get("portfolio_id") or f"portfolio-{run.run_id}-{slugify(experiment.family)}")
    default_title = str(payload.get("title") or decision.get("next_title") or f"{experiment.title} follow-up")
    default_family = str(payload.get("family") or decision.get("next_family") or experiment.family)
    default_hypothesis = str(payload.get("hypothesis") or decision.get("why") or f"Follow-up from {run.run_id}")
    default_priority = int(payload.get("priority") or (experiment.priority + int(decision.get("priority_delta", 10) or 10)))
    default_launch_mode = str(payload.get("launch_mode") or decision.get("launch_mode") or "background")
    default_knowledge_ids = [str(item) for item in payload.get("knowledge_card_ids", [])] or [
        str(item) for item in knowledge_bundle.get("knowledge_card_ids", [])
    ]

    branch_plans = [
        _canonical_branch_plan(
            config=config,
            run=run,
            experiment=experiment,
            decision=decision,
            source_config_path=source_config_path,
            source_config=source_config,
            portfolio_id=portfolio_id,
            branch_input=branch_input,
            branch_rank=index,
            default_title=default_title if index == 0 else f"{default_title} branch {index + 1}",
            default_family=default_family,
            default_hypothesis=default_hypothesis,
            default_reason=reason,
            default_launch_mode=default_launch_mode,
            default_priority=default_priority,
            default_knowledge_ids=default_knowledge_ids,
        )
        for index, branch_input in enumerate(branch_inputs)
    ]
    primary = branch_plans[0]
    return {
        "stage": "plan",
        "plan_status": "planned",
        "source_run_id": run.run_id,
        "reason": primary["reason"],
        "title": primary["title"],
        "family": primary["family"],
        "hypothesis": primary["hypothesis"],
        "config_path": primary["config_path"],
        "priority": primary["priority"],
        "depends_on": primary["depends_on"],
        "tags": primary["tags"],
        "launch_mode": primary["launch_mode"],
        "dedupe_key": primary["dedupe_key"],
        "work_type": primary["work_type"],
        "portfolio_id": portfolio_id,
        "knowledge_card_ids": primary["knowledge_card_ids"],
        "problem_frame": knowledge_bundle.get("problem_frame", {}),
        "branch_plans": branch_plans,
        "created_at": now_utc_iso(),
    }


def build_plan(config: WorkspaceConfig, state: WorkspaceState, run_id: str):
    run = next(item for item in state.runs if item.run_id == run_id)
    experiment = next(item for item in state.experiments if item.id == run.experiment_id)
    decision = latest_stage_payload(state, run_id, "decision")
    research = latest_stage_payload(state, run_id, "research")
    stage_run, input_manifest_path = begin_stage_run(
        config,
        state,
        run,
        stage_name="plan",
        input_ref=run.latest_stage_run_id or run.run_id,
    )
    knowledge_bundle = retrieve_knowledge_bundle(
        config,
        {
            "run": run.to_dict(),
            "experiment": experiment.to_dict(),
            "decision": decision,
            "research": research,
        },
        stage="plan",
    )
    write_input_manifest(
        input_manifest_path,
        {
            "run": run.to_dict(),
            "experiment": experiment.to_dict(),
            "decision": decision,
            "research": research,
            "retrieved_knowledge": compact_knowledge_bundle(knowledge_bundle),
        },
    )
    source_config_path = _resolve_source_config_path(config, experiment, decision)
    source_config = _config_payload(source_config_path)
    source_config.setdefault("experiment", {})
    source_config.setdefault("training", {})
    source_config.setdefault("data", {})
    adapted = run_configured_stage_adapter(
        config,
        state,
        stage_run,
        input_manifest_path=input_manifest_path,
        extra_env={"KAGGLE_AGENT_RUN_ID": run_id},
    )
    adapted_payload = adapted[0] if adapted is not None else None
    payload = _canonicalize_plan_payload(
        config=config,
        state=state,
        run=run,
        experiment=experiment,
        decision=decision,
        research=research,
        knowledge_bundle=knowledge_bundle,
        source_config_path=source_config_path,
        source_config=source_config,
        payload=adapted_payload,
    )

    spec_yaml_path = Path(stage_run.output_dir) / "spec.yaml"
    atomic_write_text(spec_yaml_path, yaml.safe_dump(payload, sort_keys=False))
    stage_run.spec_path = str(spec_yaml_path)
    branch_plans = payload.get("branch_plans", [])
    markdown_lines = [
        f"- Status: `{payload['plan_status']}`",
        f"- Title: {payload['title']}",
        f"- Config: `{payload['config_path']}`",
        f"- Launch mode: `{payload['launch_mode']}`",
        f"- Hypothesis: {payload['hypothesis']}",
    ]
    if isinstance(branch_plans, list) and branch_plans:
        markdown_lines.extend(["", "## Branch Portfolio"])
        for branch in branch_plans:
            if not isinstance(branch, dict):
                continue
            markdown_lines.append(
                f"- `{branch.get('branch_role', 'branch')}` | {branch.get('title', '')} | idea={branch.get('idea_class', '')} | config=`{branch.get('config_path', '')}`"
            )
    markdown = stage_markdown(f"Plan {run_id}", markdown_lines)
    complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)
    return stage_run
