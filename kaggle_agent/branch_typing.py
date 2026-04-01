from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from kaggle_agent.schema import (
    BranchTypingRecord,
    InfoGainEstimateRecord,
    ProposalTypingRecord,
    RealizedTypingRecord,
    WorkspaceState,
)
from kaggle_agent.utils import now_utc_iso, slugify


AXIS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "pseudo_label": (
        "pseudo",
        "teacher",
        "distill",
        "kd",
        "soft_label",
        "hard_pseudo",
    ),
    "class_coverage": (
        "coverage",
        "imbalance",
        "rare",
        "class_weight",
        "min_pos",
        "oversample",
        "sampler",
        "long_tail",
    ),
    "probe_head": (
        "probe",
        "head",
        "mlp",
        "pca",
        "embedding",
        "regularization",
    ),
    "prior_calibration": (
        "prior",
        "blend",
        "temperature",
        "calibration",
        "fusion",
        "threshold",
    ),
    "preprocessing_aug": (
        "pcen",
        "augment",
        "specaugment",
        "shift",
        "mel",
        "normalization",
    ),
    "backbone": (
        "backbone",
        "encoder",
        "model_name",
        "architecture",
        "v2s",
        "b0",
    ),
    "optimization": (
        "lr",
        "learning_rate",
        "epochs",
        "scheduler",
        "optimizer",
        "weight_decay",
        "batch_size",
    ),
    "data_filtering": (
        "rating",
        "filter",
        "subset",
        "quality",
        "drop",
        "curriculum",
    ),
}

SURFACE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "data": ("data.", "dataset", "sampler", "subset", "pseudo", "filter", "rating"),
    "training": ("training.", "optimizer", "scheduler", "batch", "epochs", "lr", "weight_decay"),
    "model": ("model.", "backbone", "encoder", "head", "probe", "mlp", "pca"),
    "loss": ("loss", "criterion", "asl", "focal", "distill"),
    "preprocess": ("pcen", "mel", "normalize", "specaugment", "shift", "augment"),
    "postprocess": ("prior", "blend", "temperature", "calibration", "threshold", "submission"),
}

LOW_INFORMATION_PATH_HINTS = (
    "prior",
    "blend",
    "temperature",
    "calibration",
    "threshold",
)

COST_TIER_ORDER = {"low": 0, "medium": 1, "high": 2}


def _normalize_override_ops(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "")).strip()
        if not path:
            continue
        normalized.append({"path": path, "value": item.get("value")})
    return normalized


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


def _recursive_diff(before: Any, after: Any, *, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(before, dict) and isinstance(after, dict):
        changes: list[dict[str, Any]] = []
        keys = sorted(set(before.keys()) | set(after.keys()))
        for key in keys:
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key not in before:
                changes.append({"path": next_prefix, "before": None, "after": after[key]})
                continue
            if key not in after:
                changes.append({"path": next_prefix, "before": before[key], "after": None})
                continue
            changes.extend(_recursive_diff(before[key], after[key], prefix=next_prefix))
        return changes
    if before != after:
        return [{"path": prefix, "before": before, "after": after}]
    return []


def _load_config_payload(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return {}
    try:
        payload = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _candidate_paths(branch_input: dict[str, Any], source_config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    overrides = _normalize_override_ops(branch_input.get("config_overrides"))
    if overrides:
        updated = _apply_override_ops(source_config, overrides)
        return _recursive_diff(source_config, updated), updated
    config_diff = branch_input.get("config_diff")
    if isinstance(config_diff, list):
        normalized = _normalize_override_ops(config_diff)
        updated = _apply_override_ops(source_config, normalized)
        return _recursive_diff(source_config, updated), updated
    return [], copy.deepcopy(source_config)


def _is_bookkeeping_path(path: str) -> bool:
    lowered = str(path).strip().lower()
    return lowered.startswith("experiment.")


def _infer_axis_tags(changed_paths: list[str]) -> list[str]:
    joined = " ".join(path.lower() for path in changed_paths)
    tags: list[str] = []
    for axis, keywords in AXIS_KEYWORDS.items():
        if any(keyword in joined for keyword in keywords):
            tags.append(axis)
    return tags or ["general"]


def _infer_change_surface(changed_paths: list[str]) -> list[str]:
    surfaces: list[str] = []
    lowered_paths = [path.lower() for path in changed_paths]
    for surface, keywords in SURFACE_KEYWORDS.items():
        if any(keyword in path for path in lowered_paths for keyword in keywords):
            surfaces.append(surface)
    return surfaces or (["config"] if lowered_paths else ["unknown"])


def _dominant_idea_class(axis_tags: list[str], branch_input: dict[str, Any]) -> str:
    explicit = str(branch_input.get("idea_class") or branch_input.get("target_component") or "").strip()
    if explicit:
        return explicit
    if axis_tags and axis_tags[0] != "general":
        return axis_tags[0]
    return "general"


def _pattern_tags(axis_tags: list[str], change_surface: list[str], changed_paths: list[str], branch_input: dict[str, Any]) -> list[str]:
    patterns: list[str] = []
    axis_set = set(axis_tags)
    text = " ".join([*changed_paths, str(branch_input.get("work_type", "")), str(branch_input.get("title", ""))]).lower()
    if "class_coverage" in axis_set:
        patterns.append("coverage_first")
    if "pseudo_label" in axis_set:
        patterns.append("pseudo_label_expansion")
    if "probe_head" in axis_set:
        patterns.append("probe_training_change")
    if "optimization" in axis_set:
        patterns.append("schedule_recovery")
    if "prior_calibration" in axis_set:
        if any(token in text for token in ("blend", "prior", "fusion")):
            patterns.append("blend_only")
        patterns.append("calibration_only")
    if "backbone" in axis_set:
        patterns.append("conditional_backbone_recovery")
    if str(branch_input.get("work_type", "")).strip() == "submission":
        patterns.append("submission")
    if change_surface == ["postprocess"] and "calibration_only" not in patterns:
        patterns.append("calibration_only")
    return list(dict.fromkeys(patterns))


def _signature_payload(axis_tags: list[str], change_surface: list[str], changes: list[dict[str, Any]], branch_input: dict[str, Any]) -> dict[str, Any]:
    normalized_changes = [
        {
            "path": str(item.get("path", "")),
            "after": item.get("after"),
        }
        for item in sorted(changes, key=lambda item: str(item.get("path", "")))
    ]
    return {
        "axis_tags": sorted(axis_tags),
        "change_surface": sorted(change_surface),
        "changes": normalized_changes,
        "work_type": str(branch_input.get("work_type", "") or "experiment_iteration"),
    }


def _typing_signature(signature_payload: dict[str, Any]) -> str:
    encoded = json.dumps(signature_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]
    axis_part = "+".join(signature_payload.get("axis_tags", []))
    surface_part = "+".join(signature_payload.get("change_surface", []))
    return f"{axis_part}|{surface_part}|{digest}"


def _signature_similarity(signature_payload: dict[str, Any], previous_payload: dict[str, Any]) -> float:
    current_paths = {str(item.get("path", "")) for item in signature_payload.get("changes", []) if str(item.get("path", ""))}
    previous_paths = {str(item.get("path", "")) for item in previous_payload.get("changes", []) if str(item.get("path", ""))}
    current_axes = set(signature_payload.get("axis_tags", []))
    previous_axes = set(previous_payload.get("axis_tags", []))
    if not current_paths and not previous_paths and current_axes == previous_axes:
        return 1.0
    path_union = current_paths | previous_paths
    axis_union = current_axes | previous_axes
    path_sim = len(current_paths & previous_paths) / len(path_union) if path_union else 0.0
    axis_sim = len(current_axes & previous_axes) / len(axis_union) if axis_union else 0.0
    return round(0.6 * axis_sim + 0.4 * path_sim, 3)


def _historical_signature_payloads(state: WorkspaceState, family: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for record in state.proposal_typings:
        if family and record.family and record.family != family:
            continue
        payload = record.typing_payload if isinstance(record.typing_payload, dict) else {}
        signature_payload = payload.get("signature_payload")
        if isinstance(signature_payload, dict):
            payloads.append(signature_payload)
    for record in state.realized_typings:
        if family and record.family and record.family != family:
            continue
        payload = record.typing_payload if isinstance(record.typing_payload, dict) else {}
        signature_payload = payload.get("signature_payload")
        if isinstance(signature_payload, dict):
            payloads.append(signature_payload)
    for record in state.branch_typings:
        if family and record.family and record.family != family:
            continue
        if not record.typing_signature_json:
            continue
        try:
            payloads.append(json.loads(record.typing_signature_json))
        except Exception:
            continue
    return payloads


def _novelty_score(state: WorkspaceState, family: str, signature_payload: dict[str, Any]) -> float:
    previous_payloads = _historical_signature_payloads(state, family)
    if not previous_payloads:
        return 1.0
    best_similarity = max(_signature_similarity(signature_payload, payload) for payload in previous_payloads)
    return round(max(0.0, min(1.0, 1.0 - best_similarity)), 3)


def _low_information_flag(axis_tags: list[str], change_surface: list[str], changed_paths: list[str]) -> bool:
    lowered_paths = [path.lower() for path in changed_paths]
    if not lowered_paths:
        return True
    if set(change_surface) <= {"postprocess"}:
        return True
    if set(axis_tags) <= {"prior_calibration", "general"}:
        return True
    if len(lowered_paths) <= 2 and all(any(token in path for token in LOW_INFORMATION_PATH_HINTS) for path in lowered_paths):
        return True
    return False


def _cost_tier(axis_tags: list[str], change_surface: list[str], change_count: int, grounding_mode: str, work_type: str) -> str:
    if work_type == "submission" or set(change_surface) <= {"postprocess"} or change_count <= 1:
        return "low"
    if any(axis in {"backbone", "pseudo_label"} for axis in axis_tags):
        return "high"
    if grounding_mode == "novel" and any(axis in {"backbone", "optimization"} for axis in axis_tags):
        return "high"
    if len(change_surface) >= 3 or change_count >= 6:
        return "high"
    return "medium"


def _branch_history_bias(state: WorkspaceState, family: str, idea_class: str) -> tuple[int, int]:
    strong = 0
    weak = 0
    for memory in reversed(state.branch_memories):
        if family and memory.family and memory.family != family:
            continue
        if idea_class and memory.idea_class and memory.idea_class != idea_class:
            continue
        if memory.outcome in {"leader", "improved", "submission_candidate"}:
            strong += 1
        elif memory.outcome in {"regressed", "critic_rejected", "run_failed", "validate_failed"}:
            weak += 1
        if strong + weak >= 8:
            break
    return strong, weak


def compile_proposal_typing(
    state: WorkspaceState,
    *,
    run_id: str,
    stage_run_id: str,
    family: str,
    title: str,
    branch_input: dict[str, Any],
    source_config: dict[str, Any],
) -> dict[str, Any]:
    changes, _ = _candidate_paths(branch_input, source_config)
    changed_paths = [str(item.get("path", "")) for item in changes if str(item.get("path", ""))]
    explicit_component = str(branch_input.get("idea_class") or branch_input.get("target_component") or "").strip()
    axis_tags = _infer_axis_tags(changed_paths)
    if not changed_paths and explicit_component and explicit_component != "general":
        axis_tags = [explicit_component]
    change_surface = _infer_change_surface(changed_paths)
    idea_class = _dominant_idea_class(axis_tags, branch_input)
    grounding_mode = str(branch_input.get("grounding_mode", "") or "").strip() or (
        "novel" if branch_input.get("unsupported_claims") else "grounded"
    )
    low_information_flag = _low_information_flag(axis_tags, change_surface, changed_paths)
    if not changed_paths and explicit_component and explicit_component not in {"general", "prior_calibration"}:
        low_information_flag = False
    signature_payload = _signature_payload(axis_tags, change_surface, changes, branch_input)
    typing_signature = _typing_signature(signature_payload)
    novelty_score = _novelty_score(state, family, signature_payload)
    unsupported_claims = [str(item) for item in branch_input.get("unsupported_claims", []) if str(item)]
    required_evidence = [str(item) for item in branch_input.get("required_evidence", []) if str(item)]
    requires_override = bool(branch_input.get("override_reason")) or low_information_flag or (
        grounding_mode == "novel" and not required_evidence
    )
    work_type = str(branch_input.get("work_type", "") or "experiment_iteration")
    typing_payload = {
        "typing_kind": "proposal",
        "typing_signature": typing_signature,
        "signature_payload": signature_payload,
        "idea_class": idea_class,
        "axis_tags": axis_tags,
        "change_surface": change_surface,
        "pattern_tags": _pattern_tags(axis_tags, change_surface, changed_paths, branch_input),
        "changed_paths": changed_paths,
        "change_count": len(changed_paths),
        "novelty_score": novelty_score,
        "low_information_flag": low_information_flag,
        "requires_override": requires_override,
        "grounding_mode": grounding_mode,
        "unsupported_claims": unsupported_claims,
        "required_evidence": required_evidence,
        "cost_tier": _cost_tier(axis_tags, change_surface, len(changed_paths), grounding_mode, work_type),
        "work_type": work_type,
        "structural_hint_only": bool(not changed_paths and explicit_component and explicit_component not in {"", "general"}),
    }
    proposal_typing_id = f"proposal-{run_id}-{slugify(title)}-{slugify(typing_signature)[:10]}"
    return {
        "proposal_typing_id": proposal_typing_id,
        "run_id": run_id,
        "stage_run_id": stage_run_id,
        "title": title,
        "family": family,
        "config_path": str(branch_input.get("config_path", "") or ""),
        "typing_signature": typing_signature,
        "typing_signature_json": json.dumps(signature_payload, sort_keys=True, ensure_ascii=False),
        "typing_payload": typing_payload,
        **typing_payload,
        "created_at": now_utc_iso(),
    }


def compile_realized_typing(
    *,
    run_id: str,
    stage_run_id: str,
    family: str,
    title: str,
    branch_input: dict[str, Any],
    proposal_typing: dict[str, Any] | None,
    source_config_path: str = "",
    fallback_source_config: dict[str, Any] | None = None,
    generated_config_path: str = "",
    changed_files: list[str] | None = None,
) -> dict[str, Any]:
    base_config = _load_config_payload(source_config_path) if source_config_path else {}
    if not base_config and isinstance(fallback_source_config, dict):
        base_config = copy.deepcopy(fallback_source_config)
    realized_config = _load_config_payload(generated_config_path) if generated_config_path else {}
    changes = _recursive_diff(base_config, realized_config) if base_config and realized_config else []
    changed_paths = [
        str(item.get("path", ""))
        for item in changes
        if str(item.get("path", "")) and not _is_bookkeeping_path(str(item.get("path", "")))
    ]
    for path in changed_files or []:
        text = str(path).strip()
        if text and text not in changed_paths:
            changed_paths.append(text)
    fallback_payload = proposal_typing.get("typing_payload", {}) if isinstance(proposal_typing, dict) else {}
    axis_tags = _infer_axis_tags(changed_paths) if changed_paths else [str(item) for item in fallback_payload.get("axis_tags", [])] or ["general"]
    change_surface = _infer_change_surface(changed_paths) if changed_paths else [str(item) for item in fallback_payload.get("change_surface", [])] or ["unknown"]
    idea_class = str(branch_input.get("idea_class") or branch_input.get("target_component") or fallback_payload.get("idea_class") or "general")
    grounding_mode = str(branch_input.get("grounding_mode") or fallback_payload.get("grounding_mode") or "grounded")
    work_type = str(branch_input.get("work_type") or fallback_payload.get("work_type") or "experiment_iteration")
    shadow_branch_input = dict(branch_input)
    shadow_branch_input.setdefault("work_type", work_type)
    signature_payload = _signature_payload(axis_tags, change_surface, changes, shadow_branch_input)
    typing_signature = _typing_signature(signature_payload)
    typing_payload = {
        "typing_kind": "realized",
        "typing_signature": typing_signature,
        "signature_payload": signature_payload,
        "idea_class": idea_class,
        "axis_tags": axis_tags,
        "change_surface": change_surface,
        "pattern_tags": _pattern_tags(axis_tags, change_surface, changed_paths, shadow_branch_input),
        "changed_paths": changed_paths,
        "change_count": len(changed_paths),
        "low_information_flag": _low_information_flag(axis_tags, change_surface, changed_paths),
        "grounding_mode": grounding_mode,
        "cost_tier": _cost_tier(axis_tags, change_surface, len(changed_paths), grounding_mode, work_type),
        "work_type": work_type,
        "generated_config_path": generated_config_path,
        "source_config_path": source_config_path,
    }
    realized_typing_id = f"realized-{run_id}-{slugify(title)}-{slugify(typing_signature)[:10]}"
    return {
        "realized_typing_id": realized_typing_id,
        "run_id": run_id,
        "stage_run_id": stage_run_id,
        "title": title,
        "family": family,
        "config_path": generated_config_path or str(branch_input.get("config_path", "") or ""),
        "typing_signature": typing_signature,
        "typing_payload": typing_payload,
        "proposal_typing_id": str(proposal_typing.get("proposal_typing_id", "") if isinstance(proposal_typing, dict) else ""),
        "drift_summary": [],
        **typing_payload,
        "created_at": now_utc_iso(),
    }


def estimate_info_gain(
    state: WorkspaceState,
    *,
    run_id: str,
    stage_run_id: str,
    family: str,
    title: str,
    branch_input: dict[str, Any],
    proposal_typing: dict[str, Any],
    search_envelope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    search_envelope = dict(search_envelope or {})
    idea_class = str(proposal_typing.get("idea_class", "") or "general")
    axis_tags = [str(item) for item in proposal_typing.get("axis_tags", [])]
    change_surface = [str(item) for item in proposal_typing.get("change_surface", [])]
    low_information_flag = bool(proposal_typing.get("low_information_flag", False))
    novelty_score = float(proposal_typing.get("novelty_score", 0.0) or 0.0)
    grounding_mode = str(proposal_typing.get("grounding_mode", "grounded") or "grounded")
    cost_tier = str(proposal_typing.get("cost_tier", "medium") or "medium")
    rationale: list[str] = []
    if low_information_flag:
        base = 0.15
        rationale.append("low-information branch surface")
    else:
        base = 0.42
        if any(axis in {"class_coverage", "pseudo_label", "probe_head", "backbone"} for axis in axis_tags):
            base += 0.16
            rationale.append("structural axis change")
        if len(change_surface) >= 2:
            base += 0.06
            rationale.append("multi-surface intervention")
        if grounding_mode == "novel":
            base += 0.04
            rationale.append("novel lane bonus")
    strong, weak = _branch_history_bias(state, family, idea_class)
    if weak > strong and not low_information_flag:
        base += 0.05
        rationale.append("frontier still unresolved for this idea class")
    if strong >= 2 and weak == 0:
        base -= 0.04
        rationale.append("axis already partially exploited")
    if cost_tier == "low":
        base += 0.04
        rationale.append("cheap branch")
    elif cost_tier == "high":
        base -= 0.05
        rationale.append("expensive branch")
    base += novelty_score * 0.22
    required_patterns = {str(item) for item in search_envelope.get("required_patterns", []) if str(item)}
    branch_patterns = {str(item) for item in proposal_typing.get("pattern_tags", []) if str(item)}
    if required_patterns and required_patterns.intersection(branch_patterns):
        base += 0.05
        rationale.append("matches required pattern")
    estimate = round(max(0.0, min(base, 1.0)), 3)
    estimate_id = f"gain-{run_id}-{slugify(title)}-{slugify(proposal_typing.get('typing_signature', ''))[:10]}"
    return {
        "estimate_id": estimate_id,
        "run_id": run_id,
        "stage_run_id": stage_run_id,
        "title": title,
        "family": family,
        "idea_class": idea_class,
        "grounding_mode": grounding_mode,
        "novelty_score": novelty_score,
        "estimated_gain": estimate,
        "cost_tier": cost_tier,
        "rationale": rationale or ["baseline estimate"],
        "created_at": now_utc_iso(),
        "low_information_flag": low_information_flag,
        "required_patterns_matched": sorted(required_patterns.intersection(branch_patterns)),
        "branch_role": str(branch_input.get("branch_role", "") or ""),
    }


def compare_typings(proposal_typing: dict[str, Any] | None, realized_typing: dict[str, Any] | None) -> dict[str, Any]:
    proposal_payload = proposal_typing.get("typing_payload", {}) if isinstance(proposal_typing, dict) else {}
    realized_payload = realized_typing.get("typing_payload", {}) if isinstance(realized_typing, dict) else {}
    proposal_axes = {str(item) for item in proposal_payload.get("axis_tags", []) if str(item)}
    realized_axes = {str(item) for item in realized_payload.get("axis_tags", []) if str(item)}
    proposal_patterns = {str(item) for item in proposal_payload.get("pattern_tags", []) if str(item)}
    realized_patterns = {str(item) for item in realized_payload.get("pattern_tags", []) if str(item)}
    summary: list[str] = []
    severe = False
    if proposal_payload and realized_payload:
        if str(proposal_payload.get("idea_class", "")) and str(proposal_payload.get("idea_class", "")) != str(realized_payload.get("idea_class", "")):
            severe = True
            summary.append(
                f"idea_class drift: planned `{proposal_payload.get('idea_class', '')}` but realized `{realized_payload.get('idea_class', '')}`"
            )
        added_axes = sorted(realized_axes - proposal_axes)
        lost_axes = sorted(proposal_axes - realized_axes)
        if lost_axes:
            summary.append(f"lost axes: {', '.join(lost_axes)}")
        if added_axes:
            summary.append(f"added axes: {', '.join(added_axes)}")
        if proposal_patterns != realized_patterns:
            lost_patterns = sorted(proposal_patterns - realized_patterns)
            added_patterns = sorted(realized_patterns - proposal_patterns)
            if lost_patterns:
                summary.append(f"lost patterns: {', '.join(lost_patterns)}")
            if added_patterns:
                summary.append(f"added patterns: {', '.join(added_patterns)}")
        if (
            not bool(proposal_payload.get("low_information_flag", False))
            and not bool(proposal_payload.get("structural_hint_only", False))
            and bool(realized_payload.get("low_information_flag", False))
        ):
            severe = True
            summary.append("realized branch collapsed into a low-information pattern")
    if not summary:
        summary.append("realized typing matches the proposal envelope.")
    return {
        "proposal_typing_id": str(proposal_typing.get("proposal_typing_id", "") if isinstance(proposal_typing, dict) else ""),
        "realized_typing_id": str(realized_typing.get("realized_typing_id", "") if isinstance(realized_typing, dict) else ""),
        "drifted": summary != ["realized typing matches the proposal envelope."],
        "severe": severe,
        "summary": summary,
    }


def envelope_violations(
    search_envelope: dict[str, Any] | None,
    *,
    proposal_typing: dict[str, Any] | None,
    realized_typing: dict[str, Any] | None,
    info_gain_estimate: dict[str, Any] | None,
    override_reason: str = "",
) -> list[str]:
    envelope = dict(search_envelope or {})
    realized_payload = realized_typing.get("typing_payload", {}) if isinstance(realized_typing, dict) else {}
    proposal_payload = proposal_typing.get("typing_payload", {}) if isinstance(proposal_typing, dict) else {}
    patterns = {
        str(item)
        for item in (
            realized_payload.get("pattern_tags", [])
            if isinstance(realized_payload.get("pattern_tags"), list)
            else proposal_payload.get("pattern_tags", [])
        )
        if str(item)
    }
    violations: list[str] = []
    if override_reason.strip():
        return violations
    forbidden_patterns = {str(item) for item in envelope.get("forbidden_patterns", []) if str(item)}
    required_patterns = {str(item) for item in envelope.get("required_patterns", []) if str(item)}
    minimum_information_gain_bar = float(envelope.get("minimum_information_gain_bar", 0.0) or 0.0)
    realized_low_information = bool(realized_payload.get("low_information_flag", proposal_payload.get("low_information_flag", False)))
    structural_hint_only = bool(proposal_payload.get("structural_hint_only", False))
    if forbidden_patterns.intersection(patterns):
        violations.extend(f"forbidden-pattern:{item}" for item in sorted(forbidden_patterns.intersection(patterns)))
    if required_patterns and not required_patterns.intersection(patterns):
        violations.append("missing-required-pattern")
    if realized_low_information and not structural_hint_only and not bool(proposal_payload.get("low_information_flag", False)):
        violations.append("low-information-realization")
    estimated_gain = float(info_gain_estimate.get("estimated_gain", 0.0) or 0.0) if isinstance(info_gain_estimate, dict) else 0.0
    if minimum_information_gain_bar and estimated_gain < minimum_information_gain_bar:
        violations.append(f"minimum-information-gain:{estimated_gain:.2f}<{minimum_information_gain_bar:.2f}")
    grounding_mode = str(realized_payload.get("grounding_mode", proposal_payload.get("grounding_mode", "grounded")) or "grounded")
    max_novel_cost_tier = str(envelope.get("novel_max_cost_tier", "medium") or "medium")
    cost_tier = str(realized_payload.get("cost_tier", proposal_payload.get("cost_tier", "medium")) or "medium")
    if grounding_mode == "novel" and COST_TIER_ORDER.get(cost_tier, 1) > COST_TIER_ORDER.get(max_novel_cost_tier, 1):
        violations.append(f"novel-cost-tier:{cost_tier}>{max_novel_cost_tier}")
    return violations


def persist_branch_typing(state: WorkspaceState, typing_payload: dict[str, Any]) -> BranchTypingRecord:
    payload = dict(typing_payload)
    nested = payload.get("typing_payload", {})
    if isinstance(nested, dict):
        payload = {**nested, **payload}
    typing_id = str(payload.get("typing_id") or payload.get("proposal_typing_id") or payload.get("realized_typing_id") or "")
    existing = next((item for item in state.branch_typings if item.typing_id == typing_id), None)
    record = BranchTypingRecord(
        typing_id=typing_id,
        run_id=str(payload.get("run_id", "")),
        stage_run_id=str(payload.get("stage_run_id", "")),
        title=str(payload.get("title", "")),
        family=str(payload.get("family", "")),
        config_path=str(payload.get("config_path", "")),
        typing_signature=str(payload.get("typing_signature", "")),
        typing_signature_json=str(payload.get("typing_signature_json", "") or json.dumps(payload.get("signature_payload", {}), sort_keys=True, ensure_ascii=False)),
        idea_class=str(payload.get("idea_class", "")),
        axis_tags=[str(item) for item in payload.get("axis_tags", [])],
        change_surface=[str(item) for item in payload.get("change_surface", [])],
        novelty_score=float(payload.get("novelty_score", 0.0) or 0.0),
        expected_information_gain=float(payload.get("expected_information_gain", payload.get("estimated_gain", 0.0)) or 0.0),
        low_information_flag=bool(payload.get("low_information_flag", False)),
        requires_override=bool(payload.get("requires_override", False)),
        grounding_mode=str(payload.get("grounding_mode", "grounded") or "grounded"),
        unsupported_claims=[str(item) for item in payload.get("unsupported_claims", [])],
        required_evidence=[str(item) for item in payload.get("required_evidence", [])],
        created_at=str(payload.get("created_at", "") or now_utc_iso()),
    )
    if existing is None:
        state.branch_typings.append(record)
        return record
    existing.run_id = record.run_id
    existing.stage_run_id = record.stage_run_id
    existing.title = record.title
    existing.family = record.family
    existing.config_path = record.config_path
    existing.typing_signature = record.typing_signature
    existing.typing_signature_json = record.typing_signature_json
    existing.idea_class = record.idea_class
    existing.axis_tags = record.axis_tags
    existing.change_surface = record.change_surface
    existing.novelty_score = record.novelty_score
    existing.expected_information_gain = record.expected_information_gain
    existing.low_information_flag = record.low_information_flag
    existing.requires_override = record.requires_override
    existing.grounding_mode = record.grounding_mode
    existing.unsupported_claims = record.unsupported_claims
    existing.required_evidence = record.required_evidence
    existing.created_at = record.created_at
    return existing


def persist_proposal_typing(state: WorkspaceState, typing_payload: dict[str, Any]) -> ProposalTypingRecord:
    proposal_typing_id = str(typing_payload.get("proposal_typing_id", ""))
    existing = next((item for item in state.proposal_typings if item.proposal_typing_id == proposal_typing_id), None)
    payload = ProposalTypingRecord(
        proposal_typing_id=proposal_typing_id,
        run_id=str(typing_payload.get("run_id", "")),
        stage_run_id=str(typing_payload.get("stage_run_id", "")),
        title=str(typing_payload.get("title", "")),
        family=str(typing_payload.get("family", "")),
        config_path=str(typing_payload.get("config_path", "")),
        typing_signature=str(typing_payload.get("typing_signature", "")),
        typing_payload=dict(typing_payload.get("typing_payload", {})),
        created_at=str(typing_payload.get("created_at", "") or now_utc_iso()),
    )
    if existing is None:
        state.proposal_typings.append(payload)
    else:
        existing.run_id = payload.run_id
        existing.stage_run_id = payload.stage_run_id
        existing.title = payload.title
        existing.family = payload.family
        existing.config_path = payload.config_path
        existing.typing_signature = payload.typing_signature
        existing.typing_payload = payload.typing_payload
        existing.created_at = payload.created_at
        payload = existing
    persist_branch_typing(
        state,
        {
            **typing_payload,
            "typing_id": proposal_typing_id,
        },
    )
    return payload


def persist_realized_typing(state: WorkspaceState, typing_payload: dict[str, Any]) -> RealizedTypingRecord:
    realized_typing_id = str(typing_payload.get("realized_typing_id", ""))
    existing = next((item for item in state.realized_typings if item.realized_typing_id == realized_typing_id), None)
    payload = RealizedTypingRecord(
        realized_typing_id=realized_typing_id,
        run_id=str(typing_payload.get("run_id", "")),
        stage_run_id=str(typing_payload.get("stage_run_id", "")),
        title=str(typing_payload.get("title", "")),
        family=str(typing_payload.get("family", "")),
        config_path=str(typing_payload.get("config_path", "")),
        typing_signature=str(typing_payload.get("typing_signature", "")),
        typing_payload=dict(typing_payload.get("typing_payload", {})),
        proposal_typing_id=str(typing_payload.get("proposal_typing_id", "")),
        drift_summary=[str(item) for item in typing_payload.get("drift_summary", [])],
        created_at=str(typing_payload.get("created_at", "") or now_utc_iso()),
    )
    if existing is None:
        state.realized_typings.append(payload)
        return payload
    existing.run_id = payload.run_id
    existing.stage_run_id = payload.stage_run_id
    existing.title = payload.title
    existing.family = payload.family
    existing.config_path = payload.config_path
    existing.typing_signature = payload.typing_signature
    existing.typing_payload = payload.typing_payload
    existing.proposal_typing_id = payload.proposal_typing_id
    existing.drift_summary = payload.drift_summary
    existing.created_at = payload.created_at
    return existing


def persist_info_gain_estimate(state: WorkspaceState, estimate_payload: dict[str, Any]) -> InfoGainEstimateRecord:
    estimate_id = str(estimate_payload.get("estimate_id", ""))
    existing = next((item for item in state.info_gain_estimates if item.estimate_id == estimate_id), None)
    payload = InfoGainEstimateRecord(
        estimate_id=estimate_id,
        run_id=str(estimate_payload.get("run_id", "")),
        stage_run_id=str(estimate_payload.get("stage_run_id", "")),
        title=str(estimate_payload.get("title", "")),
        family=str(estimate_payload.get("family", "")),
        idea_class=str(estimate_payload.get("idea_class", "")),
        grounding_mode=str(estimate_payload.get("grounding_mode", "grounded") or "grounded"),
        novelty_score=float(estimate_payload.get("novelty_score", 0.0) or 0.0),
        estimated_gain=float(estimate_payload.get("estimated_gain", 0.0) or 0.0),
        cost_tier=str(estimate_payload.get("cost_tier", "medium") or "medium"),
        rationale=[str(item) for item in estimate_payload.get("rationale", [])],
        created_at=str(estimate_payload.get("created_at", "") or now_utc_iso()),
    )
    if existing is None:
        state.info_gain_estimates.append(payload)
        return payload
    existing.run_id = payload.run_id
    existing.stage_run_id = payload.stage_run_id
    existing.title = payload.title
    existing.family = payload.family
    existing.idea_class = payload.idea_class
    existing.grounding_mode = payload.grounding_mode
    existing.novelty_score = payload.novelty_score
    existing.estimated_gain = payload.estimated_gain
    existing.cost_tier = payload.cost_tier
    existing.rationale = payload.rationale
    existing.created_at = payload.created_at
    return existing


def compile_branch_typing(
    state: WorkspaceState,
    *,
    run_id: str,
    stage_run_id: str,
    family: str,
    title: str,
    branch_input: dict[str, Any],
    source_config: dict[str, Any],
) -> dict[str, Any]:
    proposal = compile_proposal_typing(
        state,
        run_id=run_id,
        stage_run_id=stage_run_id,
        family=family,
        title=title,
        branch_input=branch_input,
        source_config=source_config,
    )
    estimate = estimate_info_gain(
        state,
        run_id=run_id,
        stage_run_id=stage_run_id,
        family=family,
        title=title,
        branch_input=branch_input,
        proposal_typing=proposal,
        search_envelope={},
    )
    return {
        "typing_id": proposal["proposal_typing_id"],
        **proposal,
        "expected_information_gain": estimate["estimated_gain"],
        "estimate_id": estimate["estimate_id"],
        "info_gain_estimate": estimate,
    }
