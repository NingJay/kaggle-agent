from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kaggle_agent.capabilities import invoke_capability_pack
from kaggle_agent.decision.helpers import latest_stage_payload, load_run_result
from kaggle_agent.knowledge_reducer import active_constraints, active_policy_rules, synchronize_policy_state
from kaggle_agent.layout import visible_runs
from kaggle_agent.schema import (
    BranchMemoryRecord,
    CapabilityPack,
    MemoryOp,
    SessionMemory,
    RuntimeState,
    WorkspaceConfig,
    WorkspaceState,
)
from kaggle_agent.utils import atomic_write_json, atomic_write_text, ensure_directory, now_utc_iso, slugify, truncate


SECTION_HEADING_RE = re.compile(r"^##+\s+(.*)$", re.MULTILINE)
TOKEN_RE = re.compile(r"[a-z0-9_]+")
POSITIVE_HINTS = (
    "biggest win",
    "works",
    "promising",
    "critical",
    "best",
    "validated",
    "outperform",
    "improvement",
    "improve",
    "load-bearing",
    "keep",
    "adopt",
)
NEGATIVE_HINTS = (
    "hurts",
    "hurt",
    "underperform",
    "underperforms",
    "didn't help",
    "did not help",
    "regress",
    "regression",
    "worse",
    "bad",
    "avoid",
    "reject",
    "blocked",
    "fail",
    "failed",
)
CONDITIONAL_HINTS = (
    "hypothesis",
    "maybe",
    "might",
    "if ",
    "unless",
    "open question",
    "condition",
    "expected",
    "still early",
    "needs",
)
COMPONENT_KEYWORDS = {
    "pseudo_label": ("pseudo", "teacher", "distillation", "kd", "soft label", "hard pseudo"),
    "class_coverage": ("imbalance", "coverage", "long tail", "min_pos", "rare", "class balance"),
    "probe_head": ("probe", "mlp", "embedding", "head", "pca", "logistic"),
    "prior_calibration": ("prior", "blend", "calibration", "temperature", "bayesian"),
    "preprocessing_aug": ("pcen", "specaugment", "shift", "augment", "mel", "normalization"),
    "backbone": ("backbone", "v2s", "b0", "encoder", "model"),
    "optimization": ("lr", "learning rate", "epochs", "scheduler", "optimizer"),
    "data_filtering": ("rating", "filter", "quality", "subset"),
}
SOURCE_PRIORITY_HINTS = {
    "00_experiment_rules.md": 3.0,
    "01_validated_findings.md": 5.0,
    "03_next_experiment_priors.md": 4.0,
    "04_submission_bar.md": 2.0,
    "experiment_conclusions.md": 3.0,
}
SEMANTIC_MEMORY_SUBDIRS = ("policies", "families", "issues", "playbooks")
DEFAULT_CAPABILITY_PACKS = (
    CapabilityPack(
        pack_id="ledger_miner",
        title="Ledger Miner",
        when_to_use="Use when recent run history should bias the next frontier.",
        returns="Top components, repeated failures, and strong axes mined from branch history.",
        applies_to=["research", "decision", "plan"],
        tags=["ledger", "history", "portfolio", "regression"],
        priority_hint=3.4,
    ),
    CapabilityPack(
        pack_id="veto_checker",
        title="Veto Checker",
        when_to_use="Use when deterministic policy rules and decision constraints should mechanically block weak branches.",
        returns="Forbidden patterns, override-required components, and active veto rule summaries.",
        applies_to=["research", "decision", "plan"],
        tags=["veto", "policy", "constraints", "negative"],
        priority_hint=3.8,
    ),
    CapabilityPack(
        pack_id="branch_diversifier",
        title="Branch Diversifier",
        when_to_use="Use when the next batch should preserve sibling diversity instead of serially repeating one axis.",
        returns="Recommended branch mix and target branch count.",
        applies_to=["decision", "plan"],
        tags=["parallel", "portfolio", "branch", "diversity"],
        priority_hint=3.5,
    ),
    CapabilityPack(
        pack_id="submission_bar_checker",
        title="Submission Bar Checker",
        when_to_use="Use when a branch is near packaging and you need explicit submission readiness checks.",
        returns="Leader metric snapshot and submission readiness summary.",
        applies_to=["decision", "plan", "submission"],
        tags=["submission", "leader", "bundle"],
        priority_hint=2.8,
    ),
    CapabilityPack(
        pack_id="novel_hypothesis_generator",
        title="Novel Hypothesis Generator",
        when_to_use="Use when the frontier is overfit to grounded priors and still needs one explicit novel lane.",
        returns="Unsupported claims, required evidence, and a novel component suggestion.",
        applies_to=["decision", "plan"],
        tags=["novel", "explore", "unsupported", "hypothesis"],
        priority_hint=2.9,
    ),
)


@dataclass(frozen=True)
class KnowledgeCard:
    card_id: str
    source_path: str
    title: str
    summary: str
    text: str
    stance: str
    component: str
    policy_type: str
    confidence: float
    applies_to: list[str] = field(default_factory=list)
    override_required: bool = False
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_knowledge_layout(config: WorkspaceConfig) -> None:
    ensure_directory(config.knowledge_root())
    ensure_directory(config.knowledge_path("research"))
    ensure_directory(config.knowledge_path("papers"))
    ensure_directory(config.knowledge_path("index"))
    ensure_directory(config.knowledge_memory_root())
    ensure_directory(config.capability_pack_root())
    for subdir in SEMANTIC_MEMORY_SUBDIRS:
        ensure_directory(config.knowledge_memory_root() / subdir)


def _knowledge_root_from_workspace(workspace_root: Path) -> Path:
    return workspace_root / "knowledge"


def _index_root_from_workspace(workspace_root: Path) -> Path:
    return _knowledge_root_from_workspace(workspace_root) / "index"


def _state_root_from_workspace(workspace_root: Path) -> Path:
    return workspace_root / "state"


def _session_memory_json_from_workspace(workspace_root: Path) -> Path:
    return _state_root_from_workspace(workspace_root) / "session_memory.json"


def _session_memory_markdown_from_workspace(workspace_root: Path) -> Path:
    return _state_root_from_workspace(workspace_root) / "session_memory.md"


def _knowledge_memory_root_from_workspace(workspace_root: Path) -> Path:
    return _knowledge_root_from_workspace(workspace_root) / "memory"


def _capability_pack_root_from_workspace(workspace_root: Path) -> Path:
    return _knowledge_root_from_workspace(workspace_root) / "capability_packs"


def _dedupe_strings(values: list[str], *, limit: int = 6) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
        if len(ordered) >= limit:
            break
    return ordered


def _section_summary(text: str) -> str:
    for line in text.splitlines():
        candidate = line.strip().lstrip("-*").strip()
        if candidate:
            return truncate(candidate, limit=220)
    return ""


def _tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if len(token) >= 3]


def _keywords_for_card(source_path: str, title: str, text: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for token in [*_tokenize(source_path.replace("/", " ")), *_tokenize(title), *_tokenize(text)]:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
        if len(ordered) >= 24:
            break
    return ordered


def _infer_stance(source_path: str, title: str, text: str) -> str:
    corpus = f"{source_path}\n{title}\n{text}".lower()
    positive_hits = sum(1 for hint in POSITIVE_HINTS if hint in corpus)
    negative_hits = sum(1 for hint in NEGATIVE_HINTS if hint in corpus)
    conditional_hits = sum(1 for hint in CONDITIONAL_HINTS if hint in corpus)
    if negative_hits > max(positive_hits, conditional_hits):
        return "negative"
    if positive_hits > max(negative_hits, conditional_hits):
        return "positive"
    if conditional_hits:
        return "conditional"
    if "validated" in source_path or "conclusions" in source_path:
        return "positive"
    return "general"


def _infer_component(source_path: str, title: str, text: str) -> str:
    corpus = f"{source_path}\n{title}\n{text}".lower()
    best_component = "general"
    best_score = 0
    for component, keywords in COMPONENT_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in corpus)
        if score > best_score:
            best_component = component
            best_score = score
    return best_component


def _confidence_for_card(source_path: str, title: str, text: str, stance: str) -> float:
    corpus = f"{source_path}\n{title}\n{text}".lower()
    score = 0.42
    score += min(0.24, SOURCE_PRIORITY_HINTS.get(Path(source_path).name, 0.0) * 0.04)
    if stance in {"positive", "negative"}:
        score += 0.16
    elif stance == "conditional":
        score += 0.08
    if "validated" in corpus or "biggest win" in corpus or "load-bearing" in corpus:
        score += 0.10
    if "regress" in corpus or "hurts" in corpus or "didn't help" in corpus or "did not help" in corpus:
        score += 0.10
    return max(0.35, min(0.95, round(score, 3)))


def _policy_type_for_card(stance: str, confidence: float) -> str:
    if stance == "negative":
        return "veto" if confidence >= 0.72 else "avoid"
    if stance == "positive":
        return "require" if confidence >= 0.82 else "prefer"
    if stance == "conditional":
        return "conditional"
    return "context"


def _applies_to_for_card(component: str, keywords: list[str]) -> list[str]:
    scope: list[str] = []
    if component and component != "general":
        scope.append(component)
    for keyword in keywords[:4]:
        if keyword not in scope:
            scope.append(keyword)
    return scope


def _iter_sections(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    matches = list(SECTION_HEADING_RE.finditer(text))
    if not matches:
        return [(path.stem.replace("_", " "), text)]
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((match.group(1).strip(), body))
    return sections or [(path.stem.replace("_", " "), text)]


def _compile_knowledge_index(knowledge_root: Path) -> list[KnowledgeCard]:
    cards: list[KnowledgeCard] = []
    if not knowledge_root.exists():
        return cards
    for path in sorted(knowledge_root.rglob("*.md")):
        if not path.is_file():
            continue
        if "index" in path.parts:
            continue
        source_path = path.relative_to(knowledge_root).as_posix()
        for section_title, section_text in _iter_sections(path):
            summary = _section_summary(section_text) or section_title
            card_id = slugify(f"{source_path}-{section_title}")
            cards.append(
                KnowledgeCard(
                    card_id=card_id,
                    source_path=source_path,
                    title=section_title,
                    summary=summary,
                    text=truncate(section_text, limit=1200),
                    stance=_infer_stance(source_path, section_title, section_text),
                    component=_infer_component(source_path, section_title, section_text),
                    policy_type=_policy_type_for_card(
                        _infer_stance(source_path, section_title, section_text),
                        _confidence_for_card(
                            source_path,
                            section_title,
                            section_text,
                            _infer_stance(source_path, section_title, section_text),
                        ),
                    ),
                    confidence=_confidence_for_card(
                        source_path,
                        section_title,
                        section_text,
                        _infer_stance(source_path, section_title, section_text),
                    ),
                    applies_to=_applies_to_for_card(
                        _infer_component(source_path, section_title, section_text),
                        _keywords_for_card(source_path, section_title, section_text),
                    ),
                    override_required=_policy_type_for_card(
                        _infer_stance(source_path, section_title, section_text),
                        _confidence_for_card(
                            source_path,
                            section_title,
                            section_text,
                            _infer_stance(source_path, section_title, section_text),
                        ),
                    )
                    == "veto",
                    keywords=_keywords_for_card(source_path, section_title, section_text),
                )
            )
    return cards


def compile_knowledge_index(config: WorkspaceConfig) -> list[dict[str, Any]]:
    ensure_knowledge_layout(config)
    cards = _compile_knowledge_index(config.knowledge_root())
    atomic_write_json(config.knowledge_path("index", "cards.json"), [card.to_dict() for card in cards])
    return [card.to_dict() for card in cards]


def compile_knowledge_index_from_root(workspace_root: Path) -> list[dict[str, Any]]:
    knowledge_root = _knowledge_root_from_workspace(workspace_root)
    ensure_directory(knowledge_root / "index")
    cards = _compile_knowledge_index(knowledge_root)
    atomic_write_json(_index_root_from_workspace(workspace_root) / "cards.json", [card.to_dict() for card in cards])
    return [card.to_dict() for card in cards]


def build_problem_frame(manifest: dict[str, Any], *, stage: str = "") -> dict[str, Any]:
    run = manifest.get("run", {}) if isinstance(manifest.get("run"), dict) else {}
    experiment = manifest.get("experiment", {}) if isinstance(manifest.get("experiment"), dict) else {}
    report = manifest.get("report", {}) if isinstance(manifest.get("report"), dict) else {}
    research = manifest.get("research", {}) if isinstance(manifest.get("research"), dict) else {}
    decision = manifest.get("decision", {}) if isinstance(manifest.get("decision"), dict) else {}
    plan = manifest.get("plan", {}) if isinstance(manifest.get("plan"), dict) else {}

    family = str(plan.get("family") or decision.get("next_family") or experiment.get("family") or "")
    title = str(plan.get("title") or decision.get("next_title") or experiment.get("title") or "")
    root_cause = str(
        report.get("root_cause")
        or research.get("root_cause")
        or decision.get("root_cause")
        or run.get("root_cause")
        or run.get("error")
        or ""
    )
    hypothesis = str(plan.get("hypothesis") or decision.get("why") or "")
    metric_name = str(run.get("primary_metric_name") or "val_soundscape_macro_roc_auc")
    metric_value = run.get("primary_metric_value")
    focus = str(report.get("focus") or research.get("focus") or "")
    recent_rejects = [str(item) for item in research.get("reject", []) if str(item).strip()]
    adopt_now = [str(item) for item in research.get("adopt_now", []) if str(item).strip()]
    consider = [str(item) for item in research.get("consider", []) if str(item).strip()]
    query_terms = [family, title, root_cause, hypothesis, focus, *adopt_now[:3], *recent_rejects[:3], *consider[:2]]
    tokens = []
    seen: set[str] = set()
    for token in _tokenize(" ".join(query_terms)):
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= 24:
            break
    return {
        "stage": stage or str(manifest.get("stage") or ""),
        "run_id": str(run.get("run_id", "")),
        "family": family,
        "title": title,
        "root_cause": root_cause,
        "hypothesis": hypothesis,
        "objective_metric": metric_name,
        "current_metric_value": metric_value,
        "focus": focus,
        "next_action": str(decision.get("next_action") or ""),
        "adopt_now": adopt_now[:6],
        "consider": consider[:6],
        "reject": recent_rejects[:6],
        "query_terms": tokens,
    }


def _load_session_memory_from_path(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, dict):
        return {}
    try:
        return SessionMemory.from_dict(payload).to_dict()
    except TypeError:
        return payload


def load_session_memory(config: WorkspaceConfig) -> dict[str, Any]:
    ensure_directory(config.state_root())
    return _load_session_memory_from_path(config.session_memory_json_path())


def _load_session_memory_from_workspace(workspace_root: Path) -> dict[str, Any]:
    ensure_directory(_state_root_from_workspace(workspace_root))
    return _load_session_memory_from_path(_session_memory_json_from_workspace(workspace_root))


def _render_session_memory(session_memory: dict[str, Any]) -> str:
    if not isinstance(session_memory, dict) or not session_memory:
        return ""
    lines = [
        "# Session Memory",
        "",
        f"- current_objective: {session_memory.get('current_objective', '') or 'n/a'}",
        f"- current_leader: `{session_memory.get('current_leader_run_id', '') or 'n/a'}`",
    ]
    metric_name = str(session_memory.get("current_leader_metric_name", "") or "")
    metric_value = session_memory.get("current_leader_metric_value")
    if metric_name:
        lines.append(f"- leader_metric: `{metric_name}={metric_value}`")
    for title, key in (
        ("Active Portfolios", "active_portfolios"),
        ("Top Positive Priors", "top_positive_priors"),
        ("Top Negative Vetoes", "top_negative_vetoes"),
        ("Unresolved Questions", "unresolved_questions"),
        ("Current Bottlenecks", "current_bottlenecks"),
        ("Pending Decisions", "pending_decisions"),
    ):
        values = session_memory.get(key, [])
        if not isinstance(values, list) or not values:
            continue
        lines.extend(["", f"## {title}", *(f"- {str(item)}" for item in values)])
    return "\n".join(lines).rstrip() + "\n"


def _write_session_memory_to_workspace(workspace_root: Path, session_memory: dict[str, Any]) -> dict[str, Any]:
    normalized = SessionMemory.from_dict(
        {
            "current_objective": str(session_memory.get("current_objective", "") or ""),
            "current_leader_run_id": str(session_memory.get("current_leader_run_id", "") or ""),
            "current_leader_metric_name": str(session_memory.get("current_leader_metric_name", "") or ""),
            "current_leader_metric_value": session_memory.get("current_leader_metric_value"),
            "active_portfolios": [str(item) for item in session_memory.get("active_portfolios", [])],
            "top_positive_priors": [str(item) for item in session_memory.get("top_positive_priors", [])],
            "top_negative_vetoes": [str(item) for item in session_memory.get("top_negative_vetoes", [])],
            "unresolved_questions": [str(item) for item in session_memory.get("unresolved_questions", [])],
            "current_bottlenecks": [str(item) for item in session_memory.get("current_bottlenecks", [])],
            "pending_decisions": [str(item) for item in session_memory.get("pending_decisions", [])],
            "selected_memory_files": [str(item) for item in session_memory.get("selected_memory_files", [])],
            "selected_capability_packs": [str(item) for item in session_memory.get("selected_capability_packs", [])],
            "knowledge_card_ids": [str(item) for item in session_memory.get("knowledge_card_ids", [])],
            "source_stage": str(session_memory.get("source_stage", "") or ""),
            "run_id": str(session_memory.get("run_id", "") or ""),
            "updated_at": str(session_memory.get("updated_at", "") or now_utc_iso()),
        }
    ).to_dict()
    ensure_directory(_state_root_from_workspace(workspace_root))
    atomic_write_json(_session_memory_json_from_workspace(workspace_root), normalized)
    atomic_write_text(_session_memory_markdown_from_workspace(workspace_root), _render_session_memory(normalized))
    return normalized


def write_session_memory(config: WorkspaceConfig, session_memory: dict[str, Any]) -> dict[str, Any]:
    ensure_knowledge_layout(config)
    return _write_session_memory_to_workspace(config.root, session_memory)


def _leader_snapshot(state: WorkspaceState | None) -> tuple[str, str, float | None]:
    if state is None:
        return "", "", None
    leader = None
    for run in state.runs:
        if run.status != "succeeded" or run.primary_metric_value is None:
            continue
        if leader is None or run.primary_metric_value > leader.primary_metric_value:
            leader = run
    if leader is None:
        return "", "", None
    return leader.run_id, leader.primary_metric_name, leader.primary_metric_value


def _active_portfolios(state: WorkspaceState | None) -> list[str]:
    if state is None:
        return []
    counts: dict[str, int] = {}
    for item in state.work_items:
        portfolio_id = str(item.portfolio_id or "").strip()
        if not portfolio_id or item.status not in {"queued", "running"}:
            continue
        counts[portfolio_id] = counts.get(portfolio_id, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [f"{portfolio_id} ({count} items)" for portfolio_id, count in ranked[:5]]


def _selected_memory_files(
    workspace_root: Path,
    frame: dict[str, Any],
    cards: list[dict[str, Any]],
    *,
    limit: int = 4,
) -> list[dict[str, Any]]:
    memory_root = _knowledge_memory_root_from_workspace(workspace_root)
    if not memory_root.exists():
        return []
    selected_card_sources = {
        str(card.get("source_path", ""))
        for card in cards
        if isinstance(card, dict) and str(card.get("source_path", "")).startswith("memory/")
    }
    query_terms = set(str(item) for item in frame.get("query_terms", []) if str(item))
    scored: list[dict[str, Any]] = []
    for path in sorted(memory_root.rglob("*.md")):
        relative = path.relative_to(_knowledge_root_from_workspace(workspace_root)).as_posix()
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        summary = _section_summary(text) or path.stem.replace("_", " ")
        score = 0.0
        if relative in selected_card_sources:
            score += 9.0
        corpus = f"{relative}\n{text[:1200]}".lower()
        score += sum(2.0 for token in query_terms if token and token in corpus)
        focus = str(frame.get("focus", "") or "").lower()
        if focus and focus in corpus:
            score += 2.5
        family = str(frame.get("family", "") or "").lower()
        if family and family in corpus:
            score += 1.5
        if score <= 0.0:
            continue
        scored.append(
            {
                "path": relative,
                "title": path.stem.replace("_", " "),
                "summary": truncate(summary, limit=220),
                "score": round(score, 3),
            }
        )
    scored.sort(key=lambda item: (-float(item.get("score", 0.0)), str(item.get("path", ""))))
    return [{key: value for key, value in item.items() if key != "score"} for item in scored[:limit]]


def _select_capability_packs(
    *,
    frame: dict[str, Any],
    cards: list[dict[str, Any]],
    policy_cards: list[dict[str, Any]],
    branch_memories: list[dict[str, Any]],
    contradictions: list[dict[str, Any]],
    semantic_memory_files: list[dict[str, Any]],
    stage: str,
    limit: int = 4,
) -> list[dict[str, Any]]:
    query_text = " ".join(str(item) for item in frame.get("query_terms", []))
    components = {str(card.get("component", "")) for card in cards if isinstance(card, dict)}
    negative_components = {
        str(card.get("component", ""))
        for card in policy_cards
        if isinstance(card, dict) and str(card.get("policy_type", "")) in {"veto", "avoid"}
    }
    scored: list[dict[str, Any]] = []
    for pack in DEFAULT_CAPABILITY_PACKS:
        score = float(pack.priority_hint)
        reasons: list[str] = []
        for tag in pack.tags:
            if tag and tag in query_text:
                score += 1.6
                reasons.append(f"query:{tag}")
        if pack.pack_id == "negative_prior_veto" and (negative_components or contradictions):
            score += 5.0
            reasons.append("negative-priors")
        if pack.pack_id == "coverage_expansion_playbook" and "class_coverage" in components:
            score += 5.0
            reasons.append("coverage-bottleneck")
        if pack.pack_id == "pseudo_kd_playbook" and "pseudo_label" in components:
            score += 5.0
            reasons.append("pseudo-kd")
        if pack.pack_id == "v2s_recovery_constraints" and ("backbone" in components or "v2s" in query_text):
            score += 4.5
            reasons.append("backbone-recovery")
        if pack.pack_id == "submission_bar_checker" and (
            stage == "submission" or str(frame.get("next_action", "")) == "submit_candidate"
        ):
            score += 4.0
            reasons.append("submission-gate")
        if pack.pack_id == "config_diff_reader" and stage in {"plan", "codegen", "critic"}:
            score += 3.2
            reasons.append("diff-stage")
        if pack.pack_id == "run_ledger_miner" and branch_memories:
            score += 3.8
            reasons.append("branch-history")
        if pack.pack_id == "branch_diversifier" and stage in {"decision", "plan"}:
            score += 3.5
            reasons.append("portfolio-stage")
        if pack.pack_id == "low_information_plan_detector" and stage in {"decision", "plan"}:
            score += 3.5
            reasons.append("plan-filter")
        if semantic_memory_files and pack.pack_id in {"run_ledger_miner", "branch_diversifier"}:
            score += 1.2
            reasons.append("semantic-memory")
        if score <= 0.0:
            continue
        scored.append(
            {
                **pack.to_dict(),
                "score": round(score, 3),
                "selection_reason": ", ".join(reasons[:3]) or "stage-default",
            }
        )
    scored.sort(key=lambda item: (-float(item.get("score", 0.0)), str(item.get("pack_id", ""))))
    return [{key: value for key, value in item.items() if key != "score"} for item in scored[:limit]]


def _compose_session_memory(
    *,
    frame: dict[str, Any],
    cards: list[dict[str, Any]],
    policy_cards: list[dict[str, Any]],
    branch_memories: list[dict[str, Any]],
    contradictions: list[dict[str, Any]],
    semantic_memory_files: list[dict[str, Any]],
    capability_packs: list[dict[str, Any]],
    existing: dict[str, Any],
    state: WorkspaceState | None,
) -> dict[str, Any]:
    leader_run_id, leader_metric_name, leader_metric_value = _leader_snapshot(state)
    positive_priors = [
        f"{card.get('title', '')}: {card.get('summary', '')}"
        for card in cards
        if isinstance(card, dict) and str(card.get("stance", "")) == "positive"
    ]
    negative_vetoes = [
        f"{card.get('title', '')}: {card.get('summary', '')}"
        for card in policy_cards
        if isinstance(card, dict) and str(card.get("policy_type", "")) in {"veto", "avoid"}
    ]
    unresolved_questions = [
        f"Resolve contradiction: {item.get('summary', '')}"
        for item in contradictions
        if isinstance(item, dict)
    ]
    if not unresolved_questions:
        unresolved_questions.extend(
            f"Validate conditional lead: {card.get('summary', '')}"
            for card in cards
            if isinstance(card, dict) and str(card.get("stance", "")) == "conditional"
        )
    root_cause = str(frame.get("root_cause", "") or "").strip()
    if root_cause:
        unresolved_questions.append(f"What removes the current bottleneck: {root_cause}?")
    bottlenecks = [
        value
        for value in [root_cause, str(frame.get("focus", "") or "").strip()]
        if value
    ]
    pending_decisions = []
    next_action = str(frame.get("next_action", "") or "").strip()
    if next_action:
        pending_decisions.append(f"Execute next action: {next_action}")
    family = str(frame.get("family", "") or "").strip()
    if family:
        pending_decisions.append(f"Allocate the next branch portfolio for {family}")
    memory = SessionMemory(
        current_objective=str(frame.get("focus") or frame.get("hypothesis") or frame.get("title") or existing.get("current_objective", "")),
        current_leader_run_id=leader_run_id or str(existing.get("current_leader_run_id", "") or ""),
        current_leader_metric_name=leader_metric_name or str(existing.get("current_leader_metric_name", "") or ""),
        current_leader_metric_value=leader_metric_value if leader_metric_value is not None else existing.get("current_leader_metric_value"),
        active_portfolios=_dedupe_strings([*_active_portfolios(state), *[str(item) for item in existing.get("active_portfolios", [])]], limit=5),
        top_positive_priors=_dedupe_strings([*positive_priors, *[str(item) for item in existing.get("top_positive_priors", [])]], limit=4),
        top_negative_vetoes=_dedupe_strings([*negative_vetoes, *[str(item) for item in existing.get("top_negative_vetoes", [])]], limit=4),
        unresolved_questions=_dedupe_strings([*unresolved_questions, *[str(item) for item in existing.get("unresolved_questions", [])]], limit=5),
        current_bottlenecks=_dedupe_strings([*bottlenecks, *[str(item) for item in existing.get("current_bottlenecks", [])]], limit=4),
        pending_decisions=_dedupe_strings([*pending_decisions, *[str(item) for item in existing.get("pending_decisions", [])]], limit=4),
        selected_memory_files=_dedupe_strings([str(item.get("path", "")) for item in semantic_memory_files], limit=5),
        selected_capability_packs=_dedupe_strings([str(item.get("pack_id", "")) for item in capability_packs], limit=5),
        knowledge_card_ids=_dedupe_strings([str(card.get("card_id", "")) for card in cards], limit=8),
        source_stage=str(frame.get("stage", "") or existing.get("source_stage", "")),
        run_id=str(frame.get("run_id", "") or existing.get("run_id", "")),
        updated_at=now_utc_iso(),
    )
    return memory.to_dict()


def _score_card(card: dict[str, Any], frame: dict[str, Any], *, stage: str) -> float:
    score = 0.0
    query_terms = set(str(item) for item in frame.get("query_terms", []) if str(item))
    keywords = {str(item) for item in card.get("keywords", []) if str(item)}
    overlap = len(query_terms & keywords)
    score += overlap * 3.5

    component = str(card.get("component", ""))
    if component and component != "general":
        if component in " ".join(frame.get("query_terms", [])).lower():
            score += 4.0
        if component in str(frame.get("focus", "")).lower():
            score += 3.0

    source_path = str(card.get("source_path", ""))
    for filename, bonus in SOURCE_PRIORITY_HINTS.items():
        if source_path.endswith(filename):
            score += bonus

    stance = str(card.get("stance", "general"))
    if stage in {"plan", "decision"} and stance in {"positive", "negative", "conditional"}:
        score += 1.5
    if stage == "research" and stance in {"positive", "conditional"}:
        score += 1.0
    if str(card.get("policy_type", "")) in {"veto", "require"}:
        score += 2.0
    score += float(card.get("confidence", 0.0) or 0.0) * 3.0

    summary = f"{card.get('title', '')} {card.get('summary', '')}".lower()
    if "validation" in summary and "val" in str(frame.get("objective_metric", "")).lower():
        score += 2.0
    return score


def _select_diverse_cards(cards: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(cards) <= limit:
        return cards
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for stance in ("positive", "negative", "conditional"):
        for card in cards:
            if card.get("stance") != stance:
                continue
            card_id = str(card.get("card_id", ""))
            if card_id in seen_ids:
                continue
            selected.append(card)
            seen_ids.add(card_id)
            break
    for card in cards:
        if len(selected) >= limit:
            break
        card_id = str(card.get("card_id", ""))
        if card_id in seen_ids:
            continue
        selected.append(card)
        seen_ids.add(card_id)
    return selected[:limit]


def _compact_branch_memory(memory: BranchMemoryRecord) -> dict[str, Any]:
    return {
        "memory_id": memory.memory_id,
        "run_id": memory.run_id,
        "family": memory.family,
        "portfolio_id": memory.portfolio_id,
        "idea_class": memory.idea_class,
        "branch_role": memory.branch_role,
        "branch_rank": memory.branch_rank,
        "status": memory.status,
        "outcome": memory.outcome,
        "summary": memory.summary,
        "root_cause": memory.root_cause,
        "metric_name": memory.metric_name,
        "metric_value": memory.metric_value,
        "metric_delta": memory.metric_delta,
        "signal_score": memory.signal_score,
        "critic_status": memory.critic_status,
        "validation_status": memory.validation_status,
        "submission_status": memory.submission_status,
        "policy_tags": list(memory.policy_tags),
        "knowledge_card_ids": list(memory.knowledge_card_ids),
        "created_at": memory.created_at,
    }


def _score_branch_memory(memory: BranchMemoryRecord, frame: dict[str, Any], *, stage: str) -> float:
    score = abs(float(memory.signal_score)) * 1.5
    query_terms = set(str(item) for item in frame.get("query_terms", []) if str(item))
    text = " ".join(
        [
            memory.family,
            memory.idea_class,
            memory.branch_role,
            memory.summary,
            memory.root_cause,
            " ".join(memory.policy_tags),
        ]
    ).lower()
    memory_tokens = set(_tokenize(text))
    score += len(query_terms & memory_tokens) * 3.0
    if memory.family and memory.family == str(frame.get("family", "")):
        score += 6.0
    focus = str(frame.get("focus", "")).lower()
    if focus and focus in text:
        score += 3.0
    if stage in {"plan", "decision"} and memory.outcome in {"leader", "improved", "regressed", "critic_rejected"}:
        score += 2.0
    return score


def _recent_branch_memories(
    state: WorkspaceState,
    frame: dict[str, Any],
    *,
    stage: str,
    limit: int = 6,
) -> list[dict[str, Any]]:
    scored = sorted(
        (
            {
                **_compact_branch_memory(memory),
                "_score": _score_branch_memory(memory, frame, stage=stage),
            }
            for memory in state.branch_memories
        ),
        key=lambda item: (float(item.get("_score", 0.0)), str(item.get("created_at", "")), str(item.get("memory_id", ""))),
        reverse=True,
    )
    selected = [dict(item) for item in scored if float(item.get("_score", 0.0)) > 0.0][:limit]
    if not selected:
        selected = [dict(item) for item in scored[:limit]]
    for item in selected:
        item.pop("_score", None)
    return selected


def _policy_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for card in cards:
        policy_type = str(card.get("policy_type", "context"))
        if policy_type == "context":
            continue
        policies.append(
            {
                "card_id": str(card.get("card_id", "")),
                "title": str(card.get("title", "")),
                "summary": str(card.get("summary", "")),
                "source_path": str(card.get("source_path", "")),
                "stance": str(card.get("stance", "")),
                "component": str(card.get("component", "")),
                "policy_type": policy_type,
                "confidence": float(card.get("confidence", 0.0) or 0.0),
                "applies_to": [str(item) for item in card.get("applies_to", [])],
                "override_required": bool(card.get("override_required", False)),
            }
        )
    return policies


def _policy_contradictions(policy_cards: list[dict[str, Any]], branch_memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contradictions: list[dict[str, Any]] = []
    for card in policy_cards:
        component = str(card.get("component", ""))
        policy_type = str(card.get("policy_type", ""))
        if not component or component == "general":
            continue
        for memory in branch_memories:
            if str(memory.get("idea_class", "")) != component:
                continue
            outcome = str(memory.get("outcome", ""))
            if policy_type in {"veto", "avoid"} and outcome in {"leader", "improved", "submission_candidate"}:
                contradictions.append(
                    {
                        "card_id": str(card.get("card_id", "")),
                        "memory_id": str(memory.get("memory_id", "")),
                        "type": "negative-policy-overridden-by-result",
                        "summary": f"{card.get('title', '')} conflicts with successful branch memory {memory.get('run_id', '')}.",
                    }
                )
            elif policy_type in {"require", "prefer"} and outcome in {"regressed", "critic_rejected", "run_failed", "validate_failed"}:
                contradictions.append(
                    {
                        "card_id": str(card.get("card_id", "")),
                        "memory_id": str(memory.get("memory_id", "")),
                        "type": "positive-policy-underperformed-recently",
                        "summary": f"{card.get('title', '')} conflicts with recent weak outcome {memory.get('run_id', '')}.",
                    }
                )
    return contradictions[:8]


def retrieve_knowledge_bundle_from_root(
    workspace_root: Path,
    manifest: dict[str, Any],
    *,
    stage: str = "",
    limit: int = 8,
) -> dict[str, Any]:
    cards = compile_knowledge_index_from_root(workspace_root)
    frame = build_problem_frame(manifest, stage=stage)
    existing_session_memory = _load_session_memory_from_workspace(workspace_root)
    if not cards:
        empty_bundle = {
            "problem_frame": frame,
            "cards": [],
            "knowledge_files_seen": 0,
            "knowledge_card_ids": [],
            "policy_cards": [],
            "branch_memories": [],
            "branch_memory_ids": [],
            "contradictions": [],
            "semantic_memory_files": [],
            "capability_packs": [],
        }
        session_memory = _compose_session_memory(
            frame=frame,
            cards=[],
            policy_cards=[],
            branch_memories=[],
            contradictions=[],
            semantic_memory_files=[],
            capability_packs=[],
            existing=existing_session_memory,
            state=None,
        )
        empty_bundle["session_memory"] = session_memory
        return empty_bundle
    scored = sorted(
        (
            {
                **card,
                "score": _score_card(card, frame, stage=stage or frame.get("stage", "")),
            }
            for card in cards
        ),
        key=lambda item: (float(item.get("score", 0.0)), str(item.get("source_path", "")), str(item.get("card_id", ""))),
        reverse=True,
    )
    selected = _select_diverse_cards([item for item in scored if float(item.get("score", 0.0)) > 0.0], limit)
    if not selected:
        selected = scored[:limit]
    compact_cards = [item for item in selected if isinstance(item, dict)]
    policy_cards = _policy_cards(compact_cards)
    semantic_memory_files = _selected_memory_files(workspace_root, frame, compact_cards)
    capability_packs = _select_capability_packs(
        frame=frame,
        cards=compact_cards,
        policy_cards=policy_cards,
        branch_memories=[],
        contradictions=[],
        semantic_memory_files=semantic_memory_files,
        stage=stage or str(frame.get("stage", "")),
    )
    session_memory = _compose_session_memory(
        frame=frame,
        cards=compact_cards,
        policy_cards=policy_cards,
        branch_memories=[],
        contradictions=[],
        semantic_memory_files=semantic_memory_files,
        capability_packs=capability_packs,
        existing=existing_session_memory,
        state=None,
    )
    return {
        "problem_frame": frame,
        "cards": selected,
        "knowledge_files_seen": len({str(item.get("source_path", "")) for item in selected}),
        "knowledge_card_ids": [str(item.get("card_id", "")) for item in selected],
        "policy_cards": policy_cards,
        "branch_memories": [],
        "branch_memory_ids": [],
        "contradictions": [],
        "semantic_memory_files": semantic_memory_files,
        "capability_packs": capability_packs,
        "session_memory": session_memory,
    }


def retrieve_knowledge_bundle(
    config: WorkspaceConfig,
    manifest: dict[str, Any],
    *,
    stage: str = "",
    limit: int = 8,
    state: WorkspaceState | None = None,
    memory_limit: int = 6,
) -> dict[str, Any]:
    ensure_knowledge_layout(config)
    bundle = retrieve_knowledge_bundle_from_root(config.root, manifest, stage=stage, limit=limit)
    cards = bundle.get("cards", [])
    compact_cards = [item for item in cards if isinstance(item, dict)] if isinstance(cards, list) else []
    policy_cards = _policy_cards(compact_cards)
    bundle["policy_cards"] = policy_cards
    bundle["branch_memories"] = []
    bundle["branch_memory_ids"] = []
    bundle["contradictions"] = []
    if state is not None:
        frame = bundle.get("problem_frame", {}) if isinstance(bundle.get("problem_frame"), dict) else {}
        branch_memories = _recent_branch_memories(state, frame, stage=stage or str(frame.get("stage", "")), limit=memory_limit)
        bundle["branch_memories"] = branch_memories
        bundle["branch_memory_ids"] = [str(item.get("memory_id", "")) for item in branch_memories]
        bundle["contradictions"] = _policy_contradictions(policy_cards, branch_memories)
    semantic_memory_files = _selected_memory_files(config.root, bundle.get("problem_frame", {}), compact_cards)
    capability_packs = _select_capability_packs(
        frame=bundle.get("problem_frame", {}),
        cards=compact_cards,
        policy_cards=policy_cards,
        branch_memories=[item for item in bundle.get("branch_memories", []) if isinstance(item, dict)],
        contradictions=[item for item in bundle.get("contradictions", []) if isinstance(item, dict)],
        semantic_memory_files=semantic_memory_files,
        stage=stage or str(bundle.get("problem_frame", {}).get("stage", "")),
    )
    bundle["semantic_memory_files"] = semantic_memory_files
    bundle["capability_packs"] = capability_packs
    session_memory = _compose_session_memory(
        frame=bundle.get("problem_frame", {}),
        cards=compact_cards,
        policy_cards=policy_cards,
        branch_memories=[item for item in bundle.get("branch_memories", []) if isinstance(item, dict)],
        contradictions=[item for item in bundle.get("contradictions", []) if isinstance(item, dict)],
        semantic_memory_files=semantic_memory_files,
        capability_packs=capability_packs,
        existing=load_session_memory(config),
        state=state,
    )
    bundle["session_memory"] = write_session_memory(config, session_memory)
    return bundle


def render_retrieved_knowledge(bundle: dict[str, Any], *, limit: int = 16000) -> str:
    cards = bundle.get("cards", [])
    frame = bundle.get("problem_frame", {}) if isinstance(bundle.get("problem_frame"), dict) else {}
    sections: list[str] = []
    session_memory = bundle.get("session_memory", {})
    rendered_session_memory = _render_session_memory(session_memory if isinstance(session_memory, dict) else {})
    if rendered_session_memory.strip():
        sections.append(rendered_session_memory.strip())
    if frame:
        sections.append(
            "\n".join(
                [
                    "## problem_frame",
                    f"- family: {frame.get('family', '') or 'n/a'}",
                    f"- focus: {frame.get('focus', '') or 'n/a'}",
                    f"- root_cause: {frame.get('root_cause', '') or 'n/a'}",
                    f"- objective_metric: {frame.get('objective_metric', '') or 'n/a'}",
                ]
            )
        )
    if isinstance(cards, list) and cards:
        grouped: dict[str, list[dict[str, Any]]] = {
            "positive": [],
            "negative": [],
            "conditional": [],
            "general": [],
        }
        for card in cards:
            grouped.setdefault(str(card.get("stance", "general")), []).append(card)
        for label, key in (
            ("Positive Priors", "positive"),
            ("Negative Vetoes", "negative"),
            ("Conditional Leads", "conditional"),
            ("General Context", "general"),
        ):
            bucket = grouped.get(key, [])
            if not bucket:
                continue
            lines = [f"## {label}"]
            for card in bucket:
                lines.extend(
                    [
                        f"### {card.get('title', '')}",
                        f"- card_id: `{card.get('card_id', '')}`",
                        f"- source: `knowledge/{card.get('source_path', '')}`",
                        f"- component: `{card.get('component', 'general')}`",
                        f"- policy: `{card.get('policy_type', 'context')}` @ {float(card.get('confidence', 0.0) or 0.0):.2f}",
                        f"- summary: {card.get('summary', '')}",
                    ]
                )
            sections.append("\n".join(lines))
    policy_cards = bundle.get("policy_cards", [])
    if isinstance(policy_cards, list) and policy_cards:
        lines = ["## Policy Cards"]
        for card in policy_cards:
            if not isinstance(card, dict):
                continue
            lines.append(
                f"- `{card.get('policy_type', 'context')}` | `{card.get('component', 'general')}` | `{card.get('card_id', '')}` | {card.get('summary', '')}"
            )
        sections.append("\n".join(lines))
    semantic_memory_files = bundle.get("semantic_memory_files", [])
    if isinstance(semantic_memory_files, list) and semantic_memory_files:
        lines = ["## Semantic Memory Files"]
        for item in semantic_memory_files:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- `knowledge/{item.get('path', '')}` | {item.get('summary', '')}"
            )
        sections.append("\n".join(lines))
    capability_packs = bundle.get("capability_packs", [])
    if isinstance(capability_packs, list) and capability_packs:
        lines = ["## Capability Packs"]
        for pack in capability_packs:
            if not isinstance(pack, dict):
                continue
            lines.append(
                f"- `{pack.get('pack_id', '')}` | when={pack.get('when_to_use', '')} | returns={pack.get('returns', '')} | reason={pack.get('selection_reason', 'n/a')}"
            )
        sections.append("\n".join(lines))
    branch_memories = bundle.get("branch_memories", [])
    if isinstance(branch_memories, list) and branch_memories:
        lines = ["## Recent Branch Memories"]
        for memory in branch_memories:
            if not isinstance(memory, dict):
                continue
            lines.append(
                f"- `{memory.get('run_id', '')}` | outcome={memory.get('outcome', '')} | idea={memory.get('idea_class', '')} | {memory.get('summary', '')}"
            )
        sections.append("\n".join(lines))
    contradictions = bundle.get("contradictions", [])
    if isinstance(contradictions, list) and contradictions:
        lines = ["## Contradictions"]
        for item in contradictions:
            if not isinstance(item, dict):
                continue
            lines.append(f"- `{item.get('type', '')}` | {item.get('summary', '')}")
        sections.append("\n".join(lines))
    rendered = "\n\n".join(section for section in sections if section.strip())
    return truncate(rendered, limit=limit) if len(rendered) > limit else rendered


def compact_knowledge_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    cards = bundle.get("cards", [])
    compact_cards: list[dict[str, Any]] = []
    if isinstance(cards, list):
        for card in cards:
            compact_cards.append(
                {
                    "card_id": str(card.get("card_id", "")),
                    "title": str(card.get("title", "")),
                    "summary": str(card.get("summary", "")),
                    "source_path": str(card.get("source_path", "")),
                    "stance": str(card.get("stance", "")),
                    "component": str(card.get("component", "")),
                    "policy_type": str(card.get("policy_type", "context")),
                    "confidence": float(card.get("confidence", 0.0) or 0.0),
                }
            )
    return {
        "problem_frame": bundle.get("problem_frame", {}),
        "knowledge_files_seen": int(bundle.get("knowledge_files_seen", 0) or 0),
        "knowledge_card_ids": [str(item) for item in bundle.get("knowledge_card_ids", [])],
        "branch_memory_ids": [str(item) for item in bundle.get("branch_memory_ids", [])],
        "cards": compact_cards,
        "policy_cards": [item for item in bundle.get("policy_cards", []) if isinstance(item, dict)],
        "branch_memories": [item for item in bundle.get("branch_memories", []) if isinstance(item, dict)],
        "contradictions": [item for item in bundle.get("contradictions", []) if isinstance(item, dict)],
        "semantic_memory_files": [item for item in bundle.get("semantic_memory_files", []) if isinstance(item, dict)],
        "capability_packs": [item for item in bundle.get("capability_packs", []) if isinstance(item, dict)],
        "session_memory": bundle.get("session_memory", {}) if isinstance(bundle.get("session_memory"), dict) else {},
    }


def _normalize_memory_ops(value: Any, *, stage: str, run_id: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    ops: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        target = str(item.get("target", "")).strip()
        summary = str(item.get("summary", "")).strip()
        op_name = str(item.get("op", "")).strip()
        if not target or not summary or not op_name:
            continue
        ops.append(
            MemoryOp(
                op=op_name,
                target=target,
                summary=summary,
                memory_kind=str(item.get("memory_kind", "issues") or "issues"),
                details=str(item.get("details", "") or ""),
                reason=str(item.get("reason", "") or ""),
                source_stage=str(item.get("source_stage", "") or stage),
                run_id=str(item.get("run_id", "") or run_id),
                evidence_ids=[str(entry) for entry in item.get("evidence_ids", [])],
                metadata=dict(item.get("metadata", {})) if isinstance(item.get("metadata"), dict) else {},
            ).to_dict()
        )
    return ops


def _memory_op_path(config: WorkspaceConfig, memory_op: dict[str, Any]) -> Path:
    kind = str(memory_op.get("memory_kind", "issues") or "issues")
    if kind not in SEMANTIC_MEMORY_SUBDIRS:
        kind = "issues"
    target_slug = slugify(str(memory_op.get("target", "memory-note")) or "memory-note")
    return config.knowledge_memory_root() / kind / f"{target_slug}.md"


def _existing_update_log(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    marker = "## Update Log"
    if marker not in text:
        return []
    log_text = text.split(marker, maxsplit=1)[1]
    lines = [line.rstrip() for line in log_text.splitlines() if line.strip()]
    return [line for line in lines if line.lstrip().startswith("- ")]


def apply_memory_ops(config: WorkspaceConfig, memory_ops: list[dict[str, Any]]) -> list[str]:
    ensure_knowledge_layout(config)
    touched_paths: list[str] = []
    for memory_op in memory_ops:
        path = _memory_op_path(config, memory_op)
        ensure_directory(path.parent)
        history = _existing_update_log(path)
        timestamp = now_utc_iso()
        evidence_ids = [str(item) for item in memory_op.get("evidence_ids", []) if str(item)]
        planner_effect = str(memory_op.get("metadata", {}).get("planner_effect", "") or memory_op.get("reason", "") or "")
        update_line = f"- [{timestamp}] `{memory_op.get('op', '')}` | {memory_op.get('summary', '')}"
        if evidence_ids:
            update_line += f" | evidence={', '.join(evidence_ids)}"
        history.append(update_line)
        history = history[-12:]
        support_lines = [f"- `{item}`" for item in evidence_ids] if evidence_ids else ["- none"]
        title = str(memory_op.get("target", "")).replace("_", " ").strip().title() or "Memory Note"
        lines = [
            f"# {title}",
            "",
            f"- kind: `{memory_op.get('memory_kind', 'issues')}`",
            f"- target: `{memory_op.get('target', '')}`",
            f"- last_op: `{memory_op.get('op', '')}`",
            f"- updated_at: `{timestamp}`",
            "",
            "## Current Conclusion",
            str(memory_op.get("summary", "") or "No conclusion recorded."),
            "",
            "## Planner Effect",
            planner_effect or "Use this memory as a planning constraint when related branches are generated.",
            "",
            "## Supporting Evidence",
            *support_lines,
            "",
            "## Update Log",
            *history,
            "",
        ]
        atomic_write_text(path, "\n".join(lines))
        touched_paths.append(path.relative_to(config.knowledge_root()).as_posix())
    if touched_paths:
        compile_knowledge_index(config)
    return touched_paths


def apply_knowledge_stage_outputs(
    config: WorkspaceConfig,
    *,
    run_id: str,
    stage: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    ensure_knowledge_layout(config)
    normalized_ops = _normalize_memory_ops(payload.get("memory_ops", []), stage=stage, run_id=run_id)
    touched_paths = apply_memory_ops(config, normalized_ops) if normalized_ops else []
    session_memory = payload.get("session_memory")
    if isinstance(session_memory, dict) and session_memory:
        session_memory = write_session_memory(config, session_memory)
    else:
        session_memory = load_session_memory(config)
    return {
        "memory_ops": normalized_ops,
        "touched_memory_files": touched_paths,
        "session_memory": session_memory,
    }


def _empty_workspace_state() -> WorkspaceState:
    return WorkspaceState(
        work_items=[],
        experiments=[],
        runs=[],
        stage_runs=[],
        agent_runs=[],
        specs=[],
        validations=[],
        metrics=[],
        findings=[],
        issues=[],
        research_notes=[],
        submissions=[],
        submission_results=[],
        runtime=RuntimeState(initialized_at=now_utc_iso()),
    )


def _policy_rule_payload(rule: Any) -> dict[str, Any]:
    return {
        "rule_id": str(rule.rule_id),
        "card_id": str(rule.rule_id),
        "title": str(rule.component).replace("_", " ").title(),
        "summary": str(rule.summary),
        "rationale": str(rule.rationale),
        "family": str(rule.family),
        "component": str(rule.component),
        "policy_type": str(rule.policy_type),
        "confidence": float(rule.confidence),
        "override_required": bool(rule.override_required),
        "claim_ids": [str(item) for item in rule.claim_ids],
        "contradiction_ids": [str(item) for item in rule.contradiction_ids],
        "scope_tags": [str(item) for item in rule.scope_tags],
    }


def _constraint_payload(constraint: Any) -> dict[str, Any]:
    return {
        "constraint_id": str(constraint.constraint_id),
        "scope": str(constraint.scope),
        "constraint_type": str(constraint.constraint_type),
        "summary": str(constraint.summary),
        "value": dict(constraint.value),
        "status": str(constraint.status),
    }


def _claim_payload(claim: Any) -> dict[str, Any]:
    return {
        "claim_id": str(claim.claim_id),
        "family": str(claim.family),
        "component": str(claim.component),
        "stance": str(claim.stance),
        "summary": str(claim.summary),
        "title": str(claim.title),
        "source_type": str(claim.source_type),
        "source_ref": str(claim.source_ref),
        "confidence": float(claim.confidence),
        "claim_kind": str(claim.claim_kind),
        "claim_status": str(claim.claim_status),
        "override_required": bool(claim.override_required),
        "support_ids": [str(item) for item in claim.support_ids],
        "contradict_ids": [str(item) for item in claim.contradict_ids],
    }


def _relevant_claims(state: WorkspaceState, frame: dict[str, Any], *, limit: int = 10) -> list[dict[str, Any]]:
    family = str(frame.get("family", "") or "")
    query_terms = set(str(item) for item in frame.get("query_terms", []) if str(item))
    scored: list[tuple[float, Any]] = []
    for claim in state.claims:
        if claim.claim_status in {"retired", "superseded"}:
            continue
        if family and claim.family and claim.family != family:
            continue
        score = float(claim.confidence)
        corpus = " ".join(
            [
                claim.family,
                claim.component,
                claim.title,
                claim.summary,
                claim.source_type,
            ]
        ).lower()
        score += sum(1.6 for token in query_terms if token and token in corpus)
        if claim.source_type == "branch_outcome":
            score += 1.2
        scored.append((score, claim))
    scored.sort(key=lambda item: (-item[0], item[1].component, item[1].claim_id))
    return [_claim_payload(item) for _, item in scored[:limit]]


def _policy_contradictions_from_state(state: WorkspaceState, family: str) -> list[dict[str, Any]]:
    claims_by_id = {claim.claim_id: claim for claim in state.claims}
    contradictions: list[dict[str, Any]] = []
    for rule in active_policy_rules(state, family=family):
        if not rule.contradiction_ids:
            continue
        rule_claims = [claims_by_id[claim_id] for claim_id in rule.claim_ids if claim_id in claims_by_id]
        contradiction_claims = [claims_by_id[claim_id] for claim_id in rule.contradiction_ids if claim_id in claims_by_id]
        if not contradiction_claims:
            continue
        contradiction_type = f"{rule.policy_type}_rule_has_counterevidence"
        if (
            any(item.stance == "negative" for item in rule_claims)
            and any(item.source_type == "branch_outcome" and item.stance == "positive" for item in contradiction_claims)
        ):
            contradiction_type = "negative-policy-overridden-by-result"
        elif (
            any(item.stance == "positive" for item in rule_claims)
            and any(item.source_type == "branch_outcome" and item.stance == "negative" for item in contradiction_claims)
        ):
            contradiction_type = "positive-policy-overridden-by-result"
        contradictions.append(
            {
                "rule_id": str(rule.rule_id),
                "type": contradiction_type,
                "component": str(rule.component),
                "summary": f"{rule.summary} Counterevidence: {'; '.join(truncate(item.summary, limit=120) for item in contradiction_claims[:2])}",
                "contradiction_ids": [str(item.claim_id) for item in contradiction_claims],
            }
        )
    if family:
        global_negative_by_component = {
            claim.component: claim
            for claim in state.claims
            if not claim.family and claim.component and claim.stance == "negative"
        }
        global_positive_by_component = {
            claim.component: claim
            for claim in state.claims
            if not claim.family and claim.component and claim.stance == "positive"
        }
        for claim in state.claims:
            if claim.family != family or claim.source_type != "branch_outcome" or not claim.component:
                continue
            if claim.stance == "positive" and claim.component in global_negative_by_component:
                negative_claim = global_negative_by_component[claim.component]
                contradictions.append(
                    {
                        "rule_id": "",
                        "type": "negative-policy-overridden-by-result",
                        "component": str(claim.component),
                        "summary": f"Global negative prior for {claim.component} was challenged by empirical branch result: {truncate(claim.summary, limit=120)}",
                        "contradiction_ids": [str(negative_claim.claim_id), str(claim.claim_id)],
                    }
                )
            if claim.stance == "negative" and claim.component in global_positive_by_component:
                positive_claim = global_positive_by_component[claim.component]
                contradictions.append(
                    {
                        "rule_id": "",
                        "type": "positive-policy-overridden-by-result",
                        "component": str(claim.component),
                        "summary": f"Global positive prior for {claim.component} was challenged by empirical branch result: {truncate(claim.summary, limit=120)}",
                        "contradiction_ids": [str(positive_claim.claim_id), str(claim.claim_id)],
                    }
                )
    return contradictions[:8]


def _policy_rules_for_bundle(state: WorkspaceState, family: str) -> list[dict[str, Any]]:
    return [_policy_rule_payload(rule) for rule in active_policy_rules(state, family=family)]


def _select_capability_packs(
    *,
    frame: dict[str, Any],
    policy_rules: list[dict[str, Any]],
    branch_memories: list[dict[str, Any]],
    contradictions: list[dict[str, Any]],
    stage: str,
    limit: int = 4,
) -> list[dict[str, Any]]:
    query_terms = " ".join(str(item) for item in frame.get("query_terms", [])).lower()
    stage_name = stage or str(frame.get("stage", ""))
    scored: list[dict[str, Any]] = []
    for pack in DEFAULT_CAPABILITY_PACKS:
        if stage_name and pack.applies_to and stage_name not in pack.applies_to:
            continue
        score = float(pack.priority_hint)
        reasons: list[str] = []
        if pack.pack_id == "ledger_miner" and branch_memories:
            score += 4.0
            reasons.append("branch-history")
        if pack.pack_id == "veto_checker" and any(item.get("policy_type") in {"veto", "avoid", "conditional"} for item in policy_rules):
            score += 4.2
            reasons.append("policy-rules")
        if pack.pack_id == "branch_diversifier" and stage_name in {"decision", "plan"}:
            score += 3.8
            reasons.append("portfolio-stage")
        if pack.pack_id == "submission_bar_checker" and (
            stage_name == "submission" or str(frame.get("next_action", "")) == "submit_candidate"
        ):
            score += 4.0
            reasons.append("submission-gate")
        if pack.pack_id == "novel_hypothesis_generator" and (
            contradictions or "explore" in query_terms or len(branch_memories) >= 2
        ):
            score += 3.4
            reasons.append("novel-lane")
        if score <= 0:
            continue
        scored.append(
            {
                **pack.to_dict(),
                "selection_reason": ", ".join(reasons[:3]) or "stage-default",
                "_score": round(score, 3),
            }
        )
    scored.sort(key=lambda item: (-float(item.get("_score", 0.0)), str(item.get("pack_id", ""))))
    return [{key: value for key, value in item.items() if key != "_score"} for item in scored[:limit]]


def _invoke_capability_packs(
    state: WorkspaceState | None,
    *,
    run_id: str,
    stage_name: str,
    frame: dict[str, Any],
    selected_packs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if state is None:
        return {}
    results: dict[str, dict[str, Any]] = {}
    for pack in selected_packs:
        pack_id = str(pack.get("pack_id", "")).strip()
        if not pack_id:
            continue
        results[pack_id] = invoke_capability_pack(
            state,
            run_id=run_id,
            stage_name=stage_name,
            pack_id=pack_id,
            input_summary=f"{frame.get('family', '')}::{frame.get('focus', '') or frame.get('root_cause', '')}",
            frame=frame,
            capability_results=results,
        )
    return results


def _compose_session_memory(
    *,
    frame: dict[str, Any],
    cards: list[dict[str, Any]],
    policy_rules: list[dict[str, Any]],
    branch_memories: list[dict[str, Any]],
    contradictions: list[dict[str, Any]],
    semantic_memory_files: list[dict[str, Any]],
    capability_packs: list[dict[str, Any]],
    state: WorkspaceState | None,
) -> dict[str, Any]:
    leader_run_id, leader_metric_name, leader_metric_value = _leader_snapshot(state)
    positive_priors = [
        f"{item.get('component', 'general')}: {item.get('summary', '')}"
        for item in policy_rules
        if str(item.get("policy_type", "")) in {"require", "prefer"}
    ] or [
        f"{item.get('component', 'general')}: {item.get('summary', '')}"
        for item in cards
        if str(item.get("stance", "")) == "positive"
    ]
    negative_vetoes = [
        f"{item.get('component', 'general')}: {item.get('summary', '')}"
        for item in policy_rules
        if str(item.get("policy_type", "")) in {"veto", "avoid"}
    ]
    unresolved_questions = [
        f"Resolve contradiction on {item.get('component', 'general')}: {item.get('summary', '')}"
        for item in contradictions
    ]
    root_cause = str(frame.get("root_cause", "") or "").strip()
    if root_cause:
        unresolved_questions.append(f"What removes the current bottleneck: {root_cause}?")
    pending_decisions: list[str] = []
    stage_name = str(frame.get("stage", "") or "")
    family = str(frame.get("family", "") or "")
    if family:
        pending_decisions.append(f"Choose the next {family} branch portfolio.")
    if stage_name in {"decision", "plan"}:
        pending_decisions.append("Compile grounded and novel branch slots without violating veto rules.")
    memory = SessionMemory(
        current_objective=str(frame.get("focus") or frame.get("hypothesis") or frame.get("title") or "Advance the active research frontier."),
        current_leader_run_id=leader_run_id,
        current_leader_metric_name=leader_metric_name,
        current_leader_metric_value=leader_metric_value,
        active_portfolios=_active_portfolios(state),
        top_positive_priors=_dedupe_strings(positive_priors, limit=4),
        top_negative_vetoes=_dedupe_strings(negative_vetoes, limit=4),
        unresolved_questions=_dedupe_strings(unresolved_questions, limit=5),
        current_bottlenecks=_dedupe_strings(
            [item for item in [root_cause, str(frame.get("focus", "") or "")] if item],
            limit=4,
        ),
        pending_decisions=_dedupe_strings(pending_decisions, limit=4),
        selected_memory_files=_dedupe_strings([str(item.get("path", "")) for item in semantic_memory_files], limit=5),
        selected_capability_packs=_dedupe_strings([str(item.get("pack_id", "")) for item in capability_packs], limit=5),
        knowledge_card_ids=_dedupe_strings([str(item.get("card_id", "")) for item in cards], limit=8),
        source_stage=stage_name,
        run_id=str(frame.get("run_id", "") or ""),
        updated_at=now_utc_iso(),
    )
    return memory.to_dict()


def retrieve_knowledge_bundle_from_root(
    workspace_root: Path,
    manifest: dict[str, Any],
    *,
    stage: str = "",
    limit: int = 8,
) -> dict[str, Any]:
    cards = compile_knowledge_index_from_root(workspace_root)
    frame = build_problem_frame(manifest, stage=stage)
    if not cards:
        empty_bundle = {
            "problem_frame": frame,
            "cards": [],
            "knowledge_files_seen": 0,
            "knowledge_card_ids": [],
            "policy_rules": [],
            "policy_cards": [],
            "claims": [],
            "branch_memories": [],
            "branch_memory_ids": [],
            "contradictions": [],
            "constraints": [],
            "semantic_memory_files": [],
            "capability_packs": [],
            "capability_results": {},
        }
        empty_bundle["session_memory"] = _compose_session_memory(
            frame=frame,
            cards=[],
            policy_rules=[],
            branch_memories=[],
            contradictions=[],
            semantic_memory_files=[],
            capability_packs=[],
            state=None,
        )
        return empty_bundle
    scored = sorted(
        (
            {
                **card,
                "score": _score_card(card, frame, stage=stage or frame.get("stage", "")),
            }
            for card in cards
        ),
        key=lambda item: (float(item.get("score", 0.0)), str(item.get("source_path", "")), str(item.get("card_id", ""))),
        reverse=True,
    )
    selected = _select_diverse_cards([item for item in scored if float(item.get("score", 0.0)) > 0.0], limit) or scored[:limit]
    compact_cards = [dict(item) for item in selected if isinstance(item, dict)]
    temp_state = _empty_workspace_state()
    synchronize_policy_state(temp_state, compact_cards)
    family = str(frame.get("family", "") or "")
    policy_rules = _policy_rules_for_bundle(temp_state, family)
    semantic_memory_files = _selected_memory_files(workspace_root, frame, compact_cards)
    capability_packs = _select_capability_packs(
        frame=frame,
        policy_rules=policy_rules,
        branch_memories=[],
        contradictions=[],
        stage=stage or str(frame.get("stage", "")),
    )
    session_memory = _compose_session_memory(
        frame=frame,
        cards=compact_cards,
        policy_rules=policy_rules,
        branch_memories=[],
        contradictions=[],
        semantic_memory_files=semantic_memory_files,
        capability_packs=capability_packs,
        state=None,
    )
    return {
        "problem_frame": frame,
        "cards": compact_cards,
        "knowledge_files_seen": len({str(item.get("source_path", "")) for item in compact_cards}),
        "knowledge_card_ids": [str(item.get("card_id", "")) for item in compact_cards],
        "policy_rules": policy_rules,
        "policy_cards": policy_rules,
        "claims": _relevant_claims(temp_state, frame),
        "branch_memories": [],
        "branch_memory_ids": [],
        "contradictions": [],
        "constraints": [],
        "semantic_memory_files": semantic_memory_files,
        "capability_packs": capability_packs,
        "capability_results": {},
        "session_memory": session_memory,
    }


def retrieve_knowledge_bundle(
    config: WorkspaceConfig,
    manifest: dict[str, Any],
    *,
    stage: str = "",
    limit: int = 8,
    state: WorkspaceState | None = None,
    memory_limit: int = 6,
) -> dict[str, Any]:
    ensure_knowledge_layout(config)
    bundle = retrieve_knowledge_bundle_from_root(config.root, manifest, stage=stage, limit=limit)
    frame = bundle.get("problem_frame", {}) if isinstance(bundle.get("problem_frame"), dict) else {}
    family = str(frame.get("family", "") or "")
    cards = [item for item in bundle.get("cards", []) if isinstance(item, dict)]
    if state is None:
        bundle["session_memory"] = write_session_memory(config, bundle.get("session_memory", {}))
        return bundle
    synchronize_policy_state(state, cards)
    policy_rules = _policy_rules_for_bundle(state, family)
    branch_memories = _recent_branch_memories(state, frame, stage=stage or str(frame.get("stage", "")), limit=memory_limit)
    contradictions = _policy_contradictions_from_state(state, family)
    constraints = [_constraint_payload(item) for item in active_constraints(state, family=family, run_id=str(frame.get("run_id", "") or ""))]
    semantic_memory_files = _selected_memory_files(config.root, frame, cards)
    capability_packs = _select_capability_packs(
        frame=frame,
        policy_rules=policy_rules,
        branch_memories=branch_memories,
        contradictions=contradictions,
        stage=stage or str(frame.get("stage", "")),
    )
    capability_results = _invoke_capability_packs(
        state,
        run_id=str(frame.get("run_id", "") or ""),
        stage_name=stage or str(frame.get("stage", "")),
        frame=frame,
        selected_packs=capability_packs,
    )
    session_memory = _compose_session_memory(
        frame=frame,
        cards=cards,
        policy_rules=policy_rules,
        branch_memories=branch_memories,
        contradictions=contradictions,
        semantic_memory_files=semantic_memory_files,
        capability_packs=capability_packs,
        state=state,
    )
    bundle.update(
        {
            "policy_rules": policy_rules,
            "policy_cards": policy_rules,
            "claims": _relevant_claims(state, frame),
            "branch_memories": branch_memories,
            "branch_memory_ids": [str(item.get("memory_id", "")) for item in branch_memories],
            "contradictions": contradictions,
            "constraints": constraints,
            "semantic_memory_files": semantic_memory_files,
            "capability_packs": capability_packs,
            "capability_results": capability_results,
            "session_memory": write_session_memory(config, session_memory),
        }
    )
    return bundle


def render_retrieved_knowledge(bundle: dict[str, Any], *, limit: int = 16000) -> str:
    cards = [item for item in bundle.get("cards", []) if isinstance(item, dict)]
    frame = bundle.get("problem_frame", {}) if isinstance(bundle.get("problem_frame"), dict) else {}
    sections: list[str] = []
    rendered_session_memory = _render_session_memory(bundle.get("session_memory", {}) if isinstance(bundle.get("session_memory"), dict) else {})
    if rendered_session_memory.strip():
        sections.append(rendered_session_memory.strip())
    if frame:
        sections.append(
            "\n".join(
                [
                    "## Problem Frame",
                    f"- family: {frame.get('family', '') or 'n/a'}",
                    f"- focus: {frame.get('focus', '') or 'n/a'}",
                    f"- root_cause: {frame.get('root_cause', '') or 'n/a'}",
                    f"- objective_metric: {frame.get('objective_metric', '') or 'n/a'}",
                ]
            )
        )
    if cards:
        grouped: dict[str, list[dict[str, Any]]] = {"positive": [], "negative": [], "conditional": [], "general": []}
        for card in cards:
            grouped.setdefault(str(card.get("stance", "general")), []).append(card)
        for label, key in (
            ("Positive Priors", "positive"),
            ("Negative Seeds", "negative"),
            ("Conditional Leads", "conditional"),
            ("General Context", "general"),
        ):
            bucket = grouped.get(key, [])
            if not bucket:
                continue
            lines = [f"## {label}"]
            for card in bucket:
                lines.append(
                    f"- `{card.get('card_id', '')}` | `{card.get('component', 'general')}` | {card.get('summary', '')}"
                )
            sections.append("\n".join(lines))
    policy_rules = bundle.get("policy_rules", [])
    if isinstance(policy_rules, list) and policy_rules:
        lines = ["## Policy Rules"]
        for rule in policy_rules:
            if not isinstance(rule, dict):
                continue
            lines.append(
                f"- `{rule.get('policy_type', 'context')}` | `{rule.get('component', 'general')}` | conf={float(rule.get('confidence', 0.0) or 0.0):.2f} | {rule.get('summary', '')}"
            )
        sections.append("\n".join(lines))
    claims = bundle.get("claims", [])
    if isinstance(claims, list) and claims:
        lines = ["## Relevant Claims"]
        for claim in claims[:6]:
            if not isinstance(claim, dict):
                continue
            lines.append(
                f"- `{claim.get('stance', 'general')}` | `{claim.get('component', 'general')}` | {claim.get('summary', '')}"
            )
        sections.append("\n".join(lines))
    branch_memories = bundle.get("branch_memories", [])
    if isinstance(branch_memories, list) and branch_memories:
        lines = ["## Recent Branch Memories"]
        for memory in branch_memories:
            if not isinstance(memory, dict):
                continue
            lines.append(
                f"- `{memory.get('run_id', '')}` | outcome={memory.get('outcome', '')} | idea={memory.get('idea_class', '')} | {memory.get('summary', '')}"
            )
        sections.append("\n".join(lines))
    contradictions = bundle.get("contradictions", [])
    if isinstance(contradictions, list) and contradictions:
        lines = ["## Contradictions"]
        for item in contradictions:
            if not isinstance(item, dict):
                continue
            lines.append(f"- `{item.get('component', 'general')}` | {item.get('summary', '')}")
        sections.append("\n".join(lines))
    constraints = bundle.get("constraints", [])
    if isinstance(constraints, list) and constraints:
        lines = ["## Active Constraints"]
        for item in constraints:
            if not isinstance(item, dict):
                continue
            lines.append(f"- `{item.get('constraint_type', '')}` | {item.get('summary', '')}")
        sections.append("\n".join(lines))
    capability_packs = bundle.get("capability_packs", [])
    capability_results = bundle.get("capability_results", {})
    if isinstance(capability_packs, list) and capability_packs:
        lines = ["## Capability Packs"]
        for pack in capability_packs:
            if not isinstance(pack, dict):
                continue
            result = capability_results.get(str(pack.get("pack_id", "")), {}) if isinstance(capability_results, dict) else {}
            lines.append(
                f"- `{pack.get('pack_id', '')}` | reason={pack.get('selection_reason', 'stage-default')} | result={truncate(json.dumps(result, ensure_ascii=False), limit=180) if result else 'pending'}"
            )
        sections.append("\n".join(lines))
    semantic_memory_files = bundle.get("semantic_memory_files", [])
    if isinstance(semantic_memory_files, list) and semantic_memory_files:
        lines = ["## Semantic Memory Files"]
        for item in semantic_memory_files:
            if not isinstance(item, dict):
                continue
            lines.append(f"- `knowledge/{item.get('path', '')}` | {item.get('summary', '')}")
        sections.append("\n".join(lines))
    rendered = "\n\n".join(section for section in sections if section.strip())
    return truncate(rendered, limit=limit) if len(rendered) > limit else rendered


def compact_knowledge_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem_frame": bundle.get("problem_frame", {}) if isinstance(bundle.get("problem_frame"), dict) else {},
        "knowledge_files_seen": int(bundle.get("knowledge_files_seen", 0) or 0),
        "knowledge_card_ids": [str(item) for item in bundle.get("knowledge_card_ids", [])],
        "branch_memory_ids": [str(item) for item in bundle.get("branch_memory_ids", [])],
        "cards": [dict(item) for item in bundle.get("cards", []) if isinstance(item, dict)],
        "policy_rules": [dict(item) for item in bundle.get("policy_rules", []) if isinstance(item, dict)],
        "policy_cards": [dict(item) for item in bundle.get("policy_cards", []) if isinstance(item, dict)],
        "claims": [dict(item) for item in bundle.get("claims", []) if isinstance(item, dict)],
        "branch_memories": [dict(item) for item in bundle.get("branch_memories", []) if isinstance(item, dict)],
        "contradictions": [dict(item) for item in bundle.get("contradictions", []) if isinstance(item, dict)],
        "constraints": [dict(item) for item in bundle.get("constraints", []) if isinstance(item, dict)],
        "semantic_memory_files": [dict(item) for item in bundle.get("semantic_memory_files", []) if isinstance(item, dict)],
        "capability_packs": [dict(item) for item in bundle.get("capability_packs", []) if isinstance(item, dict)],
        "capability_results": dict(bundle.get("capability_results", {})) if isinstance(bundle.get("capability_results"), dict) else {},
        "session_memory": bundle.get("session_memory", {}) if isinstance(bundle.get("session_memory"), dict) else {},
    }


def apply_knowledge_stage_outputs(
    config: WorkspaceConfig,
    *,
    run_id: str,
    stage: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    ensure_knowledge_layout(config)
    return {
        "memory_ops": [],
        "touched_memory_files": [],
        "session_memory": load_session_memory(config),
        "run_id": run_id,
        "stage": stage,
        "payload_keys": sorted(str(key) for key in payload.keys()),
    }


def read_knowledge_context(config: WorkspaceConfig) -> str:
    ensure_knowledge_layout(config)
    parts: list[str] = []
    for path in sorted(config.knowledge_root().rglob("*.md")):
        if not path.is_file():
            continue
        relative = path.relative_to(config.knowledge_root())
        parts.append(f"## {relative}\n\n{path.read_text(encoding='utf-8').strip()}")
    return "\n\n".join(part for part in parts if part.strip())


def _family_reference_metrics(state: WorkspaceState, run_id: str, family: str, *, parent_run_id: str = "") -> tuple[float | None, float | None]:
    family_best: float | None = None
    parent_metric: float | None = None
    experiment_by_id = {item.id: item for item in state.experiments}
    for candidate in state.runs:
        if candidate.run_id == run_id or candidate.status != "succeeded" or candidate.primary_metric_value is None:
            continue
        experiment = experiment_by_id.get(candidate.experiment_id)
        if experiment is None or experiment.family != family:
            continue
        family_best = candidate.primary_metric_value if family_best is None else max(family_best, candidate.primary_metric_value)
        if parent_run_id and candidate.run_id == parent_run_id:
            parent_metric = candidate.primary_metric_value
    return family_best, parent_metric


def _branch_memory_outcome(
    *,
    run,
    family_best_before: float | None,
    parent_metric: float | None,
    critic_status: str,
    validation_status: str,
    submission_status: str,
) -> tuple[str, float | None]:
    if critic_status == "rejected":
        return "critic_rejected", None
    if validation_status == "failed":
        return "validate_failed", None
    if run.status == "failed":
        return "run_failed", None
    if run.primary_metric_value is None:
        return "unscored", None
    delta_reference = parent_metric if parent_metric is not None else family_best_before
    delta = None if delta_reference is None else run.primary_metric_value - delta_reference
    if family_best_before is None or run.primary_metric_value > family_best_before + 1e-9:
        return ("submission_candidate" if submission_status == "candidate_created" else "leader"), delta
    if delta is not None and delta >= 0.002:
        return "improved", delta
    if delta is not None and delta <= -0.002:
        return "regressed", delta
    return "flat", delta


def _signal_score_for_outcome(outcome: str) -> float:
    return {
        "leader": 3.0,
        "submission_candidate": 2.6,
        "improved": 2.0,
        "flat": 0.5,
        "unscored": 0.0,
        "regressed": -1.8,
        "critic_rejected": -2.4,
        "validate_failed": -2.6,
        "run_failed": -3.0,
    }.get(outcome, 0.0)


def synchronize_branch_memory(state: WorkspaceState, run_id: str) -> BranchMemoryRecord | None:
    run = next((item for item in state.runs if item.run_id == run_id), None)
    if run is None:
        return None
    work_item = next((item for item in state.work_items if item.id == run.work_item_id), None)
    experiment = next((item for item in state.experiments if item.id == run.experiment_id), None)
    if work_item is None or experiment is None:
        return None
    critic = latest_stage_payload(state, run_id, "critic")
    validation = latest_stage_payload(state, run_id, "validate")
    decision = latest_stage_payload(state, run_id, "decision")
    submission = latest_stage_payload(state, run_id, "submission")
    report = latest_stage_payload(state, run_id, "report")
    source_run_id = str(work_item.source_run_id or "")
    family_best_before, parent_metric = _family_reference_metrics(state, run.run_id, experiment.family, parent_run_id=source_run_id)
    outcome, metric_delta = _branch_memory_outcome(
        run=run,
        family_best_before=family_best_before,
        parent_metric=parent_metric,
        critic_status=str(critic.get("status", "")),
        validation_status=str(validation.get("status", "")),
        submission_status=str(submission.get("status", "")),
    )
    summary_bits = [f"{work_item.branch_role or 'branch'} {work_item.idea_class or experiment.family} -> {outcome}"]
    if run.primary_metric_value is not None:
        summary_bits.append(f"{run.primary_metric_name or 'metric'}={run.primary_metric_value:.6f}")
    if metric_delta is not None:
        summary_bits.append(f"delta={metric_delta:+.6f}")
    if critic.get("status"):
        summary_bits.append(f"critic={critic.get('status')}")
    if validation.get("status"):
        summary_bits.append(f"validate={validation.get('status')}")
    memory = next((item for item in state.branch_memories if item.run_id == run_id), None)
    if memory is None:
        memory = BranchMemoryRecord(
            memory_id=f"memory-{state.runtime.next_branch_memory_number:04d}",
            run_id=run_id,
            work_item_id=work_item.id,
            experiment_id=experiment.id,
            family=experiment.family,
            created_at=now_utc_iso(),
        )
        state.runtime.next_branch_memory_number += 1
        state.branch_memories.append(memory)
    memory.portfolio_id = work_item.portfolio_id
    memory.idea_class = work_item.idea_class
    memory.branch_role = work_item.branch_role
    memory.branch_rank = work_item.branch_rank
    memory.source_stage_run_id = run.latest_stage_run_id
    memory.status = run.status
    memory.outcome = outcome
    memory.summary = truncate("; ".join(summary_bits), limit=320)
    memory.root_cause = str(report.get("root_cause") or decision.get("root_cause") or run.root_cause or run.error or "")
    memory.metric_name = run.primary_metric_name
    memory.metric_value = run.primary_metric_value
    memory.metric_delta = metric_delta
    memory.verify_status = str(latest_stage_payload(state, run_id, "codegen").get("verify_status", ""))
    memory.critic_status = str(critic.get("status", ""))
    memory.validation_status = str(validation.get("status", ""))
    memory.submission_status = str(submission.get("status", ""))
    memory.signal_score = _signal_score_for_outcome(outcome)
    memory.policy_tags = [
        f"outcome:{outcome}",
        *(f"role:{work_item.branch_role}" for _ in [0] if work_item.branch_role),
        *(f"idea:{work_item.idea_class}" for _ in [0] if work_item.idea_class),
    ]
    memory.knowledge_card_ids = list(work_item.knowledge_card_ids)
    memory.contradiction_ids = []
    memory.updated_at = now_utc_iso()
    return memory


def write_experiment_conclusions(config: WorkspaceConfig, state: WorkspaceState) -> Path:
    ensure_knowledge_layout(config)
    completed_runs = [run for run in visible_runs(state) if run.status in {"succeeded", "failed"}]
    completed_runs.sort(key=lambda item: (item.completed_at, item.run_id))

    lines = ["# Experiment Conclusions", ""]
    if not completed_runs:
        lines.append("- No completed experiments yet.")
    for run in completed_runs:
        result = load_run_result(run)
        decision = latest_stage_payload(state, run.run_id, "decision")
        verdict = str(result.get("verdict", "unknown"))
        root_cause = str(decision.get("root_cause") or result.get("root_cause", run.error or "unknown"))
        metric = "-" if run.primary_metric_value is None else f"{run.primary_metric_value:.6f}"
        experiment_id = run.experiment_id
        lines.extend(
            [
                f"## {run.run_id}",
                f"- Experiment: `{experiment_id}`",
                f"- Best AUC: {metric}",
                f"- Root cause: {root_cause}",
                f"- Verdict: {verdict}",
                "",
            ]
        )
    path = config.knowledge_root() / "experiment_conclusions.md"
    atomic_write_text(path, "\n".join(lines).rstrip() + "\n")
    compile_knowledge_index(config)
    return path
