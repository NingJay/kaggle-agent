from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggle_agent.decision.helpers import (
    begin_stage_run,
    complete_stage_run,
    latest_stage_payload,
    run_configured_stage_adapter,
    stage_markdown,
    write_input_manifest,
)
from kaggle_agent.knowledge import (
    apply_knowledge_stage_outputs,
    compact_knowledge_bundle,
    ensure_knowledge_layout,
    retrieve_knowledge_bundle,
)
from kaggle_agent.schema import ResearchNoteRecord, WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import atomic_write_text, now_utc_iso


def _candidate_queries(root_cause: str, family: str, knowledge_bundle: dict[str, object]) -> list[str]:
    queries: list[str] = []
    text = f"{family}\n{root_cause}".lower()
    if "probe" in text or "embedding" in text:
        queries.append("bird audio embedding probe class imbalance holdout validation")
    if "pseudo" in text or "teacher" in text or "distill" in text:
        queries.append("bird audio pseudo label distillation holdout roc auc")
    if "coverage" in text or "imbalance" in text or "rare" in text:
        queries.append("long tail multilabel acoustic event class coverage")
    if "backbone" in text or "v2s" in text:
        queries.append("bird audio backbone recovery learning rate schedule")
    capability_results = knowledge_bundle.get("capability_results", {})
    if isinstance(capability_results, dict):
        novel = capability_results.get("novel_hypothesis_generator", {})
        if isinstance(novel, dict):
            for item in novel.get("required_evidence", [])[:2]:
                if str(item).strip():
                    queries.append(str(item))
    queries.append("birdclef soundscape macro roc auc validation strategy")
    return list(dict.fromkeys(item for item in queries if item))


def _rule_summaries(policy_rules: list[dict[str, Any]], *, kinds: set[str]) -> list[str]:
    lines: list[str] = []
    for item in policy_rules:
        if str(item.get("policy_type", "")) not in kinds:
            continue
        lines.append(f"{item.get('component', 'general')}: {item.get('summary', '')}")
    return lines


def _build_research_payload(root_cause: str, family: str, knowledge_bundle: dict[str, object]) -> dict[str, object]:
    policy_rules = [item for item in knowledge_bundle.get("policy_rules", []) if isinstance(item, dict)]
    claims = [item for item in knowledge_bundle.get("claims", []) if isinstance(item, dict)]
    branch_memories = [item for item in knowledge_bundle.get("branch_memories", []) if isinstance(item, dict)]
    contradictions = [item for item in knowledge_bundle.get("contradictions", []) if isinstance(item, dict)]
    constraints = [item for item in knowledge_bundle.get("constraints", []) if isinstance(item, dict)]
    capability_packs = [item for item in knowledge_bundle.get("capability_packs", []) if isinstance(item, dict)]
    capability_results = knowledge_bundle.get("capability_results", {}) if isinstance(knowledge_bundle.get("capability_results"), dict) else {}
    session_memory = knowledge_bundle.get("session_memory", {}) if isinstance(knowledge_bundle.get("session_memory"), dict) else {}

    adopt_now = _rule_summaries(policy_rules, kinds={"require", "prefer"})
    consider = _rule_summaries(policy_rules, kinds={"conditional"})
    reject = _rule_summaries(policy_rules, kinds={"veto", "avoid"})

    strong_branch = next(
        (item for item in branch_memories if str(item.get("outcome", "")) in {"leader", "improved", "submission_candidate"}),
        None,
    )
    weak_branch = next(
        (item for item in branch_memories if str(item.get("outcome", "")) in {"regressed", "critic_rejected", "run_failed", "validate_failed"}),
        None,
    )
    if strong_branch is not None:
        adopt_now.append(f"reuse strong branch pattern: {strong_branch.get('summary', '')}")
    if weak_branch is not None:
        reject.append(f"avoid recently weak branch pattern: {weak_branch.get('summary', '')}")

    ledger = capability_results.get("ledger_miner", {})
    if isinstance(ledger, dict):
        top_components = ledger.get("top_components", [])
        if isinstance(top_components, list) and top_components:
            consider.append(
                "top ledger components: "
                + ", ".join(f"{item.get('component', '')}({item.get('net', 0)})" for item in top_components[:3] if isinstance(item, dict))
            )
    veto = capability_results.get("veto_checker", {})
    if isinstance(veto, dict):
        forbidden = [str(item) for item in veto.get("forbidden_patterns", []) if str(item)]
        if forbidden:
            reject.append("forbidden plan patterns now active: " + ", ".join(forbidden[:4]))
    novel = capability_results.get("novel_hypothesis_generator", {})
    open_questions: list[str] = [
        f"How do we resolve {item.get('component', 'general')} contradiction: {item.get('summary', '')}"
        for item in contradictions[:3]
    ]
    if isinstance(novel, dict):
        open_questions.extend(str(item) for item in novel.get("required_evidence", [])[:2] if str(item))
    if not open_questions:
        open_questions.append(f"What removes the current bottleneck without repeating low-information sweeps: {root_cause}?")

    retrieval_queries_next_turn = _candidate_queries(root_cause, family, knowledge_bundle)
    hypothesis_backlog: list[dict[str, Any]] = []
    if isinstance(novel, dict) and str(novel.get("novel_component", "")):
        hypothesis_backlog.append(
            {
                "kind": "novel_lane",
                "component": str(novel.get("novel_component", "")),
                "unsupported_claims": [str(item) for item in novel.get("unsupported_claims", []) if str(item)],
                "required_evidence": [str(item) for item in novel.get("required_evidence", []) if str(item)],
            }
        )
    if strong_branch is not None:
        hypothesis_backlog.append(
            {
                "kind": "grounded_lane",
                "component": str(strong_branch.get("idea_class", "") or "general"),
                "reason": str(strong_branch.get("summary", "")),
            }
        )

    return {
        "stage": "research",
        "root_cause": root_cause,
        "queries": retrieval_queries_next_turn,
        "adopt_now": adopt_now[:6] or ["repair the bottleneck with the strongest grounded prior first"],
        "consider": consider[:6] or ["open one adjacent branch if grounded priors are exhausted"],
        "reject": reject[:6],
        "knowledge_files_seen": int(knowledge_bundle.get("knowledge_files_seen", 0) or 0),
        "problem_frame": knowledge_bundle.get("problem_frame", {}),
        "knowledge_card_ids": [str(item) for item in knowledge_bundle.get("knowledge_card_ids", [])],
        "positive_priors": _rule_summaries(policy_rules, kinds={"require", "prefer"}),
        "negative_vetoes": _rule_summaries(policy_rules, kinds={"veto", "avoid"}),
        "conditional_leads": _rule_summaries(policy_rules, kinds={"conditional"}),
        "policy_rules": policy_rules,
        "policy_cards": policy_rules,
        "claims": claims,
        "branch_memories": branch_memories,
        "branch_memory_ids": [str(item.get("memory_id", "")) for item in branch_memories],
        "contradictions": contradictions,
        "constraints": constraints,
        "selected_memory_files": [str(item.get("path", "")) for item in knowledge_bundle.get("semantic_memory_files", []) if isinstance(item, dict)],
        "selected_capability_packs": [str(item.get("pack_id", "")) for item in capability_packs],
        "capability_results": capability_results,
        "session_memory": session_memory,
        "open_questions": open_questions[:5],
        "retrieval_queries_next_turn": retrieval_queries_next_turn,
        "hypothesis_backlog": hypothesis_backlog,
    }


def build_research(config: WorkspaceConfig, state: WorkspaceState, run_id: str):
    run = next(item for item in state.runs if item.run_id == run_id)
    experiment = next(item for item in state.experiments if item.id == run.experiment_id)
    work_item = next(item for item in state.work_items if item.id == run.work_item_id)
    report_payload = latest_stage_payload(state, run_id, "report")
    ensure_knowledge_layout(config)
    knowledge_bundle = retrieve_knowledge_bundle(
        config,
        {
            "run": run.to_dict(),
            "experiment": experiment.to_dict(),
            "work_item": work_item.to_dict(),
            "report": report_payload,
        },
        stage="research",
        state=state,
    )
    stage_run, input_manifest_path = begin_stage_run(
        config,
        state,
        run,
        stage_name="research",
        input_ref=run.latest_stage_run_id or run.run_id,
    )
    write_input_manifest(
        input_manifest_path,
        {
            "run": run.to_dict(),
            "experiment": experiment.to_dict(),
            "work_item": work_item.to_dict(),
            "report": report_payload,
            "retrieved_knowledge": compact_knowledge_bundle(knowledge_bundle),
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
        payload = _build_research_payload(
            str(report_payload.get("root_cause", run.root_cause or run.error or "missing context")),
            experiment.family,
            knowledge_bundle,
        )
        lines = [
            f"- Root cause focus: {payload['root_cause']}",
            f"- Knowledge files seen: {payload['knowledge_files_seen']}",
            f"- Retrieval next turn: {', '.join(str(item) for item in payload['retrieval_queries_next_turn'])}",
            "",
            "## Adopt Now",
            *(f"- {item}" for item in payload["adopt_now"]),
            "",
            "## Consider",
            *(f"- {item}" for item in payload["consider"]),
        ]
        if payload["reject"]:
            lines.extend(["", "## Reject For Now", *(f"- {item}" for item in payload["reject"])])
        if payload.get("policy_rules"):
            lines.extend(
                [
                    "",
                    "## Policy Rules",
                    *(
                        f"- `{item.get('policy_type', 'context')}` | `{item.get('component', 'general')}` | {item.get('summary', '')}"
                        for item in payload["policy_rules"]
                        if isinstance(item, dict)
                    ),
                ]
            )
        if payload.get("claims"):
            lines.extend(
                [
                    "",
                    "## Relevant Claims",
                    *(
                        f"- `{item.get('stance', 'general')}` | `{item.get('component', 'general')}` | {item.get('summary', '')}"
                        for item in payload["claims"][:6]
                        if isinstance(item, dict)
                    ),
                ]
            )
        if payload.get("branch_memories"):
            lines.extend(
                [
                    "",
                    "## Recent Branch Memories",
                    *(
                        f"- `{item.get('run_id', '')}` | outcome={item.get('outcome', '')} | {item.get('summary', '')}"
                        for item in payload["branch_memories"]
                        if isinstance(item, dict)
                    ),
                ]
            )
        if payload.get("contradictions"):
            lines.extend(["", "## Contradictions", *(f"- {item.get('summary', '')}" for item in payload["contradictions"] if isinstance(item, dict))])
        if payload.get("selected_capability_packs"):
            lines.extend(["", "## Capability Packs", *(f"- `{item}`" for item in payload["selected_capability_packs"])])
        if payload.get("capability_results"):
            lines.extend(
                [
                    "",
                    "## Capability Results",
                    *(
                        f"- `{key}` | {value}"
                        for key, value in payload["capability_results"].items()
                        if isinstance(payload["capability_results"], dict)
                    ),
                ]
            )
        if payload.get("hypothesis_backlog"):
            lines.extend(
                [
                    "",
                    "## Hypothesis Backlog",
                    *(
                        f"- `{item.get('kind', '')}` | component={item.get('component', '')} | unsupported={', '.join(str(entry) for entry in item.get('unsupported_claims', [])) or 'n/a'}"
                        for item in payload["hypothesis_backlog"]
                        if isinstance(item, dict)
                    ),
                ]
            )
        if payload.get("open_questions"):
            lines.extend(["", "## Open Questions", *(f"- {item}" for item in payload["open_questions"])])
        markdown = stage_markdown(f"Research Summary {run_id}", lines)
        complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)

    note_payload = latest_stage_payload(state, run_id, "research")
    apply_knowledge_stage_outputs(config, run_id=run_id, stage="research", payload=note_payload)
    summary = "; ".join(str(item) for item in note_payload.get("adopt_now", [])) or "no structured research actions"
    state.research_notes.append(
        ResearchNoteRecord(
            note_id=f"note-{state.runtime.next_note_number:04d}",
            run_id=run_id,
            title=f"Research summary for {run_id}",
            summary=summary,
            stance="adopt_now",
            source_type="internal_fallback" if not config.adapters.research_command.strip() else "adapter",
            created_at=now_utc_iso(),
        )
    )
    state.runtime.next_note_number += 1
    output_copy = config.knowledge_path("research", f"{run_id}.md")
    atomic_write_text(output_copy, Path(stage_run.output_md_path).read_text(encoding="utf-8"))
    return stage_run
