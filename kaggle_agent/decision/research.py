from __future__ import annotations

from pathlib import Path

from kaggle_agent.decision.helpers import (
    begin_stage_run,
    complete_stage_run,
    latest_stage_payload,
    run_configured_stage_adapter,
    stage_markdown,
    write_input_manifest,
)
from kaggle_agent.knowledge import compact_knowledge_bundle, ensure_knowledge_layout, retrieve_knowledge_bundle
from kaggle_agent.schema import ResearchNoteRecord, WorkspaceConfig, WorkspaceState
from kaggle_agent.utils import atomic_write_text, now_utc_iso


def _candidate_queries(root_cause: str, family: str, knowledge_bundle: dict[str, object]) -> list[str]:
    text = f"{root_cause}\n{family}".lower()
    queries: list[str] = []
    if "probe" in text or "cached" in text or "embedding" in text:
        queries.append("frozen audio embeddings linear probe bird audio classification")
    if "prior" in text or "calibration" in text or "soundscape" in text:
        queries.append("soundscape classification prior calibration passive acoustic monitoring")
    if "imbalance" in text or "sparse" in text or "long tail" in text:
        queries.append("multilabel long tail acoustic event classification")
    cards = knowledge_bundle.get("cards", [])
    if isinstance(cards, list):
        for card in cards[:5]:
            if not isinstance(card, dict):
                continue
            component = str(card.get("component", ""))
            if component == "pseudo_label":
                queries.append("bird audio pseudo label distillation teacher student")
            elif component == "class_coverage":
                queries.append("long tail class coverage passive acoustic monitoring")
            elif component == "prior_calibration":
                queries.append("holdout prior fusion calibration macro roc auc")
            elif component == "probe_head":
                queries.append("audio embedding probe classifier hidden layer birdclef")
    queries.append("birdclef soundscape macro roc auc validation strategy")
    return list(dict.fromkeys(queries))


def _string_priors(knowledge_bundle: dict[str, object], stance: str) -> list[str]:
    cards = knowledge_bundle.get("cards", [])
    priors: list[str] = []
    if not isinstance(cards, list):
        return priors
    for card in cards:
        if not isinstance(card, dict):
            continue
        if str(card.get("stance", "")) != stance:
            continue
        priors.append(
            f"{card.get('card_id', '')}: {card.get('title', '')} [{card.get('component', 'general')}] - {card.get('summary', '')}"
        )
    return priors


def _default_research_payload(root_cause: str, family: str, knowledge_bundle: dict[str, object]) -> dict[str, object]:
    queries = _candidate_queries(root_cause, family, knowledge_bundle)
    adopt_now: list[str] = []
    consider: list[str] = []
    reject: list[str] = []
    positive_priors = _string_priors(knowledge_bundle, "positive")
    negative_priors = _string_priors(knowledge_bundle, "negative")
    conditional_priors = _string_priors(knowledge_bundle, "conditional")
    policy_cards = [item for item in knowledge_bundle.get("policy_cards", []) if isinstance(item, dict)]
    branch_memories = [item for item in knowledge_bundle.get("branch_memories", []) if isinstance(item, dict)]
    contradictions = [item for item in knowledge_bundle.get("contradictions", []) if isinstance(item, dict)]
    policy_trace = [
        f"{item.get('policy_type', 'context')}:{item.get('component', 'general')}:{item.get('card_id', '')}"
        for item in policy_cards[:8]
    ]
    if positive_priors:
        adopt_now.extend(item.split(": ", maxsplit=1)[1] for item in positive_priors[:3] if ": " in item)
    if conditional_priors:
        consider.extend(item.split(": ", maxsplit=1)[1] for item in conditional_priors[:3] if ": " in item)
    if negative_priors:
        reject.extend(item.split(": ", maxsplit=1)[1] for item in negative_priors[:3] if ": " in item)
    if branch_memories:
        best_memory = branch_memories[0]
        if str(best_memory.get("outcome", "")) in {"leader", "improved", "submission_candidate"}:
            adopt_now.append(f"reuse the strongest recent branch pattern: {best_memory.get('summary', '')}")
        weak_memories = [item for item in branch_memories if str(item.get("outcome", "")) in {"regressed", "critic_rejected", "run_failed", "validate_failed"}]
        if weak_memories:
            reject.append(f"avoid repeating weak recent branch patterns: {weak_memories[0].get('summary', '')}")
    for contradiction in contradictions[:2]:
        consider.append(f"resolve contradiction: {contradiction.get('summary', '')}")
    if "probe" in family and not adopt_now:
        adopt_now.append("expand cached-probe variants along coverage or representation axes before heavier finetuning")
    if "probe" in family and not consider:
        consider.append("test probe-head changes that improve class coverage before calibration-only tuning")
    if "failed" in root_cause.lower() or "missing" in root_cause.lower():
        adopt_now.append("repair the runtime contract before spending slot budget on new variants")
        reject.append("do not schedule submission probes while the runtime is unstable")
    if not adopt_now:
        adopt_now.append("continue along the strongest validated branch and repair the primary root cause first")
    if not consider:
        consider.append("compare current findings against the reference notebook and cached Perch priors")
    return {
        "stage": "research",
        "root_cause": root_cause,
        "queries": queries,
        "adopt_now": adopt_now,
        "consider": consider,
        "reject": reject,
        "knowledge_files_seen": int(knowledge_bundle.get("knowledge_files_seen", 0) or 0),
        "problem_frame": knowledge_bundle.get("problem_frame", {}),
        "knowledge_card_ids": [str(item) for item in knowledge_bundle.get("knowledge_card_ids", [])],
        "positive_priors": positive_priors,
        "negative_priors": negative_priors,
        "conditional_priors": conditional_priors,
        "policy_cards": policy_cards,
        "policy_trace": policy_trace,
        "branch_memories": branch_memories,
        "branch_memory_ids": [str(item) for item in knowledge_bundle.get("branch_memory_ids", [])],
        "contradictions": contradictions,
    }


def build_research(config: WorkspaceConfig, state: WorkspaceState, run_id: str):
    run = next(item for item in state.runs if item.run_id == run_id)
    experiment = next(item for item in state.experiments if item.id == run.experiment_id)
    report_payload = latest_stage_payload(state, run_id, "report")
    ensure_knowledge_layout(config)
    knowledge_bundle = retrieve_knowledge_bundle(
        config,
        {
            "run": run.to_dict(),
            "experiment": experiment.to_dict(),
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
        payload = _default_research_payload(
            str(report_payload.get("root_cause", run.root_cause or run.error or "missing context")),
            experiment.family,
            knowledge_bundle,
        )
        lines = [
            f"- Root cause focus: {payload['root_cause']}",
            f"- Suggested search queries: {', '.join(str(item) for item in payload['queries'])}",
            "",
            "## Adopt Now",
            *(f"- {item}" for item in payload["adopt_now"]),
            "",
            "## Consider",
            *(f"- {item}" for item in payload["consider"]),
        ]
        if payload["reject"]:
            lines.extend(["", "## Reject For Now", *(f"- {item}" for item in payload["reject"])])
        if payload.get("knowledge_card_ids"):
            lines.extend(["", "## Retrieved Knowledge Cards", *(f"- `{item}`" for item in payload["knowledge_card_ids"])])
        policy_cards = payload.get("policy_cards")
        if isinstance(policy_cards, list) and policy_cards:
            lines.extend(
                [
                    "",
                    "## Policy Cards",
                    *(
                        f"- `{item.get('policy_type', 'context')}` | `{item.get('component', 'general')}` | `{item.get('card_id', '')}` | {item.get('summary', '')}"
                        for item in policy_cards
                        if isinstance(item, dict)
                    ),
                ]
            )
        branch_memories = payload.get("branch_memories")
        if isinstance(branch_memories, list) and branch_memories:
            lines.extend(
                [
                    "",
                    "## Recent Branch Memories",
                    *(
                        f"- `{item.get('run_id', '')}` | outcome={item.get('outcome', '')} | {item.get('summary', '')}"
                        for item in branch_memories
                        if isinstance(item, dict)
                    ),
                ]
            )
        contradictions = payload.get("contradictions")
        if isinstance(contradictions, list) and contradictions:
            lines.extend(
                [
                    "",
                    "## Contradictions",
                    *(f"- {item.get('summary', '')}" for item in contradictions if isinstance(item, dict)),
                ]
            )
        markdown = stage_markdown(f"Research Summary {run_id}", lines)
        complete_stage_run(stage_run, state=state, payload=payload, markdown=markdown)

    note_payload = latest_stage_payload(state, run_id, "research")
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
