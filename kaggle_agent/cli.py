from __future__ import annotations

import argparse
from pathlib import Path

from kaggle_agent.service import (
    build_submission,
    doctor_checks,
    dry_run_submission,
    enqueue_preflight,
    enqueue_config,
    get_status_state,
    init_workspace,
    list_ready_work_items,
    load_config,
    plan_submission,
    start_next,
    tick,
    watch,
)
from kaggle_agent.layout import (
    artifact_relative_path,
    current_attempt_slug,
    run_label_from_path,
    stage_label_from_path,
    visible_runs,
    visible_stage_runs,
    visible_work_items,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent-primary Kaggle Research OS for BirdCLEF 2026.")
    parser.add_argument("--root", default=".", help="Workspace root containing workspace.toml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize the ledger, reports, prompts, and surface docs.")
    init_parser.add_argument("--no-archive-legacy", action="store_true")
    init_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("doctor", help="Run workspace readiness checks.")
    status_parser = subparsers.add_parser("status", help="Print work-item, run, and submission status.")
    status_parser.add_argument("--include-debug", action="store_true", help="Include explicit debug/preflight work items and runs.")

    ready_parser = subparsers.add_parser("list-ready", help="List runnable work items.")
    ready_parser.add_argument("--include-debug", action="store_true", help="Include explicit debug/preflight work items.")

    enqueue_parser = subparsers.add_parser("enqueue-config", help="Queue a config as a new experiment work item.")
    enqueue_parser.add_argument("config_path")
    enqueue_parser.add_argument("--title")
    enqueue_parser.add_argument("--family", default="ad_hoc")
    enqueue_parser.add_argument("--priority", type=int, default=50)
    enqueue_parser.add_argument("--work-type", default="experiment_iteration")
    enqueue_parser.add_argument("--lifecycle-template", default="")
    enqueue_parser.add_argument("--note", action="append", default=[])
    enqueue_parser.add_argument("--depends-on", action="append", default=[])

    preflight_parser = subparsers.add_parser("enqueue-preflight", help="Queue an explicit debug preflight work item.")
    preflight_parser.add_argument("--priority", type=int, default=5)
    preflight_parser.add_argument("--allow-debug", action="store_true", help="Acknowledge that this queues a debug-only check.")

    start_parser = subparsers.add_parser("start-next", help="Start the next runnable work item.")
    start_parser.add_argument("--sync", action="store_true", help="Run synchronously instead of in background.")

    tick_parser = subparsers.add_parser("tick", help="Advance the research OS once.")
    tick_parser.add_argument("--no-auto-start", action="store_true")

    watch_parser = subparsers.add_parser("watch", help="Run the monitor loop on a cadence.")
    watch_parser.add_argument("--interval-seconds", type=int, default=600)
    watch_parser.add_argument("--iterations", type=int, default=1)
    watch_parser.add_argument("--no-auto-start", action="store_true")

    submission_parser = subparsers.add_parser("build-submission", help="Build a CPU-first submission candidate bundle.")
    submission_parser.add_argument("--run-id")

    dry_run_parser = subparsers.add_parser("dry-run-submission", help="Run the local CPU dry-run for a candidate.")
    dry_run_parser.add_argument("candidate_id")

    subparsers.add_parser("plan-submission", help="Show remaining submission slots and CPU-ready candidates.")
    return parser


def _print_status(root: Path, *, include_debug: bool = False) -> int:
    config = load_config(root)
    state = get_status_state(config)
    work_items = visible_work_items(state, include_debug=include_debug)
    runs = visible_runs(state, include_debug=include_debug)
    stage_runs = visible_stage_runs(state, include_debug=include_debug)
    print(f"Current attempt: {current_attempt_slug(state.runtime)}")
    print(f"Active runs: {', '.join(state.runtime.active_run_ids) if state.runtime.active_run_ids else 'none'}")
    print("Work Items:")
    for work_item in sorted(work_items, key=lambda item: (item.priority, item.created_at, item.id)):
        latest_stage = next((item for item in state.stage_runs if item.stage_run_id == work_item.latest_stage_run_id), None)
        stage_label = stage_label_from_path(
            latest_stage.output_dir if latest_stage is not None else "",
            latest_stage.stage_name if latest_stage is not None else "stage",
            stage_status=latest_stage.status if latest_stage is not None else "",
            validator_status=latest_stage.validator_status if latest_stage is not None else "",
        )
        run_label = next((run_label_from_path(item.run_dir) for item in runs if item.run_id == work_item.latest_run_id), "")
        print(
            f"  {work_item.id}\t{work_item.status}\tp{work_item.priority}\t{work_item.family}\t{work_item.title}"
            f"\tlifecycle={getattr(work_item, 'lifecycle_template', '') or 'n/a'}"
            f"\ttarget={getattr(work_item, 'target_run_id', '') or 'n/a'}"
            f"\trun={run_label or 'n/a'}\tstage={stage_label if latest_stage is not None else 'n/a'}"
        )
    print("Runs:")
    for run in runs[-10:]:
        metric = "-" if run.primary_metric_value is None else f"{run.primary_metric_name}={run.primary_metric_value:.6f}"
        print(
            f"  {run_label_from_path(run.run_dir) or run.run_id}\t{run.status}\t{run.stage_cursor or 'complete'}"
            f"\t{run.experiment_id}\t{metric}\tlifecycle={getattr(run, 'lifecycle_template', '') or 'n/a'}"
            f"\truntime={artifact_relative_path(run.run_dir, config.root)}"
        )
    print("Stage Runs:")
    for stage_run in stage_runs[-10:]:
        print(
            f"  {stage_label_from_path(stage_run.output_dir, stage_run.stage_name, stage_status=stage_run.status, validator_status=stage_run.validator_status)}"
            f"\t{stage_run.status}\t{run_label_from_path(next((run.run_dir for run in state.runs if run.run_id == stage_run.run_id), '')) or stage_run.run_id}"
            f"\t{artifact_relative_path(stage_run.output_dir, config.root)}"
        )
    print("Submissions:")
    for candidate in state.submissions[-5:]:
        print(f"  {candidate.id}\t{candidate.status}\tcpu_ready={candidate.cpu_ready}\tsource={candidate.source_run_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()

    if args.command == "init":
        config = load_config(root)
        init_workspace(config, archive_legacy=not args.no_archive_legacy, force=args.force)
        print(f"Initialized workspace at {config.root}")
        return 0

    if args.command == "doctor":
        config = load_config(root)
        has_failure = False
        for ok, name, detail in doctor_checks(config):
            prefix = "OK" if ok else "FAIL"
            print(f"[{prefix}] {name}: {detail}")
            has_failure = has_failure or not ok
        return 1 if has_failure else 0

    if args.command == "status":
        return _print_status(root, include_debug=args.include_debug)

    if args.command == "list-ready":
        config = load_config(root)
        ready = list_ready_work_items(config)
        if not args.include_debug:
            state = get_status_state(config)
            visible_work_item_ids = {item.id for item in visible_work_items(state)}
            ready = [item_id for item_id in ready if item_id in visible_work_item_ids]
        if not ready:
            print("No runnable work items.")
            return 0
        for work_item_id in ready:
            print(work_item_id)
        return 0

    if args.command == "enqueue-config":
        config = load_config(root)
        state = enqueue_config(
            config,
            args.config_path,
            title=args.title,
            family=args.family,
            priority=args.priority,
            work_type=args.work_type,
            lifecycle_template=args.lifecycle_template,
            notes=args.note,
            depends_on=args.depends_on,
        )
        print(f"Queued work item. Total work items: {len(state.work_items)}")
        return 0

    if args.command == "enqueue-preflight":
        config = load_config(root)
        try:
            state = enqueue_preflight(config, priority=args.priority, allow_debug=args.allow_debug)
        except ValueError as exc:
            print(str(exc))
            return 2
        preflight_items = [item for item in state.work_items if item.work_type == "preflight_check"]
        print(preflight_items[-1].id if preflight_items else "No preflight queued.")
        return 0

    if args.command == "start-next":
        config = load_config(root)
        run_id = start_next(config, background=not args.sync)
        if run_id is None:
            print("No runnable work item.")
            return 0
        print(run_id)
        return 0

    if args.command == "tick":
        config = load_config(root)
        state = tick(config, auto_start=not args.no_auto_start)
        print(f"Tick complete. Active runs: {', '.join(state.runtime.active_run_ids) if state.runtime.active_run_ids else 'none'}")
        return 0

    if args.command == "watch":
        config = load_config(root)
        watch(config, interval_seconds=args.interval_seconds, iterations=args.iterations, auto_start=not args.no_auto_start)
        print("Watch loop completed.")
        return 0

    if args.command == "build-submission":
        config = load_config(root)
        candidate_id = build_submission(config, run_id=args.run_id)
        print(candidate_id)
        return 0

    if args.command == "dry-run-submission":
        config = load_config(root)
        result = dry_run_submission(config, args.candidate_id)
        print(result["status"])
        return 0

    if args.command == "plan-submission":
        config = load_config(root)
        result = plan_submission(config)
        print(result)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
