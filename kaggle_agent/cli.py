from __future__ import annotations

import argparse
from pathlib import Path

from kaggle_agent.service import (
    build_submission,
    doctor_checks,
    dry_run_submission,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent-primary Kaggle Research OS for BirdCLEF 2026.")
    parser.add_argument("--root", default=".", help="Workspace root containing workspace.toml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize the ledger, reports, prompts, and surface docs.")
    init_parser.add_argument("--no-archive-legacy", action="store_true")
    init_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("doctor", help="Run workspace readiness checks.")
    subparsers.add_parser("status", help="Print work-item, run, and submission status.")
    subparsers.add_parser("list-ready", help="List runnable work items.")

    enqueue_parser = subparsers.add_parser("enqueue-config", help="Queue a config as a new experiment work item.")
    enqueue_parser.add_argument("config_path")
    enqueue_parser.add_argument("--title")
    enqueue_parser.add_argument("--family", default="ad_hoc")
    enqueue_parser.add_argument("--priority", type=int, default=50)

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


def _print_status(root: Path) -> int:
    config = load_config(root)
    state = get_status_state(config)
    print(f"Active runs: {', '.join(state.runtime.active_run_ids) if state.runtime.active_run_ids else 'none'}")
    print("Work Items:")
    for work_item in sorted(state.work_items, key=lambda item: (item.priority, item.created_at, item.id)):
        print(f"  {work_item.id}\t{work_item.status}\tp{work_item.priority}\t{work_item.family}\t{work_item.title}")
    print("Runs:")
    for run in state.runs[-10:]:
        metric = "-" if run.primary_metric_value is None else f"{run.primary_metric_name}={run.primary_metric_value:.6f}"
        print(f"  {run.run_id}\t{run.status}\t{run.stage_cursor or 'complete'}\t{run.experiment_id}\t{metric}")
    print("Stage Runs:")
    for stage_run in state.stage_runs[-10:]:
        print(f"  {stage_run.stage_run_id}\t{stage_run.stage_name}\t{stage_run.status}\t{stage_run.run_id}")
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
        return _print_status(root)

    if args.command == "list-ready":
        config = load_config(root)
        ready = list_ready_work_items(config)
        if not ready:
            print("No runnable work items.")
            return 0
        for work_item_id in ready:
            print(work_item_id)
        return 0

    if args.command == "enqueue-config":
        config = load_config(root)
        state = enqueue_config(config, args.config_path, title=args.title, family=args.family, priority=args.priority)
        print(f"Queued work item. Total work items: {len(state.work_items)}")
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
