from __future__ import annotations

import argparse
from pathlib import Path

from kaggle_agent.service import (
    build_submission,
    doctor_checks,
    enqueue_config,
    get_status_state,
    init_workspace,
    list_ready_experiments,
    load_config,
    start_next,
    tick,
    watch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Three-plane autonomous Kaggle research system.")
    parser.add_argument("--root", default=".", help="Workspace root containing workspace.toml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize the workspace and archive legacy artifacts.")
    init_parser.add_argument("--no-archive-legacy", action="store_true")
    init_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("doctor", help="Run workspace readiness checks.")
    subparsers.add_parser("status", help="Print experiment and run status.")
    subparsers.add_parser("list-ready", help="List runnable experiments.")

    enqueue_parser = subparsers.add_parser("enqueue-config", help="Register a config as an experiment.")
    enqueue_parser.add_argument("config_path")
    enqueue_parser.add_argument("--title")
    enqueue_parser.add_argument("--family", default="ad_hoc")
    enqueue_parser.add_argument("--priority", type=int, default=50)

    start_parser = subparsers.add_parser("start-next", help="Start the next runnable experiment.")
    start_parser.add_argument("--sync", action="store_true", help="Run synchronously instead of in background.")

    tick_parser = subparsers.add_parser("tick", help="Finalize finished runs, produce decisions, and optionally start the next run.")
    tick_parser.add_argument("--no-auto-start", action="store_true")

    watch_parser = subparsers.add_parser("watch", help="Run the scheduler/monitor loop on a cadence.")
    watch_parser.add_argument("--interval-seconds", type=int, default=600)
    watch_parser.add_argument("--iterations", type=int, default=1)
    watch_parser.add_argument("--no-auto-start", action="store_true")

    submission_parser = subparsers.add_parser("build-submission", help="Create a submission candidate scaffold from a run.")
    submission_parser.add_argument("--run-id")
    return parser


def _print_status(root: Path) -> int:
    config = load_config(root)
    state = get_status_state(config)
    print(f"Active runs: {', '.join(state.runtime.active_run_ids) if state.runtime.active_run_ids else 'none'}")
    print("Experiments:")
    for experiment in sorted(state.experiments, key=lambda item: (item.priority, item.id)):
        print(
            f"  {experiment.id}\t{experiment.status}\tp{experiment.priority}\t{experiment.family}\t{experiment.config_path}"
        )
    print("Runs:")
    for run in state.runs[-10:]:
        metric = "-" if run.primary_metric_value is None else f"{run.primary_metric_name}={run.primary_metric_value:.6f}"
        print(f"  {run.run_id}\t{run.status}\t{run.experiment_id}\t{metric}")
    print("Decisions:")
    for decision in state.decisions[-5:]:
        print(f"  {decision.decision_id}\t{decision.decision_type}\t{decision.next_action}\t{decision.source_run_id}")
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
        ready = list_ready_experiments(config)
        if not ready:
            print("No runnable experiments.")
            return 0
        for experiment_id in ready:
            print(experiment_id)
        return 0

    if args.command == "enqueue-config":
        config = load_config(root)
        state = enqueue_config(config, args.config_path, title=args.title, family=args.family, priority=args.priority)
        print(f"Registered config. Total experiments: {len(state.experiments)}")
        return 0

    if args.command == "start-next":
        config = load_config(root)
        run_id = start_next(config, background=not args.sync)
        if run_id is None:
            print("No runnable experiment.")
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
        watch(
            config,
            interval_seconds=args.interval_seconds,
            iterations=args.iterations,
            auto_start=not args.no_auto_start,
        )
        print("Watch loop completed.")
        return 0

    if args.command == "build-submission":
        config = load_config(root)
        submission_id = build_submission(config, run_id=args.run_id)
        print(submission_id)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
