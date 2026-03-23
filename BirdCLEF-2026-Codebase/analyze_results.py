from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize BirdCLEF runtime result files.")
    parser.add_argument("--output-root", default="outputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_root = Path(args.output_root).resolve()
    results = []
    for result_path in output_root.glob("*/result.json"):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        results.append(
            (
                float(payload.get("primary_metric_value", 0.0)),
                result_path.parent.name,
                payload.get("primary_metric_name", "unknown"),
            )
        )
    results.sort(reverse=True)
    if not results:
        print("No result.json files found.")
        return 0
    for rank, (metric_value, name, metric_name) in enumerate(results, start=1):
        print(f"{rank:>2}. {name:<30} {metric_name}={metric_value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
