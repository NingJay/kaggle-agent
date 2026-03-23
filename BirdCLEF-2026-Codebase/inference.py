from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parent
SRC_ROOT = RUNTIME_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from birdclef_runtime.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a BirdCLEF submission scaffold.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=False, default="")
    parser.add_argument("--output", required=False, default="submission.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data_root = (RUNTIME_ROOT / config["paths"]["data_root"]).resolve()
    sample_submission = data_root / config["data"].get("sample_submission_csv", "sample_submission.csv")
    output_path = Path(args.output).resolve()
    if sample_submission.exists():
        rows = list(csv.reader(sample_submission.open("r", encoding="utf-8", newline="")))
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(rows)
    else:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_id", "prediction"])
    print(json.dumps({"output": str(output_path), "checkpoint": args.checkpoint}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
