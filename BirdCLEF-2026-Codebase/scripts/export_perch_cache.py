from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RUNTIME_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from birdclef_runtime.config import apply_overrides, load_config
from birdclef_runtime.perch_teacher import run_perch_teacher_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export notebook-faithful Perch teacher cache artifacts.")
    parser.add_argument("--config", required=True, help="Path to runtime YAML config.")
    parser.add_argument("overrides", nargs="*", help="Optional dotted.key=value overrides.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(args.config), args.overrides)
    run_dir = (RUNTIME_ROOT / "outputs" / config["experiment"]["name"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    result = run_perch_teacher_cache(config, RUNTIME_ROOT, run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
