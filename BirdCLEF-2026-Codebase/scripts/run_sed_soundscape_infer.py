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
from birdclef_runtime.sed_v5 import run_sed_soundscape_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run notebook-faithful SED soundscape inference.")
    parser.add_argument("--config", required=True, help="Path to runtime YAML config.")
    parser.add_argument("overrides", nargs="*", help="Optional dotted.key=value overrides.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(args.config), args.overrides)
    output_dir = (RUNTIME_ROOT / "outputs" / config["experiment"]["name"] / "inference").resolve()
    result = run_sed_soundscape_inference(config, RUNTIME_ROOT, output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
