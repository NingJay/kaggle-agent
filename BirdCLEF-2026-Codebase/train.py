from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parent
SRC_ROOT = RUNTIME_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from birdclef_runtime.config import apply_overrides, load_config
from birdclef_runtime.training import run_training

DEFAULT_PRIMARY_METRIC = "val_soundscape_macro_roc_auc"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 Perch-head runtime.")
    parser.add_argument("--config", required=True, help="Path to runtime YAML config.")
    parser.add_argument("overrides", nargs="*", help="Optional dotted.key=value overrides.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config = apply_overrides(config, args.overrides)
    try:
        result = run_training(config, RUNTIME_ROOT)
    except Exception as exc:  # noqa: BLE001
        error_payload = {
            "experiment_name": config.get("experiment", {}).get("name", "unknown"),
            "config_path": config.get("_config_path", str(Path(args.config).resolve())),
            "primary_metric_name": config.get("metrics", {}).get("primary", DEFAULT_PRIMARY_METRIC),
            "primary_metric_value": 0.0,
            "secondary_metrics": {},
            "all_metrics": {},
            "root_cause": str(exc),
            "verdict": "runtime-failed",
            "artifacts": {},
            "dataset_summary": {},
            "summary_markdown": f"Runtime failed: {exc}",
            "error": str(exc),
        }
        run_dir = Path(os.environ.get("KAGGLE_AGENT_RUN_DIR", RUNTIME_ROOT / config["paths"]["output_root"] / config["experiment"]["name"])).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in [
            ("result.json", error_payload),
            ("metrics.json", {"primary": error_payload["primary_metric_name"], "metrics": {}}),
            ("artifacts.json", {}),
        ]:
            (run_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(error_payload, ensure_ascii=True))
        print(f"KAGGLE_AGENT_RESULT={json.dumps(error_payload, ensure_ascii=True)}")
        return 1
    print(json.dumps(result, ensure_ascii=True))
    print(
        "KAGGLE_AGENT_RESULT="
        + json.dumps(
            {
                "primary_metric_name": result["primary_metric_name"],
                "primary_metric_value": result["primary_metric_value"],
                "secondary_metrics": result["secondary_metrics"],
                "root_cause": result["root_cause"],
                "verdict": result["verdict"],
                "artifacts": result["artifacts"],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
