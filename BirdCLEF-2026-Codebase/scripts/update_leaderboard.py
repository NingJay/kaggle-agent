#!/usr/bin/env python3
"""Update experiment leaderboard with new results."""
import json
import sys
from pathlib import Path
from datetime import datetime

def update_leaderboard(result_json_path: Path):
    result = json.loads(result_json_path.read_text())

    leaderboard_path = Path(__file__).parent.parent / "experiments" / "leaderboard.json"
    leaderboard_path.parent.mkdir(exist_ok=True)

    leaderboard = json.loads(leaderboard_path.read_text()) if leaderboard_path.exists() else []

    entry = {
        "run_name": result_json_path.parent.name,
        "timestamp": datetime.now().isoformat(),
        "primary_metric": result["metrics"].get("val_soundscape_macro_roc_auc") or result["metrics"].get("prior_fusion_macro_roc_auc"),
        "verdict": result["verdict"],
        "fitted_classes": result["dataset_summary"]["fitted_class_count"],
        "kaggle_lb_score": None,  # To be filled after submission
    }

    leaderboard.append(entry)
    leaderboard.sort(key=lambda x: x["primary_metric"], reverse=True)

    leaderboard_path.write_text(json.dumps(leaderboard, indent=2))
    print(f"Updated leaderboard: {entry['run_name']} → {entry['primary_metric']:.4f}")

if __name__ == "__main__":
    update_leaderboard(Path(sys.argv[1]))
