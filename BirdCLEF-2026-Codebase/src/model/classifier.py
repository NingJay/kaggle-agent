from __future__ import annotations

from typing import Any

from utils.metrics import bootstrap_readiness_ratio, entropy, metadata_proxy_score


def run_baseline(config: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    metric = config.get("reporting", {}).get("metric", "bootstrap_readiness")
    mode = config.get("training", {}).get("mode", "inventory")
    existing_paths = len(inventory["existing_paths"])
    required_paths = len(inventory["required_paths"])
    missing_paths = inventory["missing_paths"]

    if mode == "inventory":
        score = bootstrap_readiness_ratio(existing_paths, required_paths)
        root_cause = "Missing data/model paths." if missing_paths else "Inventory complete."
        verdict = "blocked-no-data" if missing_paths else "inventory-ready"
        return {
            "primary_score": score,
            "metric": metric,
            "root_cause": root_cause,
            "verdict": verdict,
            "artifacts": [],
            "summary": {
                "existing_paths": existing_paths,
                "required_paths": required_paths,
            },
        }

    label_counts = list(inventory["label_counts"].values())
    score = metadata_proxy_score(
        unique_labels=inventory["unique_label_count"],
        row_count=inventory["train_row_count"],
        soundscape_label_rows=inventory["soundscape_label_row_count"],
    )
    root_cause = "Missing training metadata or soundscape labels." if inventory["train_row_count"] == 0 else "Metadata-only bootstrap baseline."
    verdict = "blocked-no-data" if inventory["train_row_count"] == 0 else "bootstrap-baseline"
    return {
        "primary_score": score,
        "metric": metric,
        "root_cause": root_cause,
        "verdict": verdict,
        "artifacts": [],
        "summary": {
            "label_entropy": entropy(label_counts),
            "unique_label_count": inventory["unique_label_count"],
        },
    }

