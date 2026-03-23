from __future__ import annotations

from typing import Any


def describe_augmentations(config: dict[str, Any]) -> list[str]:
    mode = config.get("training", {}).get("mode", "inventory")
    if mode == "metadata_prior":
        return ["none"]
    return ["inventory_only"]

