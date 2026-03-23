from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Runtime config must be a mapping.")
    data["_config_path"] = str(config_path)
    return data


def _parse_value(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = dict(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override: {override}")
        key_path, raw_value = override.split("=", maxsplit=1)
        cursor: dict[str, Any] = updated
        keys = key_path.split(".")
        for part in keys[:-1]:
            child = cursor.get(part)
            if not isinstance(child, dict):
                child = {}
                cursor[part] = child
            cursor = child
        cursor[keys[-1]] = _parse_value(raw_value)
    return updated
