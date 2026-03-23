from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    text = config_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{config_path} must be JSON-compatible YAML. "
            "Use JSON syntax inside the .yaml file or install a YAML parser."
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must decode to an object.")
    payload["_config_path"] = str(config_path)
    return payload


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override: {override}")
        key_path, raw_value = override.split("=", maxsplit=1)
        value = _parse_scalar(raw_value)
        cursor = config
        parts = key_path.split(".")
        for part in parts[:-1]:
            child = cursor.get(part)
            if not isinstance(child, dict):
                child = {}
                cursor[part] = child
            cursor = child
        cursor[parts[-1]] = value
    return config

