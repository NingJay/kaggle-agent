from __future__ import annotations

from typing import Any


class SchemaValidationError(ValueError):
    pass


def _matches_type(expected: str, value: Any) -> bool:
    if expected == "null":
        return value is None
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(value, float)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return True


def _validate_type(schema: dict[str, Any], value: Any, path: str) -> None:
    expected = schema.get("type")
    if expected is None:
        return
    expected_types = expected if isinstance(expected, list) else [expected]
    if any(_matches_type(str(item), value) for item in expected_types):
        return
    joined = ", ".join(str(item) for item in expected_types)
    raise SchemaValidationError(f"{path} expected type {joined}, got {type(value).__name__}")


def validate_payload(schema: dict[str, Any], value: Any, *, path: str = "$") -> None:
    _validate_type(schema, value, path)

    if "enum" in schema and value not in schema["enum"]:
        raise SchemaValidationError(f"{path} must be one of {schema['enum']!r}")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise SchemaValidationError(f"{path}.{key} is required")
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            next_path = f"{path}.{key}"
            if key in properties:
                validate_payload(properties[key], item, path=next_path)
                continue
            if additional is False:
                raise SchemaValidationError(f"{next_path} is not allowed")
            if isinstance(additional, dict):
                validate_payload(additional, item, path=next_path)
        return

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            raise SchemaValidationError(f"{path} must contain at least {min_items} items")
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for index, item in enumerate(value):
                validate_payload(items_schema, item, path=f"{path}[{index}]")

