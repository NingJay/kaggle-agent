from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def _falsey(value: str) -> bool:
    return value.strip().lower() in {"0", "false", "no"}


def load_user_claude_settings() -> dict[str, Any]:
    settings_path = Path.home() / ".claude" / "settings.json"
    if not settings_path.exists():
        return {}
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


@contextmanager
def claude_subprocess_env(*, isolate_home_env_var: str) -> Iterator[dict[str, str]]:
    env = os.environ.copy()
    env.pop("CLAUDE_CODE_SSE_PORT", None)

    settings = load_user_claude_settings()
    settings_env = settings.get("env", {})
    if isinstance(settings_env, dict):
        for key, value in settings_env.items():
            if isinstance(key, str) and isinstance(value, str):
                env.setdefault(key, value)

    if _falsey(os.environ.get(isolate_home_env_var, "1")):
        yield env
        return

    with tempfile.TemporaryDirectory(prefix="kaggle-agent-claude-") as tmp:
        isolated_home = Path(tmp)
        claude_dir = isolated_home / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)
        minimal_settings: dict[str, Any] = {}
        if isinstance(settings_env, dict):
            minimal_settings["env"] = {
                str(key): str(value)
                for key, value in settings_env.items()
                if isinstance(key, str) and isinstance(value, str)
            }
        model = settings.get("model")
        if isinstance(model, str) and model.strip():
            minimal_settings["model"] = model.strip()
        if minimal_settings:
            (claude_dir / "settings.json").write_text(
                json.dumps(minimal_settings, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        (claude_dir / "config.json").write_text(
            '{"hasCompletedOnboarding":true}\n',
            encoding="utf-8",
        )
        env["HOME"] = str(isolated_home)
        yield env
