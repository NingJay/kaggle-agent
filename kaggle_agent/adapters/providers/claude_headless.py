from __future__ import annotations

import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from kaggle_agent.adapters.command import parse_json_payload
from kaggle_agent.adapters.providers import ProviderResponse, ProviderUnavailable
from kaggle_agent.adapters.providers.claude_runtime import claude_subprocess_env


CLAUDE_BINARY = "claude"


@lru_cache(maxsize=1)
def _help_text() -> str:
    binary = shutil.which(CLAUDE_BINARY)
    if not binary:
        return ""
    completed = subprocess.run(
        [binary, "-p", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    return (completed.stdout or "") + (completed.stderr or "")


def _supports_flag(flag: str) -> bool:
    return flag in _help_text()


def _structured_payload(envelope: dict[str, Any]) -> dict[str, Any]:
    payload = envelope.get("structured_output")
    if isinstance(payload, dict):
        return payload
    result = envelope.get("result")
    if isinstance(result, str) and result.strip():
        try:
            parsed = parse_json_payload(result)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    return envelope


def run_claude_headless(
    *,
    prompt: str,
    schema: dict[str, Any],
    workspace_root: Path,
) -> ProviderResponse:
    binary = shutil.which(CLAUDE_BINARY)
    if not binary:
        raise ProviderUnavailable("claude binary is not available on PATH")

    args = [
        binary,
        "-p",
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(schema, separators=(",", ":"), ensure_ascii=False),
        "--tools",
        "",
        "--append-system-prompt",
        "Return only structured JSON matching the schema. Use only the supplied context. Do not call tools.",
    ]
    if _supports_flag("--no-session-persistence"):
        args.append("--no-session-persistence")
    if _supports_flag("--disable-slash-commands"):
        args.append("--disable-slash-commands")
    if _supports_flag("--no-chrome"):
        args.append("--no-chrome")
    if _supports_flag("--bare"):
        args.append("--bare")
    model = os.environ.get("KAGGLE_AGENT_CLAUDE_MODEL", "").strip()
    if model:
        args.extend(["--model", model])
    effort = os.environ.get("KAGGLE_AGENT_CLAUDE_EFFORT", "").strip()
    if effort:
        args.extend(["--effort", effort])

    with claude_subprocess_env(isolate_home_env_var="KAGGLE_AGENT_CLAUDE_ISOLATE_HOME") as env:
        completed = subprocess.run(
            args,
            input=prompt,
            capture_output=True,
            text=True,
            cwd=workspace_root,
            env=env,
            check=False,
        )
    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "claude invocation failed").strip()
        raise RuntimeError(stderr)
    try:
        envelope = parse_json_payload(completed.stdout)
    except json.JSONDecodeError as error:
        snippet = (completed.stdout or completed.stderr or "").strip()[:400] or "<empty>"
        raise RuntimeError(f"claude returned non-JSON output: {snippet}") from error
    payload = _structured_payload(envelope)
    if not isinstance(payload, dict):
        raise RuntimeError("claude did not return a structured object")
    return ProviderResponse(
        provider="claude",
        model=str(envelope.get("model", model)),
        session_id=str(envelope.get("session_id", "")),
        payload=payload,
        raw_stdout=completed.stdout,
        raw_stderr=completed.stderr,
        exit_code=completed.returncode,
        extra_meta={"isolated_home": os.environ.get("KAGGLE_AGENT_CLAUDE_ISOLATE_HOME", "1")},
    )
