from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


AMP_BINARY = "amp"


@dataclass
class AmpProbeResult:
    summary: str
    thread_id: str
    raw_stdout: str
    raw_stderr: str
    event_log_text: str
    exit_code: int


def _first_string(obj: Any, key: str) -> str:
    if isinstance(obj, dict):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
        for item in obj.values():
            found = _first_string(item, key)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _first_string(item, key)
            if found:
                return found
    return ""


def _assistant_summary(stream_json: str) -> tuple[str, str]:
    summary = ""
    thread_id = ""
    for line in stream_json.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not thread_id:
            thread_id = _first_string(event, "threadId") or _first_string(event, "thread_id")
        if event.get("type") != "assistant":
            continue
        message = event.get("message", {})
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        parts = [str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text"]
        if parts:
            summary = "\n".join(part.strip() for part in parts if part.strip()).strip()
    return summary, thread_id


def run_amp_probe(*, prompt: str, workspace_root: Path) -> AmpProbeResult | None:
    binary = shutil.which(AMP_BINARY)
    if not binary or not os.environ.get("AMP_API_KEY"):
        return None
    args = [
        binary,
        "--no-ide",
        "--no-jetbrains",
        "-x",
        "--stream-json",
    ]
    completed = subprocess.run(
        args,
        input=prompt,
        capture_output=True,
        text=True,
        cwd=workspace_root,
        check=False,
    )
    summary, thread_id = _assistant_summary(completed.stdout)
    return AmpProbeResult(
        summary=summary,
        thread_id=thread_id,
        raw_stdout=completed.stdout,
        raw_stderr=completed.stderr,
        event_log_text=completed.stdout,
        exit_code=completed.returncode,
    )

