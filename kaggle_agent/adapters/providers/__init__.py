from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ProviderUnavailable(RuntimeError):
    pass


@dataclass
class ProviderResponse:
    provider: str
    payload: dict[str, Any]
    model: str = ""
    session_id: str = ""
    thread_id: str = ""
    raw_stdout: str = ""
    raw_stderr: str = ""
    event_log_text: str = ""
    exit_code: int = 0
    extra_meta: dict[str, Any] = field(default_factory=dict)

