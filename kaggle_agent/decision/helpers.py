from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kaggle_agent.schema import RunRecord


def load_run_result(run: RunRecord) -> dict[str, Any]:
    path = run.artifact_paths.get("result", "")
    if not path:
        return {}
    result_path = Path(path)
    if not result_path.exists():
        return {}
    return json.loads(result_path.read_text(encoding="utf-8"))
