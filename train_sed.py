from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    runtime_root = Path(__file__).resolve().parent / "BirdCLEF-2026-Codebase"
    command = [sys.executable, str(runtime_root / "train.py"), *args]
    completed = subprocess.run(command, cwd=runtime_root, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

