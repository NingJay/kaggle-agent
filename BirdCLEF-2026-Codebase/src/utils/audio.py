from __future__ import annotations

from pathlib import Path


def count_audio_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*.ogg") if item.is_file())


def sample_audio_files(path: Path, limit: int = 5) -> list[str]:
    if not path.exists():
        return []
    samples: list[str] = []
    for item in sorted(path.rglob("*.ogg")):
        if item.is_file():
            samples.append(str(item))
        if len(samples) >= limit:
            break
    return samples

