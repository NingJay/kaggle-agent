from __future__ import annotations

import math


def bootstrap_readiness_ratio(existing_paths: int, required_paths: int) -> float:
    if required_paths <= 0:
        return 0.0
    return round(existing_paths / required_paths, 6)


def metadata_proxy_score(unique_labels: int, row_count: int, soundscape_label_rows: int) -> float:
    if row_count <= 0:
        return 0.0
    diversity = min(unique_labels / 234.0, 1.0)
    coverage = min(soundscape_label_rows / max(row_count, 1), 1.0)
    score = 0.7 * diversity + 0.3 * coverage
    return round(score, 6)


def entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    acc = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = count / total
        acc -= probability * math.log(probability, 2)
    return round(acc, 6)

