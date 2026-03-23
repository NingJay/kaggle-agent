from __future__ import annotations

from math import exp


def sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def _average_precision(y_true: list[int], y_score: list[float]) -> float:
    ranked = sorted(zip(y_score, y_true), key=lambda item: item[0], reverse=True)
    hits = 0
    total = 0
    precision_sum = 0.0
    positives = sum(y_true)
    if positives == 0:
        return 0.0
    for _, truth in ranked:
        total += 1
        if truth:
            hits += 1
            precision_sum += hits / total
    return precision_sum / positives


def _binary_auc(y_true: list[int], y_score: list[float]) -> float:
    positives = [score for score, truth in zip(y_score, y_true) if truth]
    negatives = [score for score, truth in zip(y_score, y_true) if not truth]
    if not positives or not negatives:
        return 0.5
    comparisons = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                comparisons += 1.0
            elif positive == negative:
                comparisons += 0.5
    return comparisons / (len(positives) * len(negatives))


def macro_roc_auc(targets: list[list[int]], predictions: list[list[float]]) -> float:
    if not targets or not predictions:
        return 0.0
    label_count = len(targets[0])
    aucs = [_binary_auc([row[index] for row in targets], [row[index] for row in predictions]) for index in range(label_count)]
    return sum(aucs) / len(aucs)


def padded_cmap(targets: list[list[int]], predictions: list[list[float]], pad_floor: float = 0.0) -> float:
    if not targets or not predictions:
        return 0.0
    label_count = len(targets[0])
    aps = [
        max(_average_precision([row[index] for row in targets], [row[index] for row in predictions]), pad_floor)
        for index in range(label_count)
    ]
    return sum(aps) / len(aps)
