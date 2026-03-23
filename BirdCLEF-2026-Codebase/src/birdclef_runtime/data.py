from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Sample:
    sample_id: str
    audio_path: Path
    target: list[int]
    metadata: dict[str, Any]


def _read_csv_rows(path: Path, max_rows: int | None = None) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            rows.append({str(key): str(value) for key, value in row.items()})
            if max_rows is not None and index + 1 >= max_rows:
                break
    return rows


def _label_column(rows: list[dict[str, str]]) -> str | None:
    if not rows:
        return None
    for candidate in ["primary_label", "label", "scientific_name", "species", "target"]:
        if candidate in rows[0]:
            return candidate
    return None


def _resolve_train_audio_path(data_root: Path, train_audio_dir: str, row: dict[str, str], label_column: str | None) -> Path:
    if row.get("filepath"):
        return (data_root / row["filepath"]).resolve()
    filename = row.get("filename") or row.get("path") or row.get("audio")
    label = row.get(label_column or "", "")
    if filename and label:
        return (data_root / train_audio_dir / label / filename).resolve()
    if filename:
        return (data_root / train_audio_dir / filename).resolve()
    return (data_root / train_audio_dir / f"missing-{hashlib.sha1(str(row).encode()).hexdigest()[:8]}.ogg").resolve()


def _resolve_soundscape_path(data_root: Path, soundscape_dir: str, row: dict[str, str]) -> Path:
    if row.get("filepath"):
        return (data_root / row["filepath"]).resolve()
    filename = row.get("filename") or row.get("soundscape") or row.get("audio") or row.get("row_id", "soundscape")
    if not filename.endswith(".ogg"):
        filename = f"{filename}.ogg"
    return (data_root / soundscape_dir / filename).resolve()


def _build_vocab(train_rows: list[dict[str, str]], taxonomy_rows: list[dict[str, str]], soundscape_rows: list[dict[str, str]], label_column: str | None) -> list[str]:
    labels: set[str] = set()
    for row in taxonomy_rows:
        for candidate in ["primary_label", "label", "scientific_name", "species"]:
            value = row.get(candidate)
            if value:
                labels.add(value)
                break
    for row in train_rows:
        if label_column and row.get(label_column):
            labels.add(row[label_column])
    for row in soundscape_rows:
        value = row.get("primary_label") or row.get("label") or row.get("species")
        if value:
            labels.add(value)
    return sorted(labels)


def _one_hot(index: int, size: int) -> list[int]:
    target = [0] * size
    if 0 <= index < size:
        target[index] = 1
    return target


def _hash_bucket(text: str, modulo: int = 100) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) % modulo


def build_dataset(config: dict[str, Any], runtime_root: Path) -> dict[str, Any]:
    paths = config["paths"]
    data_cfg = config.get("data", {})
    data_root = (runtime_root / paths["data_root"]).resolve()
    train_rows = _read_csv_rows(data_root / data_cfg.get("train_csv", "train.csv"), data_cfg.get("max_train_rows"))
    taxonomy_rows = _read_csv_rows(data_root / data_cfg.get("taxonomy_csv", "taxonomy.csv"))
    soundscape_rows = _read_csv_rows(
        data_root / data_cfg.get("train_soundscapes_labels_csv", "train_soundscapes_labels.csv"),
        data_cfg.get("max_val_rows"),
    )
    label_column = _label_column(train_rows)
    labels = _build_vocab(train_rows, taxonomy_rows, soundscape_rows, label_column)
    label_to_index = {label: index for index, label in enumerate(labels)}
    allow_missing = bool(data_cfg.get("allow_missing", False))

    train_samples: list[Sample] = []
    holdout_samples: list[Sample] = []
    for row in train_rows:
        label = row.get(label_column or "", "")
        if not label or label not in label_to_index:
            continue
        audio_path = _resolve_train_audio_path(data_root, data_cfg.get("train_audio_dir", "train_audio"), row, label_column)
        if not audio_path.exists() and not allow_missing:
            raise FileNotFoundError(f"Missing train audio: {audio_path}")
        sample = Sample(
            sample_id=row.get("id") or row.get("filename") or audio_path.stem,
            audio_path=audio_path,
            target=_one_hot(label_to_index[label], len(labels)),
            metadata={"source": "train_audio", "label": label},
        )
        if _hash_bucket(sample.sample_id) < 80:
            train_samples.append(sample)
        else:
            holdout_samples.append(sample)

    grouped_soundscapes: dict[str, Sample] = {}
    for row in soundscape_rows:
        label = row.get("primary_label") or row.get("label") or row.get("species")
        if not label or label not in label_to_index:
            continue
        sample_key = row.get("row_id") or row.get("filename") or row.get("soundscape") or label
        sample = grouped_soundscapes.get(sample_key)
        if sample is None:
            audio_path = _resolve_soundscape_path(data_root, data_cfg.get("train_soundscapes_dir", "train_soundscapes"), row)
            if not audio_path.exists() and not allow_missing:
                raise FileNotFoundError(f"Missing soundscape audio: {audio_path}")
            sample = Sample(
                sample_id=sample_key,
                audio_path=audio_path,
                target=[0] * len(labels),
                metadata={"source": "soundscape"},
            )
            grouped_soundscapes[sample_key] = sample
        sample.target[label_to_index[label]] = 1
    val_samples = list(grouped_soundscapes.values()) or holdout_samples

    if not train_samples:
        raise ValueError("No train samples resolved from train.csv.")
    if not val_samples:
        raise ValueError("No validation samples resolved from soundscape labels or holdout split.")

    return {
        "data_root": str(data_root),
        "labels": labels,
        "label_to_index": label_to_index,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "dataset_summary": {
            "train_sample_count": len(train_samples),
            "val_sample_count": len(val_samples),
            "label_count": len(labels),
            "has_soundscape_validation": bool(grouped_soundscapes),
        },
    }
