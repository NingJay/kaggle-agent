from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from utils.audio import count_audio_files, sample_audio_files


def read_csv_rows(path: Path, max_rows: int | None = None) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            rows.append({key: value for key, value in row.items()})
            if max_rows is not None and index + 1 >= max_rows:
                break
    return rows


def infer_label_key(rows: list[dict[str, str]]) -> str | None:
    if not rows:
        return None
    candidates = [
        "primary_label",
        "label",
        "scientific_name",
        "species",
        "target",
    ]
    for candidate in candidates:
        if candidate in rows[0]:
            return candidate
    return None


def build_inventory(config: dict[str, Any], runtime_root: Path) -> dict[str, Any]:
    paths = config["paths"]
    data_root = (runtime_root / paths["data_root"]).resolve()
    model_root = (runtime_root / paths["model_root"]).resolve()
    max_rows = config.get("data", {}).get("max_rows")

    train_csv = data_root / "train.csv"
    taxonomy_csv = data_root / "taxonomy.csv"
    sample_submission_csv = data_root / "sample_submission.csv"
    soundscape_labels_csv = data_root / "train_soundscapes_labels.csv"
    train_audio_dir = data_root / "train_audio"
    train_soundscapes_dir = data_root / "train_soundscapes"

    train_rows = read_csv_rows(train_csv, max_rows=max_rows)
    taxonomy_rows = read_csv_rows(taxonomy_csv, max_rows=max_rows)
    soundscape_label_rows = read_csv_rows(soundscape_labels_csv, max_rows=max_rows)
    label_key = infer_label_key(train_rows)
    label_counts = Counter(row.get(label_key, "") for row in train_rows if label_key and row.get(label_key))

    required_paths = {
        "data_root": data_root,
        "train_csv": train_csv,
        "taxonomy_csv": taxonomy_csv,
        "sample_submission_csv": sample_submission_csv,
        "train_audio_dir": train_audio_dir,
        "train_soundscapes_dir": train_soundscapes_dir,
        "model_root": model_root,
    }
    existing_paths = {name: str(path) for name, path in required_paths.items() if path.exists()}
    missing_paths = {name: str(path) for name, path in required_paths.items() if not path.exists()}
    return {
        "data_root": str(data_root),
        "model_root": str(model_root),
        "required_paths": {name: str(path) for name, path in required_paths.items()},
        "existing_paths": existing_paths,
        "missing_paths": missing_paths,
        "train_row_count": len(train_rows),
        "taxonomy_row_count": len(taxonomy_rows),
        "soundscape_label_row_count": len(soundscape_label_rows),
        "train_audio_file_count": count_audio_files(train_audio_dir),
        "train_soundscape_file_count": count_audio_files(train_soundscapes_dir),
        "sample_audio_files": sample_audio_files(train_audio_dir),
        "sample_soundscapes": sample_audio_files(train_soundscapes_dir),
        "label_key": label_key,
        "unique_label_count": len(label_counts),
        "label_histogram_top10": label_counts.most_common(10),
        "label_counts": dict(label_counts),
    }

