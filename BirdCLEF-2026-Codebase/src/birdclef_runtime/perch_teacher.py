from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from birdclef_runtime.metrics import macro_roc_auc, padded_cmap

FILENAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


@dataclass(frozen=True)
class LabelProjection:
    labels: list[str]
    direct_positions: np.ndarray
    direct_model_indices: np.ndarray
    proxy_model_indices: dict[int, np.ndarray]


def _resolve_root(runtime_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = runtime_root / path
    return path.resolve()


def _parse_soundscape_filename(name: str) -> dict[str, object]:
    match = FILENAME_RE.match(name)
    if match is None:
        return {"site": None, "hour_utc": -1}
    _clip_id, site, _ymd, hms = match.groups()
    return {"site": site, "hour_utc": int(hms[:2])}


def _parse_labels(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [token.strip() for token in str(value).split(";") if token.strip()]


def _union_labels(series: pd.Series) -> list[str]:
    labels: set[str] = set()
    for value in series:
        labels.update(_parse_labels(value))
    return sorted(labels)


def _load_label_order(data_root: Path, data_cfg: dict[str, Any]) -> tuple[list[str], pd.DataFrame]:
    submission_df = pd.read_csv(data_root / data_cfg.get("sample_submission_csv", "sample_submission.csv"))
    taxonomy = pd.read_csv(data_root / data_cfg.get("taxonomy_csv", "taxonomy.csv"))
    return submission_df.columns[1:].tolist(), taxonomy


def _aggregate_soundscape_truth(
    data_root: Path,
    data_cfg: dict[str, Any],
    labels: list[str],
    *,
    windows_per_file: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    labels_df = pd.read_csv(data_root / data_cfg.get("train_soundscapes_labels_csv", "train_soundscapes_labels.csv"))
    labels_df = labels_df.drop_duplicates().reset_index(drop=True)
    label_to_index = {label: index for index, label in enumerate(labels)}
    soundscape_df = (
        labels_df.groupby(["filename", "start", "end"])["primary_label"]
        .apply(_union_labels)
        .reset_index(name="label_list")
    )
    soundscape_df["end_sec"] = pd.to_timedelta(soundscape_df["end"]).dt.total_seconds().astype(int)
    soundscape_df["row_id"] = (
        soundscape_df["filename"].str.replace(".ogg", "", regex=False) + "_" + soundscape_df["end_sec"].astype(str)
    )
    metadata_df = soundscape_df["filename"].apply(_parse_soundscape_filename).apply(pd.Series)
    soundscape_df = pd.concat([soundscape_df, metadata_df], axis=1)
    windows = soundscape_df.groupby("filename").size()
    full_files = sorted(windows[windows == windows_per_file].index.tolist())
    full_truth = (
        soundscape_df[soundscape_df["filename"].isin(full_files)]
        .sort_values(["filename", "end_sec"])
        .reset_index(drop=False)
    )
    targets = np.zeros((len(soundscape_df), len(labels)), dtype=np.uint8)
    for row_index, row_labels in enumerate(soundscape_df["label_list"]):
        for label in row_labels:
            if label in label_to_index:
                targets[row_index, label_to_index[label]] = 1
    return full_truth, targets


def build_label_projection(model_dir: Path, labels: list[str], taxonomy: pd.DataFrame) -> LabelProjection:
    labels_path = model_dir / "assets" / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing Perch labels.csv: {labels_path}")
    bc_labels = pd.read_csv(labels_path).reset_index().rename(columns={"index": "model_index"})
    source_column = next((column for column in bc_labels.columns if column != "model_index"), "")
    if not source_column:
        raise ValueError("Perch labels.csv did not expose a class-name column")
    bc_labels = bc_labels.rename(columns={source_column: "scientific_name"})
    bc_labels["scientific_name"] = bc_labels["scientific_name"].astype(str)

    taxonomy_lookup = taxonomy[["primary_label", "scientific_name"]].copy()
    taxonomy_lookup["primary_label"] = taxonomy_lookup["primary_label"].astype(str)
    taxonomy_lookup["scientific_name"] = taxonomy_lookup["scientific_name"].astype(str)
    label_frame = pd.DataFrame({"primary_label": [str(item) for item in labels]})
    mapping = label_frame.merge(taxonomy_lookup, on="primary_label", how="left").merge(
        bc_labels[["scientific_name", "model_index"]],
        on="scientific_name",
        how="left",
    )

    direct_positions: list[int] = []
    direct_model_indices: list[int] = []
    proxy_model_indices: dict[int, np.ndarray] = {}
    for position, row in mapping.reset_index(drop=True).iterrows():
        model_index = row.get("model_index")
        if pd.notna(model_index):
            direct_positions.append(position)
            direct_model_indices.append(int(model_index))
            continue
        scientific_name = str(row.get("scientific_name", "") or "").strip()
        genus = scientific_name.split()[0] if scientific_name else ""
        if not genus or "son" in str(row.get("primary_label", "")).lower():
            continue
        proxy_hits = bc_labels[
            bc_labels["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\s", na=False)
        ]["model_index"].astype(int)
        if not proxy_hits.empty:
            proxy_model_indices[position] = proxy_hits.to_numpy(dtype=np.int32)

    return LabelProjection(
        labels=[str(item) for item in labels],
        direct_positions=np.array(direct_positions, dtype=np.int32),
        direct_model_indices=np.array(direct_model_indices, dtype=np.int32),
        proxy_model_indices=proxy_model_indices,
    )


def read_soundscape_60s(path: Path, *, sample_rate: int, file_seconds: int) -> np.ndarray:
    import librosa
    import soundfile as sf

    waveform, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if hasattr(waveform, "ndim") and waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sr != sample_rate:
        waveform = librosa.resample(np.asarray(waveform, dtype=np.float32), orig_sr=sr, target_sr=sample_rate)
    waveform = np.asarray(waveform, dtype=np.float32)
    file_samples = int(sample_rate * file_seconds)
    if len(waveform) < file_samples:
        waveform = np.pad(waveform, (0, file_samples - len(waveform)))
    return waveform[:file_samples]


class PerchSavedModel:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self._infer_fn: Any | None = None

    def _load_infer_fn(self) -> Any:
        if self._infer_fn is not None:
            return self._infer_fn
        import tensorflow as tf

        loaded = tf.saved_model.load(str(self.model_dir))
        self._infer_fn = loaded.signatures.get("serving_default") or loaded
        return self._infer_fn

    def infer(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import tensorflow as tf

        infer_fn = self._load_infer_fn()
        outputs = infer_fn(inputs=tf.convert_to_tensor(windows, dtype=tf.float32))
        if not isinstance(outputs, dict):
            raise RuntimeError("Perch SavedModel did not return a dict payload")
        return outputs["label"].numpy().astype(np.float32), outputs["embedding"].numpy().astype(np.float32)


def infer_perch_soundscapes(
    paths: list[Path],
    *,
    model_dir: Path,
    projection: LabelProjection,
    sample_rate: int,
    window_seconds: int,
    file_seconds: int,
    batch_files: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    n_windows = int(file_seconds // window_seconds)
    teacher = PerchSavedModel(model_dir)
    n_files = len(paths)
    n_rows = n_files * n_windows

    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.empty(n_rows, dtype=np.int16)
    scores = np.zeros((n_rows, len(projection.labels)), dtype=np.float32)
    embeddings = np.zeros((n_rows, 1536), dtype=np.float32)

    window_samples = sample_rate * window_seconds
    write_row = 0
    for start in range(0, n_files, batch_files):
        batch_paths = paths[start : start + batch_files]
        batch_windows = np.empty((len(batch_paths) * n_windows, window_samples), dtype=np.float32)
        batch_start = write_row
        for batch_index, path in enumerate(batch_paths):
            audio = read_soundscape_60s(path, sample_rate=sample_rate, file_seconds=file_seconds)
            batch_windows[batch_index * n_windows : (batch_index + 1) * n_windows] = audio.reshape(n_windows, window_samples)
            metadata = _parse_soundscape_filename(path.name)
            row_ids[write_row : write_row + n_windows] = [f"{path.stem}_{offset}" for offset in range(window_seconds, file_seconds + 1, window_seconds)]
            filenames[write_row : write_row + n_windows] = path.name
            sites[write_row : write_row + n_windows] = metadata["site"]
            hours[write_row : write_row + n_windows] = metadata["hour_utc"]
            write_row += n_windows

        logits, batch_embeddings = teacher.infer(batch_windows)
        batch_slice = slice(batch_start, write_row)
        scores[batch_slice, projection.direct_positions] = logits[:, projection.direct_model_indices]
        for position, proxy_indices in projection.proxy_model_indices.items():
            scores[batch_slice, position] = logits[:, proxy_indices].max(axis=1)
        embeddings[batch_slice] = batch_embeddings

    meta_df = pd.DataFrame(
        {
            "row_id": row_ids,
            "filename": filenames,
            "site": sites,
            "hour_utc": hours,
        }
    )
    return meta_df, scores, embeddings


def run_perch_teacher_cache(config: dict[str, Any], runtime_root: Path, run_dir: Path) -> dict[str, Any]:
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})

    data_root = _resolve_root(runtime_root, str(paths_cfg.get("data_root", "./birdclef-2026")))
    model_dir = _resolve_root(runtime_root, str(model_cfg.get("perch_model_dir", "")))
    if not model_dir.exists():
        raise FileNotFoundError(f"Perch model directory not found: {model_dir}")

    split_name = str(training_cfg.get("cache_split", "train_soundscapes"))
    soundscape_dir_key = "train_soundscapes_dir" if split_name == "train_soundscapes" else "test_soundscapes_dir"
    soundscape_dir = data_root / data_cfg.get(soundscape_dir_key, split_name)
    if not soundscape_dir.exists():
        raise FileNotFoundError(f"Soundscape directory not found: {soundscape_dir}")

    labels, taxonomy = _load_label_order(data_root, data_cfg)
    projection = build_label_projection(model_dir, labels, taxonomy)

    paths = sorted(soundscape_dir.glob("*.ogg"))
    sample_rate = int(training_cfg.get("sample_rate", 32000))
    window_seconds = int(training_cfg.get("window_seconds", 5))
    file_seconds = int(training_cfg.get("file_seconds", 60))
    batch_files = int(training_cfg.get("batch_files", 4))
    full_truth: pd.DataFrame | None = None
    soundscape_targets: np.ndarray | None = None
    if split_name == "train_soundscapes":
        full_truth, soundscape_targets = _aggregate_soundscape_truth(
            data_root,
            data_cfg,
            labels,
            windows_per_file=int(file_seconds // window_seconds),
        )
        labeled_filenames = set(full_truth["filename"].astype(str).tolist())
        paths = [path for path in paths if path.name in labeled_filenames]

    max_files = int(data_cfg.get("max_soundscape_files", 0) or 0)
    if max_files > 0:
        paths = paths[:max_files]
    if not paths:
        raise ValueError(f"No soundscapes found in {soundscape_dir}")

    meta_df, scores, embeddings = infer_perch_soundscapes(
        paths,
        model_dir=model_dir,
        projection=projection,
        sample_rate=sample_rate,
        window_seconds=window_seconds,
        file_seconds=file_seconds,
        batch_files=batch_files,
    )

    cache_dir = run_dir / "perch_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "full_perch_meta.parquet"
    arrays_path = cache_dir / "full_perch_arrays.npz"
    preds_path = cache_dir / "teacher_window_predictions.csv"
    meta_df.to_parquet(meta_path, index=False)
    np.savez_compressed(arrays_path, scores_full_raw=scores, emb_full=embeddings)
    predictions_df = pd.DataFrame(scores, columns=labels)
    predictions_df.insert(0, "row_id", meta_df["row_id"])
    predictions_df.to_csv(preds_path, index=False)

    metrics: dict[str, float] = {
        "teacher_cache_files": float(len(paths)),
        "teacher_cache_rows": float(len(meta_df)),
    }
    root_cause = "Notebook-faithful Perch teacher, cache export, and 5s window inference are wired in."
    if split_name == "train_soundscapes" and full_truth is not None and soundscape_targets is not None:
        aligned_truth = full_truth.set_index("row_id").loc[meta_df["row_id"]].reset_index(drop=False)
        targets = soundscape_targets[aligned_truth["index"].to_numpy()]
        metrics["soundscape_macro_roc_auc"] = macro_roc_auc(targets.tolist(), scores.tolist())
        metrics["padded_cmap"] = padded_cmap(targets.tolist(), scores.tolist())
        root_cause = "Notebook-faithful Perch teacher/cache path is running on 60s soundscapes and produces aligned 5s teacher outputs."

    summary_markdown = "\n".join(
        [
            "## Perch Teacher Cache Summary",
            f"- split: `{split_name}`",
            f"- files: {len(paths)}",
            f"- rows: {len(meta_df)}",
            f"- direct-mapped classes: {len(projection.direct_positions)}",
            f"- proxy-mapped classes: {len(projection.proxy_model_indices)}",
            *(f"- {name}={value:.6f}" for name, value in metrics.items() if "cache_" not in name),
        ]
    )
    return {
        "metrics": metrics,
        "root_cause": root_cause,
        "verdict": "baseline-ready",
        "artifacts": {
            "perch_meta": str(meta_path),
            "perch_arrays": str(arrays_path),
            "teacher_predictions": str(preds_path),
        },
        "dataset_summary": {
            "soundscape_split": split_name,
            "file_count": len(paths),
            "row_count": len(meta_df),
            "label_count": len(labels),
        },
        "summary_markdown": summary_markdown,
    }
