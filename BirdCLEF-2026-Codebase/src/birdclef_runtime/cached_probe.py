from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from birdclef_runtime.metrics import padded_cmap

FILENAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


def _resolve_root(runtime_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = runtime_root / path
    return path.resolve()


def _parse_labels(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [token.strip() for token in str(value).split(";") if token.strip()]


def _union_labels(series: pd.Series) -> list[str]:
    values: set[str] = set()
    for item in series:
        values.update(_parse_labels(item))
    return sorted(values)


def _parse_soundscape_filename(name: str) -> dict[str, object]:
    match = FILENAME_RE.match(name)
    if match is None:
        return {"site": None, "hour_utc": -1}
    _clip_id, site, _ymd, hms = match.groups()
    return {"site": site, "hour_utc": int(hms[:2])}


def _load_truth_matrix(data_root: Path, data_cfg: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    labels_df = pd.read_csv(data_root / data_cfg.get("train_soundscapes_labels_csv", "train_soundscapes_labels.csv"))
    labels_df = labels_df.drop_duplicates().reset_index(drop=True)
    submission_df = pd.read_csv(data_root / data_cfg.get("sample_submission_csv", "sample_submission.csv"))
    labels = submission_df.columns[1:].tolist()
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

    windows_per_file = soundscape_df.groupby("filename").size()
    full_files = sorted(windows_per_file[windows_per_file == 12].index.tolist())
    soundscape_df["file_fully_labeled"] = soundscape_df["filename"].isin(full_files)
    full_truth = soundscape_df[soundscape_df["file_fully_labeled"]].sort_values(["filename", "end_sec"]).reset_index(drop=False)

    targets = np.zeros((len(soundscape_df), len(labels)), dtype=np.uint8)
    for row_index, row_labels in enumerate(soundscape_df["label_list"]):
        for label in row_labels:
            if label in label_to_index:
                targets[row_index, label_to_index[label]] = 1
    return full_truth, targets, labels


def _fit_oof_probe(
    embeddings: np.ndarray,
    raw_scores: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    pca_dim: int,
    min_pos: int,
    probe_c: float,
    max_iter: int,
    n_splits: int,
    use_raw_scores: bool,
) -> tuple[np.ndarray, StandardScaler, PCA, int]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=min(pca_dim, scaled.shape[0] - 1, scaled.shape[1]))
    reduced = pca.fit_transform(scaled).astype(np.float32)
    splitter = GroupKFold(n_splits=n_splits)
    oof_predictions = raw_scores.copy() if use_raw_scores else np.zeros_like(targets, dtype=np.float32)
    fitted_classes = 0

    for _fold, (_, val_idx) in enumerate(splitter.split(reduced, groups=groups), start=1):
        train_idx = np.setdiff1d(np.arange(len(reduced)), val_idx)
        positive_counts = targets[train_idx].sum(axis=0)
        for class_index in np.where(positive_counts >= min_pos)[0]:
            train_targets = targets[train_idx, class_index]
            if train_targets.sum() == 0 or train_targets.sum() == len(train_targets):
                continue
            train_features = reduced[train_idx]
            val_features = reduced[val_idx]
            if use_raw_scores:
                train_features = np.concatenate([train_features, raw_scores[train_idx, class_index : class_index + 1]], axis=1)
                val_features = np.concatenate([val_features, raw_scores[val_idx, class_index : class_index + 1]], axis=1)
            classifier = LogisticRegression(
                C=probe_c,
                max_iter=max_iter,
                solver="liblinear",
                class_weight="balanced",
            )
            classifier.fit(train_features, train_targets)
            oof_predictions[val_idx, class_index] = classifier.predict_proba(val_features)[:, 1].astype(np.float32)
            fitted_classes += 1
    return oof_predictions, scaler, pca, fitted_classes


def _fit_full_models(
    reduced_embeddings: np.ndarray,
    raw_scores: np.ndarray,
    targets: np.ndarray,
    *,
    min_pos: int,
    probe_c: float,
    max_iter: int,
    use_raw_scores: bool,
) -> dict[int, LogisticRegression]:
    models: dict[int, LogisticRegression] = {}
    positive_counts = targets.sum(axis=0)
    for class_index in np.where(positive_counts >= min_pos)[0]:
        class_targets = targets[:, class_index]
        if class_targets.sum() == 0 or class_targets.sum() == len(class_targets):
            continue
        features = reduced_embeddings
        if use_raw_scores:
            features = np.concatenate([features, raw_scores[:, class_index : class_index + 1]], axis=1)
        classifier = LogisticRegression(
            C=probe_c,
            max_iter=max_iter,
            solver="liblinear",
            class_weight="balanced",
        )
        classifier.fit(features, class_targets)
        models[int(class_index)] = classifier
    return models


def _root_cause(primary_metric: float, fitted_classes: int, active_classes: int) -> str:
    if primary_metric >= 0.75:
        return "Cached raw+embedding probe is competitive; the next gap is stronger calibration and prior fusion."
    if fitted_classes < active_classes:
        return "The cached probe is bottlenecked by sparse positives for many active classes; lower min_pos or add class-aware priors."
    return "The cached raw+embedding probe is working, but it still lacks Bayesian prior fusion and richer temporal context from the reference notebook."


def run_cached_probe_experiment(config: dict[str, Any], runtime_root: Path, run_dir: Path) -> dict[str, Any]:
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})

    data_root = _resolve_root(runtime_root, str(paths_cfg.get("data_root", "./birdclef-2026")))
    cache_root_value = str(data_cfg.get("perch_cache_dir", "")).strip()
    if not cache_root_value:
        raise ValueError("data.perch_cache_dir is required for sklearn_cached_probe runs")
    cache_root = _resolve_root(runtime_root, cache_root_value)

    meta_path = cache_root / "full_perch_meta.parquet"
    arrays_path = cache_root / "full_perch_arrays.npz"
    if not meta_path.exists() or not arrays_path.exists():
        raise FileNotFoundError(f"Missing cached Perch artifacts in {cache_root}")

    full_truth, soundscape_targets, labels = _load_truth_matrix(data_root, data_cfg)
    meta_df = pd.read_parquet(meta_path)
    cached = np.load(arrays_path)
    raw_scores = cached["scores_full_raw"].astype(np.float32)
    embeddings = cached["emb_full"].astype(np.float32)
    aligned_truth = full_truth.set_index("row_id").loc[meta_df["row_id"]].reset_index(drop=False)
    targets = soundscape_targets[aligned_truth["index"].to_numpy()]
    groups = meta_df["site"].fillna("unknown").astype(str).to_numpy()

    probe_pca_dim = int(training_cfg.get("probe_pca_dim", 32))
    probe_min_pos = int(training_cfg.get("probe_min_pos", 8))
    probe_c = float(training_cfg.get("probe_c", 0.25))
    probe_group_folds = int(training_cfg.get("probe_group_folds", 5))
    max_iter = int(training_cfg.get("max_iter", 400))
    use_raw_scores = bool(training_cfg.get("probe_use_raw_scores", True))

    oof_predictions, scaler, pca, fitted_classes = _fit_oof_probe(
        embeddings,
        raw_scores,
        targets,
        groups,
        pca_dim=probe_pca_dim,
        min_pos=probe_min_pos,
        probe_c=probe_c,
        max_iter=max_iter,
        n_splits=probe_group_folds,
        use_raw_scores=use_raw_scores,
    )
    active_mask = targets.sum(axis=0) > 0
    macro_auc = float(roc_auc_score(targets[:, active_mask], oof_predictions[:, active_mask], average="macro"))
    cmap = float(padded_cmap(targets.tolist(), oof_predictions.tolist()))

    reduced_embeddings = pca.transform(scaler.transform(embeddings)).astype(np.float32)
    models = _fit_full_models(
        reduced_embeddings,
        raw_scores,
        targets,
        min_pos=probe_min_pos,
        probe_c=probe_c,
        max_iter=max_iter,
        use_raw_scores=use_raw_scores,
    )

    oof_path = run_dir / "oof_predictions.npz"
    np.savez_compressed(
        oof_path,
        row_ids=meta_df["row_id"].to_numpy(),
        targets=targets.astype(np.uint8),
        oof_predictions=oof_predictions.astype(np.float32),
        raw_scores=raw_scores.astype(np.float32),
    )
    bundle_path = run_dir / "probe_bundle.pkl"
    with bundle_path.open("wb") as handle:
        pickle.dump(
            {
                "labels": labels,
                "scaler": scaler,
                "pca": pca,
                "models": models,
                "probe_use_raw_scores": use_raw_scores,
                "min_pos": probe_min_pos,
                "probe_c": probe_c,
            },
            handle,
        )
    probe_metadata_path = run_dir / "probe_metadata.json"
    probe_metadata_path.write_text(
        pd.Series(
            {
                "probe_pca_dim": probe_pca_dim,
                "probe_min_pos": probe_min_pos,
                "probe_c": probe_c,
                "probe_group_folds": probe_group_folds,
                "use_raw_scores": use_raw_scores,
                "fitted_classes": fitted_classes,
                "active_classes": int(active_mask.sum()),
                "cache_rows": int(len(meta_df)),
                "full_soundscape_windows": int(len(aligned_truth)),
            }
        ).to_json(indent=2),
        encoding="utf-8",
    )

    root_cause = _root_cause(macro_auc, len(models), int(active_mask.sum()))
    summary_markdown = "\n".join(
        [
            "## Runtime Summary",
            "- Backend: sklearn_cached_probe",
            f"- Cache root: {cache_root}",
            f"- Cached rows: {len(meta_df)}",
            f"- Fully labeled windows: {len(aligned_truth)}",
            f"- Active classes: {int(active_mask.sum())}",
            f"- Fitted classes: {len(models)}",
            f"- Probe PCA dim: {probe_pca_dim}",
            f"- Probe min positives: {probe_min_pos}",
            f"- Probe C: {probe_c}",
            f"- Use raw Perch scores: {use_raw_scores}",
            f"- soundscape_macro_roc_auc={macro_auc:.6f}",
            f"- padded_cmap={cmap:.6f}",
        ]
    )
    return {
        "metrics": {
            "soundscape_macro_roc_auc": macro_auc,
            "padded_cmap": cmap,
        },
        "artifacts": {
            "oof_predictions": str(oof_path),
            "probe_bundle": str(bundle_path),
            "probe_metadata": str(probe_metadata_path),
        },
        "dataset_summary": {
            "cache_row_count": int(len(meta_df)),
            "fully_labeled_windows": int(len(aligned_truth)),
            "fully_labeled_files": int(aligned_truth["filename"].nunique()),
            "active_class_count": int(active_mask.sum()),
            "fitted_class_count": int(len(models)),
        },
        "root_cause": root_cause,
        "summary_markdown": summary_markdown,
        "verdict": "baseline-ready" if macro_auc >= 0.55 else "needs-tuning",
    }
