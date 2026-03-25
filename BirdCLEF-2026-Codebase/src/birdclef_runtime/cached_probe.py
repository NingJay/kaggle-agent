from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from birdclef_runtime.metrics import padded_cmap

FILENAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")
TEXTURE_TAXA = {"Amphibia", "Insecta"}


@dataclass
class MappingContext:
    taxonomy: pd.DataFrame
    labels: list[str]
    class_name_map: dict[str, str]
    class_family: dict[int, str]
    family_idx_map: dict[str, np.ndarray]
    active_mask: np.ndarray
    idx_active_texture: np.ndarray
    idx_active_event: np.ndarray
    idx_mapped_active_texture: np.ndarray
    idx_mapped_active_event: np.ndarray
    idx_selected_proxy_active_texture: np.ndarray
    idx_selected_prioronly_active_texture: np.ndarray
    idx_selected_prioronly_active_event: np.ndarray
    idx_unmapped_inactive: np.ndarray


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


def _load_truth_matrix(data_root: Path, data_cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str], pd.DataFrame]:
    taxonomy = pd.read_csv(data_root / data_cfg.get("taxonomy_csv", "taxonomy.csv"))
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
    return full_truth, soundscape_df, targets, labels, taxonomy


def _build_mapping_context(
    *,
    labels: list[str],
    targets: np.ndarray,
    taxonomy: pd.DataFrame,
    runtime_root: Path,
    config: dict[str, Any],
) -> MappingContext:
    class_name_map = taxonomy.set_index("primary_label")["class_name"].astype(str).to_dict()
    class_family = {index: class_name_map.get(label, "Unknown") for index, label in enumerate(labels)}
    family_groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        family_groups.setdefault(class_name_map.get(label, "Unknown"), []).append(index)
    family_idx_map = {name: np.array(indices, dtype=np.int32) for name, indices in family_groups.items()}

    active_mask = targets.sum(axis=0) > 0
    active_indices = np.where(active_mask)[0].astype(np.int32)
    idx_active_texture = np.array(
        [index for index in active_indices if class_name_map.get(labels[index]) in TEXTURE_TAXA],
        dtype=np.int32,
    )
    idx_active_event = np.array(
        [index for index in active_indices if class_name_map.get(labels[index]) not in TEXTURE_TAXA],
        dtype=np.int32,
    )

    mapped_mask = np.ones(len(labels), dtype=bool)
    selected_proxy_pos = np.array([], dtype=np.int32)
    if config.get("model", {}).get("perch_model_dir", ""):
        model_dir = _resolve_root(runtime_root, str(config["model"]["perch_model_dir"]))
        bc_labels_path = model_dir / "assets" / "labels.csv"
        if bc_labels_path.exists() and "scientific_name" in taxonomy.columns:
            bc_labels = (
                pd.read_csv(bc_labels_path)
                .reset_index()
                .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
            )
            no_label_index = len(bc_labels)
            taxonomy_ = taxonomy[["primary_label", "scientific_name"]].copy()
            taxonomy_["scientific_name"] = taxonomy_["scientific_name"].astype(str)
            mapping = taxonomy_.merge(bc_labels[["scientific_name", "bc_index"]], on="scientific_name", how="left")
            mapping["bc_index"] = mapping["bc_index"].fillna(no_label_index).astype(int)
            label_to_bc = mapping.set_index("primary_label")["bc_index"].to_dict()
            bc_indices = np.array([int(label_to_bc.get(label, no_label_index)) for label in labels], dtype=np.int32)
            mapped_mask = bc_indices != no_label_index

            proxy_map: dict[str, list[int]] = {}
            unmapped_df = mapping[mapping["bc_index"] == no_label_index].copy()
            unmapped_non_sonotype = unmapped_df[
                ~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)
            ].copy()
            for _, row in unmapped_non_sonotype.iterrows():
                scientific_name = str(row.get("scientific_name", "")).strip()
                genus = scientific_name.split()[0] if scientific_name else ""
                if not genus:
                    continue
                hits = bc_labels[
                    bc_labels["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\s", na=False)
                ]
                if len(hits) > 0:
                    proxy_map[str(row["primary_label"])] = hits["bc_index"].astype(int).tolist()
            selected_proxy_targets = sorted(label for label in proxy_map if active_mask[labels.index(label)])
            selected_proxy_pos = np.array([labels.index(label) for label in selected_proxy_targets], dtype=np.int32)

    unmapped_pos = np.where(~mapped_mask)[0].astype(np.int32)
    idx_mapped_active_texture = idx_active_texture[mapped_mask[idx_active_texture]]
    idx_mapped_active_event = idx_active_event[mapped_mask[idx_active_event]]
    idx_unmapped_active_texture = idx_active_texture[~mapped_mask[idx_active_texture]]
    idx_unmapped_active_event = idx_active_event[~mapped_mask[idx_active_event]]
    idx_selected_proxy_active_texture = np.intersect1d(selected_proxy_pos, idx_active_texture)
    idx_selected_prioronly_active_texture = np.setdiff1d(idx_unmapped_active_texture, selected_proxy_pos)
    idx_selected_prioronly_active_event = np.setdiff1d(idx_unmapped_active_event, selected_proxy_pos)
    idx_unmapped_inactive = np.array([index for index in unmapped_pos if not active_mask[index]], dtype=np.int32)

    return MappingContext(
        taxonomy=taxonomy,
        labels=labels,
        class_name_map=class_name_map,
        class_family=class_family,
        family_idx_map=family_idx_map,
        active_mask=active_mask,
        idx_active_texture=idx_active_texture,
        idx_active_event=idx_active_event,
        idx_mapped_active_texture=idx_mapped_active_texture,
        idx_mapped_active_event=idx_mapped_active_event,
        idx_selected_proxy_active_texture=idx_selected_proxy_active_texture,
        idx_selected_prioronly_active_texture=idx_selected_prioronly_active_texture,
        idx_selected_prioronly_active_event=idx_selected_prioronly_active_event,
        idx_unmapped_inactive=idx_unmapped_inactive,
    )


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

    for _, (_, val_idx) in enumerate(splitter.split(reduced, groups=groups), start=1):
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


def _fit_prior_tables(prior_df: pd.DataFrame, targets: np.ndarray) -> dict[str, Any]:
    prior_df = prior_df.reset_index(drop=True)
    global_p = targets.mean(axis=0).astype(np.float32)

    site_keys = sorted(prior_df["site"].dropna().astype(str).unique())
    hour_keys = sorted(prior_df["hour_utc"].dropna().astype(int).unique())

    site_to_i: dict[str, int] = {}
    site_n: list[int] = []
    site_p: list[np.ndarray] = []
    for site in site_keys:
        mask = prior_df["site"].astype(str).values == site
        site_to_i[site] = len(site_n)
        site_n.append(int(mask.sum()))
        site_p.append(targets[mask].mean(axis=0))

    hour_to_i: dict[int, int] = {}
    hour_n: list[int] = []
    hour_p: list[np.ndarray] = []
    for hour in hour_keys:
        mask = prior_df["hour_utc"].astype(int).values == hour
        hour_to_i[int(hour)] = len(hour_n)
        hour_n.append(int(mask.sum()))
        hour_p.append(targets[mask].mean(axis=0))

    sh_to_i: dict[tuple[str, int], int] = {}
    sh_n: list[int] = []
    sh_p: list[np.ndarray] = []
    for (site, hour), indices in prior_df.groupby(["site", "hour_utc"]).groups.items():
        sh_to_i[(str(site), int(hour))] = len(sh_n)
        idx = np.array(list(indices))
        sh_n.append(len(idx))
        sh_p.append(targets[idx].mean(axis=0))

    return {
        "global_p": global_p,
        "site_to_i": site_to_i,
        "site_n": np.array(site_n, dtype=np.float32),
        "site_p": np.stack(site_p).astype(np.float32) if site_p else np.zeros((0, targets.shape[1]), dtype=np.float32),
        "hour_to_i": hour_to_i,
        "hour_n": np.array(hour_n, dtype=np.float32),
        "hour_p": np.stack(hour_p).astype(np.float32) if hour_p else np.zeros((0, targets.shape[1]), dtype=np.float32),
        "sh_to_i": sh_to_i,
        "sh_n": np.array(sh_n, dtype=np.float32),
        "sh_p": np.stack(sh_p).astype(np.float32) if sh_p else np.zeros((0, targets.shape[1]), dtype=np.float32),
    }


def _prior_logits(sites: np.ndarray, hours: np.ndarray, tables: dict[str, Any], *, eps: float = 1e-4) -> np.ndarray:
    n_rows = len(sites)
    probabilities = np.repeat(tables["global_p"][None, :], n_rows, axis=0).astype(np.float32, copy=True)

    site_indices = np.fromiter((tables["site_to_i"].get(str(site), -1) for site in sites), np.int32, n_rows)
    hour_indices = np.fromiter(
        (tables["hour_to_i"].get(int(hour), -1) if int(hour) >= 0 else -1 for hour in hours),
        np.int32,
        n_rows,
    )
    sh_indices = np.fromiter(
        (
            tables["sh_to_i"].get((str(site), int(hour)), -1) if int(hour) >= 0 else -1
            for site, hour in zip(sites, hours, strict=False)
        ),
        np.int32,
        n_rows,
    )

    valid = hour_indices >= 0
    if valid.any():
        counts = tables["hour_n"][hour_indices[valid]][:, None]
        probabilities[valid] = counts / (counts + 8.0) * tables["hour_p"][hour_indices[valid]] + (
            1.0 - counts / (counts + 8.0)
        ) * probabilities[valid]

    valid = site_indices >= 0
    if valid.any():
        counts = tables["site_n"][site_indices[valid]][:, None]
        probabilities[valid] = counts / (counts + 8.0) * tables["site_p"][site_indices[valid]] + (
            1.0 - counts / (counts + 8.0)
        ) * probabilities[valid]

    valid = sh_indices >= 0
    if valid.any():
        counts = tables["sh_n"][sh_indices[valid]][:, None]
        probabilities[valid] = counts / (counts + 4.0) * tables["sh_p"][sh_indices[valid]] + (
            1.0 - counts / (counts + 4.0)
        ) * probabilities[valid]

    np.clip(probabilities, eps, 1.0 - eps, out=probabilities)
    return (np.log(probabilities) - np.log1p(-probabilities)).astype(np.float32)


def _smooth_cols(scores: np.ndarray, columns: np.ndarray, *, windows_per_file: int, alpha: float) -> np.ndarray:
    if alpha <= 0.0 or len(columns) == 0:
        return scores.copy()
    smoothed = scores.copy()
    view = smoothed.reshape(-1, windows_per_file, smoothed.shape[1])
    subset = view[:, :, columns]
    previous = np.concatenate([subset[:, :1, :], subset[:, :-1, :]], axis=1)
    next_values = np.concatenate([subset[:, 1:, :], subset[:, -1:, :]], axis=1)
    view[:, :, columns] = (1.0 - alpha) * subset + 0.5 * alpha * (previous + next_values)
    return smoothed


def _fuse_scores(
    base_scores: np.ndarray,
    *,
    sites: np.ndarray,
    hours: np.ndarray,
    tables: dict[str, Any],
    mapping: MappingContext,
    lambda_event: float,
    lambda_texture: float,
    lambda_proxy_texture: float,
    smooth_texture_alpha: float,
    windows_per_file: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = base_scores.copy()
    prior = _prior_logits(sites, hours, tables)

    if len(mapping.idx_mapped_active_event):
        scores[:, mapping.idx_mapped_active_event] += lambda_event * prior[:, mapping.idx_mapped_active_event]
    if len(mapping.idx_mapped_active_texture):
        scores[:, mapping.idx_mapped_active_texture] += lambda_texture * prior[:, mapping.idx_mapped_active_texture]
    if len(mapping.idx_selected_proxy_active_texture):
        scores[:, mapping.idx_selected_proxy_active_texture] += (
            lambda_proxy_texture * prior[:, mapping.idx_selected_proxy_active_texture]
        )
    if len(mapping.idx_selected_prioronly_active_event):
        scores[:, mapping.idx_selected_prioronly_active_event] = (
            lambda_event * prior[:, mapping.idx_selected_prioronly_active_event]
        )
    if len(mapping.idx_selected_prioronly_active_texture):
        scores[:, mapping.idx_selected_prioronly_active_texture] = (
            lambda_texture * prior[:, mapping.idx_selected_prioronly_active_texture]
        )
    if len(mapping.idx_unmapped_inactive):
        scores[:, mapping.idx_unmapped_inactive] = -8.0

    scores = _smooth_cols(
        scores,
        mapping.idx_active_texture,
        windows_per_file=windows_per_file,
        alpha=smooth_texture_alpha,
    )
    return scores.astype(np.float32), prior.astype(np.float32)


def _seq_features_1d(values: np.ndarray, *, windows_per_file: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reshaped = values.reshape(-1, windows_per_file)
    previous = np.concatenate([reshaped[:, :1], reshaped[:, :-1]], axis=1).reshape(-1)
    next_values = np.concatenate([reshaped[:, 1:], reshaped[:, -1:]], axis=1).reshape(-1)
    mean_values = np.repeat(reshaped.mean(axis=1), windows_per_file)
    max_values = np.repeat(reshaped.max(axis=1), windows_per_file)
    min_values = np.repeat(reshaped.min(axis=1), windows_per_file)
    range_values = max_values - min_values
    return previous, next_values, mean_values, max_values, min_values, range_values


def _cosine_sim_to_prototype(embeddings: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    prototype_norm = prototype / (np.linalg.norm(prototype) + 1e-8)
    return (embeddings_norm @ prototype_norm).astype(np.float32)


def _build_class_features(
    reduced_embeddings: np.ndarray,
    *,
    raw_scores: np.ndarray,
    prior_scores: np.ndarray,
    base_scores: np.ndarray,
    windows_per_file: int,
    use_raw_scores: bool,
    use_prior_scores: bool,
    use_sequential_features: bool,
    prototype_similarity: np.ndarray | None = None,
    family_mean: np.ndarray | None = None,
) -> np.ndarray:
    parts: list[np.ndarray] = [reduced_embeddings.astype(np.float32)]
    if use_raw_scores:
        parts.append(raw_scores[:, None].astype(np.float32))
    if use_prior_scores:
        parts.append(prior_scores[:, None].astype(np.float32))
    parts.append(base_scores[:, None].astype(np.float32))
    if use_sequential_features:
        previous, next_values, mean_values, max_values, min_values, range_values = _seq_features_1d(
            base_scores,
            windows_per_file=windows_per_file,
        )
        parts.extend(
            [
                previous[:, None].astype(np.float32),
                next_values[:, None].astype(np.float32),
                mean_values[:, None].astype(np.float32),
                max_values[:, None].astype(np.float32),
                min_values[:, None].astype(np.float32),
                range_values[:, None].astype(np.float32),
            ]
        )
    if prototype_similarity is not None:
        parts.append(prototype_similarity[:, None].astype(np.float32))
    if family_mean is not None:
        parts.append(family_mean[:, None].astype(np.float32))
    return np.concatenate(parts, axis=1).astype(np.float32)


def _build_class_prototypes(reduced_embeddings: np.ndarray, targets: np.ndarray, *, min_pos: int) -> dict[int, np.ndarray]:
    prototypes: dict[int, np.ndarray] = {}
    positive_counts = targets.sum(axis=0)
    for class_index in np.where(positive_counts >= min_pos)[0]:
        mask = targets[:, class_index] == 1
        if mask.sum() >= min_pos:
            prototypes[int(class_index)] = reduced_embeddings[mask].mean(axis=0).astype(np.float32)
    return prototypes


def _family_mean_features(
    class_index: int,
    base_scores: np.ndarray,
    mapping: MappingContext,
    indices: np.ndarray,
) -> np.ndarray | None:
    family_name = mapping.class_family.get(class_index, "Unknown")
    family_indices = mapping.family_idx_map.get(family_name, np.array([], dtype=np.int32))
    other_indices = family_indices[family_indices != class_index]
    if len(other_indices) == 0:
        return None
    return base_scores[indices][:, other_indices].mean(axis=1).astype(np.float32)


def _group_kfold(groups: np.ndarray, n_splits: int) -> GroupKFold:
    unique_groups = np.unique(groups)
    splits = min(max(2, n_splits), len(unique_groups))
    return GroupKFold(n_splits=splits)


def _macro_auc(targets: np.ndarray, predictions: np.ndarray) -> float:
    active_mask = targets.sum(axis=0) > 0
    return float(roc_auc_score(targets[:, active_mask], predictions[:, active_mask], average="macro"))


def _fit_reference_pipeline(
    *,
    embeddings: np.ndarray,
    raw_scores: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    meta_df: pd.DataFrame,
    soundscape_df: pd.DataFrame,
    soundscape_targets: np.ndarray,
    mapping: MappingContext,
    pca_dim: int,
    min_pos: int,
    probe_c: float,
    max_iter: int,
    n_splits: int,
    use_raw_scores: bool,
    probe_alpha: float,
    lambda_event: float,
    lambda_texture: float,
    lambda_proxy_texture: float,
    smooth_texture_alpha: float,
    windows_per_file: int,
    use_prototype_similarity: bool,
    use_family_mean: bool,
    use_sequential_features: bool,
) -> tuple[dict[str, Any], dict[str, LogisticRegression]]:
    oof_base = np.zeros_like(raw_scores, dtype=np.float32)
    oof_prior = np.zeros_like(raw_scores, dtype=np.float32)
    splitter = _group_kfold(groups, n_splits)
    for _, val_idx in splitter.split(raw_scores, groups=groups):
        val_idx = np.sort(val_idx)
        val_sites = set(meta_df.iloc[val_idx]["site"].astype(str).tolist())
        prior_mask = ~soundscape_df["site"].astype(str).isin(val_sites).values
        tables = _fit_prior_tables(soundscape_df.loc[prior_mask].reset_index(drop=True), soundscape_targets[prior_mask])
        base_fold, prior_fold = _fuse_scores(
            raw_scores[val_idx],
            sites=meta_df.iloc[val_idx]["site"].to_numpy(),
            hours=meta_df.iloc[val_idx]["hour_utc"].to_numpy(),
            tables=tables,
            mapping=mapping,
            lambda_event=lambda_event,
            lambda_texture=lambda_texture,
            lambda_proxy_texture=lambda_proxy_texture,
            smooth_texture_alpha=smooth_texture_alpha,
            windows_per_file=windows_per_file,
        )
        oof_base[val_idx] = base_fold
        oof_prior[val_idx] = prior_fold
    prior_auc = _macro_auc(targets, oof_base)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    n_components = max(1, min(pca_dim, scaled.shape[0] - 1, scaled.shape[1]))
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(scaled).astype(np.float32)

    oof_predictions = oof_base.copy()
    fitted_classes = 0
    for _, val_idx in splitter.split(reduced, groups=groups):
        train_idx = np.setdiff1d(np.arange(len(reduced)), val_idx)
        positive_counts = targets[train_idx].sum(axis=0)
        fold_targets = targets[train_idx]
        fold_prototypes = (
            _build_class_prototypes(reduced[train_idx], fold_targets, min_pos=min_pos) if use_prototype_similarity else {}
        )
        for class_index in np.where(positive_counts >= min_pos)[0]:
            train_targets = targets[train_idx, class_index]
            if train_targets.sum() == 0 or train_targets.sum() == len(train_targets):
                continue
            prototype_train = prototype_val = None
            if use_prototype_similarity and class_index in fold_prototypes:
                prototype = fold_prototypes[class_index]
                prototype_train = _cosine_sim_to_prototype(reduced[train_idx], prototype)
                prototype_val = _cosine_sim_to_prototype(reduced[val_idx], prototype)
            family_mean_train = _family_mean_features(class_index, oof_base, mapping, train_idx) if use_family_mean else None
            family_mean_val = _family_mean_features(class_index, oof_base, mapping, val_idx) if use_family_mean else None
            x_train = _build_class_features(
                reduced[train_idx],
                raw_scores=raw_scores[train_idx, class_index],
                prior_scores=oof_prior[train_idx, class_index],
                base_scores=oof_base[train_idx, class_index],
                windows_per_file=windows_per_file,
                use_raw_scores=use_raw_scores,
                use_prior_scores=True,
                use_sequential_features=use_sequential_features,
                prototype_similarity=prototype_train,
                family_mean=family_mean_train,
            )
            x_val = _build_class_features(
                reduced[val_idx],
                raw_scores=raw_scores[val_idx, class_index],
                prior_scores=oof_prior[val_idx, class_index],
                base_scores=oof_base[val_idx, class_index],
                windows_per_file=windows_per_file,
                use_raw_scores=use_raw_scores,
                use_prior_scores=True,
                use_sequential_features=use_sequential_features,
                prototype_similarity=prototype_val,
                family_mean=family_mean_val,
            )
            classifier = LogisticRegression(
                C=probe_c,
                max_iter=max_iter,
                solver="liblinear",
                class_weight="balanced",
            )
            classifier.fit(x_train, train_targets)
            probe_logits = classifier.decision_function(x_val).astype(np.float32)
            oof_predictions[val_idx, class_index] = (1.0 - probe_alpha) * oof_base[val_idx, class_index] + probe_alpha * probe_logits
            fitted_classes += 1

    macro_auc = _macro_auc(targets, oof_predictions)
    full_prototypes = _build_class_prototypes(reduced, targets, min_pos=min_pos) if use_prototype_similarity else {}
    full_models: dict[int, LogisticRegression] = {}
    positive_counts = targets.sum(axis=0)
    for class_index in np.where(positive_counts >= min_pos)[0]:
        class_targets = targets[:, class_index]
        if class_targets.sum() == 0 or class_targets.sum() == len(class_targets):
            continue
        prototype_full = None
        if use_prototype_similarity and class_index in full_prototypes:
            prototype_full = _cosine_sim_to_prototype(reduced, full_prototypes[class_index])
        family_mean_full = _family_mean_features(class_index, oof_base, mapping, np.arange(len(oof_base))) if use_family_mean else None
        features = _build_class_features(
            reduced,
            raw_scores=raw_scores[:, class_index],
            prior_scores=oof_prior[:, class_index],
            base_scores=oof_base[:, class_index],
            windows_per_file=windows_per_file,
            use_raw_scores=use_raw_scores,
            use_prior_scores=True,
            use_sequential_features=use_sequential_features,
            prototype_similarity=prototype_full,
            family_mean=family_mean_full,
        )
        classifier = LogisticRegression(
            C=probe_c,
            max_iter=max_iter,
            solver="liblinear",
            class_weight="balanced",
        )
        classifier.fit(features, class_targets)
        full_models[int(class_index)] = classifier

    final_tables = _fit_prior_tables(soundscape_df.reset_index(drop=True), soundscape_targets)
    return (
        {
            "oof_predictions": oof_predictions.astype(np.float32),
            "oof_base": oof_base.astype(np.float32),
            "oof_prior": oof_prior.astype(np.float32),
            "prior_macro_auc": prior_auc,
            "macro_auc": macro_auc,
            "scaler": scaler,
            "pca": pca,
            "reduced_embeddings": reduced,
            "fitted_classes": fitted_classes,
            "class_prototypes": full_prototypes,
            "final_tables": final_tables,
        },
        full_models,
    )


def _root_cause(primary_metric: float, fitted_classes: int, active_classes: int, *, reference_pipeline: bool) -> str:
    if reference_pipeline:
        if primary_metric >= 0.90:
            return "The notebook-derived Bayesian prior + embedding probe pipeline is live; next steps are submission bundle parity and calibration."
        if fitted_classes < active_classes:
            return "The reference pipeline is running, but sparse positives still limit how many classes can support probe heads."
        return "The reference pipeline is wired in, but the remaining gap is calibration and submission-path parity with the notebook."
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

    full_truth, soundscape_df, soundscape_targets, labels, taxonomy = _load_truth_matrix(data_root, data_cfg)
    meta_df = pd.read_parquet(meta_path)
    cached = np.load(arrays_path)
    raw_scores = cached["scores_full_raw"].astype(np.float32)
    embeddings = cached["emb_full"].astype(np.float32)
    aligned_truth = full_truth.set_index("row_id").loc[meta_df["row_id"]].reset_index(drop=False)
    targets = soundscape_targets[aligned_truth["index"].to_numpy()]
    groups = meta_df["site"].fillna("unknown").astype(str).to_numpy()

    mapping = _build_mapping_context(
        labels=labels,
        targets=targets,
        taxonomy=taxonomy,
        runtime_root=runtime_root,
        config=config,
    )

    probe_pca_dim = int(training_cfg.get("probe_pca_dim", 32))
    probe_min_pos = int(training_cfg.get("probe_min_pos", 8))
    probe_c = float(training_cfg.get("probe_c", 0.25))
    probe_group_folds = int(training_cfg.get("probe_group_folds", 5))
    max_iter = int(training_cfg.get("max_iter", 400))
    use_raw_scores = bool(training_cfg.get("probe_use_raw_scores", True))

    reference_pipeline = bool(training_cfg.get("reference_bayesian_pipeline", False))
    probe_alpha = float(training_cfg.get("probe_alpha", 0.40))
    lambda_event = float(training_cfg.get("prior_lambda_event", 0.4))
    lambda_texture = float(training_cfg.get("prior_lambda_texture", 1.0))
    lambda_proxy_texture = float(training_cfg.get("prior_lambda_proxy_texture", 0.8))
    smooth_texture_alpha = float(training_cfg.get("smooth_texture_alpha", 0.35))
    use_prototype_similarity = bool(training_cfg.get("probe_use_prototype_similarity", True))
    use_family_mean = bool(training_cfg.get("probe_use_family_mean", True))
    use_sequential_features = bool(training_cfg.get("probe_use_sequential_features", True))
    final_gaussian_smoothing = bool(training_cfg.get("final_gaussian_smoothing", True))
    gaussian_weights = np.array(training_cfg.get("gaussian_weights", [0.1, 0.2, 0.4, 0.2, 0.1]), dtype=np.float32)
    windows_per_file = int(training_cfg.get("windows_per_file", 12))

    if reference_pipeline:
        reference_result, models = _fit_reference_pipeline(
            embeddings=embeddings,
            raw_scores=raw_scores,
            targets=targets,
            groups=groups,
            meta_df=meta_df,
            soundscape_df=soundscape_df,
            soundscape_targets=soundscape_targets,
            mapping=mapping,
            pca_dim=probe_pca_dim,
            min_pos=probe_min_pos,
            probe_c=probe_c,
            max_iter=max_iter,
            n_splits=probe_group_folds,
            use_raw_scores=use_raw_scores,
            probe_alpha=probe_alpha,
            lambda_event=lambda_event,
            lambda_texture=lambda_texture,
            lambda_proxy_texture=lambda_proxy_texture,
            smooth_texture_alpha=smooth_texture_alpha,
            windows_per_file=windows_per_file,
            use_prototype_similarity=use_prototype_similarity,
            use_family_mean=use_family_mean,
            use_sequential_features=use_sequential_features,
        )
        macro_auc = float(reference_result["macro_auc"])
        prior_auc = float(reference_result["prior_macro_auc"])
        oof_predictions = reference_result["oof_predictions"]
        scaler = reference_result["scaler"]
        pca = reference_result["pca"]
        fitted_classes = len(models)
        extra_metrics = {"prior_fusion_macro_roc_auc": prior_auc}
    else:
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
        macro_auc = _macro_auc(targets, oof_predictions)
        prior_auc = None
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
        reference_result = {
            "oof_base": raw_scores.astype(np.float32),
            "oof_prior": np.zeros_like(raw_scores, dtype=np.float32),
            "final_tables": {},
            "class_prototypes": {},
        }
        extra_metrics = {}

    cmap = float(padded_cmap(targets.tolist(), oof_predictions.tolist()))
    active_classes = int(mapping.active_mask.sum())

    oof_path = run_dir / "oof_predictions.npz"
    np.savez_compressed(
        oof_path,
        row_ids=meta_df["row_id"].to_numpy(),
        targets=targets.astype(np.uint8),
        oof_predictions=oof_predictions.astype(np.float32),
        raw_scores=raw_scores.astype(np.float32),
        oof_base=reference_result["oof_base"].astype(np.float32),
        oof_prior=reference_result["oof_prior"].astype(np.float32),
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
                "reference_bayesian_pipeline": reference_pipeline,
                "probe_alpha": probe_alpha,
                "min_pos": probe_min_pos,
                "probe_c": probe_c,
                "prior_tables": reference_result["final_tables"],
                "class_prototypes": reference_result["class_prototypes"],
                "class_family": mapping.class_family,
                "family_idx_map": mapping.family_idx_map,
                "gaussian_weights": gaussian_weights,
                "final_gaussian_smoothing": final_gaussian_smoothing,
                "mapping_context": {
                    "idx_active_texture": mapping.idx_active_texture.tolist(),
                    "idx_active_event": mapping.idx_active_event.tolist(),
                    "idx_mapped_active_texture": mapping.idx_mapped_active_texture.tolist(),
                    "idx_mapped_active_event": mapping.idx_mapped_active_event.tolist(),
                    "idx_selected_proxy_active_texture": mapping.idx_selected_proxy_active_texture.tolist(),
                    "idx_selected_prioronly_active_texture": mapping.idx_selected_prioronly_active_texture.tolist(),
                    "idx_selected_prioronly_active_event": mapping.idx_selected_prioronly_active_event.tolist(),
                    "idx_unmapped_inactive": mapping.idx_unmapped_inactive.tolist(),
                },
                "config": {
                    "probe_pca_dim": probe_pca_dim,
                    "probe_min_pos": probe_min_pos,
                    "probe_c": probe_c,
                    "probe_group_folds": probe_group_folds,
                    "windows_per_file": windows_per_file,
                    "prior_lambda_event": lambda_event,
                    "prior_lambda_texture": lambda_texture,
                    "prior_lambda_proxy_texture": lambda_proxy_texture,
                    "smooth_texture_alpha": smooth_texture_alpha,
                    "use_sequential_features": use_sequential_features,
                    "use_prototype_similarity": use_prototype_similarity,
                    "use_family_mean": use_family_mean,
                },
            },
            handle,
        )
    probe_metadata_path = run_dir / "probe_metadata.json"
    probe_metadata_path.write_text(
        json.dumps(
            {
                "probe_pca_dim": probe_pca_dim,
                "probe_min_pos": probe_min_pos,
                "probe_c": probe_c,
                "probe_group_folds": probe_group_folds,
                "use_raw_scores": use_raw_scores,
                "reference_bayesian_pipeline": reference_pipeline,
                "probe_alpha": probe_alpha,
                "fitted_classes": len(models),
                "active_classes": active_classes,
                "cache_rows": int(len(meta_df)),
                "full_soundscape_windows": int(len(aligned_truth)),
                "prior_fusion_macro_roc_auc": prior_auc,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "## Runtime Summary",
        f"- Backend: sklearn_cached_probe{' (reference_bayesian_pipeline)' if reference_pipeline else ''}",
        f"- Cache root: {cache_root}",
        f"- Cached rows: {len(meta_df)}",
        f"- Fully labeled windows: {len(aligned_truth)}",
        f"- Active classes: {active_classes}",
        f"- Fitted classes: {len(models)}",
        f"- Probe PCA dim: {probe_pca_dim}",
        f"- Probe min positives: {probe_min_pos}",
        f"- Probe C: {probe_c}",
        f"- Use raw Perch scores: {use_raw_scores}",
    ]
    if reference_pipeline:
        summary_lines.extend(
            [
                f"- Probe alpha: {probe_alpha}",
                f"- Prior lambda event: {lambda_event}",
                f"- Prior lambda texture: {lambda_texture}",
                f"- Prior lambda proxy texture: {lambda_proxy_texture}",
                f"- Smooth texture alpha: {smooth_texture_alpha}",
                f"- prior_fusion_macro_roc_auc={prior_auc:.6f}",
            ]
        )
    summary_lines.extend(
        [
            f"- soundscape_macro_roc_auc={macro_auc:.6f}",
            f"- padded_cmap={cmap:.6f}",
        ]
    )
    summary_markdown = "\n".join(summary_lines)

    metrics = {
        "soundscape_macro_roc_auc": macro_auc,
        "padded_cmap": cmap,
        **extra_metrics,
    }
    root_cause = _root_cause(
        macro_auc,
        len(models),
        active_classes,
        reference_pipeline=reference_pipeline,
    )
    threshold = float(config.get("metrics", {}).get("submission_candidate_threshold", 0.75))
    verdict = "submission-candidate" if macro_auc >= threshold else "baseline-ready"
    return {
        "metrics": metrics,
        "artifacts": {
            "oof_predictions": str(oof_path),
            "probe_bundle": str(bundle_path),
            "probe_metadata": str(probe_metadata_path),
        },
        "dataset_summary": {
            "cache_row_count": int(len(meta_df)),
            "fully_labeled_windows": int(len(aligned_truth)),
            "fully_labeled_files": int(aligned_truth["filename"].nunique()),
            "active_class_count": active_classes,
            "fitted_class_count": int(len(models)),
        },
        "root_cause": root_cause,
        "summary_markdown": summary_markdown,
        "verdict": verdict,
    }
