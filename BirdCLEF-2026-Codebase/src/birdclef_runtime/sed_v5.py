from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from birdclef_runtime.metrics import macro_roc_auc, padded_cmap


@dataclass(frozen=True)
class AudioExample:
    sample_id: str
    audio_path: Path
    target: np.ndarray
    start_sec: float | None = None
    duration_sec: float = 5.0
    source: str = "train_audio"
    metadata: dict[str, Any] = field(default_factory=dict)
    target_kind: str = "hard"
    sampling_weight: float = 1.0
    loss_weight: float = 1.0


@dataclass(frozen=True)
class SEDSettings:
    sample_rate: int = 32000
    chunk_duration: float = 5.0
    n_mels: int = 256
    n_fft: int = 2048
    hop_length: int = 512
    fmin: int = 0
    fmax: int = 16000
    image_size: int = 256
    num_classes: int = 234
    backbone_name: str = "tf_efficientnet_b0.ns_jft_in1k"
    backbone_trainable: bool = True
    dropout: float = 0.1
    gem_p_init: float = 3.0
    learning_rate: float = 1e-3
    batch_size: int = 4
    eval_batch_size: int = 4
    epochs: int = 1
    steps_per_epoch: int = 0
    clip_loss_weight: float = 0.5
    frame_loss_weight: float = 0.5
    mixup_alpha: float = 0.2
    specaugment_time_mask_ratio: float = 0.08
    specaugment_freq_mask_ratio: float = 0.08
    specaugment_num_masks: int = 2
    gain_db_range: float = 6.0
    gaussian_noise_std: float = 0.003
    clip_loss_name: str = "bce"
    frame_loss_name: str = "bce"
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 0.0
    asl_clip: float = 0.05
    seed: int = 42
    require_full_soundscapes: bool = True
    max_train_rows: int = 0
    max_val_rows: int = 0
    max_val_files: int = 0
    val_file_offset: int = 0
    max_pseudo_rows: int = 0
    max_pseudo_files: int = 0
    exclude_validation_soundscapes_from_pseudo: bool = True
    pseudo_sampling_weight: float = 1.0
    pseudo_loss_weight: float = 1.0
    pseudo_merge_strategy: str = "mean"

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_duration)


def _resolve_root(runtime_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = runtime_root / path
    return path.resolve()


def _load_label_order(data_root: Path, data_cfg: dict[str, Any]) -> list[str]:
    submission_df = pd.read_csv(data_root / data_cfg.get("sample_submission_csv", "sample_submission.csv"))
    return submission_df.columns[1:].tolist()


def _read_train_rows(data_root: Path, data_cfg: dict[str, Any], max_rows: int) -> pd.DataFrame:
    nrows = max_rows if max_rows > 0 else None
    return pd.read_csv(data_root / data_cfg.get("train_csv", "train.csv"), nrows=nrows)


def _read_soundscape_rows(data_root: Path, data_cfg: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(data_root / data_cfg.get("train_soundscapes_labels_csv", "train_soundscapes_labels.csv"))


def _parse_labels(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [token.strip() for token in str(value).split(";") if token.strip()]


def _union_labels(series: pd.Series) -> list[str]:
    labels: set[str] = set()
    for value in series:
        labels.update(_parse_labels(value))
    return sorted(labels)


def _resolve_train_audio_path(data_root: Path, train_audio_dir: str, row: dict[str, Any]) -> Path:
    filename = str(row.get("filename") or row.get("path") or row.get("audio") or "")
    label = str(row.get("primary_label") or row.get("label") or "")
    filename_path = Path(filename)
    if filename and len(filename_path.parts) > 1:
        return (data_root / train_audio_dir / filename_path).resolve()
    if filename and label:
        return (data_root / train_audio_dir / label / filename).resolve()
    if filename:
        return (data_root / train_audio_dir / filename).resolve()
    raise ValueError(f"Unable to resolve audio path for row: {row}")


def _build_train_examples(data_root: Path, data_cfg: dict[str, Any], labels: list[str], settings: SEDSettings) -> list[AudioExample]:
    train_rows = _read_train_rows(data_root, data_cfg, settings.max_train_rows)
    label_to_index = {label: index for index, label in enumerate(labels)}
    examples: list[AudioExample] = []
    for row in train_rows.to_dict(orient="records"):
        label = str(row.get("primary_label") or row.get("label") or "")
        if label not in label_to_index:
            continue
        audio_path = _resolve_train_audio_path(data_root, data_cfg.get("train_audio_dir", "train_audio"), row)
        if not audio_path.exists():
            continue
        target = np.zeros(len(labels), dtype=np.float32)
        target[label_to_index[label]] = 1.0
        examples.append(
            AudioExample(
                sample_id=str(row.get("filename") or audio_path.stem),
                audio_path=audio_path,
                target=target,
                source="train_audio",
                metadata={"label": label},
            )
        )
    return examples


def _build_soundscape_validation_examples(
    data_root: Path,
    data_cfg: dict[str, Any],
    labels: list[str],
    settings: SEDSettings,
) -> list[AudioExample]:
    soundscape_rows = _read_soundscape_rows(data_root, data_cfg).drop_duplicates().reset_index(drop=True)
    label_to_index = {label: index for index, label in enumerate(labels)}
    grouped = (
        soundscape_rows.groupby(["filename", "start", "end"])["primary_label"]
        .apply(_union_labels)
        .reset_index(name="label_list")
    )
    grouped["start_sec"] = pd.to_timedelta(grouped["start"]).dt.total_seconds().astype(float)
    grouped["end_sec"] = pd.to_timedelta(grouped["end"]).dt.total_seconds().astype(float)
    grouped["row_id"] = grouped["filename"].str.replace(".ogg", "", regex=False) + "_" + grouped["end_sec"].astype(int).astype(str)

    if settings.require_full_soundscapes:
        windows_per_file = grouped.groupby("filename").size()
        full_files = set(windows_per_file[windows_per_file == int(60 / settings.chunk_duration)].index.tolist())
        grouped = grouped[grouped["filename"].isin(full_files)].reset_index(drop=True)

    ordered_files = sorted(grouped["filename"].unique())
    if settings.val_file_offset > 0:
        ordered_files = ordered_files[settings.val_file_offset :]
    if settings.max_val_files > 0:
        keep_files = set(ordered_files[: settings.max_val_files])
        grouped = grouped[grouped["filename"].isin(keep_files)].reset_index(drop=True)
    if settings.max_val_rows > 0:
        grouped = grouped.head(settings.max_val_rows).reset_index(drop=True)

    examples: list[AudioExample] = []
    for row in grouped.to_dict(orient="records"):
        audio_path = (data_root / data_cfg.get("train_soundscapes_dir", "train_soundscapes") / str(row["filename"])).resolve()
        if not audio_path.exists():
            continue
        target = np.zeros(len(labels), dtype=np.float32)
        for label in row["label_list"]:
            if label in label_to_index:
                target[label_to_index[label]] = 1.0
        examples.append(
            AudioExample(
                sample_id=str(row["row_id"]),
                audio_path=audio_path,
                target=target,
                start_sec=float(row["start_sec"]),
                duration_sec=float(row["end_sec"] - row["start_sec"]),
                source="soundscape_validation",
                metadata={"filename": str(row["filename"]), "row_id": str(row["row_id"])},
            )
        )
    return examples


def _resolve_pseudo_source_paths(data_cfg: dict[str, Any], runtime_root: Path) -> list[Path]:
    raw_sources = data_cfg.get("pseudo_source_paths")
    if raw_sources is None:
        raw_single = str(data_cfg.get("pseudo_source_path", "") or "").strip()
        raw_sources = [raw_single] if raw_single else []
    if isinstance(raw_sources, str):
        raw_sources = [raw_sources]
    resolved: list[Path] = []
    for raw_source in raw_sources or []:
        source = str(raw_source or "").strip()
        if not source:
            continue
        resolved.append(_resolve_root(runtime_root, source))
    return resolved


def _parse_window_end_sec(row_id: str) -> float:
    stem, separator, suffix = row_id.rpartition("_")
    if not separator or not suffix:
        raise ValueError(f"Pseudo row_id must end with an integer second suffix: {row_id}")
    try:
        return float(int(suffix))
    except ValueError as exc:
        raise ValueError(f"Pseudo row_id must end with an integer second suffix: {row_id}") from exc


def _build_soundscape_filename_lookup(data_root: Path, data_cfg: dict[str, Any]) -> dict[str, str]:
    soundscape_dir = (data_root / data_cfg.get("train_soundscapes_dir", "train_soundscapes")).resolve()
    lookup: dict[str, str] = {}
    for path in sorted(soundscape_dir.glob("*.ogg")):
        lookup[path.stem] = path.name
    return lookup


def _load_pseudo_rows_from_source(
    source_path: Path,
    labels: list[str],
    filename_lookup: dict[str, str],
) -> list[dict[str, Any]]:
    if source_path.is_dir():
        meta_path = source_path / "full_perch_meta.parquet"
        arrays_path = source_path / "full_perch_arrays.npz"
        if not meta_path.exists() or not arrays_path.exists():
            raise FileNotFoundError(f"Pseudo cache directory is missing required files: {source_path}")
        meta = pd.read_parquet(meta_path)
        arrays = np.load(arrays_path)
        if "scores_full_raw" not in arrays.files:
            raise KeyError(f"Pseudo cache {arrays_path} does not contain scores_full_raw")
        scores = 1.0 / (1.0 + np.exp(-np.asarray(arrays["scores_full_raw"], dtype=np.float32)))
        if len(meta) != len(scores):
            raise ValueError(f"Pseudo cache row count mismatch for {source_path}")
        rows: list[dict[str, Any]] = []
        for row, target in zip(meta.to_dict(orient="records"), scores, strict=False):
            rows.append(
                {
                    "row_id": str(row["row_id"]),
                    "filename": str(row["filename"]),
                    "target": np.asarray(target, dtype=np.float32),
                }
            )
        return rows
    pseudo_df = pd.read_csv(source_path)
    label_columns = [label for label in labels if label in pseudo_df.columns]
    if "row_id" not in pseudo_df.columns or not label_columns:
        raise ValueError(f"Pseudo source {source_path} must contain row_id and label columns")
    if len(label_columns) != len(labels):
        missing = [label for label in labels if label not in pseudo_df.columns]
        raise ValueError(f"Pseudo source {source_path} is missing label columns: {missing[:8]}")
    rows = []
    for record in pseudo_df.to_dict(orient="records"):
        row_id = str(record["row_id"])
        stem, _separator, _suffix = row_id.rpartition("_")
        filename = filename_lookup.get(stem)
        if not filename:
            continue
        target = np.asarray([float(record[label]) for label in labels], dtype=np.float32)
        rows.append({"row_id": row_id, "filename": filename, "target": target})
    return rows


def _merge_pseudo_targets(rows: list[dict[str, Any]], merge_strategy: str) -> list[dict[str, Any]]:
    if merge_strategy == "concat":
        return rows
    if merge_strategy not in {"mean"}:
        raise ValueError(f"Unsupported pseudo merge strategy: {merge_strategy}")
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        existing = grouped.get(row["row_id"])
        if existing is None:
            grouped[row["row_id"]] = {
                "row_id": row["row_id"],
                "filename": row["filename"],
                "target_sum": np.asarray(row["target"], dtype=np.float32),
                "count": 1,
            }
            continue
        existing["target_sum"] = existing["target_sum"] + np.asarray(row["target"], dtype=np.float32)
        existing["count"] += 1
    merged_rows: list[dict[str, Any]] = []
    for item in grouped.values():
        divisor = float(item["count"]) if merge_strategy == "mean" else 1.0
        merged_rows.append(
            {
                "row_id": item["row_id"],
                "filename": item["filename"],
                "target": (item["target_sum"] / divisor).astype(np.float32),
                "merge_count": int(item["count"]),
            }
        )
    return sorted(merged_rows, key=lambda row: str(row["row_id"]))


def _build_pseudo_examples(
    runtime_root: Path,
    data_root: Path,
    data_cfg: dict[str, Any],
    labels: list[str],
    settings: SEDSettings,
    *,
    excluded_filenames: set[str],
) -> tuple[list[AudioExample], dict[str, Any]]:
    source_paths = _resolve_pseudo_source_paths(data_cfg, runtime_root)
    if not source_paths:
        return [], {"pseudo_source_paths": [], "pseudo_source_count": 0, "pseudo_example_count": 0}
    filename_lookup = _build_soundscape_filename_lookup(data_root, data_cfg)
    aggregated_rows: list[dict[str, Any]] = []
    for source_path in source_paths:
        source_rows = _load_pseudo_rows_from_source(source_path, labels, filename_lookup)
        for row in source_rows:
            if settings.exclude_validation_soundscapes_from_pseudo and str(row["filename"]) in excluded_filenames:
                continue
            aggregated_rows.append({**row, "source_path": str(source_path)})
    merged_rows = _merge_pseudo_targets(aggregated_rows, settings.pseudo_merge_strategy)
    if settings.max_pseudo_files > 0:
        ordered_files = []
        seen_files: set[str] = set()
        for row in merged_rows:
            filename = str(row["filename"])
            if filename in seen_files:
                continue
            seen_files.add(filename)
            ordered_files.append(filename)
        allowed = set(ordered_files[: settings.max_pseudo_files])
        merged_rows = [row for row in merged_rows if str(row["filename"]) in allowed]
    if settings.max_pseudo_rows > 0:
        merged_rows = merged_rows[: settings.max_pseudo_rows]

    soundscape_dir = (data_root / data_cfg.get("train_soundscapes_dir", "train_soundscapes")).resolve()
    examples: list[AudioExample] = []
    for row in merged_rows:
        filename = str(row["filename"])
        audio_path = (soundscape_dir / filename).resolve()
        if not audio_path.exists():
            continue
        end_sec = _parse_window_end_sec(str(row["row_id"]))
        start_sec = max(0.0, end_sec - settings.chunk_duration)
        examples.append(
            AudioExample(
                sample_id=str(row["row_id"]),
                audio_path=audio_path,
                target=np.asarray(row["target"], dtype=np.float32),
                start_sec=start_sec,
                duration_sec=settings.chunk_duration,
                source="soft_pseudo",
                metadata={
                    "filename": filename,
                    "row_id": str(row["row_id"]),
                    "merge_count": int(row.get("merge_count", 1)),
                },
                target_kind="soft",
                sampling_weight=settings.pseudo_sampling_weight,
                loss_weight=settings.pseudo_loss_weight,
            )
        )
    summary = {
        "pseudo_source_paths": [str(path) for path in source_paths],
        "pseudo_source_count": len(source_paths),
        "pseudo_example_count": len(examples),
        "pseudo_merge_strategy": settings.pseudo_merge_strategy,
        "pseudo_sampling_weight": settings.pseudo_sampling_weight,
        "pseudo_loss_weight": settings.pseudo_loss_weight,
    }
    return examples, summary


class AudioCache:
    def __init__(self, *, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._cache: dict[Path, np.ndarray] = {}

    def read(self, path: Path) -> np.ndarray:
        if path in self._cache:
            return self._cache[path]
        import librosa
        import soundfile as sf

        waveform, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if hasattr(waveform, "ndim") and waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        waveform = np.asarray(waveform, dtype=np.float32)
        if sr != self.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
        self._cache[path] = np.asarray(waveform, dtype=np.float32)
        return self._cache[path]


def _load_clip(example: AudioExample, settings: SEDSettings, cache: AudioCache, *, rng: np.random.RandomState) -> np.ndarray:
    waveform = cache.read(example.audio_path)
    clip_samples = settings.chunk_samples
    if example.start_sec is None:
        if len(waveform) <= clip_samples:
            start = 0
        else:
            start = int(rng.randint(0, len(waveform) - clip_samples + 1))
        clip = waveform[start : start + clip_samples]
    else:
        start = int(example.start_sec * settings.sample_rate)
        clip = waveform[start : start + clip_samples]
    if len(clip) < clip_samples:
        clip = np.pad(clip, (0, clip_samples - len(clip)))
    return clip.astype(np.float32)


def _apply_waveform_augmentations(batch: np.ndarray, settings: SEDSettings, rng: np.random.RandomState) -> np.ndarray:
    augmented = batch.astype(np.float32, copy=True)
    if settings.gain_db_range > 0:
        gains = 10.0 ** (rng.uniform(-settings.gain_db_range, settings.gain_db_range, size=(len(batch), 1)) / 20.0)
        augmented *= gains.astype(np.float32)
    if settings.gaussian_noise_std > 0:
        augmented += rng.normal(0.0, settings.gaussian_noise_std, size=augmented.shape).astype(np.float32)
    return np.clip(augmented, -1.0, 1.0)


def _apply_mixup(images: np.ndarray, clip_targets: np.ndarray, alpha: float, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    if alpha <= 0 or len(images) < 2:
        return images, clip_targets
    lam = float(rng.beta(alpha, alpha))
    indices = rng.permutation(len(images))
    mixed_images = lam * images + (1.0 - lam) * images[indices]
    mixed_targets = lam * clip_targets + (1.0 - lam) * clip_targets[indices]
    return mixed_images.astype(np.float32), mixed_targets.astype(np.float32)


def _apply_specaugment(images: np.ndarray, settings: SEDSettings, rng: np.random.RandomState) -> np.ndarray:
    if settings.specaugment_num_masks <= 0:
        return images
    augmented = images.astype(np.float32, copy=True)
    _, freq_bins, time_bins, _ = augmented.shape
    for sample_index in range(len(augmented)):
        for _ in range(settings.specaugment_num_masks):
            if settings.specaugment_freq_mask_ratio > 0:
                mask = max(1, int(freq_bins * settings.specaugment_freq_mask_ratio * rng.uniform(0.35, 1.0)))
                start = int(rng.randint(0, max(freq_bins - mask + 1, 1)))
                augmented[sample_index, start : start + mask, :, :] = 0.0
            if settings.specaugment_time_mask_ratio > 0:
                mask = max(1, int(time_bins * settings.specaugment_time_mask_ratio * rng.uniform(0.35, 1.0)))
                start = int(rng.randint(0, max(time_bins - mask + 1, 1)))
                augmented[sample_index, :, start : start + mask, :] = 0.0
    return augmented


class MelSpectrogramTransform:
    def __init__(self, settings: SEDSettings) -> None:
        import tensorflow as tf

        self.tf = tf
        self.settings = settings
        self.mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=settings.n_mels,
            num_spectrogram_bins=settings.n_fft // 2 + 1,
            sample_rate=settings.sample_rate,
            lower_edge_hertz=settings.fmin,
            upper_edge_hertz=settings.fmax,
        )

    def __call__(self, waveforms: np.ndarray) -> Any:
        tf = self.tf
        waveforms_tensor = tf.convert_to_tensor(waveforms, dtype=tf.float32)
        stft = tf.signal.stft(
            waveforms_tensor,
            frame_length=self.settings.n_fft,
            frame_step=self.settings.hop_length,
            fft_length=self.settings.n_fft,
            pad_end=True,
        )
        power = tf.math.square(tf.abs(stft))
        mel = tf.tensordot(power, self.mel_weight_matrix, axes=1)
        mel = tf.transpose(mel, perm=[0, 2, 1])
        mel = tf.math.log(tf.maximum(mel, 1e-6))
        mel = mel[..., tf.newaxis]
        mel = tf.image.resize(mel, [self.settings.image_size, self.settings.image_size])
        flat = tf.reshape(mel, [tf.shape(mel)[0], -1])
        min_values = tf.reshape(tf.reduce_min(flat, axis=1), [-1, 1, 1, 1])
        max_values = tf.reshape(tf.reduce_max(flat, axis=1), [-1, 1, 1, 1])
        mel = (mel - min_values) / (max_values - min_values + 1e-6)
        return tf.repeat(mel, repeats=3, axis=-1)


class GEMFreqPoolLayer:  # pragma: no cover - alias placeholder for typing tools
    pass


def _register_custom_layers() -> tuple[type[Any], type[Any]]:
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="birdclef_runtime")
    class GEMFreqPool(tf.keras.layers.Layer):
        def __init__(self, p_init: float = 3.0, eps: float = 1e-6, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.p_init = p_init
            self.eps = eps

        def build(self, input_shape: Any) -> None:
            self.p = self.add_weight(
                name="p",
                shape=(),
                initializer=tf.keras.initializers.Constant(self.p_init),
                trainable=True,
            )
            super().build(input_shape)

        def call(self, features: Any) -> Any:
            power = tf.maximum(self.p, 1.0)
            features = tf.maximum(features, self.eps)
            features = tf.pow(features, power)
            features = tf.reduce_mean(features, axis=1)
            return tf.pow(features, 1.0 / power)

        def get_config(self) -> dict[str, Any]:
            return {**super().get_config(), "p_init": self.p_init, "eps": self.eps}

    @tf.keras.utils.register_keras_serializable(package="birdclef_runtime")
    class AttentionSEDHead(tf.keras.layers.Layer):
        def __init__(self, feat_dim: int, num_classes: int, dropout: float, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.feat_dim = feat_dim
            self.num_classes = num_classes
            self.dropout = dropout
            self.fc = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(feat_dim),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Dropout(dropout),
                ]
            )
            self.att_conv = tf.keras.layers.Conv1D(num_classes, kernel_size=1)
            self.cls_conv = tf.keras.layers.Conv1D(num_classes, kernel_size=1)

        def call(self, inputs: Any, training: bool = False) -> dict[str, Any]:
            hidden = self.fc(inputs, training=training)
            attention = tf.nn.softmax(tf.math.tanh(self.att_conv(hidden)), axis=1)
            segmentwise_logits = self.cls_conv(hidden)
            clipwise_logits = tf.reduce_sum(attention * segmentwise_logits, axis=1)
            return {
                "clipwise_logits": clipwise_logits,
                "clipwise_prob": tf.nn.sigmoid(clipwise_logits),
                "segmentwise_logits": segmentwise_logits,
                "segmentwise_prob": tf.nn.sigmoid(segmentwise_logits),
            }

        def get_config(self) -> dict[str, Any]:
            return {
                **super().get_config(),
                "feat_dim": self.feat_dim,
                "num_classes": self.num_classes,
                "dropout": self.dropout,
            }

    return GEMFreqPool, AttentionSEDHead


def _build_backbone(settings: SEDSettings) -> Any:
    import tensorflow as tf

    input_shape = (settings.image_size, settings.image_size, 3)
    backbone_name = settings.backbone_name.lower()
    if backbone_name in {"tf_efficientnet_b0.ns_jft_in1k", "efficientnet_b0", "tf_efficientnet_b0"}:
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=input_shape,
        )
        backbone.trainable = settings.backbone_trainable
        return backbone
    if backbone_name == "tiny_conv":
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
        x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
        return tf.keras.Model(inputs, x, name="tiny_conv_backbone")
    raise ValueError(f"Unsupported SED backbone: {settings.backbone_name}")


def build_sed_model(settings: SEDSettings) -> Any:
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(settings.image_size, settings.image_size, 3), name="mel_image")
    backbone = _build_backbone(settings)
    features = backbone(inputs)
    gem_pool_cls, attention_head_cls = _register_custom_layers()
    pooled = gem_pool_cls(p_init=settings.gem_p_init, name="gem_freq_pool")(features)
    feat_dim = int(backbone.output_shape[-1])
    outputs = attention_head_cls(feat_dim, settings.num_classes, settings.dropout, name="attention_sed_head")(pooled)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="sed_v5")
    return model


def _build_settings(config: dict[str, Any], labels: list[str]) -> SEDSettings:
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    return SEDSettings(
        sample_rate=int(training_cfg.get("sample_rate", 32000)),
        chunk_duration=float(training_cfg.get("chunk_duration", 5.0)),
        n_mels=int(model_cfg.get("n_mels", 256)),
        n_fft=int(model_cfg.get("n_fft", 2048)),
        hop_length=int(model_cfg.get("hop_length", 512)),
        fmin=int(model_cfg.get("fmin", 0)),
        fmax=int(model_cfg.get("fmax", 16000)),
        image_size=int(model_cfg.get("image_size", 256)),
        num_classes=len(labels),
        backbone_name=str(model_cfg.get("backbone_name", "tf_efficientnet_b0.ns_jft_in1k")),
        backbone_trainable=bool(model_cfg.get("backbone_trainable", True)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        gem_p_init=float(model_cfg.get("gem_p_init", 3.0)),
        learning_rate=float(training_cfg.get("learning_rate", 1e-3)),
        batch_size=int(training_cfg.get("batch_size", 4)),
        eval_batch_size=int(training_cfg.get("eval_batch_size", training_cfg.get("batch_size", 4))),
        epochs=int(training_cfg.get("epochs", 1)),
        steps_per_epoch=int(training_cfg.get("steps_per_epoch", 0)),
        clip_loss_weight=float(training_cfg.get("clip_loss_weight", 0.5)),
        frame_loss_weight=float(training_cfg.get("frame_loss_weight", 0.5)),
        mixup_alpha=float(training_cfg.get("mixup_alpha", 0.2)),
        specaugment_time_mask_ratio=float(training_cfg.get("specaugment_time_mask_ratio", 0.08)),
        specaugment_freq_mask_ratio=float(training_cfg.get("specaugment_freq_mask_ratio", 0.08)),
        specaugment_num_masks=int(training_cfg.get("specaugment_num_masks", 2)),
        gain_db_range=float(training_cfg.get("gain_db_range", 6.0)),
        gaussian_noise_std=float(training_cfg.get("gaussian_noise_std", 0.003)),
        clip_loss_name=str(training_cfg.get("clip_loss_name", "bce")).lower(),
        frame_loss_name=str(training_cfg.get("frame_loss_name", "bce")).lower(),
        asl_gamma_neg=float(training_cfg.get("asl_gamma_neg", 4.0)),
        asl_gamma_pos=float(training_cfg.get("asl_gamma_pos", 0.0)),
        asl_clip=float(training_cfg.get("asl_clip", 0.05)),
        seed=int(config.get("experiment", {}).get("seed", 42)),
        require_full_soundscapes=bool(data_cfg.get("require_full_soundscapes", True)),
        max_train_rows=int(data_cfg.get("max_train_rows", 0) or 0),
        max_val_rows=int(data_cfg.get("max_val_rows", 0) or 0),
        max_val_files=int(data_cfg.get("max_val_files", 0) or 0),
        val_file_offset=int(data_cfg.get("val_file_offset", 0) or 0),
        max_pseudo_rows=int(data_cfg.get("max_pseudo_rows", 0) or 0),
        max_pseudo_files=int(data_cfg.get("max_pseudo_files", 0) or 0),
        exclude_validation_soundscapes_from_pseudo=bool(data_cfg.get("exclude_validation_soundscapes_from_pseudo", True)),
        pseudo_sampling_weight=float(data_cfg.get("pseudo_sampling_weight", 1.0) or 1.0),
        pseudo_loss_weight=float(training_cfg.get("pseudo_loss_weight", 1.0) or 1.0),
        pseudo_merge_strategy=str(data_cfg.get("pseudo_merge_strategy", "mean") or "mean").lower(),
    )


def _build_dataset_bundle(config: dict[str, Any], runtime_root: Path) -> tuple[list[str], list[AudioExample], list[AudioExample], dict[str, Any]]:
    data_cfg = config.get("data", {})
    paths_cfg = config.get("paths", {})
    data_root = _resolve_root(runtime_root, str(paths_cfg.get("data_root", "./birdclef-2026")))
    labels = _load_label_order(data_root, data_cfg)
    settings = _build_settings(config, labels)
    train_examples = _build_train_examples(data_root, data_cfg, labels, settings)
    val_examples = _build_soundscape_validation_examples(data_root, data_cfg, labels, settings)
    val_filenames = {str(example.metadata.get("filename", example.audio_path.name)) for example in val_examples}
    pseudo_examples, pseudo_summary = _build_pseudo_examples(
        runtime_root,
        data_root,
        data_cfg,
        labels,
        settings,
        excluded_filenames=val_filenames,
    )
    train_examples = train_examples + pseudo_examples
    if not train_examples:
        raise ValueError("No train audio examples were resolved for the SED v5 runtime.")
    if not val_examples:
        raise ValueError("No soundscape validation examples were resolved for the SED v5 runtime.")
    dataset_summary = {
        "data_root": str(data_root),
        "train_sample_count": len(train_examples),
        "val_sample_count": len(val_examples),
        "label_count": len(labels),
        "train_sources": sorted({example.source for example in train_examples}),
        "val_sources": sorted({example.source for example in val_examples}),
        "source_breakdown": {
            source: sum(1 for example in train_examples if example.source == source)
            for source in sorted({example.source for example in train_examples})
        },
        **pseudo_summary,
    }
    return labels, train_examples, val_examples, dataset_summary


def _sample_train_batch(
    examples: list[AudioExample],
    settings: SEDSettings,
    cache: AudioCache,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_weights = np.asarray([max(example.sampling_weight, 1e-6) for example in examples], dtype=np.float64)
    probabilities = sample_weights / sample_weights.sum()
    indices = rng.choice(
        len(examples),
        size=settings.batch_size,
        replace=len(examples) < settings.batch_size,
        p=probabilities,
    )
    selected_examples = [examples[int(index)] for index in indices]
    batch_waveforms = np.stack([_load_clip(example, settings, cache, rng=rng) for example in selected_examples]).astype(np.float32)
    batch_targets = np.stack([example.target for example in selected_examples]).astype(np.float32)
    batch_loss_weights = np.asarray([example.loss_weight for example in selected_examples], dtype=np.float32)
    batch_waveforms = _apply_waveform_augmentations(batch_waveforms, settings, rng)
    return batch_waveforms, batch_targets, batch_loss_weights


def _batched(items: list[AudioExample], batch_size: int) -> list[list[AudioExample]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def evaluate_soundscape_validation(
    model: Any,
    mel_transform: MelSpectrogramTransform,
    val_examples: list[AudioExample],
    settings: SEDSettings,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    predictions: list[list[float]] = []
    targets: list[list[float]] = []
    rows: list[dict[str, Any]] = []
    cache = AudioCache(sample_rate=settings.sample_rate)
    for batch in _batched(val_examples, settings.eval_batch_size):
        batch_waveforms = np.stack(
            [_load_clip(example, settings, cache, rng=np.random.RandomState(settings.seed)) for example in batch]
        ).astype(np.float32)
        batch_images = mel_transform(batch_waveforms)
        batch_outputs = model(batch_images, training=False)
        batch_probs = batch_outputs["clipwise_prob"].numpy().astype(np.float32)
        for example, probs in zip(batch, batch_probs, strict=False):
            predictions.append(probs.tolist())
            targets.append(example.target.tolist())
            rows.append(
                {
                    "row_id": str(example.metadata.get("row_id", example.sample_id)),
                    "filename": str(example.metadata.get("filename", example.audio_path.name)),
                    **{f"pred_{index}": float(value) for index, value in enumerate(probs)},
                }
            )
    metrics = {
        "soundscape_macro_roc_auc": macro_roc_auc(targets, predictions),
        "val_soundscape_macro_roc_auc": macro_roc_auc(targets, predictions),
        "padded_cmap": padded_cmap(targets, predictions),
    }
    return metrics, rows


def _asymmetric_multilabel_loss(
    tf: Any,
    labels: Any,
    logits: Any,
    *,
    gamma_neg: float,
    gamma_pos: float,
    clip: float,
) -> Any:
    probabilities = tf.nn.sigmoid(logits)
    pos_probabilities = tf.clip_by_value(probabilities, 1e-8, 1.0 - 1e-8)
    neg_probabilities = 1.0 - probabilities
    if clip > 0:
        neg_probabilities = tf.minimum(1.0, neg_probabilities + clip)
    neg_probabilities = tf.clip_by_value(neg_probabilities, 1e-8, 1.0 - 1e-8)
    base_loss = labels * tf.math.log(pos_probabilities) + (1.0 - labels) * tf.math.log(neg_probabilities)
    focus = tf.pow(
        1.0 - (pos_probabilities * labels + neg_probabilities * (1.0 - labels)),
        gamma_pos * labels + gamma_neg * (1.0 - labels),
    )
    return -base_loss * focus


def _multilabel_loss(tf: Any, labels: Any, logits: Any, settings: SEDSettings, *, loss_name: str) -> Any:
    if loss_name == "bce":
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    if loss_name == "asl":
        return _asymmetric_multilabel_loss(
            tf,
            labels,
            logits,
            gamma_neg=settings.asl_gamma_neg,
            gamma_pos=settings.asl_gamma_pos,
            clip=settings.asl_clip,
        )
    raise ValueError(f"Unsupported multilabel loss: {loss_name}")


def _reduce_weighted_loss(tf: Any, loss_tensor: Any, sample_weights: Any) -> Any:
    reduced = loss_tensor
    while len(reduced.shape) > 1:
        reduced = tf.reduce_mean(reduced, axis=-1)
    sample_weights = tf.cast(sample_weights, dtype=reduced.dtype)
    normalizer = tf.maximum(tf.reduce_sum(sample_weights), tf.cast(1e-6, reduced.dtype))
    return tf.reduce_sum(reduced * sample_weights) / normalizer


def load_sed_model(checkpoint_path: Path, settings: SEDSettings) -> Any:
    model = build_sed_model(settings)
    dummy = np.zeros((1, settings.image_size, settings.image_size, 3), dtype=np.float32)
    model(dummy, training=False)
    model.load_weights(checkpoint_path)
    return model


def run_sed_v5_training(config: dict[str, Any], runtime_root: Path, run_dir: Path) -> dict[str, Any]:
    import tensorflow as tf

    labels, train_examples, val_examples, dataset_summary = _build_dataset_bundle(config, runtime_root)
    settings = _build_settings(config, labels)
    tf.keras.utils.set_random_seed(settings.seed)
    rng = np.random.RandomState(settings.seed)
    model = build_sed_model(settings)
    mel_transform = MelSpectrogramTransform(settings)
    optimizer = tf.keras.optimizers.Adam(learning_rate=settings.learning_rate)
    audio_cache = AudioCache(sample_rate=settings.sample_rate)
    steps_per_epoch = settings.steps_per_epoch or max(1, math.ceil(len(train_examples) / settings.batch_size))

    best_metric = -1.0
    history: list[dict[str, float]] = []
    checkpoint_path = run_dir / "best_sed_v5.weights.h5"
    for epoch in range(settings.epochs):
        epoch_losses: list[float] = []
        clip_losses: list[float] = []
        frame_losses: list[float] = []
        for _step in range(steps_per_epoch):
            batch_waveforms, clip_targets, loss_weights = _sample_train_batch(train_examples, settings, audio_cache, rng)
            batch_images = mel_transform(batch_waveforms).numpy()
            batch_images = _apply_specaugment(batch_images, settings, rng)
            batch_images, clip_targets = _apply_mixup(batch_images, clip_targets, settings.mixup_alpha, rng)
            batch_images_tensor = tf.convert_to_tensor(batch_images, dtype=tf.float32)
            clip_targets_tensor = tf.convert_to_tensor(clip_targets, dtype=tf.float32)
            loss_weights_tensor = tf.convert_to_tensor(loss_weights, dtype=tf.float32)
            with tf.GradientTape() as tape:
                outputs = model(batch_images_tensor, training=True)
                clipwise_logits = outputs["clipwise_logits"]
                segmentwise_logits = outputs["segmentwise_logits"]
                frame_targets = tf.repeat(clip_targets_tensor[:, tf.newaxis, :], tf.shape(segmentwise_logits)[1], axis=1)
                clip_loss = _reduce_weighted_loss(
                    tf,
                    _multilabel_loss(
                        tf,
                        clip_targets_tensor,
                        clipwise_logits,
                        settings,
                        loss_name=settings.clip_loss_name,
                    ),
                    loss_weights_tensor,
                )
                frame_loss = _reduce_weighted_loss(
                    tf,
                    _multilabel_loss(
                        tf,
                        frame_targets,
                        segmentwise_logits,
                        settings,
                        loss_name=settings.frame_loss_name,
                    ),
                    loss_weights_tensor,
                )
                loss = settings.clip_loss_weight * clip_loss + settings.frame_loss_weight * frame_loss
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_losses.append(float(loss.numpy()))
            clip_losses.append(float(clip_loss.numpy()))
            frame_losses.append(float(frame_loss.numpy()))

        val_metrics, val_rows = evaluate_soundscape_validation(model, mel_transform, val_examples, settings)
        epoch_record = {
            "epoch": float(epoch + 1),
            "train_loss": float(np.mean(epoch_losses)),
            "clip_loss": float(np.mean(clip_losses)),
            "frame_loss": float(np.mean(frame_losses)),
            **val_metrics,
        }
        history.append(epoch_record)
        if val_metrics["val_soundscape_macro_roc_auc"] > best_metric:
            best_metric = val_metrics["val_soundscape_macro_roc_auc"]
            model.save_weights(checkpoint_path)
            (run_dir / "validation_predictions.json").write_text(json.dumps(val_rows, indent=2), encoding="utf-8")

    history_path = run_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    settings_path = run_dir / "sed_v5_settings.json"
    settings_path.write_text(json.dumps(settings.__dict__, indent=2), encoding="utf-8")
    metrics = dict(history[-1])
    metrics.pop("epoch", None)
    summary_markdown = "\n".join(
        [
            "## SED v5-like Summary",
            f"- backbone: `{settings.backbone_name}`",
            f"- clip_loss: `{settings.clip_loss_name}`",
            f"- frame_loss: `{settings.frame_loss_name}`",
            f"- epochs: {settings.epochs}",
            f"- steps_per_epoch: {steps_per_epoch}",
            f"- train_samples: {len(train_examples)}",
            f"- val_samples: {len(val_examples)}",
            f"- pseudo_samples: {dataset_summary.get('pseudo_example_count', 0)}",
            f"- val_soundscape_macro_roc_auc={metrics['val_soundscape_macro_roc_auc']:.6f}",
            f"- padded_cmap={metrics['padded_cmap']:.6f}",
        ]
    )
    feature_parts = []
    if settings.clip_loss_name == "asl" or settings.frame_loss_name == "asl":
        feature_parts.append("ASL")
    if dataset_summary.get("pseudo_example_count", 0):
        feature_parts.append("soft pseudo")
    feature_text = " + ".join(feature_parts) if feature_parts else "dual BCE"
    return {
        "metrics": metrics,
        "root_cause": f"B0 SED training, soundscape validation, and {feature_text} supervision are wired into the runtime.",
        "verdict": "baseline-ready",
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "history": str(history_path),
            "settings": str(settings_path),
            "validation_predictions": str(run_dir / "validation_predictions.json"),
        },
        "dataset_summary": dataset_summary,
        "summary_markdown": summary_markdown,
    }


def run_sed_soundscape_inference(config: dict[str, Any], runtime_root: Path, output_dir: Path) -> dict[str, Any]:
    import tensorflow as tf

    paths_cfg = config.get("paths", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    data_root = _resolve_root(runtime_root, str(paths_cfg.get("data_root", "./birdclef-2026")))
    labels = _load_label_order(data_root, data_cfg)
    settings = _build_settings(config, labels)
    checkpoint_path = _resolve_root(runtime_root, str(model_cfg.get("checkpoint_path", "")))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SED checkpoint not found: {checkpoint_path}")

    model = load_sed_model(checkpoint_path, settings)
    mel_transform = MelSpectrogramTransform(settings)
    soundscape_dir = data_root / data_cfg.get("test_soundscapes_dir", "test_soundscapes")
    if not soundscape_dir.exists() or not list(soundscape_dir.glob("*.ogg")):
        soundscape_dir = data_root / data_cfg.get("train_soundscapes_dir", "train_soundscapes")
    paths = sorted(soundscape_dir.glob("*.ogg"))
    max_files = int(data_cfg.get("max_infer_files", 0) or 0)
    if max_files > 0:
        paths = paths[:max_files]
    if not paths:
        raise ValueError(f"No soundscapes found in {soundscape_dir}")

    cache = AudioCache(sample_rate=settings.sample_rate)
    rows: list[dict[str, Any]] = []
    for path in paths:
        waveform = cache.read(path)
        n_chunks = max(1, math.ceil(len(waveform) / settings.chunk_samples))
        padded_len = n_chunks * settings.chunk_samples
        if len(waveform) < padded_len:
            waveform = np.pad(waveform, (0, padded_len - len(waveform)))
        else:
            waveform = waveform[:padded_len]
        chunks = waveform.reshape(n_chunks, settings.chunk_samples).astype(np.float32)
        batch_images = mel_transform(chunks)
        outputs = model(batch_images, training=False)
        probs = outputs["clipwise_prob"].numpy().astype(np.float32)
        for index, values in enumerate(probs, start=1):
            row = {"row_id": f"{path.stem}_{int(index * settings.chunk_duration)}"}
            row.update({label: float(value) for label, value in zip(labels, values, strict=False)})
            rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "sed_soundscape_predictions.csv"
    pd.DataFrame(rows).to_csv(submission_path, index=False)
    return {
        "rows": len(rows),
        "soundscapes": len(paths),
        "submission_csv": str(submission_path),
    }
