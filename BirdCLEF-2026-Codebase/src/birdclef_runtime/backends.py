from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from birdclef_runtime.data import Sample


class MockHashBackbone:
    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim

    def embed(self, sample: Sample) -> list[float]:
        seed = f"{sample.sample_id}|{sample.audio_path}|{sample.metadata}"
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        values: list[float] = []
        for index in range(self.embedding_dim):
            byte = digest[index % len(digest)]
            values.append((byte / 255.0) * 2.0 - 1.0)
        return values


class PerchSavedModelBackbone:
    def __init__(self, model_dir: Path, embedding_dim: int) -> None:
        self.model_dir = model_dir
        self.embedding_dim = embedding_dim
        self._model = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import tensorflow as tf
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("tensorflow is required for perch_saved_model backbone") from exc
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Perch model directory not found: {self.model_dir}")
        self._model = tf.saved_model.load(str(self.model_dir))
        return self._model

    def embed(self, sample: Sample) -> list[float]:
        try:
            import soundfile as sf
            import tensorflow as tf
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("soundfile and tensorflow are required for perch_saved_model backbone") from exc
        waveform, sample_rate = sf.read(str(sample.audio_path), dtype="float32", always_2d=False)
        if hasattr(waveform, "ndim") and waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sample_rate != 32000:
            raise RuntimeError(f"Expected 32kHz audio for perch backend, got {sample_rate} for {sample.audio_path}")
        waveform = waveform[:160000]
        if len(waveform) < 160000:
            padding = tf.zeros([160000 - len(waveform)], dtype=tf.float32)
            waveform = tf.concat([tf.convert_to_tensor(waveform), padding], axis=0)
        else:
            waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
        model = self._load_model()
        output = model(tf.expand_dims(waveform, axis=0))
        if isinstance(output, dict):
            tensor = next(iter(output.values()))
        else:
            tensor = output
        vector = tf.reshape(tensor, [-1]).numpy().tolist()
        if len(vector) >= self.embedding_dim:
            return [float(value) for value in vector[: self.embedding_dim]]
        return [float(value) for value in vector] + [0.0] * (self.embedding_dim - len(vector))


def build_backbone(config: dict[str, Any], runtime_root: Path) -> Any:
    model_cfg = config.get("model", {})
    provider = model_cfg.get("backbone_provider", "mock_hash")
    embedding_dim = int(model_cfg.get("embedding_dim", 32))
    if provider == "mock_hash":
        return MockHashBackbone(embedding_dim)
    if provider == "perch_saved_model":
        model_dir = (runtime_root / "models" / model_cfg.get("perch_model_dir", "")).resolve()
        return PerchSavedModelBackbone(model_dir, embedding_dim)
    raise ValueError(f"Unsupported backbone provider: {provider}")
