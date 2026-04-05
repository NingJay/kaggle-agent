from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from birdclef_runtime.backends import build_backbone
from birdclef_runtime.data import Sample, build_dataset
from birdclef_runtime.metrics import macro_roc_auc, padded_cmap, sigmoid

DEFAULT_PRIMARY_METRIC = "val_soundscape_macro_roc_auc"


def _dot(lhs: list[float], rhs: list[float]) -> float:
    return sum(left * right for left, right in zip(lhs, rhs))


def _train_python_debug(
    train_embeddings: list[list[float]],
    train_targets: list[list[int]],
    val_embeddings: list[list[float]],
    *,
    epochs: int,
    learning_rate: float,
) -> tuple[list[list[float]], list[list[float]], list[float]]:
    label_count = len(train_targets[0])
    feature_dim = len(train_embeddings[0])
    weights = [[0.0 for _ in range(feature_dim)] for _ in range(label_count)]
    biases = [0.0 for _ in range(label_count)]
    for _ in range(epochs):
        for features, target in zip(train_embeddings, train_targets):
            for label_index in range(label_count):
                score = biases[label_index] + _dot(weights[label_index], features)
                probability = sigmoid(score)
                error = probability - target[label_index]
                biases[label_index] -= learning_rate * error
                row = weights[label_index]
                for feature_index, feature_value in enumerate(features):
                    row[feature_index] -= learning_rate * error * feature_value
    predictions: list[list[float]] = []
    for features in val_embeddings:
        row_scores = []
        for label_index in range(label_count):
            row_scores.append(sigmoid(biases[label_index] + _dot(weights[label_index], features)))
        predictions.append(row_scores)
    return weights, predictions, biases


def _train_tensorflow_keras(
    train_embeddings: list[list[float]],
    train_targets: list[list[int]],
    val_embeddings: list[list[float]],
    *,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    dropout: float,
) -> tuple[Any, list[list[float]]]:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("tensorflow is required for tensorflow_keras training backend") from exc
    label_count = len(train_targets[0])
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(len(train_embeddings[0]),)),
            tf.keras.layers.Dense(hidden_dim, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(label_count, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
    )
    model.fit(train_embeddings, train_targets, epochs=epochs, verbose=0)
    predictions = model.predict(val_embeddings, verbose=0).tolist()
    return model, predictions


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _disable_output_mirror() -> bool:
    return os.environ.get("KAGGLE_AGENT_DISABLE_OUTPUT_MIRROR", "").strip() == "1"


def _prepare_output_dirs(config: dict[str, Any], runtime_root: Path) -> tuple[Path, Path]:
    run_dir_env = os.environ.get("KAGGLE_AGENT_RUN_DIR", "").strip()
    experiment_name = str(config["experiment"]["name"])
    output_root = (runtime_root / config["paths"]["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    default_dir = output_root / experiment_name
    run_dir = Path(run_dir_env).resolve() if run_dir_env else default_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if _disable_output_mirror():
        return run_dir, run_dir
    mirrored_dir = default_dir
    mirrored_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, mirrored_dir


def _mirror_files(source_dir: Path, mirrored_dir: Path, filenames: list[str]) -> None:
    if _disable_output_mirror() or source_dir == mirrored_dir:
        return
    for filename in filenames:
        source = source_dir / filename
        if source.exists():
            destination = mirrored_dir / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def _select_primary_metric(config: dict[str, Any], metrics: dict[str, float]) -> tuple[str, float]:
    configured = str(
        config.get("metrics", {}).get(
            "primary",
            os.environ.get("KAGGLE_AGENT_PRIMARY_METRIC", DEFAULT_PRIMARY_METRIC),
        )
    )
    if "val_soundscape_macro_roc_auc" in metrics:
        primary_name = "val_soundscape_macro_roc_auc"
    elif configured in metrics:
        primary_name = configured
    elif "soundscape_macro_roc_auc" in metrics:
        primary_name = "soundscape_macro_roc_auc"
    else:
        primary_name = next(iter(metrics))
    return primary_name, float(metrics[primary_name])


def run_training(config: dict[str, Any], runtime_root: Path) -> dict[str, Any]:
    training_cfg = config.get("training", {})
    backend = training_cfg.get("backend", "python_debug")
    run_dir, mirrored_dir = _prepare_output_dirs(config, runtime_root)

    if backend == "perch_teacher_cache":
        from birdclef_runtime.perch_teacher import run_perch_teacher_cache

        teacher_result = run_perch_teacher_cache(config, runtime_root, run_dir)
        metrics = teacher_result["metrics"]
        primary_metric, primary_value = _select_primary_metric(config, metrics)
        secondary_names = list(config.get("metrics", {}).get("secondary", []))
        secondary_metrics = {
            name: float(metrics[name])
            for name in secondary_names
            if name in metrics and name != primary_metric
        }
        result = {
            "experiment_name": config["experiment"]["name"],
            "config_path": config["_config_path"],
            "primary_metric_name": primary_metric,
            "primary_metric_value": primary_value,
            "secondary_metrics": secondary_metrics,
            "all_metrics": metrics,
            "root_cause": teacher_result["root_cause"],
            "verdict": teacher_result["verdict"],
            "artifacts": teacher_result["artifacts"],
            "dataset_summary": teacher_result["dataset_summary"],
            "summary_markdown": teacher_result["summary_markdown"],
        }
        _write_json(run_dir / "result.json", result)
        _write_json(run_dir / "metrics.json", {"primary": primary_metric, "primary_value": primary_value, "metrics": metrics})
        _write_json(run_dir / "artifacts.json", teacher_result["artifacts"])
        _write_text(run_dir / "summary.md", teacher_result["summary_markdown"] + "\n")
        _mirror_files(
            run_dir,
            mirrored_dir,
            ["result.json", "metrics.json", "artifacts.json", "summary.md"],
        )
        return result

    if backend == "sklearn_cached_probe":
        from birdclef_runtime.cached_probe import run_cached_probe_experiment

        cached_probe = run_cached_probe_experiment(config, runtime_root, run_dir)
        metrics = cached_probe["metrics"]
        primary_metric, primary_value = _select_primary_metric(config, metrics)
        secondary_names = list(config.get("metrics", {}).get("secondary", []))
        secondary_metrics = {
            name: float(metrics[name])
            for name in secondary_names
            if name in metrics and name != primary_metric
        }
        result = {
            "experiment_name": config["experiment"]["name"],
            "config_path": config["_config_path"],
            "primary_metric_name": primary_metric,
            "primary_metric_value": primary_value,
            "secondary_metrics": secondary_metrics,
            "all_metrics": metrics,
            "root_cause": cached_probe["root_cause"],
            "verdict": cached_probe["verdict"],
            "artifacts": cached_probe["artifacts"],
            "dataset_summary": cached_probe["dataset_summary"],
            "summary_markdown": cached_probe["summary_markdown"],
        }
        _write_json(run_dir / "result.json", result)
        _write_json(run_dir / "metrics.json", {"primary": primary_metric, "primary_value": primary_value, "metrics": metrics})
        _write_json(run_dir / "artifacts.json", cached_probe["artifacts"])
        _write_text(run_dir / "summary.md", cached_probe["summary_markdown"] + "\n")
        _mirror_files(
            run_dir,
            mirrored_dir,
            ["result.json", "metrics.json", "artifacts.json", "summary.md", "oof_predictions.npz", "probe_bundle.pkl", "probe_metadata.json"],
        )
        return result

    if backend == "tensorflow_sed_v5":
        from birdclef_runtime.sed_v5 import run_sed_v5_training

        sed_result = run_sed_v5_training(config, runtime_root, run_dir)
        metrics = sed_result["metrics"]
        primary_metric, primary_value = _select_primary_metric(config, metrics)
        secondary_names = list(config.get("metrics", {}).get("secondary", []))
        secondary_metrics = {
            name: float(metrics[name])
            for name in secondary_names
            if name in metrics and name != primary_metric
        }
        result = {
            "experiment_name": config["experiment"]["name"],
            "config_path": config["_config_path"],
            "primary_metric_name": primary_metric,
            "primary_metric_value": primary_value,
            "secondary_metrics": secondary_metrics,
            "all_metrics": metrics,
            "root_cause": sed_result["root_cause"],
            "verdict": sed_result["verdict"],
            "artifacts": sed_result["artifacts"],
            "dataset_summary": sed_result["dataset_summary"],
            "summary_markdown": sed_result["summary_markdown"],
        }
        _write_json(run_dir / "result.json", result)
        _write_json(run_dir / "metrics.json", {"primary": primary_metric, "primary_value": primary_value, "metrics": metrics})
        _write_json(run_dir / "artifacts.json", sed_result["artifacts"])
        _write_text(run_dir / "summary.md", sed_result["summary_markdown"] + "\n")
        _mirror_files(
            run_dir,
            mirrored_dir,
            ["result.json", "metrics.json", "artifacts.json", "summary.md", "best_sed_v5.weights.h5", "history.json", "validation_predictions.json", "sed_v5_settings.json"],
        )
        return result

    if backend == "tensorflow_sed_v5_infer":
        from birdclef_runtime.sed_v5 import run_sed_soundscape_inference

        inference_result = run_sed_soundscape_inference(config, runtime_root, run_dir / "inference")
        metrics = {
            "soundscapes_processed": float(inference_result["soundscapes"]),
            "inference_rows": float(inference_result["rows"]),
        }
        primary_metric, primary_value = _select_primary_metric(config, metrics)
        secondary_names = list(config.get("metrics", {}).get("secondary", []))
        secondary_metrics = {
            name: float(metrics[name])
            for name in secondary_names
            if name in metrics and name != primary_metric
        }
        summary_markdown = "\n".join(
            [
                "## SED v5-like Inference Summary",
                f"- soundscapes_processed: {int(metrics['soundscapes_processed'])}",
                f"- inference_rows: {int(metrics['inference_rows'])}",
                f"- submission_csv: `{inference_result['submission_csv']}`",
            ]
        )
        result = {
            "experiment_name": config["experiment"]["name"],
            "config_path": config["_config_path"],
            "primary_metric_name": primary_metric,
            "primary_metric_value": primary_value,
            "secondary_metrics": secondary_metrics,
            "all_metrics": metrics,
            "root_cause": "Notebook-faithful SED inference path is runnable as an independent OS-managed experiment.",
            "verdict": "inference-ready",
            "artifacts": {"submission_csv": inference_result["submission_csv"]},
            "dataset_summary": {
                "soundscapes_processed": int(metrics["soundscapes_processed"]),
                "inference_rows": int(metrics["inference_rows"]),
            },
            "summary_markdown": summary_markdown,
        }
        _write_json(run_dir / "result.json", result)
        _write_json(run_dir / "metrics.json", {"primary": primary_metric, "primary_value": primary_value, "metrics": metrics})
        _write_json(run_dir / "artifacts.json", result["artifacts"])
        _write_text(run_dir / "summary.md", summary_markdown + "\n")
        _mirror_files(
            run_dir,
            mirrored_dir,
            ["result.json", "metrics.json", "artifacts.json", "summary.md", "inference/sed_soundscape_predictions.csv"],
        )
        return result

    dataset = build_dataset(config, runtime_root)
    backbone = build_backbone(config, runtime_root)
    train_samples: list[Sample] = dataset["train_samples"]
    val_samples: list[Sample] = dataset["val_samples"]
    train_embeddings = [backbone.embed(sample) for sample in train_samples]
    val_embeddings = [backbone.embed(sample) for sample in val_samples]
    train_targets = [sample.target for sample in train_samples]
    val_targets = [sample.target for sample in val_samples]
    epochs = int(training_cfg.get("epochs", 3))
    learning_rate = float(training_cfg.get("learning_rate", 0.05))
    hidden_dim = int(training_cfg.get("hidden_dim", 128))
    dropout = float(training_cfg.get("dropout", 0.0))

    artifacts: dict[str, str] = {}
    if backend == "python_debug":
        weights, predictions, biases = _train_python_debug(
            train_embeddings,
            train_targets,
            val_embeddings,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        checkpoint_payload = {"weights": weights, "biases": biases, "labels": dataset["labels"]}
        checkpoint_path = run_dir / "checkpoint.json"
        _write_json(checkpoint_path, checkpoint_payload)
        artifacts["checkpoint"] = str(checkpoint_path)
    elif backend == "tensorflow_keras":
        model, predictions = _train_tensorflow_keras(
            train_embeddings,
            train_targets,
            val_embeddings,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        checkpoint_path = run_dir / "model.keras"
        model.save(str(checkpoint_path))
        artifacts["checkpoint"] = str(checkpoint_path)
    else:
        raise ValueError(f"Unsupported training backend: {backend}")

    metrics = {
        "soundscape_macro_roc_auc": macro_roc_auc(val_targets, predictions),
        "val_soundscape_macro_roc_auc": macro_roc_auc(val_targets, predictions),
        "padded_cmap": padded_cmap(val_targets, predictions),
    }
    primary_metric, primary_value = _select_primary_metric(config, metrics)
    secondary_names = list(config.get("metrics", {}).get("secondary", []))
    secondary_metrics = {name: float(metrics[name]) for name in secondary_names if name in metrics and name != primary_metric}
    summary_markdown = "\n".join(
        [
            "## Runtime Summary",
            f"- Backbone provider: {config['model']['backbone_provider']}",
            f"- Training backend: {backend}",
            f"- Train samples: {dataset['dataset_summary']['train_sample_count']}",
            f"- Val samples: {dataset['dataset_summary']['val_sample_count']}",
            f"- Label count: {dataset['dataset_summary']['label_count']}",
            f"- Primary metric: {primary_metric}={primary_value:.6f}",
        ]
    )
    threshold = float(config.get("metrics", {}).get("submission_candidate_threshold", 0.85))
    verdict = "submission-candidate" if primary_value >= threshold else "baseline-ready"
    result = {
        "experiment_name": config["experiment"]["name"],
        "config_path": config["_config_path"],
        "primary_metric_name": primary_metric,
        "primary_metric_value": primary_value,
        "secondary_metrics": secondary_metrics,
        "all_metrics": metrics,
        "root_cause": "runtime completed",
        "verdict": verdict,
        "artifacts": artifacts,
        "dataset_summary": dataset["dataset_summary"],
        "summary_markdown": summary_markdown,
    }
    _write_json(run_dir / "result.json", result)
    _write_json(run_dir / "metrics.json", {"primary": primary_metric, "primary_value": primary_value, "metrics": metrics})
    _write_json(run_dir / "artifacts.json", artifacts)
    _write_text(run_dir / "summary.md", summary_markdown + "\n")
    _mirror_files(run_dir, mirrored_dir, ["result.json", "metrics.json", "artifacts.json", "summary.md"])
    return result
