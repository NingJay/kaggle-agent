# BirdCLEF 2026 Runtime

This runtime is the `BirdCLEF Runtime` plane inside `kaggle_agent`.

It now has two paths:

- `configs/debug.yaml`: smoke path using a mock embedding backbone and a pure-Python trainer
- `configs/default.yaml`: real Perch-head path using a SavedModel backbone and TensorFlow head training

## Entrypoints

```bash
python train.py --config configs/debug.yaml
python train.py --config configs/default.yaml
python inference.py --config configs/default.yaml --checkpoint /abs/path/to/checkpoint
python analyze_results.py --output-root outputs
```

## Expected Data Layout

```text
BirdCLEF-2026-Codebase/
├── birdclef-2026/
│   ├── train_audio/
│   ├── train_soundscapes/
│   ├── train.csv
│   ├── taxonomy.csv
│   ├── train_soundscapes_labels.csv
│   └── sample_submission.csv
└── models/
    └── bird-vocalization-classifier-tensorflow2-perch_v2-v2/
```

## Artifact Contract

Each run writes:

- `result.json`
- `metrics.json`
- `artifacts.json`
- `summary.md`

When launched by the control plane, these files are written into `artifacts/runs/<run_id>/` and mirrored to `outputs/<experiment_name>/`.
