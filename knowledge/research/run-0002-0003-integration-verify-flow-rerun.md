Integration verify flow rerun reproduced the exact same random-guess baseline (ROC-AUC=0.5, cMAP=0.0085) as all previous debug runs. Pipeline infrastructure is stable and deterministic, but the core training mechanics failure remains unresolved.

The model executes without errors but produces zero learning signal across multiple attempts. This is a training configuration issue, not a data or architecture problem. The Perch classification head is either frozen, excluded from the optimizer, or receiving misaligned labels.

The immediate priority is fixing training mechanics: verify the head is trainable, confirm optimizer targets head parameters, validate label-output alignment, and add gradient flow diagnostics. All downstream work—tuning, post-processing, ensemble, architecture changes—is blocked until the model learns something beyond random guessing.

Debug config is validated as stable for integration testing but should never be used for model evaluation or experiment comparison.
