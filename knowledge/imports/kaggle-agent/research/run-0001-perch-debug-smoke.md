Perch-head debug smoke completed with 0.5 macro ROC-AUC and 0.0085 padded cMAP. Both metrics indicate the model is producing random-guess predictions with no discriminative power.

The root cause is a training mechanics failure: either the Perch classification head is frozen, the optimizer isn't targeting it, or the data pipeline is feeding mismatched labels. The next move is to inspect the debug config and train_sed.py to verify the head is trainable, receiving gradients, and connected to correct training labels.

Fix the training loop before any downstream work. Threshold tuning, post-processing, ensemble, or architecture changes are all premature when the model hasn't learned anything yet.
