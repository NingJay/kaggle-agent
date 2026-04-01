Preflight debug run completed with identical random-guess performance across three consecutive attempts (ROC-AUC=0.5, cMAP=0.0085). Pipeline infrastructure is stable but the model produces no discriminative signal.

The training loop executes without errors but the model isn't learning. This is a training mechanics failure, not a data or architecture problem. Three likely root causes: the Perch classification head is frozen, the optimizer isn't configured to update head parameters, or the data pipeline is feeding mismatched labels.

Fix training mechanics first. Verify the head is trainable, optimizer config includes head parameters, and labels match model output shape. Add gradient flow diagnostics to catch frozen layers early and sanity-check that training loss decreases within the first few batches.

All downstream work—threshold tuning, post-processing, ensemble, architecture changes—is blocked until the model produces non-random predictions.
