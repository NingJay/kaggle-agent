## Research Stage: run-0004-0003 Class Coverage Fix

### Current State

The class coverage fix is the new leader at **val_soundscape_macro_roc_auc = 0.6727**, a +0.0078 improvement over the baseline. The run verdict is **submission-required**.

### Root Cause

Coverage expansion from 52→71 fitted classes was the primary gain driver. However, two structural problems remain:
1. **4 classes still unfitted** — 5% of the active target set has no probe representation.
2. **Severe train/val gap** — train soundscape AUC 0.997 vs val 0.673 (gap = 0.324). This indicates the probe head is overfitting to training soundscapes or there is a distribution shift.

### Adopt Now

1. **Build submission bundle from the current leader.** The verdict is submission-required and bundle parity is the stated objective. This should happen before any further experimentation.
2. **Close the remaining 4-class coverage gap** as the next grounded branch. This is the highest-priority structural improvement per the experiment priority order (priority 2: improve class coverage and fitted-class count).
3. **Prefer grounded branches only.** Ledger mining confirms grounded = net positive (+1), novel = flat (0). All branches should derive from the current leader code state.

### Consider (guarded)

- **Probe head regularization** (dropout, weight decay, early stopping) to attack the 0.324 train/val gap. This is a valid axis but the policy confidence is low (0.28, override required). Pursue only as a second parallel branch after submission and coverage are addressed.
- **Prior fusion calibration** could squeeze marginal gains since the Bayesian prior already generalizes well (val 0.662 > train 0.487). However, never couple calibration with coverage changes — round 3 showed this destabilizes fitting.
- **Domain robustness** augmentation if the train/val gap has a distribution-shift component.

### Reject

- **Pseudo-labeling** already tested, produced +0.00013 (flat). Not viable as a primary strategy.
- **Cosmetic sweeps** (threshold/temperature alone) flagged as low-information by the branch typing compiler.
- **Coupled structural changes** — round 3 demonstrated this collapses prior fusion and destabilizes probe fitting.
- **Novel idea-class branches** — policy confidence 0.32, no empirical support.

### Policy Constraints

| Component | Policy | Confidence | Override Required |
|-----------|--------|------------|-------------------|
| grounded | prefer | 0.50 | no |
| novel | conditional | 0.32 | no |
| class_coverage | conditional | 0.28 | yes |
| prior_calibration | conditional | 0.28 | yes |
| probe_head | conditional | 0.28 | yes |

### Capability Signals

- **Veto checker**: No active vetoes or forbidden patterns.
- **Ledger miner**: grounded is the only net-positive component. general is net-negative (-1).
- **Branch typing**: Low-information patterns include calibration_only, blend_only, temperature, threshold. Avoid branches that only touch these axes.

### Contradictions to Resolve Downstream

- **class_coverage**: Mixed evidence between workspace run (positive) and imported round-3 regression.
- **prior_calibration**: Round 3 calibration coupled with coverage caused regression; standalone calibration evidence is incomplete.
- **probe_head**: Mixup augmentation evidence from exp-0008 is the counterevidence; needs retesting on current code state.
