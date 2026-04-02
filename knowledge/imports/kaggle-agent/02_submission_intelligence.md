# Submission Intelligence

## Hard Constraints

- BirdCLEF scored submissions are CPU-only notebooks with internet off.
- The notebook is rerun by Kaggle; local `submission.csv` is only a dry-run contract check.
- Daily submission slots and final selection slots are scarce resources.

## Candidate Gating

Do not promote a candidate to submission just because local validation improved slightly.

Only promote when at least one of these is true:

- it is the current best clean holdout candidate,
- it adds genuinely different signal to an ensemble,
- it is a deliberate probe to test a calibration hypothesis,
- it validates notebook packaging or inference runtime risk.

## Anchor Rules

- Keep anchor rows that connect local metric, post-processed local metric, predicted LB, and actual LB.
- Use anchors to calibrate PP gains and probe strategy.
- Do not assume the best local raw model is the best leaderboard model.

## Post-Process Rules

- Record both raw and post-processed scores.
- Record PP local gain separately from expected LB gain.
- If a PP gain is confirmed on LB once, treat it as evidence for that lane, not as a universal rule.

## Notebook Packaging Rules

- The scored bundle should be deterministic and contract-checked locally before any online push.
- Historical examples that assume GPU notebook submission are obsolete for the current competition contract.
- Dataset upload and kernel push are mechanics; the real intelligence is deciding which candidate deserves a slot.
