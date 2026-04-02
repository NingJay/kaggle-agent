# Validated Findings

## Validation And Metric Semantics

- Holdout validation is the most trustworthy local signal for iteration.
- Small soundscape validation subsets can be useful, but they are not a universal ranking metric across different training regimes.
- Historical reports repeatedly show that one metric can look weak while another better-matched split shows the model is actually strong.

## Perch And Embedding Findings

- Embedding-head style models can add complementary signal even when their standalone score is not the best.
- Joint use of focal and soundscape-derived information is valuable because it improves coverage and representation alignment across domains.
- When fitted class count is much lower than active class count, the next move should target coverage first, not just threshold tuning.

## Post-Processing Findings

- Post-processing can create clear gains on leaderboard-facing behavior, but local raw gain does not transfer 1:1 to LB gain.
- Treat post-processing as a calibrated outer-loop tool, not as a standalone reason to promote a model.
- If PP helps one lane strongly, verify the effect again with anchors before generalizing it to another lane.

## Submission Findings

- Submission is not just notebook packaging.
- Candidate choice, predicted LB, actual LB, anchor history, and scarce slot usage all matter.
- Historical reports show that anchor-based calibration can uncover useful candidates that local metrics alone would mis-rank.

## Domain Generalization Findings

- The right framing is DG: source focal recordings and target soundscapes differ in device response, density, noise, geography, and label structure.
- Useful existing DG-like behavior already appears in joint PCA alignment, model soup, output blending, and primitive soundscape simulation.
- Future DG work should build on those strengths rather than restarting from abstract theory.

## SED Findings

- Public SED references are useful mainly as adoption checklists.
- The highest-value public gaps are frontend and training semantics, not ornamental architecture changes.
- The most transferable upgrades are PCEN-like frontend behavior, better secondary-label loss semantics, waveform-side augmentation, and stronger optimization discipline.
