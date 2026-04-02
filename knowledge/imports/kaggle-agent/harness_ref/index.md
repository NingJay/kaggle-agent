# Harness Reference Index

This folder organizes the local reference set imported from:

- `/home/staff/jiayining/kaggle/harness_ref/kaggle_exist`

This layer is archival support, not the primary operating memory.

For future experiments, read these first instead:

- `knowledge/00_experiment_rules.md`
- `knowledge/01_validated_findings.md`
- `knowledge/02_submission_intelligence.md`
- `knowledge/03_next_experiment_priors.md`

The normalized full-text extracts live under `knowledge/harness_ref/raw/`.

## How To Use This Archive

- Treat the report family as historical evidence used to derive the current operating memory.
- Treat the legacy codebase README and automatic submission workflow as historical implementation references, not as the current repo contract.
- When a historical source conflicts with the current BirdCLEF contract, prefer the current repo runtime and `COMPETITION.md`.

## Source Groups

### Competition And Legacy Codebase

- `BirdCLEF2026_Analysis_Report.html`
  Focus: competition framing, CPU-only inference, macro ROC-AUC, dataset challenges, prior-solution scan.
- `README.md`
  Focus: older Perch fine-tuning codebase surface.
  Caution: still describes padded cMAP as the metric and contains legacy training assumptions.

### Validation, Progress, And Core Reporting

- `experiment_report_20260316.html`
  Focus: data split and validation discipline, explicit holdout construction.
- `progress_report.html`
  Focus: holdout-first progress tracking and no-leak language.
- `experiment_report.html`
  Focus: combined LB, proxy validation, and holdout reporting.
- `master_report_20260318.html`
  Focus: control-room overview, running experiments, recommendations, known issues.
- `discovery_report_20260319.html`
  Focus: promoted ideas, failed ideas, next moves, anchor context.

### Submission Intelligence And Calibration

- `nohuman_pp_eval_20260315_2158.html`
  Focus: local raw vs post-process eval, LB calibration, anchor-based decision making.
- `automatic_submission_workflow.txt`
  Focus: dataset upload and kernel push flow for Kaggle code competitions.
  Caution: historical example includes `enable_gpu = true`, which does not match the current BirdCLEF CPU-only scored path.

### Modeling And Research Workstreams

- `public_sed_comparison_report.html`
  Focus: public SED gap analysis and adoption candidates.
- `sed_improvement_plan.html`
  Focus: SED roadmap from current baseline toward stronger LB.
- `sed_model_comparison_20260320.html`
  Focus: backbone and training recipe comparison under soundscape validation.
- `domain_generalization_report_20260320.html`
  Focus: DG framing, target-domain gap, why SFDA is the wrong mental model here.

### Earlier Experimental Snapshots

- `experiment_report (1).html`
  Focus: earlier embedding-head report with mixed local metrics.
- `experiment_report_v2.html`
  Focus: label-head, nohuman, post-processing, pseudo-label strategy.

## Durable Operating Themes

- Report artifacts are first-class working memory, not decorative outputs.
- Validation must distinguish holdout, proxy-target, raw local, post-process, and LB-anchor signals.
- Submission planning is its own workstream: anchor tracking, calibration, and scarce slot use matter.
- Public SED adoption and Perch/domain-generalization work should remain separate but connected research lanes.
- Historical sources often mix multiple metric types in one dashboard; future summaries should always label metric scope and trust.

## Raw Extracts

- Inventory: `knowledge/harness_ref/raw_inventory.md`
- Full normalized text: `knowledge/harness_ref/raw/*.txt`
