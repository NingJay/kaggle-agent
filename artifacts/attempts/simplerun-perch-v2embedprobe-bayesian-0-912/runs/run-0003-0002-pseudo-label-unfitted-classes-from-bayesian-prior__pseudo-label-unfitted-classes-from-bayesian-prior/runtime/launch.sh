#!/usr/bin/env bash
set +e
source /home/staff/jiayining/miniconda3/etc/profile.d/conda.sh
conda activate kaggle-agent
cd /home/staff/jiayining/kaggle/worktrees/kaggle-agent-branch-search/artifacts/attempts/simplerun-perch-v2embedprobe-bayesian-0-912/runs/run-0003-0002-pseudo-label-unfitted-classes-from-bayesian-prior__pseudo-label-unfitted-classes-from-bayesian-prior/runtime/code_state_workspace
export KAGGLE_AGENT_WORKSPACE_ROOT=/home/staff/jiayining/kaggle/worktrees/kaggle-agent-branch-search
export KAGGLE_AGENT_WORK_ITEM_ID=workitem-0002-pseudo-label-unfitted-classes-from-bayesian-prior
export KAGGLE_AGENT_EXPERIMENT_ID=exp-0002-pseudo-label-unfitted-classes-from-bayesian-prior
export KAGGLE_AGENT_RUN_ID=run-0003-0002-pseudo-label-unfitted-classes-from-bayesian-prior
export KAGGLE_AGENT_RUN_DIR=/home/staff/jiayining/kaggle/worktrees/kaggle-agent-branch-search/artifacts/attempts/simplerun-perch-v2embedprobe-bayesian-0-912/runs/run-0003-0002-pseudo-label-unfitted-classes-from-bayesian-prior__pseudo-label-unfitted-classes-from-bayesian-prior/runtime
export KAGGLE_AGENT_SPEC_ID=spec-0003
export KAGGLE_AGENT_PRIMARY_METRIC=val_soundscape_macro_roc_auc
export KAGGLE_AGENT_SECONDARY_METRICS='["soundscape_macro_roc_auc", "padded_cmap", "prior_fusion_macro_roc_auc", "val_prior_fusion_macro_roc_auc"]'
export KAGGLE_AGENT_CODE_STATE_REF=/home/staff/jiayining/kaggle/worktrees/kaggle-agent-branch-search/state/worktrees/codegen/06-codegen__running/workspace
python /home/staff/jiayining/kaggle/worktrees/kaggle-agent-branch-search/artifacts/attempts/simplerun-perch-v2embedprobe-bayesian-0-912/runs/run-0003-0002-pseudo-label-unfitted-classes-from-bayesian-prior__pseudo-label-unfitted-classes-from-bayesian-prior/runtime/code_state_workspace/train_sed.py --config /home/staff/jiayining/kaggle/worktrees/kaggle-agent-branch-search/artifacts/attempts/simplerun-perch-v2embedprobe-bayesian-0-912/runs/run-0003-0002-pseudo-label-unfitted-classes-from-bayesian-prior__pseudo-label-unfitted-classes-from-bayesian-prior/runtime/code_state_workspace/BirdCLEF-2026-Codebase/configs/generated/pseudo-label-unfitted-classes-from-bayesian-prior.yaml paths.data_root=/home/staff/jiayining/kaggle/kaggle_comp_seed/BirdCLEF-2026/dataset data.train_csv=train.csv data.taxonomy_csv=taxonomy.csv data.sample_submission_csv=sample_submission.csv data.train_audio_dir=train_audio data.train_soundscapes_dir=train_soundscapes data.train_soundscapes_labels_csv=train_soundscapes_labels.csv data.test_soundscapes_dir=test_soundscapes data.perch_cache_dir=/home/staff/jiayining/kaggle/kaggle_comp_seed/BirdCLEF-2026/input/perch-meta model.perch_model_dir=/home/staff/jiayining/kaggle/kaggle_comp_seed/BirdCLEF-2026/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1
status=$?
echo "$status" > /home/staff/jiayining/kaggle/worktrees/kaggle-agent-branch-search/artifacts/attempts/simplerun-perch-v2embedprobe-bayesian-0-912/runs/run-0003-0002-pseudo-label-unfitted-classes-from-bayesian-prior__pseudo-label-unfitted-classes-from-bayesian-prior/runtime/exit_code.txt
exit "$status"
