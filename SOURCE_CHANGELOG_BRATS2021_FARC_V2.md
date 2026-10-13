# Source changelog — BraTS2021 FARC-v2

- Added native BraTS 2021 four-modality NIfTI discovery and WT/TC/ET conversion.
- Added 3D U-Net source model and residual 3D U-Net replication model.
- Added six-channel FP/FN error-map model for WT/TC/ET.
- Added interpretable 3D correction action bank.
- Added action-outcome table generation using GT only as development/evaluation targets.
- Added action-specific gain regression and harmful-action classification.
- Added policy calibration with validation-only grid search and Clopper–Pearson upper risk estimate.
- Added `human_review` when no action passes benefit/harm constraints and source quality is insufficient.
- Added leakage-safe split with separate `error_train` and `controller` cohorts.
- Added locked-test baseline suite: keep-all, entropy referral, quality referral, failure routing, gain-only, no-human, always-correct, oracle.
- Added feature-ablation suite.
- Added direct H1 test: matching correction to true failure type is not necessarily safe/useful.
- Added effective post-resize voxel spacing for approximate physical-mm HD95.
- Added one-entry HPC execution through `run_uit.slurm`.
