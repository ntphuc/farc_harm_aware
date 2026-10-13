# IMPORTANT DATASET LABEL UPDATE (Fix 1701)

This source supports both legacy BraTS 2021 labels (`ET=4`) and BraTS-GLI 2023+ labels (`ET=3`). The local folder may still be named `BraTS2021`; the code detects the label convention from the segmentation masks and stores the resolved scheme in the manifest. Do not hard-code ET=4 for `BraTS-GLI-...-t1n/t1c/t2w/t2f` data.

Default is `label_scheme: auto`. You can force a convention only for debugging with `label_scheme: legacy_0124` or `label_scheme: gli_0123`. A mismatch is treated as a fatal error instead of silently training on the wrong target.

# BraTS2021 FARC-v2 — Failure-Aware Action/Harm Controller

## Central research question

Can a brain-tumor segmentation system predict **which corrective action will help or harm an individual case before applying it**, then choose among:

- keep the original prediction,
- recover likely false negatives,
- remove likely false positives,
- refine boundaries,
- clean small tails/components,
- or defer the case to human review,

without using ground truth at inference time?

This repository is intentionally designed to test that hypothesis, not to claim novelty for uncertainty maps, error maps, morphological corrections, or human referral separately.

## Important novelty boundary

Prior work already covers major pieces of the pipeline:

- uncertainty / calibration / error alignment;
- reference-free segmentation quality estimation;
- voxel-level error localization;
- FP/FN-specific correction;
- uncertainty-guided refinement;
- human referral / risk-controlled triage;
- conformal risk control.

The hypothesis tested here is narrower: **failure type alone may be insufficient for safe correction; the outcome of each candidate intervention should be estimated per case, including expected benefit and probability of harm, before automatic intervention.**

## BraTS 2021 data expected

Default root:

```text
/datastore/phucngt/datasets/BraTS2021
```

The scanner expects one folder per labeled training case, e.g.

```text
BraTS2021_00000/
  BraTS2021_00000_t1.nii.gz
  BraTS2021_00000_t1ce.nii.gz
  BraTS2021_00000_t2.nii.gz
  BraTS2021_00000_flair.nii.gz
  BraTS2021_00000_seg.nii.gz
```

`t1gd` is also accepted as an alias for `t1ce`.

Only cases with local segmentation labels are used. The hidden official BraTS test labels are not assumed.

## One-command HPC run

```bash
sbatch run_uit.slurm
```

The Slurm file remains only the scheduler configuration plus the shell entry point.

Default environment:

```text
DATA_ROOT=/datastore/phucngt/datasets/BraTS2021
VENV_ROOT=/datastore/phucngt/datasets/venv_cu128
```

If needed:

```bash
/datastore/phucngt/datasets/venv_cu128/bin/python -m pip install -r requirements_brats2021.txt
```

## Leakage-safe patient splits

The 1,251 labeled BraTS 2021 training cases are locally split by patient into independent cohorts:

```text
train         50%  source segmentor training
source_val     8%  source early stopping + error-map validation
error_train   12%  error-map training only
controller    12%  action gain/harm controller training only
val            8%  policy thresholds/calibration only
test          10%  locked final test
```

The exact rounded counts are written to `data/brats2021_manifest.json`.

The separation between `error_train` and `controller` is deliberate: the downstream action controller must not be trained using overfit error-map predictions generated on the same patients used to train the error-map network.

## BraTS targets

The official labels are converted to overlapping regions:

```text
WT = labels {1,2,4}
TC = labels {1,4}
ET = label  {4}
```

The model predicts WT/TC/ET as three sigmoid channels.

## Pipeline

```text
4-modality MRI
    ↓
3D source segmentor
    ↓
WT / TC / ET probability maps
    ↓
uncertainty + morphology + error-risk features
    ↓
FP/FN error map (auxiliary, not claimed novel)
    ↓
construct candidate actions
    ↓
for each action predict:
    expected utility gain
    probability of harm
    ↓
risk-aware controller
    ↓
keep / correction-A / correction-B / correction-C / correction-D / human_review
```

## Candidate actions

The action bank is intentionally simple and interpretable:

```text
0 no_correction
1 recover_fn
2 remove_fp
3 boundary_align
4 tail_cleanup
```

The novelty test is not whether these morphological operations are new. It is whether the controller can predict *which one is useful or harmful for this patient/region*.

## Ground truth usage contract

Ground truth **is allowed only** for:

- source/error-map/controller training targets;
- measuring candidate-action outcomes during development;
- diagnostic/oracle baselines;
- locked final evaluation.

Ground truth is **not** used by deployment-visible controller features or by the action policy at inference time.

## Main outputs

For each backbone/seed:

```text
outputs/brats2021_farc_v2/<backbone>/seed*/
  segmentor/
  error_map/
  action_outcomes/
  controller/
  policy/
  locked_test/
    summary.json
    decisions.csv
    end_to_end/summary.json   # final ET⊂TC⊂WT-consistent inference-style evaluation
    baselines/baseline_comparison.csv
    feature_ablations/feature_ablation.csv
    hypotheses/hypothesis_summary.json
```

Aggregate files:

```text
outputs/brats2021_farc_v2/analysis/by_seed.csv
outputs/brats2021_farc_v2/analysis/summary_mean.csv
```

## Experiments automatically run

The suite tests:

1. source segmentation on WT / TC / ET;
2. learned FP/FN error maps versus entropy;
3. action-gain regression and harm classification;
4. frozen policy calibration on validation only;
5. locked-test FARC policy;
6. source-keep-all baseline;
7. quality-only referral;
8. entropy referral at matched review rate;
9. diagnostic GT failure-type routing (upper-bound diagnostic only);
10. reference-free error-map failure routing;
11. gain-only controller without harm model;
12. full controller without human referral;
13. always-apply each corrective action;
14. oracle best-action upper bound;
15. feature ablations: no error-map, no uncertainty, no region identity, geometry/action-only;
16. direct hypothesis test that matching an action to the true failure type does not always improve a case;
17. true end-to-end locked-test evaluation with ET⊂TC⊂WT consistency enforced after action selection.

## Central hypothesis tests

### H1 — Failure type is not enough

If the GT-derived matching correction still harms a non-zero subset of cases or fails to benefit all cases, then knowing the failure type does not guarantee safe correction.

### H2 — Action outcome is predictable

Action-specific `delta_utility` should have useful rank correlation / low MAE, and the harm model should achieve useful AUROC on validation.

### H3 — Explicit harm modeling matters

`full_action_harm_referral` should reduce harmful-action rate compared with `gain_only_no_harm` and failure-type routing while retaining useful automatic coverage.

### H4 — Referral has a reason

Human referral is used when no candidate action clears the predicted benefit/harm constraints and the source prediction is not sufficiently high-quality—not simply when entropy crosses one threshold.

## Metrics to emphasize in the paper

Do not make mean Dice the only headline. Report at least:

```text
Dice WT / TC / ET
HD95 (approx. physical mm after resize-spacing correction)
action gain MAE / Spearman
harm AUROC
harmful correction rate
beneficial correction rate
automatic coverage
case-level human-review rate
automated-case failure rate + Clopper–Pearson upper bound
oracle action gap
```

## Important limitation

The default implementation resizes 3D scans to 128^3 for tractable experimentation. Effective voxel spacing is tracked so HD95 is expressed approximately in physical mm, but this is **not an official BraTS leaderboard reproduction**. For a final high-tier submission, repeat the key experiment with a strong full-resolution baseline such as nnU-Net and external data.

## Recommended paper claim

A defensible claim is:

> We formulate post-hoc segmentation correction as a case-specific action-selection problem, learning the expected benefit and harm of alternative corrections from inference-time-visible signals and deferring cases when no action is predicted to be sufficiently safe.

Do **not** claim that this is the first work on uncertainty, error localization, FP/FN correction, automatic refinement, or human referral.
