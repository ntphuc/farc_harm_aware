# Research-gap validation — checked October 2026

## What is already occupied by prior work

The following broad claims are not safe novelty claims:

- "the model knows when it is wrong";
- "the model knows where it is wrong";
- "the model predicts quality without ground truth";
- "the model uses uncertainty to correct FP/FN";
- "the model sends uncertain cases to a human";
- "the model uses uncertainty/risk for quality control".

Examples that make those broad claims unsafe include:

- Jungo et al.: uncertainty/error localization and failure detection for brain-tumor segmentation;
- QCResUNet: subject-level quality + voxel-level error maps, including BraTS 2021 and external cohorts;
- FNPC-SAM: uncertainty-guided false-negative/false-positive correction;
- TEGDA (MICCAI 2025): reference-free quality estimation guiding refinement/adaptation, including BraTS 2023;
- EviRisk-Seg (2026): error-aligned uncertainty and reference-free high-risk regions;
- Badr et al. (MICCAI UNSURE 2026): region-aware risk-controlled human triage for post-treatment glioma;
- Gerber et al., Scientific Reports 2026: uncertainty-based 3D error localization, clinical-metric impact estimation, and prioritized correction regions on hip MRI;
- conformal 3D lesion/tumor segmentation work: explicit statistical risk control.

## Closest 2026 paper found

Gerber et al., "Automated error localisation and correction techniques for deep-learning-based segmentation of 3D MRI sequences based on feature-derived-region aggregation" (Scientific Reports, 2026) is especially close conceptually.

It:

- localizes likely 3D segmentation errors;
- aggregates uncertainty into feature-derived regions;
- estimates impact of regions on clinical metrics;
- prioritizes regions for correction;
- evaluates guided automatic/manual-like correction.

The remaining distinction for FARC-v2 is that it treats **several alternative corrective actions as competing interventions** and tries to predict the *patient/region-specific consequence of each action before selecting it*, including an explicit harm probability and the option to defer when no automated action is acceptable.

## Narrow gap retained for this project

A more defensible research question is:

> Can a reference-free medical-segmentation controller predict the expected benefit and harm of multiple alternative corrective interventions for an individual prediction, and use those estimates to choose between retaining the prediction, applying a specific correction, or deferring to human review?

This is narrower than "failure-aware segmentation" and should remain phrased as a gap supported by a literature search, not as an absolute claim that no prior work anywhere has ever considered action value.

## Scientific observation required before claiming novelty

The paper becomes scientifically stronger if the data demonstrate:

> Failure type does not uniquely determine correction success.

For example, even among cases that are truly FN-dominant, `recover_fn` should sometimes help, sometimes do little, and sometimes harm. That motivates action-specific outcome prediction instead of a hand-coded error-type-to-action lookup.

If the experiment does **not** show this heterogeneity, the central motivation weakens substantially.

## What would make the paper look like module stacking

If the final method is effectively:

```text
FN score high -> recover_fn
FP score high -> remove_fp
uncertainty high -> human
```

then novelty is limited because every ingredient has close precedent.

The key technical/scientific burden is therefore to show:

1. action outcomes are heterogeneous even conditional on failure type;
2. action benefit/harm is predictable from deployment-visible signals;
3. explicit harm prediction changes decisions and reduces harmful intervention;
4. this improvement remains after matched referral/coverage comparisons;
5. the effect is reproducible across seeds and a second segmentation backbone.

## Next Q1-strengthening step

After the BraTS 2021 proof, the highest-value extension is action-policy reliability under domain shift:

- learn/calibrate on BraTS 2021;
- test on an external glioma cohort or another appropriately harmonized BraTS-domain dataset;
- measure whether predicted action gain/harm remains calibrated;
- determine whether referral catches action-policy failures.

That asks a deeper question than ordinary segmentation generalization: does the system still know which intervention is safe after the data distribution changes?
