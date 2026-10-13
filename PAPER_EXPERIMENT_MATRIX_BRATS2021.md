# BraTS 2021 paper experiment matrix

| Experiment | Question | Key output | What would support the hypothesis? |
|---|---|---|---|
| E0 Source segmentation | Is the starting model usable? | Dice/HD95 WT,TC,ET | Reasonable non-trivial baseline |
| E1 Error-map vs entropy | Can learned error cues add signal beyond generic uncertainty? | FP/FN AUPRC/AUROC | Error-map improves AUPRC, at least for meaningful regions |
| E2 Failure-type heterogeneity | Does knowing the true failure type guarantee its matching correction works? | benefit/harm rates by failure type | benefit <100% and/or harm >0% |
| E3 Action gain prediction | Can we forecast correction utility before applying it? | MAE, Spearman | positive useful rank correlation |
| E4 Harm prediction | Can harmful corrections be identified? | AUROC | clearly above chance and stable |
| E5 Full policy | Does action-aware policy improve safety? | harm rate, auto Dice, coverage, review rate | lower harm at useful coverage |
| E6 Gain-only ablation | Is explicit harm prediction necessary? | full vs gain-only | full lowers harmful corrections |
| E7 Failure routing baseline | Is error type alone enough? | GT diagnostic route + reference-free route | FARC safer than routing |
| E8 Entropy referral | Is a simple uncertainty threshold enough? | matched-referral baseline | FARC better at similar review burden |
| E9 No-referral ablation | Does human deferral add safety? | full vs no-human | lower auto failure with referral |
| E10 Always-correct | Why not just refine everything? | always action baselines | demonstrates avoidable harm |
| E11 Oracle | How much headroom remains? | oracle best action | quantifies action-selection gap |
| E12 Feature ablation | Which evidence matters? | no error-map/no UQ/no region/etc. | interpretable contribution |
| E13 Seeds | Is effect stable? | 3 U-Net seeds | consistent direction |
| E14 Backbone replication | Is result model-specific? | ResUNet replication | same qualitative conclusion |

## Primary outcome hierarchy

Primary safety outcome:

```text
harmful_action_rate among automatically managed corrected units
```

Co-primary operational outcomes:

```text
automatic coverage
case-level human review rate
auto-managed failure rate
```

Secondary outcomes:

```text
Dice / HD95
action benefit rate
action gain MAE / Spearman
harm AUROC
error-map AUPRC
```

## Important comparison logic

The most convincing comparison is not simply FARC vs source Dice. It is:

```text
failure-type routing
      vs
action gain only
      vs
action gain + harm
      vs
action gain + harm + referral
```

This sequence directly tests the scientific thesis instead of merely reporting a larger model.
