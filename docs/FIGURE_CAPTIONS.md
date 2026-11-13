# Figure captions and panel guide

All figures are generated from the repeated stratified out-of-fold evaluation in
`results/local_eval.json` (5 folds × 4 repeats) and saved at 300 dpi as
PNG, PDF and SVG.

## Figure 1 — Cohort and GARF-H architecture (`fig1_cohort_and_architecture`)

**a.** Clinical trajectory across the three tasks: pre-biopsy MRI/PSA decision,
post-biopsy management, and post-surgery recurrence risk.

**b.** Cohort composition: labeled vs unlabeled cases in each task.

**c.** Endpoint distribution. Task 1: biopsy yes/no. Task 2: active surveillance
(AS), active treatment (AT), continued surveillance (CS), watchful waiting (WW).
Task 3: biochemical recurrence event vs censoring.

**d.** Availability of MRI, biopsy WSI, and prostatectomy WSI neural
representations in the labeled cohorts.

**e.** Rule-anchored multimodal architecture. Inputs (structured prompt, clinical
notes, MRI/WSI sets) pass through a feature layer (PSA kinetics, pathology
concepts, text signals, neural set statistics), a guideline prior plus exception
model, and finally decision/survival/reasoning outputs. Task 3 uses a rank
ensemble of pathology risk, neural event probability and hazard trajectory.

## Figure 2 — Task 1 biopsy decision (`fig2_task1_biopsy`)

**a.** ROC curve with 95% bootstrap band. The diamond marks the operating point
of the PI-RADS guideline rule.

**b.** Precision–recall curve with average precision and the cohort prevalence.

**c.** Calibration curve (quantile bins) with Brier score.

**d.** Confusion matrix; cells show counts and overall percentages (the four
cells sum to 100%).

**e.** Operating characteristics across probability thresholds (positive F1,
accuracy, precision, recall). The dot marks the maximum-F1 threshold.

**f.** Case-level separation: cases sorted by predicted P(biopsy), coloured by
observed outcome; open circles mark misclassifications.

## Figure 3 — Task 2 management (`fig3_task2_management`)

**a.** Management class distribution.

**b.** Confusion matrix with counts.

**c.** Per-class precision, recall and F1; the legend is placed above the bars.

**d.** Row-normalised error composition: for each true class, the proportion
assigned to each predicted class.

**e.** Per-class F1 with 95% bootstrap confidence intervals. The wide interval for
WW reflects its support of only two cases.

**f.** Misclassified cases table with three labelled columns: Case, True → Pred,
and Confidence.

## Figure 4 — Task 3 recurrence (`fig4_task3_recurrence`)

**a.** Kaplan–Meier recurrence-free survival by predicted-risk tertile, with
censoring ticks and a low-vs-high log-rank test.

**b.** Component ablation of the current GARF-H model. Harrell C-index with 95%
bootstrap CI for pathology only, neural event model, hazard trajectory, their
combinations, Cox PH, the clinical-only hybrid, and the final rank ensemble.

**c.** Predicted risk distribution by observed event status.

**d.** Event-probability calibration: observed event rate by predicted-probability
quartile with Wilson confidence intervals.

**e.** Cumulative/dynamic AUC at 12, 24, 36, 48 and 60 months.

**f.** Model-implied cumulative recurrence incidence over time by risk tertile.

## Figure 5 — Model performance and reasoning (`fig5_model_comparison`)

**a.** GARF-H vs clinical anchor (guideline rule for Tasks 1/2; pathology score
for Task 3) on the primary metric: Task 1 positive F1, Task 2 weighted F1,
Task 3 C-index. Error bars are 95% bootstrap CIs.

**b.** Task 1 reasoning components of GARF-H: confidence, variable weights, tool
order, rationale, and composite.

**c.** Task 2 reasoning components of GARF-H (same definitions).

**d.** Case-score composition for Task 1 and Task 2. Task 3 has no reference reasoning in the release and is reported as C-index in panels a and f.

**e.** Per-case reasoning composite distribution for Task 1 and Task 2; box plots
and points show the cohort spread, the white marker shows the mean.

**f.** Task-level summary table: decision metric, clinical anchor, GARF-H with
95% CI, accuracy, reasoning, and case score.
