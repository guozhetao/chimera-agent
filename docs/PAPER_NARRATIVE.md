# Paper narrative and figure plan

## Working title

**A guideline-anchored multimodal agent for prostate cancer decision-making across the diagnostic-to-postoperative continuum**

## One-sentence story

Clinical guidelines encode strong, interpretable priors, but they are rigid; multimodal data are high-dimensional, incomplete, and available in small cohorts. We introduce **GARF-H** (Guideline-Anchored Residual Fusion – Hybrid), which keeps the guideline as an anchor, learns only its exceptions, fuses pathology and neural evidence through a rank-robust survival ensemble, and generates structured clinical reasoning. The model is evaluated on three linked tasks: biopsy decision, post-biopsy management, and post-surgery recurrence risk.

The benchmark against traditional ML/DL and LLM APIs is a **supporting validation**, not the paper's main contribution. The main claim is the architecture and the clinical reasoning design.

## Why this is not "just another benchmark"

1. **Guideline anchoring rather than guideline replacement.** On small, noisy cohorts, a classifier trained from scratch is consistently worse than the PI-RADS/ISUP rule. We model the *residual*: the guideline remains the prior, and a low-variance exception layer learns when to override it.
2. **Rank-robust survival fusion.** The pathology risk score is heavy-tailed (PSA up to 190 ng/mL). A z-score blend lets outliers dominate. We fuse pathology risk, a neural event model, and a discrete-time hazard trajectory in rank space, which is stable and interpretable.
3. **Supervised structured reasoning.** The reference traces contain confidence, ordinal variable weights, tool-reveal order, and a short rationale. We learn the dominant structure and generate a case-specific rationale, instead of hard-coding it.
4. **One model family, three clinical stages.** The same design principles—anchor, residual, rank fusion, structured reasoning—are applied to pre-biopsy, post-biopsy, and post-surgery decisions.

## Contributions

- A guideline-anchored exception learner for binary biopsy decisions (Task 1).
- A rule-preserving management strategy with supervised structured reasoning for four-class management (Task 2).
- A pathology-anchored rank-ensemble survival model with neural event evidence and a monotone hazard trajectory (Task 3).
- A reproducible evaluation protocol with repeated stratified CV, bootstrap CIs, calibration, decision-curve, Kaplan–Meier, and time-dependent AUC analyses.
- A supporting comparison against logistic regression, random forest, gradient boosting, MLP, Cox PH, survival/event baselines, and two LLM APIs (DeepSeek-Flash, GLM-5.3).

## Figure plan

| Figure | Role in the story |
|---|---|
| Figure 1 | Cohort, clinical trajectory, and the GARF-H architecture. |
| Figure 2 | Task 1: discrimination, calibration, operating characteristics, case-level separation. |
| Figure 3 | Task 2: class imbalance, per-class performance, error composition, uncertainty. |
| Figure 4 | Task 3: risk-stratified survival, component ablation, calibration, time-dependent AUC. |
| Figure 5 | Rule baseline vs GARF-H, reasoning improvement, case-score decomposition, task-level summary. |
| Figure 6 (supplementary) | Benchmark against traditional ML/DL and LLM APIs. |

The benchmark figure is supplementary. In the main text it should be framed as:

> "To contextualize performance, we compared with conventional tabular models, survival baselines, and two general-purpose LLM agents under the same case subset and evaluation script. GARF-H remained the strongest method on all three tasks; the comparison is descriptive, and the study is not powered to rank LLM APIs."

## Main results (OOF, 5-fold × 4 repeats)

| Task | Metric | Guideline/anchor | GARF-H |
|---|---|---:|---:|
| 1 | Positive F1 | 0.806 | **0.825** (95% CI 0.750–0.892) |
| 1 | Accuracy | 0.714 | **0.758** |
| 1 | ROC AUC | ≈0.77 | **0.827** (0.733–0.909) |
| 2 | Weighted F1 | 0.850 | **0.861** (95% CI 0.773–0.944) |
| 2 | Accuracy | 0.847 | **0.861** |
| 3 | Harrell C-index | 0.807 | **0.917** (0.872–0.958) |

Reasoning composite: Task 1 **0.667**; Task 2 **0.636**.

## Task 3 component ablation

| Component | C-index |
|---|---:|
| Pathology anchor | 0.807 |
| Neural event model | 0.837 |
| Hazard trajectory | 0.832 |
| Pathology + event | 0.890 |
| Full component blend | 0.898 |
| Final rank ensemble | **0.917** |
| Cox PH | 0.822 |
| Clinical-only hybrid | 0.819 |

## Why a compact model rather than an LLM-only solution?

General-purpose LLMs are flexible but non-deterministic, expensive at cohort scale, and prone to uncalibrated or inconsistent structured output. The proposed model is small, deterministic, locally deployable, and calibrated on the target cohort; it also exposes structured variable weights and a rationale. The LLM benchmark is therefore a **stress test**: if a compact, guideline-anchored model remains competitive with general-purpose agents, that is evidence for the design rather than against it.

## Benchmark framing

- **Question:** does the proposed architecture still hold an advantage when compared with conventional learners and general-purpose LLM agents?
- **Design:** the same stratified subset per task (n=40), the same 5-fold × 4-repeat splits for local models, and the same two-stage prompt for LLM APIs.
- **Local baselines:** logistic regression, random forest, gradient boosting, MLP; Cox PH, random-forest event model, MLP event model, ExtraTrees survival for Task 3.
- **LLM agents:** DeepSeek-Flash and GLM-5.3, called through the same structured two-stage prompt; keys read from environment variables and never stored in the repository.
- **Primary metrics:** positive F1 (Task 1), weighted F1 (Task 2), Harrell C-index (Task 3), with bootstrap CIs.
- **Key findings:** GARF-H remains strongest on all three tasks. DeepSeek-Flash completed all cases but underperformed (Task 1 F1 0.781; Task 2 weighted F1 0.503; Task 3 C-index 0.777). GLM-5.3 initially lost Task 1/2 cases to truncated JSON; after cache validation, a 16000-token budget, strict parsing, decision normalisation and one repair call, it completed 40/40 on all tasks (Task 1 F1 0.794; Task 2 weighted F1 0.369; Task 3 C-index 0.757). With complete evidence it became markedly more aggressive in Task 2, assigning most surveillance candidates to active treatment; this output variance is itself part of the deployment finding.
- **Interpretation:** the benchmark is descriptive and supporting. It is not a leaderboard, and the small cohort means CIs overlap for some methods. The contrast between deterministic, calibrated GARF-H and format-fragile general-purpose agents is part of the argument for a compact, guideline-anchored model.

## Limitations to state explicitly

- Small labeled cohorts (91/72/75); Task 2 has only two watchful-waiting cases; Task 3 has 19 events.
- Task 1 labels contain genuine clinical uncertainty; the exception model improves precision more than recall.
- Task 2 keeps the rule as a prior but blends it with a seed-averaged ensemble; direct classifiers were worse in stratified CV, and the blend improves weighted F1 over the rule.
- LLM benchmark uses a subset and a single pass per case; API outputs are stochastic and cost-limited.
- The benchmark is not the contribution; it is a sanity check that the architecture is not outperformed by generic baselines.
