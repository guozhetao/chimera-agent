# Changelog

## 2026-11-13 — Task 2 residual fusion and figure refinements

- **Task 2 model:** replaced the rule-only decision path with a rule-prior + residual blend. The guideline probabilities are fused in logit space (rule weight 0.65) with a three-seed soft-voting ExtraTrees ensemble, and the rule label is added as a categorical feature.
- **Task 2 results:** full OOF weighted F1 0.850 → **0.861** (95% CI 0.773–0.944), accuracy 0.847 → **0.861**; 40-case benchmark weighted F1 0.875 → **0.896** (0.795–0.976).
- **Figure 1:** returned to the 2×3 layout and kept the architecture panel deliberately simple (inputs → feature layer → rule anchor → outputs).
- **Figure 5:** panel d now shows case-score composition for Tasks 1/2 only; Task 3 has no reference reasoning and is reported as C-index in panels a and f.
- Regenerated all figures, OOF tables, benchmark tables and Figure 6 from the updated evaluation.
- **GLM-5.3 reliability:** cached responses are validated before reuse; the completion budget is 16000 tokens with a length-triggered retry; the parser accepts only complete top-level objects; decision labels are normalised and one minimal repair call is made when needed. All three tasks now complete **40/40** (Task 1 F1 0.794, Task 2 weighted F1 0.369, Task 3 C-index 0.757).
- Fixed the evaluator crash when a model returned `reveal_sequence` as objects instead of strings.
- Added `scripts/retry_glm_benchmark.py` and `tests/test_direct_api.py`.

## 2026-11-07 — Benchmark, LLM agents, release packaging

- Added the supporting benchmark against logistic regression, random forest, gradient boosting, MLP, Cox PH, event/survival baselines, DeepSeek-Flash and GLM-5.3.
- Added `scripts/run_benchmark.py`, `run_llm_benchmark.py`, `combine_benchmark_tables.py`, `make_benchmark_figure.py`, `rebuild_llm_results.py`.
- Added benchmark tables and Figure 6.
- GLM-5.3 Task 1/2 evidence-extraction JSON truncation was diagnosed and fixed (3000 → 8000 tokens, robust JSON parser); retry recovered Task 1/2 to 39/40 and 37/40.
- Reworked the main figure suite: consistent panel labels, legends inside panels, Task 1 overall-percentage confusion matrix, Task 2 error-table headers, Task 3 ablation, Task 5 summary table.
- Added the Chinese project report and GitHub release documentation.

## 2026-11-06 — Optimized GARF-H pipeline

- Replaced the rule/model logit blend with a rule-anchored exception model for Task 1.
- Preserved the guideline rule for Task 2; moved the improvement to supervised structured reasoning.
- Replaced the Task 3 z-score blend with a rank ensemble of pathology risk, neural event probability and hazard trajectory.
- Added stratified repeated CV, bootstrap CIs, calibration/decision-curve/Kaplan–Meier/time-dependent AUC figures.
- Added `HybridSurvivalModel`, regularized Cox PH, supervised reasoning generation and the current test suite.

## 2026-10 — Initial prototype

- Dataset loading and audit, guideline rules, local evaluator, API agents, initial plotting and submission scripts.
