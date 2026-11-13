# Supplementary Figure S1 — Benchmark against traditional ML/DL and LLM agents

**a.** Task 1 biopsy decision: positive F1 (95% bootstrap CI) for the guideline
rule, logistic regression, random forest, gradient boosting, MLP, and GARF-H.

**b.** Task 2 management decision: weighted F1 for the same methods. GARF-H
blends the guideline prior with a seed-averaged ExtraTrees ensemble; it improves
weighted F1 over the rule (0.896 vs. 0.875) and adds structured reasoning
(composite 0.638 vs. no reasoning baseline for the traditional models).

**c.** Task 3 recurrence risk: Harrell C-index for the pathology anchor, Cox PH,
random-forest event model, MLP event model, ExtraTrees survival, and GARF-H.

**d.** Benchmark summary table. The table labels each method, its family, and its
primary metric on each task; “—” indicates that the method was not run on that
task.

## Protocol

- The same stratified subset per task (n = 40) is used for all methods.
- Local models use 5-fold × 4-repeat stratified cross-validation; LLM agents make
  one pass over the same cases.
- LLM agents use the same two-stage structured prompt (evidence extraction →
  decision/reasoning); keys are read from environment variables or an interactive
  terminal prompt and are never stored in the repository.
- Primary metrics: positive F1 (Task 1), weighted F1 (Task 2), Harrell C-index
  (Task 3); 95% bootstrap CIs.
- The benchmark is a supporting validation, not a leaderboard; the small cohort
  and the single LLM pass mean the comparison is descriptive.

## Key results (local models)

| Task | Guideline/anchor | Best traditional ML/DL | GARF-H |
|---|---:|---:|---:|
| 1 — positive F1 | 0.814 | 0.766 (MLP) | **0.906** (0.808–0.978) |
| 2 — weighted F1 | 0.875 | 0.852 (random forest) | **0.896** (0.795–0.976; reasoning 0.638) |
| 3 — C-index | 0.815 (pathology) | 0.781 (Cox PH) | **0.875** (0.781–0.957) |

## LLM-agent results

| Task | DeepSeek-Flash | GLM-5.3 | GARF-H |
|---|---:|---:|---:|
| 1 — positive F1 | 0.781 (40/40) | 0.794 (40/40) | **0.906** |
| 2 — weighted F1 | 0.503 (40/40) | 0.369 (40/40) | **0.896** |
| 3 — C-index | 0.777 (40/40) | 0.757 (40/40) | **0.875** |

GLM-5.3 initially lost Task 1/2 cases to truncated JSON. The client now validates
cached responses, uses a 16000-token budget with a length-triggered retry, parses
only complete objects, normalises decision labels and makes one minimal repair
call if needed; all three tasks then completed 40/40. The comparison is
descriptive, not a ranking of APIs.
