# Benchmark: GARF-H vs Traditional ML/DL and LLM Agents

## 1. Purpose and positioning

This benchmark is a **supporting validation**, not the paper's main contribution. The main contribution is the GARF-H architecture: guideline-anchored exception learning, rank-ensemble survival, and supervised structured reasoning.

The benchmark answers a narrower question:

> Under the same case subset and the same evaluation script, does GARF-H still hold an advantage over conventional learners and general-purpose LLM agents?

It is not a leaderboard. The cohort is small, the LLM arm uses one pass per case, and the confidence intervals are wide.

---

## 2. Protocol

| Item | Design |
|---|---|
| Subset | 40 cases per task, stratified by label/event; Task 2 keeps at least two watchful-waiting cases |
| Local validation | 5-fold × 4-repeat stratified CV, out-of-fold predictions |
| LLM validation | One pass over the same subset, same two-stage prompt |
| Task 1 metric | Positive F1 (primary), accuracy, AUC |
| Task 2 metric | Weighted F1 (primary), accuracy, macro F1 |
| Task 3 metric | Harrell C-index (primary) |
| Uncertainty | 95% bootstrap CIs |

The exact case IDs are stored in `results/tables/cases_task1.json`, `cases_task2.json`, and `cases_task3.json`.

---

## 3. Methods

### 3.1 Anchors and traditional ML/DL

- **Guideline rule** (Task 1/2): PI-RADS ≥ 4 → biopsy; ISUP-based management rule.
- **Pathology score** (Task 3): surgical pathology + preoperative PSA.
- **Logistic regression**: standardized tabular features, balanced class weights.
- **Random forest**: 400 trees, class-balanced subsampling.
- **Gradient boosting**: 200 trees, depth 2.
- **MLP**: 32→16 hidden units, L2 regularization.

### 3.2 Survival baselines (Task 3)

- **Cox PH**: ridge-regularized Breslow partial likelihood on curated clinical/pathology features.
- **Random-forest event**: random forest event classifier on clinical + compact neural set statistics; risk = P(event).
- **MLP event**: MLP event classifier on clinical features; risk = P(event).
- **ExtraTrees survival**: ExtraTrees regression on log follow-up time; risk = −prediction.

### 3.3 LLM agents

- **DeepSeek-Flash** (`deepseek-flash`).
- **GLM-5.3** (`glm-5.3`).
- Two-stage prompt: structured clinical case → evidence extraction JSON → decision/reasoning JSON.
- Same case subset, temperature 0, responses cached.
- Keys are read from `DEEPSEEK_API_KEY` / `ZHIPU_API_KEY` or prompted with `getpass`; they are never stored in the repository.

---

## 4. Results

### 4.1 Task 1 — positive F1

| Method | n | Positive F1 (95% CI) | Accuracy |
|---|---:|---:|---:|
| **GARF-H** | 40 | **0.906 (0.808–0.978)** | 0.875 |
| Guideline rule | 40 | 0.814 (0.692–0.903) | 0.725 |
| DeepSeek-Flash | 40 | 0.781 (0.655–0.879) | 0.650 |
| GLM-5.3 | 40 | 0.794 (0.678–0.886) | 0.675 |
| MLP | 40 | 0.766 (0.622–0.880) | 0.725 |
| Gradient boosting | 40 | 0.755 (0.596–0.862) | 0.675 |
| Random forest | 40 | 0.745 (0.591–0.857) | 0.675 |
| Logistic regression | 40 | 0.735 (0.578–0.851) | 0.675 |

### 4.2 Task 2 — weighted F1

| Method | n | Weighted F1 (95% CI) | Accuracy |
|---|---:|---:|---:|
| **GARF-H** | 40 | **0.896 (0.795–0.976)** | 0.900 |
| Guideline rule | 40 | 0.875 (0.768–0.975) | 0.875 |
| Random forest | 40 | 0.852 (0.732–0.975) | 0.875 |
| Logistic regression | 40 | 0.822 (0.681–0.938) | 0.825 |
| Gradient boosting | 40 | 0.776 (0.638–0.913) | 0.800 |
| MLP | 40 | 0.729 (0.590–0.873) | 0.750 |
| DeepSeek-Flash | 40 | 0.503 (0.313–0.671) | 0.550 |
| GLM-5.3 | 40 | 0.369 (0.210–0.556) | 0.500 |

GARF-H blends the guideline prior with a three-seed ExtraTrees ensemble in probability space (alpha = 0.65) and adds the rule label as a categorical feature. It therefore improves on the rule (weighted F1 0.896 vs. 0.875; accuracy 0.900 vs. 0.875) and retains the structured-reasoning advantage: composite 0.638, versus 0.368 for DeepSeek-Flash and 0.409 for GLM-5.3.

### 4.3 Task 3 — C-index

| Method | n | Harrell C-index (95% CI) |
|---|---:|---:|
| **GARF-H** | 40 | **0.875 (0.781–0.957)** |
| Pathology score | 40 | 0.815 (0.685–0.934) |
| Cox PH | 40 | 0.781 (0.598–0.928) |
| DeepSeek-Flash | 40 | 0.777 (0.597–0.919) |
| GLM-5.3 | 40 | 0.757 (0.564–0.925) |
| Random-forest event | 40 | 0.726 (0.516–0.909) |
| MLP event | 40 | 0.717 (0.461–0.920) |
| ExtraTrees survival | 40 | 0.670 (0.449–0.853) |

---

## 5. GLM-5.3 JSON reliability

The first GLM-5.3 run lost 19/40 Task 1 and 18/40 Task 2 cases. The cause was
**output truncation, not authentication**: the evidence-extraction JSON exceeded
the 3000-token completion budget, so the response ended mid-object. Two further
defects kept the failures alive: the cache replayed the truncated content on
every retry, and the parser accepted an inner dictionary (e.g. `variable_weights`)
as the whole answer, yielding `decision = None`.

Fixes now in the released client:

- a stricter parser that only accepts a top-level object containing the required
  key (`decision` for Tasks 1/2, `patient` for evidence extraction);
- cache entries are validated before reuse; malformed or truncated entries are
  refetched instead of replayed;
- the GLM completion budget is 16000 tokens, and a `finish_reason = "length"`
  response is retried once with a doubled budget;
- decision labels are normalised (e.g. `AS`, `active surveillance`,
  `watchful-waiting`);
- if the decision is still missing, one minimal schema-only repair call is made;
- API errors (408/429/5xx) are retried with backoff, and tool traces returned as
  objects are normalised before reasoning scoring.

With these changes, GLM-5.3 completed **40/40 Task 1**, **40/40 Task 2** and
**40/40 Task 3** cases. With complete evidence, its Task 1 positive F1 improved
to 0.794, while its Task 2 weighted F1 fell to 0.369: the model becomes more
aggressive and assigns most surveillance candidates to active treatment once
the full PSA kinetics and pathology are visible. This behaviour is itself part
of the descriptive comparison; no prompt was tuned per model.

## 6. Interpretation

- GARF-H leads on all three tasks.
- The guideline/pathology anchors are strong baselines; GARF-H improves on them in all three tasks, including Task 2 after the rule-prior residual blend.
- Traditional ML/DL methods are competitive in Task 2 but fall behind in Task 1 and Task 3.
- DeepSeek-Flash completes all cases but underperforms GARF-H on all three tasks.
- GLM-5.3 completes every case after the JSON hardening, scores above GARF-H on none, and is markedly more aggressive in Task 2 (most surveillance cases are assigned to active treatment); its format fragility and output variance remain deployment concerns.
- The benchmark supports the paper's narrative that a compact, guideline-anchored, deterministic model is a strong alternative to general-purpose agents in small clinical cohorts.

---

## 7. Reproduce

```bash
# Local models
python scripts/run_benchmark.py \
  --data-root /path/to/train_release \
  --output-dir results/benchmark \
  --n-per-task 40 --splits 5 --repeats 4

# LLM agents (interactive keys; not stored)
python scripts/run_llm_benchmark.py \
  --data-root /path/to/train_release \
  --models deepseek-flash glm-5.3 --interactive-keys

# Summary table and Figure 6
python scripts/combine_benchmark_tables.py   --benchmark results/benchmark/benchmark_results.json   --llm results/benchmark/llm_results.json   --output results/benchmark/benchmark_combined

python scripts/make_benchmark_figure.py   --benchmark results/benchmark/benchmark_results.json   --llm results/benchmark/llm_results.json   --output-dir results/benchmark/figures
```

Generated artifacts (pre-generated copies are in `results/tables/` and `results/figures/`):

- `results/tables/benchmark_results.json/.csv/.md`
- `results/tables/llm_results.json/.md`
- `results/tables/benchmark_combined.md/.csv`
- `results/figures/fig6_benchmark_comparison.*`

Running the scripts writes the same files under `results/benchmark/` by default.

---

## 8. Limitations

- Small subset (n=40 per task); CIs are wide.
- Single LLM pass per case; API outputs are stochastic and cost-limited.
- The LLM arm is descriptive and not powered to rank APIs.
- Task 2 has only two watchful-waiting cases; class-level conclusions are fragile.
- Task 3 has only ~10 events in the subset.
- The benchmark is not the paper's contribution; it is a sanity check that the architecture is not outperformed by generic baselines.
