# Historical document — legacy analysis workflow

> This file describes the pre-2026-11 analysis workflow and its old figure layout. The current pipeline is documented in [PROJECT_REPORT_CN.md](../PROJECT_REPORT_CN.md) and [BENCHMARK.md](../BENCHMARK.md); the current figures are in `results/figures/`.

# CHIMERA-Agent analysis workflow and plotting guide

> **2026-11 update:** the current optimized pipeline, results and figure suite are
> documented in `docs/PROJECT_REPORT_CN.md`, `docs/BENCHMARK.md` and `scripts/make_report_figures.py`. This document is
> retained as the original workflow/API-benchmark guide.

This document describes the complete analysis loop used to compare local models and API agents.

## 1. End-to-end workflow


1. **Audit the dataset**  
   Count labeled/unlabeled cases, labels, missing modalities, and reasoning traces.

2. **Establish guideline baselines**  
   - Task 1: PI-RADS rule.  
   - Task 2: biopsy ISUP rule.  
   - Task 3: surgical pathology risk score.

3. **Train GARF-M3**  
   Combine guideline priors with clinical, pathology, and neural-representation residual learners.

4. **Add neural/semi-supervised learning**  
   Use MRI and WSI set statistics plus conservative pseudo-labeling.

5. **Benchmark API agents**  
   Call DeepSeek/GLM through direct Python HTTP requests. Each case uses:
   - Stage 1: clinical evidence extraction;
   - Stage 2: structured decision prediction.

6. **Score reasoning**  
   Evaluate confidence, variable weights, tool trace, and rationale.

7. **Perform error analysis**  
   Inspect false positives, false negatives, rare-class failures, and calibration.

8. **Plot results**  
   Generate metric summaries, confusion matrices, ROC curves, survival-risk plots, and reasoning radar charts.

## 2. Architecture


The architecture is:

```text
Dataset
  ├─ guideline rules
  ├─ clinical/pathology parsing
  ├─ neural representation set statistics
  ├─ residual fusion
  ├─ reasoning generator
  ├─ local evaluator
  └─ API comparison
```

## 3. Direct API benchmark

The direct API module does not require the OpenAI SDK.

### DeepSeek

```powershell
$env:DEEPSEEK_API_KEY = "..."
python scripts/run_direct_api.py `
  --data-root "E:/Projects/chimera-agent/train_release_updated_25_08/train_release" `
  --task 1 `
  --provider deepseek `
  --model deepseek-flash `
  --max-cases 20 `
  --output results/api_deepseek_flash_task1.json
```

Available DeepSeek aliases:

```text
deepseek-flash
deepseek-v4-pro
```

### GLM

```powershell
$env:ZHIPU_API_KEY = "..."
python scripts/run_direct_api.py `
  --data-root "E:/Projects/chimera-agent/train_release_updated_25_08/train_release" `
  --task 2 `
  --provider zhipu `
  --model glm-flash `
  --max-cases 20 `
  --output results/api_glm_flash_task2.json
```

Available GLM aliases:

```text
glm-flash  -> glm-5.3-flash
glm-flashx -> glm-5.3-flashx
glm-pro    -> glm-5.3
glm-legacy-air -> glm-4.5-air
glm-legacy -> glm-4.5
glm-image  -> glm-4v-flash
```

The API call is two-stage:

```text
clinical JSON
  -> evidence extraction JSON
  -> decision/reasoning JSON
```

Responses are cached under:

```text
work/api_cache/<provider>/<model>/
```

## 4. Plot generation

After local and API evaluations:

```powershell
python scripts/plot_analysis.py `
  --local results/local_eval_smoke.json `
  --local-neural results/local_eval_neural_semi_smoke.json `
  --api-glob "results/api_*_task*.json" `
  --output-dir results/figures
```

Generated figures:

| File | Meaning |
|---|---|
| `analysis_workflow.png` | End-to-end analysis pipeline |
| `architecture.png` | Model and evaluation architecture |
| `primary_metrics.png` | Task 1/2/3 primary metric comparison |
| `case_scores.png` | Local case-score comparison |
| `task1_confusion.png` | Task 1 confusion matrix |
| `task1_roc.png` | Task 1 out-of-fold ROC curve |
| `task2_confusion.png` | Task 2 confusion matrix |
| `task3_risk_vs_time.png` | Task 3 predicted risk vs observed time |
| `task3_event_rate_by_risk.png` | Event rate by predicted-risk quartile |
| `reasoning_radar.png` | Confidence, weights, tool trace, and rationale |

## 5. Current local plots






