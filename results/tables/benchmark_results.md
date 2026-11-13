# Baseline and LLM benchmark

All methods use the same stratified subset and repeated folds. Values are out-of-fold estimates; 95% bootstrap CIs are shown for the primary metric.

| Task | Method | Type | n | Metric | Value (95% CI) | Accuracy | Weighted F1 | Reasoning | Case score |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| 1 | Guideline rule | Rule | 40 | f1_positive | 0.814 | 0.725 | 0.687 |  |  |
| 1 | Logistic regression | Traditional ML | 40 | f1_positive | 0.735 | 0.675 | 0.677 |  |  |
| 1 | Random forest | Traditional ML | 40 | f1_positive | 0.745 | 0.675 | 0.673 |  |  |
| 1 | Gradient boosting | Traditional ML | 40 | f1_positive | 0.755 | 0.675 | 0.666 |  |  |
| 1 | MLP | Deep learning | 40 | f1_positive | 0.766 | 0.725 | 0.729 |  |  |
| 1 | Ours (GARF-H) | Proposed | 40 | f1_positive | 0.906 | 0.875 | 0.872 | 0.673 | 0.847 |
| 2 | Guideline rule | Rule | 40 | f1_weighted | 0.875 | 0.875 | 0.875 |  |  |
| 2 | Logistic regression | Traditional ML | 40 | f1_weighted | 0.822 | 0.825 | 0.822 |  |  |
| 2 | Random forest | Traditional ML | 40 | f1_weighted | 0.852 | 0.875 | 0.852 |  |  |
| 2 | Gradient boosting | Traditional ML | 40 | f1_weighted | 0.776 | 0.800 | 0.776 |  |  |
| 2 | MLP | Deep learning | 40 | f1_weighted | 0.729 | 0.750 | 0.729 |  |  |
| 2 | Ours (GARF-H) | Proposed | 40 | f1_weighted | 0.896 (0.795–0.976) | 0.900 | 0.896 | 0.636 | 0.831 |
| 3 | Pathology score | Clinical anchor | 40 | harrell_c_index | 0.815 (0.685–0.934) |  |  |  |  |
| 3 | Cox PH | Traditional survival | 40 | harrell_c_index | 0.781 (0.598–0.928) |  |  |  |  |
| 3 | RF event | Traditional ML | 40 | harrell_c_index | 0.726 (0.516–0.909) |  |  |  |  |
| 3 | MLP event | Deep learning | 40 | harrell_c_index | 0.717 (0.461–0.920) |  |  |  |  |
| 3 | ET survival | Traditional survival | 40 | harrell_c_index | 0.670 (0.449–0.853) |  |  |  |  |
| 3 | Ours (GARF-H) | Proposed | 40 | harrell_c_index | 0.875 (0.781–0.957) |  |  |  |  |

> Case score is shown only for methods that produce structured reasoning (GARF-H and LLM agents).  
> The benchmark is descriptive and supporting, not a leaderboard.
