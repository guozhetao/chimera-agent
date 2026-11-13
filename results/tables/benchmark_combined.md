# Combined benchmark table

Local models: 5-fold × 4-repeat stratified CV on the shared subset. LLM agents: one pass over the same subset with the same two-stage prompt. Values are out-of-fold estimates; 95% bootstrap CIs are shown where available.

| Task | Method | Type | n | Metric | Value (95% CI) | Accuracy | Weighted F1 | Reasoning | Case score | Note |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| 1 | Ours (GARF-H) | Proposed | 40 | f1_positive | 0.906 (0.808-0.978) | 0.875 | 0.872 | 0.673 | 0.847 |  |
| 1 | Guideline rule | Rule | 40 | f1_positive | 0.814 (0.692-0.903) | 0.725 | 0.687 |  |  |  |
| 1 | GLM-5.3 | LLM API | 40 | f1_positive | 0.794 (0.678-0.886) | 0.675 | 0.584 | 0.585 | 0.741 |  |
| 1 | DeepSeek-Flash | LLM API | 40 | f1_positive | 0.781 (0.655-0.879) | 0.650 | 0.535 | 0.485 | 0.707 |  |
| 1 | MLP | Deep learning | 40 | f1_positive | 0.766 (0.622-0.880) | 0.725 | 0.729 |  |  |  |
| 1 | Gradient boosting | Traditional ML | 40 | f1_positive | 0.755 (0.596-0.862) | 0.675 | 0.666 |  |  |  |
| 1 | Random forest | Traditional ML | 40 | f1_positive | 0.745 (0.591-0.857) | 0.675 | 0.673 |  |  |  |
| 1 | Logistic regression | Traditional ML | 40 | f1_positive | 0.735 (0.578-0.851) | 0.675 | 0.677 |  |  |  |
| 2 | Ours (GARF-H) | Proposed | 40 | f1_weighted | 0.896 (0.795-0.976) | 0.900 | 0.896 | 0.636 | 0.831 |  |
| 2 | Guideline rule | Rule | 40 | f1_weighted | 0.875 (0.768-0.975) | 0.875 | 0.875 |  |  |  |
| 2 | Random forest | Traditional ML | 40 | f1_weighted | 0.852 (0.732-0.975) | 0.875 | 0.852 |  |  |  |
| 2 | Logistic regression | Traditional ML | 40 | f1_weighted | 0.822 (0.681-0.938) | 0.825 | 0.822 |  |  |  |
| 2 | Gradient boosting | Traditional ML | 40 | f1_weighted | 0.776 (0.638-0.913) | 0.800 | 0.776 |  |  |  |
| 2 | MLP | Deep learning | 40 | f1_weighted | 0.729 (0.590-0.873) | 0.750 | 0.729 |  |  |  |
| 2 | DeepSeek-Flash | LLM API | 40 | f1_weighted | 0.503 (0.313-0.671) | 0.550 | 0.503 | 0.368 | 0.470 |  |
| 2 | GLM-5.3 | LLM API | 40 | f1_weighted | 0.369 (0.210-0.556) | 0.500 | 0.369 | 0.390 | 0.374 |  |
| 3 | Ours (GARF-H) | Proposed | 40 | harrell_c_index | 0.875 (0.781-0.957) |  |  |  |  |  |
| 3 | Pathology score | Clinical anchor | 40 | harrell_c_index | 0.815 (0.685-0.934) |  |  |  |  |  |
| 3 | Cox PH | Traditional survival | 40 | harrell_c_index | 0.781 (0.598-0.928) |  |  |  |  |  |
| 3 | DeepSeek-Flash | LLM API | 40 | harrell_c_index | 0.777 (0.597-0.919) |  |  |  |  |  |
| 3 | GLM-5.3 | LLM API | 40 | harrell_c_index | 0.757 (0.564-0.925) |  |  |  |  |  |
| 3 | RF event | Traditional ML | 40 | harrell_c_index | 0.726 (0.516-0.909) |  |  |  |  |  |
| 3 | MLP event | Deep learning | 40 | harrell_c_index | 0.717 (0.461-0.920) |  |  |  |  |  |
| 3 | ET survival | Traditional survival | 40 | harrell_c_index | 0.670 (0.449-0.853) |  |  |  |  |  |

> Case score is shown only for methods that produce structured reasoning.  
> The benchmark is supporting evidence, not a leaderboard; partial LLM rows are marked.
