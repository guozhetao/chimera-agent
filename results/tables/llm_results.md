# LLM-agent benchmark

The same case subset and two-stage prompt were used for every API model. Keys are read from environment variables and are never written to disk.

| Task | Model | n | Primary metric | Value (95% CI) | Accuracy | Reasoning | Case score |
|---|---|---:|---|---:|---:|---:|---:|
| 1 | DeepSeek-Flash | 40 | f1_positive | 0.781 (0.655–0.879) | 0.650 | 0.485 | 0.707 |
| 2 | DeepSeek-Flash | 40 | f1_weighted | 0.503 (0.313–0.671) | 0.550 | 0.368 | 0.470 |
| 3 | DeepSeek-Flash | 40 | harrell_c_index | 0.777 (0.597–0.919) |  |  |  |
| 1 | GLM-5.3 | 40 | f1_positive | 0.794 (0.678–0.886) | 0.675 | 0.585 | 0.741 |
| 2 | GLM-5.3 | 40 | f1_weighted | 0.369 (0.210–0.556) | 0.500 | 0.390 | 0.374 |
| 3 | GLM-5.3 | 40 | harrell_c_index | 0.757 (0.564–0.925) |  |  |  |

> Case score includes the structured reasoning score.  
> The LLM arm is descriptive and not powered to rank APIs.
