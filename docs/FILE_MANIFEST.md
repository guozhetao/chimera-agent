# Release file manifest

Upload the `chimera-agent/` folder (or extract `chimera_agent_github_release.zip`) as the repository root. The manifest below lists every file included in the release.

## Not included

- Raw training data (`train_release/`), generated `submission/` directories, API caches and per-model raw API outputs.
- Legacy figures and the original zipped dataset. Historical scripts are under `archive/legacy_scripts/` only for provenance.

## Key files

| Path | Purpose |
|---|---|
| `README.md` | Project overview, results, installation, quickstart |
| `docs/PROJECT_REPORT_CN.md` | Detailed Chinese project report |
| `docs/BENCHMARK.md` | Benchmark protocol, baselines and LLM results |
| `docs/PAPER_NARRATIVE.md` | Paper storyline and figure plan |
| `docs/FIGURE_CAPTIONS.md` | Panel-by-panel figure captions |
| `results/figures/` | Figures 1–6 in PNG/PDF/SVG |
| `results/tables/` | Benchmark and evaluation summary tables |
| `results/local_eval.json` | Full OOF predictions for reproducibility |
| `chimera/` | Core package |
| `scripts/` | Evaluation, submission, plotting and benchmark entrypoints |
| `tests/` | Regression tests |

## File tree

```text
.gitignore
archive/legacy_scripts/build_task_comparison.py
archive/legacy_scripts/check_figure_layout.py
archive/legacy_scripts/check_figures.py
archive/legacy_scripts/make_publication_figures.py
archive/legacy_scripts/make_task_figures.py
archive/legacy_scripts/plot_analysis.py
archive/README.md
chimera/__init__.py
chimera/agents/api_agents.py
chimera/agents/comparison.py
chimera/agents/direct_api.py
chimera/data/dataset.py
chimera/evaluation/benchmark.py
chimera/evaluation/cv.py
chimera/evaluation/local_evaluator.py
chimera/evaluation/survival_metrics.py
chimera/evaluation/validation.py
chimera/features/clinical.py
chimera/features/neural.py
chimera/features/pathology_parser.py
chimera/features/text.py
chimera/models/cox.py
chimera/models/features.py
chimera/models/guideline.py
chimera/models/hybrid.py
chimera/models/residual.py
chimera/models/semisupervised.py
chimera/models/survival.py
chimera/reasoning/generator.py
chimera/reasoning/supervised.py
chimera/utils/jsonio.py
CITATION.cff
configs/experiments.yaml
data/README.md
docs/archive/ANALYSIS_WORKFLOW.md
docs/BENCHMARK.md
docs/BENCHMARK_FIGURE_CAPTION.md
docs/CHANGELOG.md
docs/FIGURE_CAPTIONS.md
docs/FIGURE_INDEX.md
docs/FILE_MANIFEST.md
docs/PAPER_NARRATIVE.md
docs/PROJECT_REPORT_CN.md
examples/output_format/README.md
examples/output_format/task1_decision.json
examples/output_format/task1_reasoning.json
examples/output_format/task2_decision.json
examples/output_format/task2_reasoning.json
examples/output_format/task3_decision.json
examples/output_format/task3_reasoning.json
GITHUB_UPLOAD_GUIDE.md
LICENSE
pyproject.toml
README.md
requirements.txt
results/audit.json
results/figures/fig1_cohort_and_architecture.pdf
results/figures/fig1_cohort_and_architecture.png
results/figures/fig1_cohort_and_architecture.svg
results/figures/fig2_task1_biopsy.pdf
results/figures/fig2_task1_biopsy.png
results/figures/fig2_task1_biopsy.svg
results/figures/fig3_task2_management.pdf
results/figures/fig3_task2_management.png
results/figures/fig3_task2_management.svg
results/figures/fig4_task3_recurrence.pdf
results/figures/fig4_task3_recurrence.png
results/figures/fig4_task3_recurrence.svg
results/figures/fig5_model_comparison.pdf
results/figures/fig5_model_comparison.png
results/figures/fig5_model_comparison.svg
results/figures/fig6_benchmark_comparison.pdf
results/figures/fig6_benchmark_comparison.png
results/figures/fig6_benchmark_comparison.svg
results/local_eval.json
results/tables/benchmark_combined.csv
results/tables/benchmark_combined.md
results/tables/benchmark_results.csv
results/tables/benchmark_results.json
results/tables/benchmark_results.md
results/tables/cases_task1.json
results/tables/cases_task2.json
results/tables/cases_task3.json
results/tables/llm_results.json
results/tables/llm_results.md
results/tables/local_eval_summary.csv
results/tables/local_eval_summary.md
scripts/audit_data.py
scripts/combine_benchmark_tables.py
scripts/create_submission.py
scripts/make_benchmark_figure.py
scripts/make_report_figures.py
scripts/rebuild_llm_results.py
scripts/retry_glm_benchmark.py
scripts/run_agents.py
scripts/run_benchmark.py
scripts/run_direct_api.py
scripts/run_llm_benchmark.py
scripts/run_local_eval.py
scripts/run_pipeline.py
tests/test_direct_api.py
tests/test_optimized_pipeline.py
tests/test_smoke.py
```
