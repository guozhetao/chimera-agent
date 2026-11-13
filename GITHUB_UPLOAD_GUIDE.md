# GitHub upload guide

This folder is a curated, GitHub-ready snapshot of the project. It contains code, tests, documentation, final figures, benchmark tables and the full OOF evaluation JSON. It does **not** contain raw data, generated submissions, API caches or credentials.

## 1. Pre-upload checklist

- [ ] No `.env`, API keys, tokens or credentials in the tree.
- [ ] No raw patient data under `data/` (only `data/README.md`).
- [ ] No `work/api_cache`, `results/submission*` or `results/benchmark/llm/` raw outputs.
- [ ] `results/figures/` contains the six final figures.
- [ ] `results/tables/` contains the combined benchmark table and LLM table.
- [ ] `.gitignore` is present.
- [ ] `LICENSE` and `CITATION.cff` are present.
- [ ] The repository name/URL in `CITATION.cff` and README is updated.

## 2. Create a repository on GitHub

Create an empty repository, for example `chimera-agent`, without a README or `.gitignore` (this folder already has them).

## 3. Initialise and push

```bash
cd github_release/chimera-agent
git init
git add .
git status
git commit -m "Initial release: CHIMERA-Agent GARF-H"
git branch -M main
git remote add origin https://github.com/<your-org>/chimera-agent.git
git push -u origin main
```

If you use the GitHub CLI:

```bash
gh repo create <your-org>/chimera-agent --private --source=. --remote=origin --push
```

## 4. Recommended repository settings

- **Description**: Guideline-anchored multimodal decision agent for prostate-cancer biopsy, management and recurrence risk.
- **Topics**: `prostate-cancer`, `clinical-decision-support`, `multimodal-learning`, `survival-analysis`, `guideline-anchored`, `structured-reasoning`, `benchmark`.
- **Visibility**: start private if the data-use agreement is unclear. The code is MIT-licensed; the data are not included.
- **Branch protection**: require review and tests for `main` if collaborators are added.

## 5. Release tag (optional)

```bash
git tag -a v0.1.0 -m "CHIMERA-Agent v0.1.0"
git push origin v0.1.0
```

## 6. What not to upload

- Raw `train_release` data or generated `submission` directories.
- `work/api_cache` or any API response cache.
- `.env`, API key files, shell history or chat transcripts.
- The original dataset zip or generated submission zip.

If you need to share data with reviewers, use a private repository or a controlled data-access route that follows the original data-use agreement.

## 7. Security note

The DeepSeek and GLM keys used during development were entered interactively and were not written to the repository. Because keys were also shared in a chat during development, rotate them before publishing or deploying.
