# Data layout

Raw data are **not** included in this repository. Point `--data-root` to a directory with the following structure:

```text
<data-root>/
  task1/
    <case_id>/
      structured-prompt.json
      prostate-biopsy-decision-clinical-data.json
      prostate-modality-level-neural-representations.json
      prostate-biopsy-decision.json
      prostate-biopsy-decision-reasoning.json
  task2/
    <case_id>/
      structured-prompt.json
      prostate-treatment-decision-clinical-data.json
      prostate-modality-level-neural-representations.json
      prostate-treatment-decision.json
      prostate-treatment-decision-reasoning.json
  task3/
    <case_id>/
      structured-prompt.json
      prostate-time-to-recurrence-or-last-follow-up-clinical-data.json
      prostate-modality-level-neural-representations.json
      prostate-time-to-recurrence-or-last-follow-up.json
      prostate-time-to-recurrence-or-last-follow-up-reasoning.json
```

## Labeled cohort sizes

| Task | Total | Labeled | Unlabeled | Labels |
|---|---:|---:|---:|---|
| Task 1 | 195 | 91 | 104 | `yes` 56 / `no` 35 |
| Task 2 | 153 | 72 | 81 | `active_surveillance` 25 / `active_treatment` 31 / `continued_surveillance` 14 / `watchful_waiting` 2 |
| Task 3 | 75 | 75 | 0 | `event` 19 / censored 56; follow-up 1.3–110 months |

## Modalities

| Task | MRI | Biopsy WSI | Prostatectomy WSI |
|---|---:|---:|---:|
| Task 1 | 191 | 0 | 0 |
| Task 2 | 149 | 147 | 0 |
| Task 3 | 75 | 51 | 75 |

The neural representation files contain variable-length vectors:

- MRI: 1024 dimensions per image;
- biopsy WSI: 960 dimensions per slide;
- prostatectomy WSI: 960 dimensions per slide.

The code converts them to fixed-size set statistics (mean/std/max/count, or a compact mean/std/count variant). Missing modalities are encoded as zero vectors plus a count feature.

## Data use

The data are pseudo-anonymized for the release. Do not redistribute the raw data or generated case-level outputs publicly without confirming the original data-use agreement. The repository includes only aggregate metrics, figures and derived tables.
