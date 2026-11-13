# Output format examples

These files show the required JSON structure for submission outputs. They use placeholder case IDs and values; they are not real patients.

## Task 1 — biopsy decision

`task1_decision.json`

```json
"yes"
```

`task1_reasoning.json`

```json
{
  "confidence": "clear",
  "variable_weights": {
    "pirads": "decisive",
    "psa": "important",
    "psad": "important",
    "vol": "noted",
    "dre": "noted",
    "fh": "noted",
    "comorbidity": "noted",
    "age": "important",
    "bx": "important",
    "cspca": "noted"
  },
  "reveal_sequence": ["previous_notes", "laboratory_results", "psa_trend", "radiology_report"],
  "free_text": "Biopsy is recommended because the MRI shows a PI-RADS 5 lesion with elevated PSA density and a rising PSA trajectory."
}
```

## Task 2 — management decision

`task2_decision.json`

```json
"active_treatment"
```

`task2_reasoning.json`

```json
{
  "confidence": "clear",
  "variable_weights": {
    "pirads": "important",
    "psa": "important",
    "psad": "noted",
    "age": "noted",
    "comorbidity": "noted",
    "fh": "noted",
    "ct": "important",
    "cspca": "important",
    "bx_isup": "decisive",
    "bx_gl_prim": "important",
    "bx_gl_sec": "important"
  },
  "reveal_sequence": [],
  "free_text": "Biopsy shows ISUP grade group 3 disease with adverse pathology, supporting definitive active treatment rather than surveillance."
}
```

## Task 3 — recurrence risk

`task3_decision.json`

```json
{
  "months_to_recurrence": 18.0,
  "event": 1
}
```

`task3_reasoning.json`

```json
{
  "confidence": "clear",
  "variable_weights": {
    "age": "noted",
    "psa": "important",
    "isup_grade_group": "decisive",
    "gleason": "important",
    "pt_stage": "important",
    "positive_margin": "important",
    "extraprostatic_extension": "important",
    "seminal_vesicle_invasion": "noted",
    "lymph_node_metastasis": "noted",
    "lymphovascular_invasion": "noted"
  },
  "reveal_sequence": ["surgical_pathology_report", "pathology_report", "radiology_report"],
  "free_text": "After radical prostatectomy, ISUP grade group 3 disease with positive margins and extraprostatic extension places the patient in a higher-risk trajectory; close PSA monitoring is warranted."
}
```
