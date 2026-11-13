from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from ..data.dataset import Case
from ..features.clinical import base_structured_features
from ..features.pathology_parser import surgical_risk_features
from ..features.text import text_signal_features


TASK_LABELS = {
    1: ["no", "yes"],
    2: ["active_surveillance", "active_treatment", "continued_surveillance", "watchful_waiting"],
}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _radiology_values(case: Case) -> dict[str, float]:
    report = str(case.clinical.get("radiology_report") or "")
    result: dict[str, float] = {}
    for key, pattern in {
        "radio_pirads": r"PI-RADS:\s*(\d)",
        "radio_psad": r"PSA density:\s*(\d+(?:\.\d+)?)",
        "radio_cspca": r"prostate cancer[^:]*:\s*(\d+(?:\.\d+)?)",
    }.items():
        match = re.search(pattern, report, flags=re.I)
        result[key] = float(match.group(1)) if match else float("nan")
    return result


def _clinical_stage(case: Case) -> float:
    value = str(case.prompt.get("ct") or "").lower()
    match = re.search(r"ct\s*([0-4])", value)
    if not match:
        dre = str(case.prompt.get("dre") or "").lower()
        match = re.search(r"clinical t-stage[^.]*?ct\s*([0-4])", dre)
    return float(match.group(1)) if match else float("nan")


def case_feature_dict(case: Case) -> dict[str, Any]:
    """Auditable tabular representation used by the decision models."""

    features = base_structured_features(case.prompt, case.clinical)
    features.update(surgical_risk_features(case.clinical))
    features.update(text_signal_features(case))
    features.update(_radiology_values(case))

    prompt = case.prompt
    pirads = str(prompt.get("pirads")).strip()
    pirads_num = float(pirads) if pirads.isdigit() else float("nan")
    psa = _safe_float(prompt.get("psa"))
    psad = _safe_float(prompt.get("psad"))
    volume = _safe_float(prompt.get("vol"))
    cspca = _safe_float(prompt.get("cspca"))
    isup = _safe_float(prompt.get("bx_isup"))
    gleason_primary = _safe_float(prompt.get("bx_gl_prim"))
    gleason_secondary = _safe_float(prompt.get("bx_gl_sec"))

    prior_biopsy = str(prompt.get("bx") or "").strip().lower()
    features.update(
        {
            "pirads_num": pirads_num,
            "bx_prior_positive": float(prior_biopsy == "positive"),
            "bx_prior_none": float(prior_biopsy in {"none", ""}),
            "bx_prior_negative": float(prior_biopsy == "negative"),
            "dre_abnormal": float(str(prompt.get("dre") or "") in {"Nodus", "Suspicious", "Abnormal"}),
            "psa_log": float(np.log1p(max(psa, 0.0))) if np.isfinite(psa) else float("nan"),
            "psad_log": float(np.log1p(max(psad, 0.0))) if np.isfinite(psad) else float("nan"),
            "vol_log": float(np.log1p(max(volume, 0.0))) if np.isfinite(volume) else float("nan"),
            "cspca_gap": float(abs(cspca - 0.5)) if np.isfinite(cspca) else float("nan"),
            "pirads_x_prior_positive": pirads_num * float(prior_biopsy == "positive"),
            "pirads_x_prior_none": pirads_num * float(prior_biopsy in {"none", ""}),
            "cspca_x_pirads": cspca * pirads_num,
            "psad_x_pirads": psad * pirads_num,
            "isup_group": isup,
            "isup_high": float(isup >= 2),
            "gleason_high": float((gleason_secondary >= 4) or (isup >= 2)),
            "clinical_stage": _clinical_stage(case),
            "psa_density_high": float(psad >= 0.15),
            "psa_velocity_high": float(_safe_float(prompt.get("psav")) >= 2.0),
        }
    )

    # Narrative flags used by earlier releases are retained for continuity.
    reports = [case.clinical.get("radiology_report", ""), case.clinical.get("pathology_report", "")]
    prior_notes = case.clinical.get("previous_notes", [])
    if isinstance(prior_notes, list):
        reports.extend(str(x.get("text", "")) for x in prior_notes if isinstance(x, dict))
    all_text = " ".join(x or "" for x in reports).lower()
    features.update(
        {
            "text_mentions_cribriform": float("cribriform" in all_text),
            "text_mentions_intraductal": float("intraductal" in all_text),
            "text_mentions_upgrade": float("upgrade" in all_text or "progression" in all_text),
            "text_mentions_stable": float("stable" in all_text or "no progression" in all_text),
            "radiology_report_length": float(len(case.clinical.get("radiology_report") or "")),
            "pathology_report_length": float(len(case.clinical.get("pathology_report") or "")),
            "surgical_report_length": float(len(case.clinical.get("surgical_pathology_report") or "")),
        }
    )
    return features


# Curated, low-dimensional feature set for the regularised Cox model.  Keeping
# the set small matters more than raw accuracy with only 19 observed events.
COX_COLUMNS = [
    "age",
    "psa_log",
    "psad",
    "cspca",
    "pirads",
    "isup_grade_group",
    "gleason_primary",
    "gleason_secondary",
    "pt_stage_major",
    "extraprostatic_extension",
    "positive_margin",
    "seminal_vesicle_invasion",
    "lymphovascular_invasion",
    "lymph_node_metastasis",
    "biopsy_isup",
    "biopsy_gleason_primary",
    "biopsy_gleason_secondary",
    "biopsy_cribriform",
    "biopsy_intraductal",
    "txt_prior_isup_max",
    "txt_prior_gleason_sum",
]


def survival_feature_dict(case: Case) -> dict[str, float]:
    raw = case_feature_dict(case)
    fallback = {"pirads": "radio_pirads", "psad": "radio_psad", "cspca": "radio_cspca"}
    output: dict[str, float] = {}
    for key in COX_COLUMNS:
        value = _safe_float(raw.get(key))
        if not np.isfinite(value) and key in fallback:
            value = _safe_float(raw.get(fallback[key]))
        output[key] = value
    return output


def build_dataframe(cases: list[Case], neural: bool = False, neural_mode: str = "stats") -> pd.DataFrame:
    rows = [case_feature_dict(c) for c in cases]
    df = pd.DataFrame(rows)
    if neural:
        from ..features.neural import SetStatisticsEncoder

        encoder = SetStatisticsEncoder(mode=neural_mode)
        embedding = encoder.transform([c.neural for c in cases])
        embedding_df = pd.DataFrame(embedding, columns=[f"neural_{i}" for i in range(embedding.shape[1])])
        df = pd.concat([df.reset_index(drop=True), embedding_df.reset_index(drop=True)], axis=1)
    return df


def harmonize_columns(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([train, test], ignore_index=True, sort=False)
    combined = combined.fillna(np.nan)
    train_out = combined.iloc[: len(train)].reset_index(drop=True)
    test_out = combined.iloc[len(train):].reset_index(drop=True)
    return train_out, test_out


def make_preprocessor(df: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # An all-missing column carries no information and breaks median imputation.
    usable = [column for column in df.columns if not df[column].isna().all()]
    frame = df[usable]
    numeric = list(frame.select_dtypes(include=[np.number, "bool"]).columns)
    categorical = [c for c in frame.select_dtypes(exclude=[np.number, "bool"]).columns if c != "case_id"]

    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=3),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers)
