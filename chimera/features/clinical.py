from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any


MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_date(value: str | None) -> float | None:
    if not value:
        return None
    s = str(value).strip()
    for fmt in ("%d %b %Y", "%d %B %Y", "%b %Y", "%B %Y", "%Y-%m-%d", "%Y"):
        try:
            return float(datetime.strptime(s, fmt).toordinal())
        except ValueError:
            pass
    return None


def psa_trend_features(trend: list[dict[str, Any]] | None) -> dict[str, float | None]:
    points: list[tuple[float, float]] = []
    for item in trend or []:
        date = _parse_date(item.get("date"))
        try:
            val = float(item.get("val"))
        except (TypeError, ValueError):
            continue
        if date is not None:
            points.append((date, val))
    points.sort()
    if not points:
        return {
            "psa_points": 0, "psa_first": None, "psa_last": None,
            "psa_delta": None, "psa_range": None, "psa_std": None,
            "psa_slope_per_year": None, "psa_doubling_years": None,
        }
    dates = [p[0] for p in points]
    vals = [p[1] for p in points]
    slope = None
    if len(points) >= 2:
        xbar, ybar = sum(dates) / len(dates), sum(vals) / len(vals)
        denom = sum((x - xbar) ** 2 for x in dates)
        slope = sum((x - xbar) * (y - ybar) for x, y in points) / denom if denom else 0.0
        slope *= 365.0
    first, last = vals[0], vals[-1]
    doubling = None
    if slope and first > 0:
        doubling = math.log(2) / (slope / first)
    return {
        "psa_points": float(len(points)),
        "psa_first": first,
        "psa_last": last,
        "psa_delta": last - first,
        "psa_range": max(vals) - min(vals),
        "psa_std": (sum((v - ybar) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0,
        "psa_slope_per_year": slope,
        "psa_doubling_years": doubling,
    }


def laboratory_features(labs: list[dict[str, Any]] | None) -> dict[str, float | None]:
    out = {"free_psa": None, "percent_free_psa": None, "egfr_numeric": None}
    for item in labs or []:
        name = str(item.get("name", "")).lower()
        raw = item.get("val")
        number = None
        if raw is not None:
            m = re.search(r"\d+(?:\.\d+)?", str(raw))
            number = float(m.group()) if m else None
        if "free psa" in name and "%" not in str(raw):
            out["free_psa"] = number
        elif "free psa" in name and "%" in str(raw):
            out["percent_free_psa"] = number
    return out


def radiology_text_features(report: str | None) -> dict[str, float | None]:
    text = (report or "").lower()
    diameters = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*(?:mm|millimet(?:er|re)s)", text)]
    volumes = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*(?:ml|cc|millilit(?:er|re)s)", text)]
    return {
        "mri_max_diameter_mm": max(diameters) if diameters else None,
        "mri_min_diameter_mm": min(diameters) if diameters else None,
        "mri_n_diameters": float(len(diameters)),
        "mri_report_volume_ml": max(volumes) if volumes else None,
        "mri_restricted_diffusion": float("restrict" in text),
        "mri_early_enhancement": float("early" in text and "enhanc" in text),
        "mri_extracapsular": float("extracapsular" in text or "extraprostatic" in text),
        "mri_seminal_invasion": float("seminal" in text and "invas" in text),
        "mri_lymphadenopathy": float("lymphadenopathy" in text or "suspicious lymph" in text),
        "mri_bone_lesion": float("bone" in text or "osseous" in text),
        "mri_no_suspicious_lesion": float("no suspicious" in text or "no pi-rads 3" in text),
        "mri_homogeneous": float("homogeneous" in text),
        "mri_ill_defined": float("ill-defined" in text or "ill defined" in text),
    }


def social_history_features(prompt: dict[str, Any], clinical: dict[str, Any]) -> dict[str, float | None]:
    medhx = str(prompt.get("medhx", "") or "")
    pmhx = prompt.get("pmhx") or []
    comorbidity_count = len([x for x in pmhx if str(x).strip()]) or (0 if medhx in {"", "Not reported", "None"} else len([x for x in medhx.split(",") if x.strip()]))
    family = str(clinical.get("family_history", prompt.get("family_history", ""))).lower()
    return {
        "comorbidity_count": float(comorbidity_count),
        "family_history_positive": float(family in {"yes", "true", "positive"}),
        "active_treatment_flag": float(prompt.get("active_treatment_flag") or 0),
    }


def base_structured_features(prompt: dict[str, Any], clinical: dict[str, Any]) -> dict[str, Any]:
    numeric = [
        "age", "psa", "psad", "psav", "psap", "vol", "cspca", "months",
        "bx_isup", "bx_gl_prim", "bx_gl_sec", "active_treatment_flag",
    ]
    categorical = [
        "pirads", "dre", "bx", "ct", "medhx", "meds", "enc_type",
    ]
    features: dict[str, Any] = {}
    for key in numeric:
        try:
            features[key] = None if prompt.get(key) is None else float(prompt[key])
        except (TypeError, ValueError):
            features[key] = None
    for key in categorical:
        features[key] = None if prompt.get(key) is None else str(prompt[key])
    features.update(psa_trend_features(clinical.get("psa_trend")))
    features.update(laboratory_features(clinical.get("laboratory_results")))
    features.update(radiology_text_features(clinical.get("radiology_report")))
    features.update(social_history_features(prompt, clinical))
    return features
