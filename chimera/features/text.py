"""Text-derived clinical signals shared by the decision and survival models.

The release contains narrative radiology, pathology and GP notes.  These are
turned into small, auditable numeric signals instead of opaque embeddings so
that the downstream tree models remain interpretable on very small cohorts.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

_NUMBER = r"(\d+(?:\.\d+)?)"


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)


def clinical_text(case: Any) -> str:
    """Concatenate every narrative field that is available at decision time.

    Reference reasoning is deliberately excluded: only inputs a clinician would
    have before making the decision are used.
    """

    clinical = getattr(case, "clinical", {}) or {}
    prompt = getattr(case, "prompt", {}) or {}
    parts: list[str] = [
        _as_text(clinical.get("radiology_report")),
        _as_text(clinical.get("pathology_report")),
        _as_text(clinical.get("surgical_pathology_report")),
        _as_text(prompt.get("notes")),
        _as_text(prompt.get("medhx")),
        _as_text(prompt.get("ipss")),
        _as_text(prompt.get("enc_type")),
    ]
    for item in clinical.get("previous_notes") or []:
        if isinstance(item, dict):
            parts.append(_as_text(item.get("text")))
    for item in prompt.get("note_sections") or []:
        if isinstance(item, dict):
            parts.append(_as_text(item.get("t")))
    return " ".join(part for part in parts if part)


def token_set(text: str) -> set[str]:
    """Lowercase lexical tokens used for lightweight retrieval."""

    return set(re.findall(r"[a-z]{4,}|\d+(?:\.\d+)?", (text or "").lower()))


def _all_ints(pattern: str, text: str) -> list[int]:
    return [int(x) for x in re.findall(pattern, text)]


def text_signal_features(case: Any) -> dict[str, float]:
    """Interpretable language signals from the available clinical narratives."""

    text = clinical_text(case).lower()

    isup = _all_ints(r"isup(?:\s*grade\s*group)?\s*(?:gg)?\s*(?:of\s*)?(\d)", text)
    isup += _all_ints(r"\bgg\s*(\d)", text)
    gleason = [int(a) + int(b) for a, b in re.findall(r"gleason\s*(?:score\s*)?(\d)\s*[+\-]\s*(\d)", text)]
    psa_values = [float(x) for x in re.findall(r"psa[^0-9]{0,30}" + _NUMBER, text)]

    treatment = bool(
        re.search(
            r"curative treatment|radiotherapy|radiation therapy|hormone therapy|"
            r"androgen deprivation|psma|prostatectomy|start treatment|should be treated|"
            r"recommend treatment|systemic treatment",
            text,
        )
    )
    surveillance = bool(re.search(r"active surveillance|watchful waiting|surveillance protocol", text))
    growth = bool(re.search(r"grow|increas|progress|climb|rising|upward trend|rapid", text))
    stable = bool(re.search(r"stable|plateau|unchanged|no progression|static", text))
    negative_biopsy = bool(re.search(r"negative biopsy|benign|no malignancy", text))
    positive_biopsy = bool(re.search(r"confirmed adenocarcinoma|adenocarcinoma|cancer diagnosis|malignant", text))
    missing_info = bool(re.search(r"missing|unclear|not available|no information|need to know|not reported", text))
    cribriform = "cribriform" in text
    intraductal = "intraductal" in text

    return {
        "txt_prior_isup_max": float(max(isup)) if isup else float("nan"),
        "txt_prior_gleason_sum": float(max(gleason)) if gleason else float("nan"),
        "txt_n_isup_mentions": float(len(isup)),
        "txt_prior_psa_max": float(max(psa_values)) if psa_values else float("nan"),
        "txt_active_surveillance": float(surveillance),
        "txt_treatment_language": float(treatment),
        "txt_growth_language": float(growth),
        "txt_stable_language": float(stable),
        "txt_negative_biopsy": float(negative_biopsy),
        "txt_positive_biopsy": float(positive_biopsy),
        "txt_missing_information": float(missing_info),
        "txt_cribriform": float(cribriform),
        "txt_intraductal": float(intraductal),
        "txt_biopsy_mentions": float(len(re.findall(r"biops", text))),
        "txt_length": float(len(text)),
    }


def first_number(pattern: str, text: str, default: float = float("nan")) -> float:
    match = re.search(pattern, text or "", flags=re.I)
    return float(match.group(1)) if match else default


def keyword_sentence(terms: Iterable[str]) -> str:
    """Turn retrieved keywords into one short, readable sentence."""

    clean: list[str] = []
    for term in terms:
        term = term.strip()
        if not term or term in clean:
            continue
        clean.append(term)
        if len(clean) >= 8:
            break
    if not clean:
        return ""
    return "Relevant factors included " + ", ".join(clean[:-1]) + (", and " if len(clean) > 1 else "") + clean[-1] + "."
