from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


@dataclass
class PathologyFindings:
    gleason_primary: int | None = None
    gleason_secondary: int | None = None
    isup_grade_group: int | None = None
    pt_stage: str | None = None
    extraprostatic_extension: bool = False
    positive_margin: bool = False
    seminal_vesicle_invasion: bool = False
    lymph_node_metastasis: bool = False
    lymphovascular_invasion: bool = False
    perineural_invasion: bool = False
    cribriform: bool = False
    intraductal: bool = False
    adverse_pattern: bool = False
    n_timepoints: int = 0
    histologically_stable: bool = False
    upgrade_across_timepoints: bool = False


def _int(value: str | None) -> int | None:
    return int(value) if value is not None else None


def parse_gleason(text: str) -> tuple[int | None, int | None]:
    patterns = [
        r"Gleason(?:\s+score)?(?:\s+of)?\s*(\d)\s*\+\s*(\d)",
        r"Gleason(?:\s+score)?(?:\s+of)?\s*(\d)\s*-\s*(\d)",
        r"Gleason(?:\s+score)?\s*(\d)\s*/\s*(\d)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.I)
        if m:
            return _int(m.group(1)), _int(m.group(2))
    return None, None


def parse_isup(text: str) -> int | None:
    m = re.search(r"(?:ISUP|grade group)\s*(?:of)?\s*(\d)", text, flags=re.I)
    return _int(m.group(1)) if m else None


def _positive_finding(text: str, concept_pattern: str) -> bool:
    """True only when a concept is explicitly present, not merely mentioned as absent."""
    concept = re.compile(concept_pattern, flags=re.I)
    for match in concept.finditer(text):
        window = text[max(0, match.start() - 70): min(len(text), match.end() + 100)]
        if re.search(r"\babsent\b|\bnot present\b|\bno\b|\bnegative\b", window, flags=re.I):
            continue
        if re.search(r"\bpresent\b|\bpositive\b|\bidentified\b|\binvaded\b|\binvolved\b|\bmetastasis\b", window, flags=re.I):
            return True
    return False


def parse_pathology(text: str | None, surgical: bool = False) -> PathologyFindings:
    text = text or ""
    folded = text.lower()
    g1, g2 = parse_gleason(text)
    isup = parse_isup(text)
    if isup is None and g1 is not None and g2 is not None:
        total = g1 + g2
        isup = 1 if total <= 6 else 2 if total == 7 else 3 if total == 8 else 4 if total == 9 else 5
    pt = None
    m = re.search(r"\bp\s*T\s*(?:[0-4])(?:[a-c]?)\b", text, flags=re.I)
    if m:
        pt = m.group().replace(" ", "").lower()
    n_timepoints = len(re.findall(r"(?:biopsy|timepoint|specimen)\s*(?:\d+|\(|[0-9]{2}\s+[A-Za-z]{3}\s+[0-9]{4})", folded, flags=re.I))
    findings = PathologyFindings(
        gleason_primary=g1,
        gleason_secondary=g2,
        isup_grade_group=isup,
        pt_stage=pt,
        extraprostatic_extension=_positive_finding(text, r"extraprostatic extension|extracapsular extension"),
        positive_margin=bool(re.search(r"(?:surgical )?margins? (?:were |are )?positive|positive (?:surgical )?margin", folded)),
        seminal_vesicle_invasion=_positive_finding(text, r"seminal vesicles?|seminal vesicle invasion"),
        lymph_node_metastasis=_positive_finding(text, r"lymph node metastasis|lymph nodes?|nodal metastasis"),
        lymphovascular_invasion=_positive_finding(text, r"lymphovascular invasion"),
        perineural_invasion=_positive_finding(text, r"perineural invasion"),
        cribriform=_positive_finding(text, r"cribriform"),
        intraductal=_positive_finding(text, r"intraductal"),
        n_timepoints=max(n_timepoints, 1 if text else 0),
        histologically_stable=bool(re.search(r"stable|no (?:histological )?(?:upgrade|progression)", folded)),
        upgrade_across_timepoints=bool(re.search(r"upgrade|upgrad|progression", folded)),
    )
    findings.adverse_pattern = findings.cribriform or findings.intraductal
    return findings


def surgical_risk_features(clinical: dict[str, Any]) -> dict[str, float | None]:
    surgical = parse_pathology(clinical.get("surgical_pathology_report"), surgical=True)
    biopsy = parse_pathology(clinical.get("pathology_report"), surgical=False)
    d = asdict(surgical)
    d["biopsy_isup"] = biopsy.isup_grade_group
    d["biopsy_gleason_primary"] = biopsy.gleason_primary
    d["biopsy_gleason_secondary"] = biopsy.gleason_secondary
    d["biopsy_cribriform"] = biopsy.cribriform
    d["grade_upgrading"] = (
        (surgical.isup_grade_group or 0) > (biopsy.isup_grade_group or 0)
    )
    d["gleason_sum"] = (
        surgical.gleason_primary + surgical.gleason_secondary
        if surgical.gleason_primary is not None and surgical.gleason_secondary is not None
        else None
    )
    output: dict[str, float | None] = {}
    for key, value in d.items():
        if isinstance(value, bool):
            output[key] = float(value)
        elif isinstance(value, (int, float)) or value is None:
            output[key] = None if value is None else float(value)
        else:
            # Keep stage as one-hot friendly numeric prefixes.
            m = re.search(r"t([0-4])([a-c]?)", str(value))
            output["pt_stage_major"] = float(m.group(1)) if m else None
            output["pt_stage_sub"] = float(ord(m.group(2)) - ord("a") + 1) if m and m.group(2) else None
    return output


def pathology_risk_score(clinical: dict[str, Any], psa: float | None = None) -> float:
    f = surgical_risk_features(clinical)
    # Weights selected by greedy coordinate ascent on comparable Harrell pairs.
    # Continuous values are scaled to the training support before weighting.
    score = 2.0 * f.get("positive_margin", 0)
    score += 3.0 * (f.get("biopsy_isup") or 0) / 5.0
    score += 3.0 * (float(psa) / 190.0 if psa is not None else 0.0)
    score += 3.0 * f.get("lymph_node_metastasis", 0)
    score += 3.0 * f.get("extraprostatic_extension", 0)
    return float(score)
