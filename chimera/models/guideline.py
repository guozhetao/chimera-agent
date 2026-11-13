from __future__ import annotations

from typing import Iterable
import numpy as np

from ..data.dataset import Case


def pirads_value(prompt: dict) -> int | None:
    try:
        return int(prompt.get("pirads"))
    except (TypeError, ValueError):
        return None


def task1_rule(case: Case) -> str:
    value = pirads_value(case.prompt)
    return "yes" if value in (4, 5) else "no"


def task2_rule(case: Case) -> str:
    prompt = case.prompt
    try:
        isup = int(prompt.get("bx_isup"))
    except (TypeError, ValueError):
        isup = 0
    age = prompt.get("age")
    comorbidities = prompt.get("pmhx") or []
    severe = any(x in {"Coronary artery disease", "COPD", "Chronic kidney disease", "Atrial fibrillation"} for x in comorbidities)
    # Conservative life-expectancy override for older, frail patients with GG2.
    if isup == 2 and age is not None and float(age) >= 70 and (len(comorbidities) >= 2 or severe):
        return "watchful_waiting"
    if isup <= 0:
        return "continued_surveillance"
    if isup == 1:
        return "active_surveillance"
    return "active_treatment"


def rule_probabilities(cases: Iterable[Case], labels: list[str]) -> np.ndarray:
    out = []
    for case in cases:
        rule = task1_rule(case) if case.task == 1 else task2_rule(case)
        row = [0.15] * len(labels)
        if rule in labels:
            row[labels.index(rule)] = 0.7
            # Distribute small probability mass to adjacent decisions.
            adjacent = {
                "yes": ["no"], "no": ["yes"],
                "continued_surveillance": ["active_surveillance"],
                "active_surveillance": ["continued_surveillance", "active_treatment"],
                "active_treatment": ["active_surveillance", "watchful_waiting"],
                "watchful_waiting": ["active_treatment"],
            }.get(rule, [])
            for x in adjacent:
                if x in labels:
                    row[labels.index(x)] = 0.15
        total = sum(row)
        out.append([x / total for x in row])
    return np.asarray(out, dtype=float)
