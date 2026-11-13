from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils.jsonio import read_json


TASK2_LABELS = {"active_surveillance", "active_treatment", "continued_surveillance", "watchful_waiting"}


def validate_outputs(directory: str | Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    root = Path(directory)
    if not root.exists():
        return False, [f"Output directory does not exist: {root}"]
    required_dirs = [root / f"task{t}" for t in (1, 2, 3)]
    for task_dir in required_dirs:
        if not task_dir.exists():
            errors.append(f"Missing directory: {task_dir}")
            continue
        for case_dir in task_dir.iterdir():
            if not case_dir.is_dir():
                continue
            if task_dir.name == "task1":
                dec = read_json(case_dir / "prostate-biopsy-decision.json")
                if dec not in {"yes", "no"}:
                    errors.append(f"{case_dir.name}: invalid Task 1 decision {dec!r}")
                reason = read_json(case_dir / "prostate-biopsy-decision-reasoning.json")
                errors.extend(validate_reasoning(case_dir.name, reason))
            elif task_dir.name == "task2":
                dec = read_json(case_dir / "prostate-treatment-decision.json")
                if dec not in TASK2_LABELS:
                    errors.append(f"{case_dir.name}: invalid Task 2 decision {dec!r}")
                reason = read_json(case_dir / "prostate-treatment-decision-reasoning.json")
                errors.extend(validate_reasoning(case_dir.name, reason))
            else:
                dec = read_json(case_dir / "prostate-time-to-recurrence-or-last-follow-up.json")
                if not isinstance(dec, dict) or "months_to_recurrence" not in dec or "event" not in dec:
                    errors.append(f"{case_dir.name}: Task 3 outcome must contain months_to_recurrence and event")
                elif dec["event"] not in {0, 1}:
                    errors.append(f"{case_dir.name}: Task 3 event must be 0 or 1")
    return not errors, errors


def validate_reasoning(case_id: str, obj: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        errors.append(f"{case_id}: reasoning is not an object")
        return errors
    for key in ("confidence", "variable_weights", "reveal_sequence", "free_text"):
        if key not in obj:
            errors.append(f"{case_id}: reasoning missing {key}")
    if obj.get("confidence") not in {"clear", "borderline", "uncertain"}:
        errors.append(f"{case_id}: invalid confidence {obj.get('confidence')!r}")
    if not isinstance(obj.get("variable_weights"), dict):
        errors.append(f"{case_id}: variable_weights is not an object")
    if not isinstance(obj.get("reveal_sequence"), list):
        errors.append(f"{case_id}: reveal_sequence is not a list")
    if not isinstance(obj.get("free_text"), str) or not obj.get("free_text", "").strip():
        errors.append(f"{case_id}: free_text is empty")
    return errors
