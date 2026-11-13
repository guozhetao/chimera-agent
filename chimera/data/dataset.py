from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.jsonio import read_json


TASK_FILES = {
    1: {
        "prompt": "structured-prompt.json",
        "clinical": "prostate-biopsy-decision-clinical-data.json",
        "neural": "prostate-modality-level-neural-representations.json",
        "decision": "prostate-biopsy-decision.json",
        "reasoning": "prostate-biopsy-decision-reasoning.json",
    },
    2: {
        "prompt": "structured-prompt.json",
        "clinical": "prostate-treatment-decision-clinical-data.json",
        "neural": "prostate-modality-level-neural-representations.json",
        "decision": "prostate-treatment-decision.json",
        "reasoning": "prostate-treatment-decision-reasoning.json",
    },
    3: {
        "prompt": "structured-prompt.json",
        "clinical": "prostate-time-to-recurrence-or-last-follow-up-clinical-data.json",
        "neural": "prostate-modality-level-neural-representations.json",
        "decision": "prostate-time-to-recurrence-or-last-follow-up.json",
        "reasoning": "prostate-time-to-recurrence-or-last-follow-up-reasoning.json",
    },
}


@dataclass
class Case:
    task: int
    case_id: str
    directory: Path
    prompt: dict[str, Any]
    clinical: dict[str, Any]
    neural: dict[str, Any]
    label: Any = None
    reasoning: dict[str, Any] | None = None

    @property
    def has_label(self) -> bool:
        return self.label is not None


class ChimeraDataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

    def cases(self, task: int, labeled_only: bool = False) -> list[Case]:
        if task not in TASK_FILES:
            raise ValueError("task must be 1, 2, or 3")
        spec = TASK_FILES[task]
        task_dir = self.root / f"task{task}"
        out: list[Case] = []
        for d in sorted(task_dir.iterdir() if task_dir.exists() else []):
            if not d.is_dir():
                continue
            prompt = read_json(d / spec["prompt"], {})
            if not prompt:
                continue
            label = read_json(d / spec["decision"])
            reasoning = read_json(d / spec["reasoning"])
            if labeled_only and label is None:
                continue
            out.append(
                Case(
                    task=task,
                    case_id=str(prompt.get("case_id", d.name)),
                    directory=d,
                    prompt=prompt,
                    clinical=read_json(d / spec["clinical"], {}) or {},
                    neural=read_json(d / spec["neural"], {}) or {},
                    label=label,
                    reasoning=reasoning,
                )
            )
        return out

    def counts(self) -> dict[int, dict[str, int]]:
        result: dict[int, dict[str, int]] = {}
        for task in (1, 2, 3):
            cases = self.cases(task)
            result[task] = {
                "total": len(cases),
                "labeled": sum(c.has_label for c in cases),
                "unlabeled": sum(not c.has_label for c in cases),
            }
        return result
