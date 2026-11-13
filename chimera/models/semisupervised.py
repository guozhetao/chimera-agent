from __future__ import annotations

import numpy as np


class ConservativeSelfTraining:
    """Adds only high-confidence pseudo-labels while preserving class balance."""

    def __init__(self, confidence_threshold: float = 0.78, max_rounds: int = 2, max_pseudo_ratio: float = 0.35):
        self.confidence_threshold = confidence_threshold
        self.max_rounds = max_rounds
        self.max_pseudo_ratio = max_pseudo_ratio
        self.history: list[dict] = []

    def fit(self, model, cases, labels: list[str], unlabeled_cases: list) -> object:
        remaining = list(unlabeled_cases)
        accepted_cases = list(cases)
        accepted_labels = list(labels)
        max_total = int(self.max_pseudo_ratio * len(cases))
        for round_id in range(self.max_rounds):
            if not remaining or len(accepted_cases) - len(cases) >= max_total:
                break
            prob = model.predict_proba(remaining)
            conf = prob.max(axis=1)
            pred = np.asarray(model.labels)[prob.argmax(axis=1)]
            accepted = conf >= self.confidence_threshold
            # Do not let one class take over pseudo-labels.
            target_per_class = {x: max(1, int(0.35 * labels.count(x))) for x in sorted(set(labels))}
            chosen: list[int] = []
            seen: dict[str, int] = {}
            for idx in np.argsort(-conf):
                if not accepted[idx] or len(chosen) >= max_total:
                    continue
                cls = pred[idx]
                if seen.get(cls, 0) < target_per_class.get(cls, 1):
                    chosen.append(int(idx))
                    seen[cls] = seen.get(cls, 0) + 1
            if not chosen:
                break
            chosen_set = set(chosen)
            new_cases = [remaining[i] for i in sorted(chosen_set)]
            new_labels = [pred[i] for i in sorted(chosen_set)]
            accepted_cases.extend(new_cases)
            accepted_labels.extend(new_labels)
            remaining = [x for i, x in enumerate(remaining) if i not in chosen_set]
            model.fit(accepted_cases, accepted_labels)
            self.history.append({
                "round": round_id + 1,
                "accepted": len(new_cases),
                "distribution": {x: int(new_labels.count(x)) for x in sorted(set(new_labels))},
            })
        return model
