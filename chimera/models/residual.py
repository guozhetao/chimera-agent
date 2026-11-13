from __future__ import annotations

from ..data.dataset import Case
from .hybrid import BlendMetrics, RuleAnchoredClassifier


class GuidelineAnchoredResidualModel(RuleAnchoredClassifier):
    """Backward-compatible name for the rule-anchored decision model.

    The current model keeps the guideline rule as the anchor.  Task 1 learns
    only its exceptions; Task 2 blends the rule prior with a seed-averaged
    ExtraTrees ensemble in logit space and feeds the rule label as a feature.
    Both variants are materially better behaved than a classifier trained from
    scratch on these small cohorts.
    """

    def __init__(
        self,
        labels: list[str],
        estimator: str = "gb",
        rule_weight_grid: tuple[float, ...] = (0.80, 0.85, 0.90, 1.0),
        neural: bool = False,
        random_state: int = 42,
        task: int | None = None,
        mode: str = "auto",
        threshold: float | None = None,
    ):
        super().__init__(
            labels=labels,
            task=task,
            estimator=estimator,
            neural=neural,
            neural_mode="compact",
            mode=mode,
            random_state=random_state,
            threshold=threshold,
        )
        self.rule_weight_grid = rule_weight_grid

    def fit(self, cases: list[Case], y: list[str], unlabeled: list[Case] | None = None):
        return super().fit(cases, y, unlabeled)


__all__ = ["GuidelineAnchoredResidualModel", "BlendMetrics", "RuleAnchoredClassifier"]
