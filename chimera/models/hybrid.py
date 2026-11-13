"""Rule-anchored decision models.

The guideline rules are strong on this small, noisy release.  A conventional
classifier trained from scratch is consistently worse than the rule.  The model
therefore learns only the *exceptions* to the rule: a low-variance, depth-one
boosted model estimates P(label != rule) and the final decision flips the rule
only when that probability is high.  This keeps the clinical prior while still
learning systematic corrections from the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data.dataset import Case
from .features import build_dataframe, make_preprocessor
from .guideline import rule_probabilities, task1_rule, task2_rule


@dataclass
class BlendMetrics:
    alpha: float
    score: float
    threshold: float = 0.5
    mode: str = "exception"


def _rule_label(case: Case) -> str:
    return task1_rule(case) if case.task == 1 else task2_rule(case)


def _align(prob: np.ndarray, source_classes: np.ndarray, target_classes: list[str]) -> np.ndarray:
    output = np.zeros((len(prob), len(target_classes)), dtype=float)
    for index, label in enumerate(source_classes):
        if label in target_classes:
            output[:, target_classes.index(label)] = prob[:, index]
    row_sum = output.sum(axis=1, keepdims=True)
    return output / np.where(row_sum > 0, row_sum, 1.0)


class RuleAnchoredClassifier:
    """Guideline prior with a learned exception layer (and optional blend)."""

    def __init__(
        self,
        labels: list[str],
        task: int | None = None,
        estimator: str = "gb",
        neural: bool = False,
        neural_mode: str = "compact",
        mode: str = "auto",
        random_state: int = 42,
        threshold: float | None = None,
        blend_alpha: float = 0.65,
    ):
        self.labels = labels
        self.task = task
        self.estimator_name = estimator
        self.neural = neural
        self.neural_mode = neural_mode
        self.mode = mode
        self.random_state = random_state
        self.threshold = threshold
        self.blend_alpha = blend_alpha

        self.exception_model = None
        self.direct_model = None
        self.preprocessor = None
        self.rule_probability_: np.ndarray | None = None
        self.train_model_probability_: np.ndarray | None = None
        self.blend_metrics: BlendMetrics | None = None

    # ----------------------------------------------------------------- api
    def fit(self, cases: list[Case], y: list[str], unlabeled: list[Case] | None = None):
        if not cases:
            raise ValueError("at least one training case is required")
        self.task = self.task or cases[0].task
        mode = self.mode
        if mode == "auto":
            mode = "exception" if self.task == 1 else "blend"
        self.mode = mode

        frame = build_dataframe(cases, neural=self.neural, neural_mode=self.neural_mode)
        if mode == "blend":
            frame = frame.copy()
            frame["rule_label"] = [str(_rule_label(case)) for case in cases]
        self.preprocessor = make_preprocessor(frame)
        self.rule_probability_ = rule_probabilities(cases, self.labels)
        rule_labels = np.asarray([_rule_label(case) for case in cases])
        y_array = np.asarray([str(value) for value in y])

        if mode == "exception":
            self._fit_exception_layer(frame, y_array, rule_labels)
        elif mode == "blend":
            self._fit_direct_layer(frame, y_array)
        self.train_model_probability_ = self.predict_proba(cases)
        return self

    def predict_proba(self, cases: list[Case]) -> np.ndarray:
        if self.rule_probability_ is None and self.task is None:
            raise RuntimeError("model has not been fitted")
        base = rule_probabilities(cases, self.labels).astype(float)

        if self.mode == "exception" and self.exception_model is not None:
            frame = build_dataframe(cases, neural=self.neural, neural_mode=self.neural_mode)
            p_exception = self._positive_probability(self.exception_model, frame)
            positive = base[:, 1] * (1.0 - p_exception) + base[:, 0] * p_exception
            base = np.column_stack([1.0 - positive, positive])

        if self.mode == "blend" and self.direct_model is not None:
            frame = build_dataframe(cases, neural=self.neural, neural_mode=self.neural_mode)
            frame = frame.copy()
            frame["rule_label"] = [str(_rule_label(case)) for case in cases]
            model_prob = self.direct_model.predict_proba(frame)
            model_prob = _align(model_prob, self.direct_model.named_steps["model"].classes_, self.labels)
            alpha = self.blend_metrics.alpha if self.blend_metrics else 0.85
            base = self._logit_blend(alpha, base, model_prob)
        return base

    def predict(self, cases: list[Case]) -> list[str]:
        if self.mode == "exception" and self.exception_model is not None:
            frame = build_dataframe(cases, neural=self.neural, neural_mode=self.neural_mode)
            p_exception = self._positive_probability(self.exception_model, frame)
            rule_labels = np.asarray([_rule_label(case) for case in cases])
            predicted = rule_labels.copy()
            threshold = self.threshold if self.threshold is not None else 0.5
            # Flip a positive rule conservatively; a negative rule needs much
            # stronger evidence before it is overridden.
            predicted[(rule_labels == "yes") & (p_exception >= threshold)] = "no"
            predicted[(rule_labels == "no") & (p_exception >= 0.90)] = "yes"
            return predicted.tolist()
        probabilities = self.predict_proba(cases)
        return [self.labels[index] for index in probabilities.argmax(axis=1)]

    # ------------------------------------------------------------- internals
    def _make_exception_estimator(self):
        from sklearn.ensemble import GradientBoostingClassifier

        if self.estimator_name in {"logreg", "lr"}:
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(C=0.2, max_iter=5000, class_weight="balanced")
        if self.estimator_name in {"extra", "et"}:
            from sklearn.ensemble import ExtraTreesClassifier

            return ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=4,
                max_features=0.5,
                class_weight="balanced",
                random_state=self.random_state,
            )
        return GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.04,
            max_depth=1,
            random_state=self.random_state,
        )

    def _fit_exception_layer(self, frame, y_array: np.ndarray, rule_labels: np.ndarray) -> None:
        exception = (y_array != rule_labels).astype(int)
        if len(set(exception.tolist())) < 2:
            self.exception_model = None
            self.threshold = 0.5
            self.blend_metrics = BlendMetrics(1.0, 1.0, 0.5, "rule")
            return

        estimator = self._make_exception_estimator()
        from sklearn.pipeline import Pipeline

        self.exception_model = Pipeline([("preprocess", self.preprocessor), ("model", estimator)])
        self.exception_model.fit(frame, exception)

        if self.threshold is None:
            self.threshold = self._select_threshold(frame, y_array, rule_labels, exception)
        exception_probability = self._positive_probability(self.exception_model, frame)
        score = self._threshold_score(y_array, rule_labels, exception_probability, self.threshold)
        self.blend_metrics = BlendMetrics(1.0, float(score), float(self.threshold), "exception")

    def _fit_direct_layer(self, frame, y_array: np.ndarray) -> None:
        from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
        from sklearn.pipeline import Pipeline

        # Three seeds averaged with soft voting: on 72 cases a single tree
        # ensemble is high-variance, and the seed spread is the cheapest
        # regulariser that does not need another CV loop.
        estimators = [
            (
                f"et{seed}",
                ExtraTreesClassifier(
                    n_estimators=400,
                    min_samples_leaf=3,
                    max_features=0.5,
                    class_weight="balanced",
                    random_state=self.random_state + seed,
                ),
            )
            for seed in range(3)
        ]
        self.direct_model = Pipeline(
            [
                ("preprocess", self.preprocessor),
                ("model", VotingClassifier(estimators=estimators, voting="soft")),
            ]
        )
        self.direct_model.fit(frame, y_array)
        self.blend_metrics = BlendMetrics(0.65, 0.0, 0.5, "blend")

    def _select_threshold(self, frame, y_array, rule_labels, exception) -> float:
        from sklearn.metrics import f1_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import Pipeline

        if len(set(y_array.tolist())) < 2 or np.bincount(exception).min() < 2:
            return 0.5
        try:
            splits = list(
                StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state).split(frame, y_array)
            )
        except ValueError:
            return 0.5

        scores: dict[float, list[float]] = {
            round(float(value), 2): [] for value in np.arange(0.30, 0.76, 0.05)
        }
        for train_index, valid_index in splits:
            pipeline = Pipeline(
                [
                    ("preprocess", make_preprocessor(frame.iloc[train_index])),
                    ("model", self._make_exception_estimator()),
                ]
            )
            pipeline.fit(frame.iloc[train_index], exception[train_index])
            probability = self._positive_probability(pipeline, frame.iloc[valid_index])
            truth = y_array[valid_index]
            rule = rule_labels[valid_index]
            for threshold in scores:
                predicted = rule.copy()
                predicted[(rule == "yes") & (probability >= threshold)] = "no"
                predicted[(rule == "no") & (probability >= 0.90)] = "yes"
                scores[threshold].append(f1_score(truth, predicted, pos_label="yes", zero_division=0))

        best = max(scores.items(), key=lambda item: float(np.mean(item[1])))
        baseline = float(np.mean(scores[0.50]))
        return float(best[0]) if float(np.mean(best[1])) > baseline + 0.003 else 0.5

    @staticmethod
    def _positive_probability(model, frame) -> np.ndarray:
        probability = model.predict_proba(frame)
        classes = list(model.named_steps["model"].classes_)
        if 1 not in classes:
            return np.zeros(len(frame))
        return probability[:, classes.index(1)]

    @staticmethod
    def _threshold_score(y_array, rule_labels, exception_probability, threshold: float) -> float:
        from sklearn.metrics import f1_score

        predicted = rule_labels.copy()
        predicted[(rule_labels == "yes") & (exception_probability >= threshold)] = "no"
        predicted[(rule_labels == "no") & (exception_probability >= 0.90)] = "yes"
        return f1_score(y_array, predicted, pos_label="yes", zero_division=0)

    @staticmethod
    def _logit_blend(alpha: float, rule: np.ndarray, model: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        rule = np.clip(rule, eps, 1.0 - eps)
        model = np.clip(model, eps, 1.0 - eps)
        logit = alpha * np.log(rule / (1.0 - rule)) + (1.0 - alpha) * np.log(model / (1.0 - model))
        logit -= logit.max(axis=1, keepdims=True)
        exponent = np.exp(logit)
        return exponent / exponent.sum(axis=1, keepdims=True)
