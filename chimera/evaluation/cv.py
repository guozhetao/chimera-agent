"""Cross-validation and uncertainty helpers.

The original runner used unstratified shuffled folds.  With four Task 2 classes
(and only two watchful-waiting cases) and 19 Task 3 events, unstratified folds
produce unstable and occasionally degenerate estimates.  Everything here is
stratified and repeated, and every reported metric is accompanied by a
bootstrap confidence interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from .survival_metrics import concordance_index


@dataclass
class Fold:
    repeat: int
    fold: int
    train: np.ndarray
    test: np.ndarray


def stratified_repeated_folds(
    y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 4,
    seed: int = 42,
    stratify: bool = True,
) -> Iterator[Fold]:
    """Yield deterministic repeated folds, stratified whenever possible."""

    from sklearn.model_selection import KFold, StratifiedKFold

    y = np.asarray(y)
    counts = np.unique(y, return_counts=True)[1] if len(y) else np.array([])
    can_stratify = stratify and len(counts) > 1 and counts.min() >= n_splits
    for repeat in range(n_repeats):
        splitter = (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
            if can_stratify
            else KFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        )
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y)):
            yield Fold(repeat=repeat, fold=fold, train=np.asarray(train), test=np.asarray(test))


def summarize(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(array):
        return {"mean": float("nan"), "std": float("nan"), "sem": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "sem": float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "n": int(len(array)),
    }


def bootstrap_ci(
    metric: Callable[[np.ndarray], float],
    n: int,
    n_boot: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap an index-based metric."""

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        indices = rng.integers(0, n, n)
        try:
            value = float(metric(indices))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        if np.isfinite(value):
            values.append(value)
    if not values:
        return {"low": float("nan"), "high": float("nan")}
    alpha = (1.0 - confidence) / 2.0
    return {
        "low": float(np.percentile(values, 100 * alpha)),
        "high": float(np.percentile(values, 100 * (1 - alpha))),
    }


def classification_bootstrap(
    y_true: list[str],
    y_pred: list[str],
    probabilities: list[list[float]] | None = None,
    task: int = 1,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    truth = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    output: dict[str, dict[str, float]] = {}

    def add(name: str, metric: Callable[[np.ndarray], float]) -> None:
        point = float(metric(np.arange(len(truth))))
        output[name] = {"value": point, **bootstrap_ci(metric, len(truth), n_boot=n_boot, seed=seed)}

    add("accuracy", lambda idx: accuracy_score(truth[idx], predicted[idx]))
    add(
        "f1_weighted",
        lambda idx: f1_score(truth[idx], predicted[idx], average="weighted", zero_division=0),
    )
    add("f1_macro", lambda idx: f1_score(truth[idx], predicted[idx], average="macro", zero_division=0))
    if task == 1:
        add(
            "f1_positive",
            lambda idx: f1_score(truth[idx], predicted[idx], pos_label="yes", zero_division=0),
        )
        if probabilities is not None:
            positive = np.asarray([row[1] for row in probabilities])
            def auc(index: np.ndarray) -> float:
                labels = (truth[index] == "yes").astype(int)
                if labels.sum() in (0, len(labels)):
                    return float("nan")
                return float(roc_auc_score(labels, positive[index]))
            add("roc_auc", auc)
    return output


def survival_bootstrap(
    months: list[float],
    events: list[int],
    risk: list[float],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    time = np.asarray(months, dtype=float)
    event = np.asarray(events, dtype=int)
    score = np.asarray(risk, dtype=float)

    def c_index(index: np.ndarray) -> float:
        if event[index].sum() == 0 or len(np.unique(time[index])) < 2:
            return float("nan")
        return concordance_index(time[index], event[index], score[index])

    point = c_index(np.arange(len(time)))
    return {
        "harrell_c_index": {
            "value": float(point),
            **bootstrap_ci(c_index, len(time), n_boot=n_boot, seed=seed),
        }
    }


def paired_bootstrap_difference(
    metric: Callable[[np.ndarray, np.ndarray], float],
    n: int,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap a paired difference between two model score vectors."""

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        indices = rng.integers(0, n, n)
        try:
            values.append(float(metric(indices)))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
    if not values:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "low": float(np.percentile(values, 2.5)),
        "high": float(np.percentile(values, 97.5)),
    }
