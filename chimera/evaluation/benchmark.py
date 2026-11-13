"""Benchmark harness for traditional ML/DL baselines and LLM agents.

All methods are evaluated on the same stratified subset and the same repeated
folds.  The benchmark is deliberately a *supporting* experiment: the paper's
main contribution is the rule-anchored architecture, not a leaderboard.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..data.dataset import Case
from ..features.pathology_parser import pathology_risk_score
from ..models.cox import CoxPHModel
from ..models.features import build_dataframe, make_preprocessor
from ..models.guideline import rule_probabilities, task1_rule, task2_rule
from ..models.residual import GuidelineAnchoredResidualModel
from ..models.survival import HybridSurvivalModel, concordance_index
from .cv import classification_bootstrap, stratified_repeated_folds, survival_bootstrap


def stratified_subset(cases: list[Case], labels: list[str], n: int, seed: int = 42) -> list[int]:
    """Sample a class-balanced subset, guaranteeing rare classes are represented."""

    if n >= len(cases):
        return list(range(len(cases)))
    groups: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    rng = np.random.default_rng(seed)
    total = len(cases)
    allocation: dict[str, int] = {}
    for label, indices in groups.items():
        share = n * len(indices) / total
        allocation[label] = min(len(indices), max(1, int(round(share))))
        if len(indices) >= 2:
            allocation[label] = max(2, allocation[label])
    # Trim or fill to exactly n, preferring the largest classes.
    difference = n - sum(allocation.values())
    while difference != 0:
        if difference > 0:
            candidates = [label for label, indices in groups.items() if allocation[label] < len(indices)]
            if not candidates:
                break
            label = max(candidates, key=lambda item: len(groups[item]))
            allocation[label] += 1
            difference -= 1
        else:
            candidates = [label for label in groups if allocation[label] > 1]
            if not candidates:
                break
            label = max(candidates, key=lambda item: allocation[item])
            allocation[label] -= 1
            difference += 1
    selected: list[int] = []
    for label, indices in groups.items():
        selected.extend(rng.choice(indices, size=allocation[label], replace=False).tolist())
    return sorted(selected)


def _align(probability: np.ndarray, source_classes: np.ndarray, target_classes: list[str]) -> np.ndarray:
    output = np.zeros((len(probability), len(target_classes)), dtype=float)
    for index, label in enumerate(source_classes):
        if label in target_classes:
            output[:, target_classes.index(label)] = probability[:, index]
    row_sum = output.sum(axis=1, keepdims=True)
    return output / np.where(row_sum > 0, row_sum, 1.0)


def _majority_vote(predictions: np.ndarray, labels: list[str]) -> np.ndarray:
    output = np.empty(predictions.shape[1], dtype=object)
    for index in range(predictions.shape[1]):
        values = predictions[:, index].tolist()
        counts = {label: values.count(label) for label in labels}
        output[index] = max(labels, key=lambda label: (counts[label], -labels.index(label)))
    return output


def _classification_record(
    task: int,
    method: str,
    method_type: str,
    y_true: list[str],
    y_pred: list[str],
    probabilities: list[list[float]],
    seed: int,
    reasoning_true: list | None = None,
    reasoning_pred: list | None = None,
) -> dict:
    from .local_evaluator import evaluate_task

    metric = evaluate_task(
        task,
        y_true,
        y_pred,
        reasoning_true=reasoning_true,
        reasoning_pred=reasoning_pred,
        probabilities=probabilities,
    )
    bootstrap = classification_bootstrap(y_true, y_pred, probabilities, task=task, seed=seed)
    record = {
        "task": task,
        "method": method,
        "type": method_type,
        "n": len(y_true),
        "metric": "f1_positive" if task == 1 else "f1_weighted",
        "value": metric.metrics.get("f1_positive" if task == 1 else "f1_weighted"),
        "accuracy": metric.metrics.get("accuracy"),
        "f1_weighted": metric.metrics.get("f1_weighted"),
        "f1_macro": metric.metrics.get("f1_macro"),
        "roc_auc": metric.metrics.get("roc_auc"),
        "reasoning_composite": metric.reasoning.get("composite") if reasoning_pred is not None else None,
        "case_score": metric.case_score if reasoning_pred is not None else None,
        "bootstrap": bootstrap,
    }
    for key in ("f1_positive", "f1_weighted", "accuracy", "roc_auc"):
        if key in bootstrap:
            record[f"{key}_low"] = bootstrap[key]["low"]
            record[f"{key}_high"] = bootstrap[key]["high"]
    primary = "f1_positive" if task == 1 else "f1_weighted"
    if primary in bootstrap:
        record["low"] = bootstrap[primary]["low"]
        record["high"] = bootstrap[primary]["high"]
    return record


def _survival_record(
    method: str,
    method_type: str,
    months: list[float],
    events: list[int],
    risk: np.ndarray,
    seed: int,
) -> dict:
    from .local_evaluator import evaluate_task

    metric = evaluate_task(
        3,
        [None] * len(months),
        [None] * len(months),
        months_true=months,
        events_true=events,
        predicted_risk=risk.tolist(),
    )
    confidence = survival_bootstrap(months, events, risk.tolist(), seed=seed)["harrell_c_index"]
    return {
        "task": 3,
        "method": method,
        "type": method_type,
        "n": len(months),
        "metric": "harrell_c_index",
        "value": metric.metrics["harrell_c_index"],
        "low": confidence["low"],
        "high": confidence["high"],
        "bootstrap": {"harrell_c_index": confidence},
    }


def benchmark_rule(task: int, cases: list[Case], seed: int = 42) -> dict:
    labels = ["no", "yes"] if task == 1 else [
        "active_surveillance",
        "active_treatment",
        "continued_surveillance",
        "watchful_waiting",
    ]
    y_true = [str(case.label) for case in cases]
    y_pred = [task1_rule(case) if task == 1 else task2_rule(case) for case in cases]
    probabilities = rule_probabilities(cases, labels).tolist()
    return _classification_record(task, "Guideline rule", "Rule", y_true, y_pred, probabilities, seed)


def benchmark_classifier(
    task: int,
    cases: list[Case],
    method: str,
    estimator_factory: Callable,
    splits: int = 5,
    repeats: int = 4,
    seed: int = 42,
) -> dict:
    from sklearn.pipeline import Pipeline

    labels = ["no", "yes"] if task == 1 else [
        "active_surveillance",
        "active_treatment",
        "continued_surveillance",
        "watchful_waiting",
    ]
    y_true = np.asarray([str(case.label) for case in cases])
    frame = build_dataframe(cases, neural=False)
    folds = list(stratified_repeated_folds(y_true, n_splits=splits, n_repeats=repeats, seed=seed))
    probabilities = np.zeros((repeats, len(cases), len(labels)))
    predictions = np.empty((repeats, len(cases)), dtype=object)

    for repeat in range(repeats):
        for fold in [item for item in folds if item.repeat == repeat]:
            train_frame = frame.iloc[fold.train]
            test_frame = frame.iloc[fold.test]
            pipeline = Pipeline(
                [
                    ("preprocess", make_preprocessor(train_frame)),
                    ("model", estimator_factory()),
                ]
            )
            pipeline.fit(train_frame, y_true[fold.train])
            probability = pipeline.predict_proba(test_frame)
            probability = _align(probability, pipeline.named_steps["model"].classes_, labels)
            probabilities[repeat, fold.test] = probability
            predictions[repeat, fold.test] = np.asarray(labels)[probability.argmax(axis=1)]

    final_predictions = _majority_vote(predictions, labels)
    final_probabilities = probabilities.mean(axis=0)
    return _classification_record(
        task, method, "Traditional ML" if method not in {"MLP"} else "Deep learning",
        y_true.tolist(), final_predictions.tolist(), final_probabilities.tolist(), seed,
    )


def benchmark_our_classifier(
    task: int,
    cases: list[Case],
    splits: int = 5,
    repeats: int = 4,
    seed: int = 42,
) -> dict:
    labels = ["no", "yes"] if task == 1 else [
        "active_surveillance",
        "active_treatment",
        "continued_surveillance",
        "watchful_waiting",
    ]
    from ..reasoning.supervised import SupervisedReasoningGenerator

    y_true = np.asarray([str(case.label) for case in cases])
    folds = list(stratified_repeated_folds(y_true, n_splits=splits, n_repeats=repeats, seed=seed))
    probabilities = np.zeros((repeats, len(cases), len(labels)))
    predictions = np.empty((repeats, len(cases)), dtype=object)
    reasoning_predictions = np.empty((repeats, len(cases)), dtype=object)
    for repeat in range(repeats):
        for fold in [item for item in folds if item.repeat == repeat]:
            train_cases = [cases[index] for index in fold.train]
            test_cases = [cases[index] for index in fold.test]
            model = GuidelineAnchoredResidualModel(
                labels, task=task, neural=False, random_state=seed + repeat
            ).fit(train_cases, y_true[fold.train].tolist())
            probability = model.predict_proba(test_cases)
            decision = model.predict(test_cases)
            probabilities[repeat, fold.test] = probability
            predictions[repeat, fold.test] = decision
            generator = SupervisedReasoningGenerator(task=task).fit(train_cases)
            for position, case, decision_value, probability_row in zip(
                fold.test, test_cases, decision, probability
            ):
                reasoning_predictions[repeat, position] = generator.generate(
                    case, decision_value, float(np.max(probability_row)),
                    exclude_case_id=case.case_id,
                )
    final_predictions = _majority_vote(predictions, labels)
    final_probabilities = probabilities.mean(axis=0)
    final_reasoning = []
    for index in range(len(cases)):
        chosen = None
        for repeat in range(repeats):
            if predictions[repeat, index] == final_predictions[index]:
                chosen = reasoning_predictions[repeat, index]
                break
        final_reasoning.append(chosen or reasoning_predictions[0, index])
    return _classification_record(
        task, "Ours (GARF-H)", "Proposed", y_true.tolist(), final_predictions.tolist(),
        final_probabilities.tolist(), seed,
        reasoning_true=[case.reasoning for case in cases],
        reasoning_pred=final_reasoning,
    )


def _survival_folds(events: np.ndarray, splits: int, repeats: int, seed: int):
    return list(stratified_repeated_folds(events, n_splits=splits, n_repeats=repeats, seed=seed))


def benchmark_pathology(cases: list[Case], months: list[float], events: list[int], seed: int = 42) -> dict:
    risk = np.asarray([
        pathology_risk_score(case.clinical, case.prompt.get("psa")) for case in cases
    ], dtype=float)
    return _survival_record("Pathology score", "Clinical anchor", months, events, risk, seed)


def benchmark_cox(
    cases: list[Case],
    months: list[float],
    events: list[int],
    splits: int = 5,
    repeats: int = 4,
    seed: int = 42,
) -> dict:
    event_array = np.asarray(events, dtype=int)
    folds = _survival_folds(event_array, splits, repeats, seed)
    risk = np.zeros((repeats, len(cases)))
    for repeat in range(repeats):
        for fold in [item for item in folds if item.repeat == repeat]:
            model = CoxPHModel(l2=50, random_state=seed + repeat).fit(
                [cases[index] for index in fold.train],
                [months[index] for index in fold.train],
                [events[index] for index in fold.train],
            )
            risk[repeat, fold.test] = model.predict_risk([cases[index] for index in fold.test])
    mean_risk = np.mean([_rank(risk[repeat]) for repeat in range(repeats)], axis=0)
    return _survival_record("Cox PH", "Traditional survival", months, events, mean_risk, seed)


def _rank(values: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(values) / len(values)


def benchmark_event_classifier(
    cases: list[Case],
    months: list[float],
    events: list[int],
    method: str,
    estimator_factory: Callable,
    use_neural: bool,
    splits: int = 5,
    repeats: int = 4,
    seed: int = 42,
) -> dict:
    from sklearn.pipeline import Pipeline

    event_array = np.asarray(events, dtype=int)
    frame = build_dataframe(cases, neural=use_neural, neural_mode="compact")
    folds = _survival_folds(event_array, splits, repeats, seed)
    risk = np.zeros((repeats, len(cases)))
    for repeat in range(repeats):
        for fold in [item for item in folds if item.repeat == repeat]:
            train_frame = frame.iloc[fold.train]
            test_frame = frame.iloc[fold.test]
            pipeline = Pipeline(
                [("preprocess", make_preprocessor(train_frame)), ("model", estimator_factory())]
            )
            pipeline.fit(train_frame, event_array[fold.train])
            probability = pipeline.predict_proba(test_frame)
            classes = list(pipeline.named_steps["model"].classes_)
            risk[repeat, fold.test] = (
                probability[:, classes.index(1)] if 1 in classes else 0.0
            )
    mean_risk = np.mean([_rank(risk[repeat]) for repeat in range(repeats)], axis=0)
    method_type = "Deep learning" if method == "MLP event" else "Traditional ML"
    return _survival_record(method, method_type, months, events, mean_risk, seed)


def benchmark_survival_regressor(
    cases: list[Case],
    months: list[float],
    events: list[int],
    splits: int = 5,
    repeats: int = 4,
    seed: int = 42,
) -> dict:
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.pipeline import Pipeline

    event_array = np.asarray(events, dtype=int)
    frame = build_dataframe(cases, neural=False)
    folds = _survival_folds(event_array, splits, repeats, seed)
    risk = np.zeros((repeats, len(cases)))
    target = np.log(np.clip(months, 1.0, 120.0))
    for repeat in range(repeats):
        for fold in [item for item in folds if item.repeat == repeat]:
            train_frame = frame.iloc[fold.train]
            test_frame = frame.iloc[fold.test]
            pipeline = Pipeline(
                [
                    ("preprocess", make_preprocessor(train_frame)),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=300, min_samples_leaf=3, max_features=0.5,
                            random_state=seed + repeat,
                        ),
                    ),
                ]
            )
            pipeline.fit(train_frame, target[fold.train])
            risk[repeat, fold.test] = -pipeline.predict(test_frame)
    mean_risk = np.mean([_rank(risk[repeat]) for repeat in range(repeats)], axis=0)
    return _survival_record("ET survival", "Traditional survival", months, events, mean_risk, seed)


def benchmark_our_survival(
    cases: list[Case],
    months: list[float],
    events: list[int],
    neural: bool = True,
    splits: int = 5,
    repeats: int = 4,
    seed: int = 42,
) -> dict:
    event_array = np.asarray(events, dtype=int)
    folds = _survival_folds(event_array, splits, repeats, seed)
    risk = np.zeros((repeats, len(cases)))
    for repeat in range(repeats):
        for fold in [item for item in folds if item.repeat == repeat]:
            model = HybridSurvivalModel(neural=neural, random_state=seed + repeat).fit(
                [cases[index] for index in fold.train],
                [months[index] for index in fold.train],
                [events[index] for index in fold.train],
            )
            risk[repeat, fold.test] = model.predict_components(
                [cases[index] for index in fold.test]
            )["risk"]
    mean_risk = np.mean([_rank(risk[repeat]) for repeat in range(repeats)], axis=0)
    return _survival_record("Ours (GARF-H)", "Proposed", months, events, mean_risk, seed)
