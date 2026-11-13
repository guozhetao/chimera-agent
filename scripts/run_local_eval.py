from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import numpy as np
from scipy.stats import rankdata

from chimera.data.dataset import ChimeraDataset
from chimera.evaluation.cv import classification_bootstrap, stratified_repeated_folds, summarize, survival_bootstrap
from chimera.evaluation.local_evaluator import evaluate_task
from chimera.models.cox import CoxPHModel
from chimera.models.features import TASK_LABELS
from chimera.models.guideline import task1_rule, task2_rule
from chimera.models.residual import GuidelineAnchoredResidualModel
from chimera.models.semisupervised import ConservativeSelfTraining
from chimera.models.survival import HybridSurvivalModel
from chimera.reasoning.generator import ReasoningGenerator, SurvivalReasoningGenerator
from chimera.reasoning.supervised import SupervisedReasoningGenerator
from chimera.utils.jsonio import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default=root / "results" / "local_eval.json")
    parser.add_argument("--neural", action="store_true", help="Use MRI/WSI set statistics for survival.")
    parser.add_argument("--semi", action="store_true", help="Use conservative self-training for Tasks 1/2.")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Use 2 repeats instead of the requested value.")
    return parser.parse_args()


def _metric_values(runs: list[dict], key: str) -> list[float]:
    values = []
    for run in runs:
        value = run.get(key, run.get("metrics", {}).get(key))
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return values


def _reasoning_values(runs: list[dict], key: str) -> list[float]:
    return [float(run.get("reasoning", {}).get(key, 0.0)) for run in runs if np.isfinite(run.get("reasoning", {}).get(key, 0.0))]


def _summarize_runs(runs: list[dict]) -> dict:
    decision_keys = [
        "accuracy",
        "balanced_accuracy",
        "f1_weighted",
        "f1_macro",
        "f1_positive",
        "roc_auc",
    ]
    output: dict[str, dict] = {}
    for key in decision_keys:
        values = _metric_values(runs, key)
        if values:
            output[key] = summarize(values)
    case_scores = _metric_values(runs, "case_score")
    if case_scores:
        output["case_score"] = summarize(case_scores)
    reasoning_keys = ["confidence", "variable_weights", "tool_sequence", "rationale", "composite"]
    output["reasoning"] = {}
    for key in reasoning_keys:
        values = _reasoning_values(runs, key)
        if values:
            output["reasoning"][key] = summarize(values)
    return output


def _majority_vote(predictions: np.ndarray, labels: list[str]) -> np.ndarray:
    output = np.empty(predictions.shape[1], dtype=object)
    for index in range(predictions.shape[1]):
        counts = Counter(predictions[:, index].tolist())
        # Deterministic tie-break follows the canonical label order.
        output[index] = max(labels, key=lambda label: (counts.get(label, 0), -labels.index(label)))
    return output


def run_classification(
    dataset: ChimeraDataset,
    task: int,
    splits: int,
    repeats: int,
    seed: int,
    semi: bool,
) -> dict:
    cases = dataset.cases(task, labeled_only=True)
    labels = TASK_LABELS[task]
    truth = np.asarray([str(case.label) for case in cases])
    case_ids = [case.case_id for case in cases]
    folds = list(stratified_repeated_folds(truth, n_splits=splits, n_repeats=repeats, seed=seed))

    probabilities = np.zeros((repeats, len(cases), len(labels)))
    predictions = np.empty((repeats, len(cases)), dtype=object)
    reasoning_supervised = [None] * len(cases)
    reasoning_base = [None] * len(cases)
    runs: list[dict] = []
    base_runs: list[dict] = []

    unlabeled = [case for case in dataset.cases(task) if not case.has_label]

    for repeat in range(repeats):
        repeat_probabilities = np.zeros((len(cases), len(labels)))
        repeat_predictions = np.empty(len(cases), dtype=object)
        repeat_reasoning = [None] * len(cases)
        repeat_reasoning_base = [None] * len(cases)
        for fold in [item for item in folds if item.repeat == repeat]:
            train_cases = [cases[index] for index in fold.train]
            test_cases = [cases[index] for index in fold.test]
            train_labels = truth[fold.train].tolist()

            model = GuidelineAnchoredResidualModel(
                labels,
                task=task,
                neural=False,
                random_state=seed + repeat,
            ).fit(train_cases, train_labels)
            if semi:
                ConservativeSelfTraining().fit(model, train_cases, train_labels, unlabeled)

            fold_probabilities = model.predict_proba(test_cases)
            fold_predictions = model.predict(test_cases)

            generator = SupervisedReasoningGenerator(task=task).fit(train_cases)
            baseline_generator = ReasoningGenerator()
            for position, case, decision, probability in zip(
                fold.test, test_cases, fold_predictions, fold_probabilities
            ):
                repeat_probabilities[position] = probability
                repeat_predictions[position] = decision
                confidence = float(np.max(probability))
                repeat_reasoning[position] = generator.generate(
                    case, decision, confidence, exclude_case_id=case.case_id
                )
                repeat_reasoning_base[position] = baseline_generator.generate(case, decision, confidence)

        probabilities[repeat] = repeat_probabilities
        predictions[repeat] = repeat_predictions
        if repeat == 0:
            reasoning_supervised = repeat_reasoning
            reasoning_base = repeat_reasoning_base

        runs.append(
            evaluate_task(
                task,
                truth.tolist(),
                repeat_predictions.tolist(),
                reasoning_true=[case.reasoning for case in cases],
                reasoning_pred=repeat_reasoning,
                probabilities=repeat_probabilities.tolist(),
                case_ids=case_ids,
            ).to_dict()
        )
        base_runs.append(
            evaluate_task(
                task,
                truth.tolist(),
                repeat_predictions.tolist(),
                reasoning_true=[case.reasoning for case in cases],
                reasoning_pred=repeat_reasoning_base,
                probabilities=repeat_probabilities.tolist(),
                case_ids=case_ids,
            ).to_dict()
        )

    final_predictions = _majority_vote(predictions, labels)
    final_probabilities = probabilities.mean(axis=0)
    final_metric = evaluate_task(
        task,
        truth.tolist(),
        final_predictions.tolist(),
        reasoning_true=[case.reasoning for case in cases],
        reasoning_pred=reasoning_supervised,
        probabilities=final_probabilities.tolist(),
        case_ids=case_ids,
    )
    rule_predictions = [
        task1_rule(case) if task == 1 else task2_rule(case) for case in cases
    ]
    rule_metric = evaluate_task(task, truth.tolist(), rule_predictions, case_ids=case_ids)

    summary = _summarize_runs(runs)
    summary["pooled"] = {
        "decision": final_metric.metrics,
        "reasoning": final_metric.reasoning,
        "case_score": final_metric.case_score,
        "bootstrap": classification_bootstrap(
            truth.tolist(),
            final_predictions.tolist(),
            final_probabilities.tolist(),
            task=task,
            seed=seed,
        ),
    }
    summary["baseline_rule"] = {
        "decision": {key: value for key, value in rule_metric.metrics.items() if not isinstance(value, (list, dict))},
        "case_score": rule_metric.case_score,
    }
    summary["reasoning_baseline"] = _summarize_runs(base_runs).get("reasoning", {})

    prediction_records = []
    for index, case in enumerate(cases):
        prediction_records.append(
            {
                "case_id": case.case_id,
                "true": truth[index],
                "predicted": final_predictions[index],
                "probability": final_probabilities[index].tolist(),
                "reasoning": reasoning_supervised[index],
                "reasoning_baseline": reasoning_base[index],
                "fold_predictions": predictions[:, index].tolist(),
            }
        )

    return {
        "summary": summary,
        "runs": runs,
        "baseline_runs": base_runs,
        "predictions": prediction_records,
    }


def _rank(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.full(len(values), 0.5)
    return rankdata(values) / len(values)


def _survival_ablation(months, events, risk, seed: int) -> dict:
    point = evaluate_task(
        3,
        [None] * len(months),
        [None] * len(months),
        months_true=months,
        events_true=events,
        predicted_risk=risk.tolist(),
    ).metrics["harrell_c_index"]
    confidence = survival_bootstrap(months, events, risk.tolist(), seed=seed)["harrell_c_index"]
    return {"value": float(point), **confidence}


def run_survival(
    dataset: ChimeraDataset,
    splits: int,
    repeats: int,
    seed: int,
    neural: bool,
) -> dict:
    cases = dataset.cases(3, labeled_only=True)
    months = np.asarray([float(case.label["months_to_recurrence"]) for case in cases])
    events = np.asarray([int(case.label["event"]) for case in cases])
    folds = list(stratified_repeated_folds(events, n_splits=splits, n_repeats=repeats, seed=seed))

    risk = np.zeros((repeats, len(cases)))
    event_probability = np.zeros((repeats, len(cases)))
    hazard = np.zeros((repeats, len(cases), 7))
    pathology = np.zeros((repeats, len(cases)))
    predicted_months = np.zeros((repeats, len(cases)))
    predicted_event = np.zeros((repeats, len(cases)), dtype=int)
    cox_risk = np.zeros((repeats, len(cases)))
    clinical_risk = np.zeros((repeats, len(cases)))
    reasoning = [None] * len(cases)
    generator = SurvivalReasoningGenerator()
    runs: list[dict] = []

    for repeat in range(repeats):
        for fold in [item for item in folds if item.repeat == repeat]:
            train_cases = [cases[index] for index in fold.train]
            test_cases = [cases[index] for index in fold.test]
            model = HybridSurvivalModel(neural=neural, random_state=seed + repeat).fit(
                train_cases,
                months[fold.train].tolist(),
                events[fold.train].tolist(),
            )
            components = model.predict_components(test_cases)
            predictions = model.predict(test_cases)

            risk[repeat, fold.test] = components["risk"]
            event_probability[repeat, fold.test] = components["event_probability"]
            pathology[repeat, fold.test] = components["pathology"]
            hazard[repeat, fold.test, : components["hazard"].shape[1]] = components["hazard"]
            predicted_months[repeat, fold.test] = [item.predicted_months for item in predictions]
            predicted_event[repeat, fold.test] = [item.event for item in predictions]
            if repeat == 0:
                for index, case, item in zip(fold.test, test_cases, predictions):
                    reasoning[index] = generator.generate(
                        case, item.event_probability, item.predicted_months, item.event
                    )

            cox = CoxPHModel(l2=50, random_state=seed + repeat).fit(
                train_cases,
                months[fold.train].tolist(),
                events[fold.train].tolist(),
            )
            cox_risk[repeat, fold.test] = cox.predict_risk(test_cases)

            clinical = HybridSurvivalModel(neural=False, random_state=seed + repeat).fit(
                train_cases,
                months[fold.train].tolist(),
                events[fold.train].tolist(),
            )
            clinical_risk[repeat, fold.test] = clinical.predict_components(test_cases)["risk"]

        metric = evaluate_task(
            3,
            [None] * len(cases),
            [None] * len(cases),
            months_true=months.tolist(),
            events_true=events.tolist(),
            predicted_risk=risk[repeat].tolist(),
            case_ids=[case.case_id for case in cases],
        )
        runs.append(metric.to_dict())

    mean_risk = np.mean([_rank(risk[repeat]) for repeat in range(repeats)], axis=0)
    mean_event = event_probability.mean(axis=0)
    mean_pathology = pathology.mean(axis=0)
    mean_hazard = hazard.mean(axis=0)[:, -1]
    mean_cox = np.mean([_rank(cox_risk[repeat]) for repeat in range(repeats)], axis=0)
    mean_clinical = np.mean([_rank(clinical_risk[repeat]) for repeat in range(repeats)], axis=0)

    ablations = {
        "pathology_only": _survival_ablation(months.tolist(), events.tolist(), _rank(mean_pathology), seed),
        "event_only": _survival_ablation(months.tolist(), events.tolist(), _rank(mean_event), seed),
        "hazard_only": _survival_ablation(months.tolist(), events.tolist(), _rank(mean_hazard), seed),
        "pathology_event": _survival_ablation(
            months.tolist(), events.tolist(), 0.5 * _rank(mean_pathology) + 0.5 * _rank(mean_event), seed
        ),
        "pathology_event_hazard": _survival_ablation(
            months.tolist(),
            events.tolist(),
            0.45 * _rank(mean_pathology) + 0.45 * _rank(mean_event) + 0.10 * _rank(mean_hazard),
            seed,
        ),
        "cox_ph": _survival_ablation(months.tolist(), events.tolist(), _rank(mean_cox), seed),
        "clinical_hybrid": _survival_ablation(months.tolist(), events.tolist(), _rank(mean_clinical), seed),
    }

    summary = _summarize_runs(runs)
    summary["pooled"] = {
        "harrell_c_index": _survival_ablation(months.tolist(), events.tolist(), mean_risk, seed),
        "bootstrap": survival_bootstrap(months.tolist(), events.tolist(), mean_risk.tolist(), seed=seed),
    }
    summary["ablation"] = ablations

    prediction_records = []
    for index, case in enumerate(cases):
        prediction_records.append(
            {
                "case_id": case.case_id,
                "true_event": int(events[index]),
                "true_months": float(months[index]),
                "risk": float(mean_risk[index]),
                "event_probability": float(mean_event[index]),
                "pathology_risk": float(mean_pathology[index]),
                "cumulative_hazard": float(mean_hazard[index]),
                "cox_risk": float(mean_cox[index]),
                "clinical_risk": float(mean_clinical[index]),
                "predicted_event": int(np.mean(predicted_event[:, index]) >= 0.5),
                "predicted_months": float(np.mean(predicted_months[:, index])),
                "reasoning": reasoning[index],
            }
        )

    return {
        "summary": summary,
        "runs": runs,
        "predictions": prediction_records,
        "components": {
            "months": months.tolist(),
            "events": events.tolist(),
            "risk_by_repeat": risk.tolist(),
            "event_probability_by_repeat": event_probability.tolist(),
            "hazard_by_repeat": hazard.tolist(),
            "pathology_by_repeat": pathology.tolist(),
            "cox_risk_by_repeat": cox_risk.tolist(),
            "clinical_risk_by_repeat": clinical_risk.tolist(),
        },
    }


def main() -> None:
    args = parse_args()
    repeats = 2 if args.quick else args.repeats
    dataset = ChimeraDataset(args.data_root)
    result = {
        "configuration": {
            "neural": args.neural,
            "semi": args.semi,
            "splits": args.splits,
            "repeats": repeats,
            "seed": args.seed,
            "cv": "repeated stratified k-fold",
        },
        "counts": dataset.counts(),
    }
    for task in (1, 2):
        print(f"Evaluating Task {task}...")
        result[f"task{task}"] = run_classification(
            dataset,
            task,
            splits=args.splits,
            repeats=repeats,
            seed=args.seed,
            semi=args.semi,
        )
    print("Evaluating Task 3...")
    result["task3"] = run_survival(
        dataset,
        splits=args.splits,
        repeats=repeats,
        seed=args.seed,
        neural=args.neural,
    )
    write_json(args.output, result)
    print(f"Wrote {args.output}")
    for task in (1, 2, 3):
        print(task, result[f"task{task}"]["summary"])


if __name__ == "__main__":
    main()
