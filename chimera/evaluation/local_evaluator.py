from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Iterable
import numpy as np

from .survival_metrics import concordance_index


WEIGHT_LEVELS = {"not_used": 0, "noted": 1, "important": 2, "decisive": 3}


@dataclass
class TaskMetrics:
    task: int
    n: int
    metrics: dict[str, float]
    reasoning: dict[str, float]
    case_score: float
    errors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classification_metrics(y_true: list[Any], y_pred: list[Any], task: int) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, classification_report,
        confusion_matrix, f1_score, roc_auc_score,
    )

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if task == 1:
        metrics["f1_positive"] = float(f1_score(y_true, y_pred, pos_label="yes", zero_division=0))
        labels = ["no", "yes"]
    else:
        labels = ["active_surveillance", "active_treatment", "continued_surveillance", "watchful_waiting"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["labels"] = labels
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    metrics["per_class"] = report
    return metrics


def survival_metrics(months: list[float], events: list[int], predicted_risk: list[float]) -> dict[str, float]:
    time = np.asarray(months, dtype=float)
    event = np.asarray(events, dtype=int)
    risk = np.asarray(predicted_risk, dtype=float)
    return {
        "harrell_c_index": float(concordance_index(time, event, risk)),
        "n_events": int(event.sum()),
        "n_censored": int((event == 0).sum()),
        "event_accuracy": float(np.mean((risk > np.median(risk[event == 1])) == event)) if event.sum() else float("nan"),
    }


def confidence_score(true: list[Any] | None, pred: list[Any] | None) -> float:
    if not true or not pred:
        return 0.0
    return float(np.mean([a == b for a, b in zip(true, pred)]))


def variable_weight_score(true: dict[str, Any] | None, pred: dict[str, Any] | None) -> float:
    if not true or not pred:
        return 0.0
    keys = list(true.keys())
    if not keys:
        return 0.0
    distances = []
    for key in keys:
        a = WEIGHT_LEVELS.get(str(true.get(key, "not_used")), 1)
        b = WEIGHT_LEVELS.get(str(pred.get(key, "not_used")), 1)
        distances.append(1.0 - abs(a - b) / 3.0)
    return float(np.mean(distances))


def _string_list(value: Any) -> list[str]:
    """Normalise model-produced tool traces to hashable strings.

    Some APIs return ``reveal_sequence`` as a list of objects rather than a
    list of tool names; scoring must not crash on that shape.
    """
    if not isinstance(value, (list, tuple)):
        return []
    output: list[str] = []
    for item in value:
        if isinstance(item, str):
            output.append(item)
        elif isinstance(item, dict):
            for key in ("tool", "name", "step", "variable"):
                candidate = item.get(key)
                if isinstance(candidate, str) and candidate:
                    output.append(candidate)
                    break
            else:
                output.append(json.dumps(item, sort_keys=True, ensure_ascii=False))
        else:
            output.append(str(item))
    return output


def tool_sequence_score(true: list[str] | None, pred: list[str] | None) -> dict[str, float]:
    true = _string_list(true)
    pred = _string_list(pred)
    true_set, pred_set = set(true), set(pred)
    if not pred_set:
        precision = 1.0 if not true_set else 0.0
    else:
        precision = len(true_set & pred_set) / len(pred_set)
    recall = len(true_set & pred_set) / len(true_set) if true_set else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    order = _sequence_order_score(true or [], pred or [])
    return {"precision": precision, "recall": recall, "f1": f1, "order": order, "composite": 0.6 * f1 + 0.4 * order}


def _sequence_order_score(true: list[str], pred: list[str]) -> float:
    if not true or not pred:
        return 0.0
    common = [x for x in pred if x in true]
    if not common:
        return 0.0
    positions = [true.index(x) for x in common]
    inversions = sum(1 for i in range(len(positions)) for j in range(i + 1, len(positions)) if positions[i] > positions[j])
    max_inv = len(positions) * (len(positions) - 1) / 2
    return 1.0 - inversions / max_inv if max_inv else 1.0


def rationale_score(true: str | None, pred: str | None) -> float:
    """A local deterministic surrogate for the official LLM rationale judge."""
    if not true or not pred:
        return 0.0
    true_tokens = set(re.findall(r"[a-z]{4,}|\d+(?:\.\d+)?", str(true).lower()))
    pred_tokens = set(re.findall(r"[a-z]{4,}|\d+(?:\.\d+)?", str(pred).lower()))
    if not true_tokens:
        return 0.0
    coverage = len(true_tokens & pred_tokens) / len(true_tokens)
    length = len(str(pred).split())
    length_penalty = 0.0 if 15 <= length <= 120 else min(1.0, abs(length - 40) / 100.0)
    evidence_bonus = float(any(x in str(pred).lower() for x in ["psa", "pi-rads", "pirads", "isup", "gleason"]))
    return float(max(0.0, min(1.0, 0.78 * coverage + 0.12 * evidence_bonus - 0.10 * length_penalty)))


def evaluate_task(
    task: int,
    y_true: list[Any],
    y_pred: list[Any],
    reasoning_true: list[dict[str, Any] | None] | None = None,
    reasoning_pred: list[dict[str, Any] | None] | None = None,
    probabilities: list[list[float]] | None = None,
    months_true: list[float] | None = None,
    events_true: list[int] | None = None,
    predicted_risk: list[float] | None = None,
    case_ids: list[str] | None = None,
    decision_weight: float = 0.75,
) -> TaskMetrics:
    case_ids = case_ids or [str(i) for i in range(len(y_true))]
    if task in (1, 2):
        metrics = classification_metrics(y_true, y_pred, task)
        if probabilities is not None and task == 1:
            metrics["roc_auc"] = float(roc_auc_safe(y_true, np.asarray(probabilities)[:, 1], positive="yes"))
        decision_score = metrics["f1_positive"] if task == 1 else metrics["f1_weighted"]
    elif task == 3 and months_true is not None and events_true is not None and predicted_risk is not None:
        metrics = survival_metrics(months_true, events_true, predicted_risk)
        decision_score = metrics["harrell_c_index"]
    else:
        raise ValueError("Task 3 requires months_true, events_true, and predicted_risk")

    reasoning_scores: dict[str, float] = {}
    if reasoning_true is not None and reasoning_pred is not None:
        conf = [confidence_score(a, b) for a, b in zip(_field(reasoning_true, "confidence"), _field(reasoning_pred, "confidence"))]
        weights = [variable_weight_score(a, b) for a, b in zip(_field(reasoning_true, "variable_weights"), _field(reasoning_pred, "variable_weights"))]
        tools = [tool_sequence_score(a, b) for a, b in zip(_field(reasoning_true, "reveal_sequence"), _field(reasoning_pred, "reveal_sequence"))]
        rationale = [rationale_score(a, b) for a, b in zip(_field(reasoning_true, "free_text"), _field(reasoning_pred, "free_text"))]
        reasoning_scores = {
            "confidence": float(np.mean(conf or [0])),
            "variable_weights": float(np.mean(weights or [0])),
            "tool_sequence": float(np.mean([x["composite"] for x in tools] or [0])),
            "rationale": float(np.mean(rationale or [0])),
            "composite": float(
                0.15 * np.mean(conf or [0])
                + 0.25 * np.mean(weights or [0])
                + 0.25 * np.mean([x["composite"] for x in tools] or [0])
                + 0.35 * np.mean(rationale or [0])
            ),
        }
        case_score = decision_weight * decision_score + (1 - decision_weight) * reasoning_scores["composite"]
    else:
        reasoning_scores = {"enabled": 0.0}
        case_score = decision_score

    errors: list[dict[str, Any]] = []
    if task in (1, 2):
        for cid, a, b in zip(case_ids, y_true, y_pred):
            if a != b:
                errors.append({"case_id": cid, "true": a, "predicted": b})
    else:
        # Store largest time-to-event ranking mistakes separately if desired.
        errors = []
    return TaskMetrics(task, len(y_true), metrics, reasoning_scores, float(case_score), errors)


def _field(records: Iterable[dict[str, Any] | None], key: str) -> list[Any | None]:
    return [x.get(key) if isinstance(x, dict) else None for x in records]


def roc_auc_safe(y_true: list[Any], probabilities: np.ndarray, positive: str = "yes") -> float:
    from sklearn.metrics import roc_auc_score

    y = [1 if x == positive else 0 for x in y_true]
    if len(set(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, probabilities))
