from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from ..data.dataset import Case, ChimeraDataset
from ..evaluation.local_evaluator import evaluate_task
from ..utils.jsonio import write_json
from .api_agents import AgentPrediction, BaseApiAgent


@dataclass
class AgentRunResult:
    name: str
    n: int
    case_score: float | None
    metrics: dict[str, Any]
    elapsed_seconds: float
    failures: list[str]


def compare_agents(
    dataset: ChimeraDataset,
    task: int,
    agents: dict[str, BaseApiAgent],
    max_cases: int | None = None,
    output_path: str | Path | None = None,
) -> list[AgentRunResult]:
    cases = dataset.cases(task, labeled_only=True)
    if max_cases:
        cases = cases[:max_cases]
    results: list[AgentRunResult] = []
    for name, agent in agents.items():
        start = time.time()
        predictions: list[AgentPrediction] = []
        failures: list[str] = []
        for case in cases:
            try:
                predictions.append(agent.predict(case))
            except Exception as exc:
                failures.append(f"{case.case_id}: {type(exc).__name__}: {exc}")
        evaluated = [p for p in predictions if p.case_id in {c.case_id for c in cases}]
        if not evaluated:
            results.append(AgentRunResult(name, len(cases), None, {"error": "no predictions"}, time.time() - start, failures))
            continue
        if task in (1, 2):
            truth = [c.label for c in cases if c.case_id in {p.case_id for p in evaluated}]
            pred = []
            reasoning_pred = []
            reasoning_true = []
            by_id = {p.case_id: p for p in evaluated}
            for c in cases:
                if c.case_id not in by_id:
                    continue
                pred.append(by_id[c.case_id].decision)
                reasoning_pred.append(by_id[c.case_id].reasoning)
                reasoning_true.append(c.reasoning)
            metric = evaluate_task(
                task, truth, pred, reasoning_true=reasoning_true, reasoning_pred=reasoning_pred,
                case_ids=[c.case_id for c in cases if c.case_id in by_id],
            )
            results.append(AgentRunResult(name, len(evaluated), metric.case_score, metric.to_dict(), time.time() - start, failures))
        else:
            truth_cases = [c for c in cases if c.case_id in {p.case_id for p in evaluated}]
            by_id = {p.case_id: p for p in evaluated}
            months = [float(c.label["months_to_recurrence"]) for c in truth_cases]
            events = [int(c.label["event"]) for c in truth_cases]
            risk = [float(by_id[c.case_id].decision.get("risk", by_id[c.case_id].decision["months_to_recurrence"])) for c in truth_cases]
            metric = evaluate_task(
                3, [None] * len(truth_cases), [None] * len(truth_cases),
                months_true=months, events_true=events, predicted_risk=risk,
                case_ids=[c.case_id for c in truth_cases],
            )
            results.append(AgentRunResult(name, len(evaluated), metric.case_score, metric.to_dict(), time.time() - start, failures))
    if output_path:
        write_json(output_path, [r.__dict__ for r in results])
    return results
