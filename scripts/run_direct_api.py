from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chimera.agents.direct_api import make_agent, parse_json
from chimera.data.dataset import ChimeraDataset
from chimera.evaluation.local_evaluator import evaluate_task
from chimera.utils.jsonio import write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--task", type=int, choices=(1, 2, 3), required=True)
    p.add_argument("--provider", choices=("deepseek", "zhipu"), required=True)
    p.add_argument("--model", required=True, help="deepseek-flash, deepseek-v4-pro, glm-flash, glm-flashx, glm-pro, glm-legacy-air, glm-legacy, glm-legacy-flash")
    p.add_argument("--api-key", default=None, help="Prefer DEEPSEEK_API_KEY / ZHIPU_API_KEY instead of passing this.")
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--case-ids", default=None, help="JSON file containing a list of case IDs to evaluate.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output", required=True)
    p.add_argument("--fail-fast", action="store_true")
    return p.parse_args()


def normalize_output(task: int, result: dict[str, Any]):
    if task in (1, 2):
        return result.get("decision"), result.get("reasoning")
    risk_value = result.get("risk")
    if risk_value is None:
        risk_value = result.get("event", 0.5)
    try:
        risk = float(risk_value)
    except (TypeError, ValueError):
        risk = 0.5
    months_value = result.get("months_to_recurrence")
    try:
        months = float(months_value)
    except (TypeError, ValueError):
        # APIs often refuse an exact time. Derive a deterministic time from risk so C-index remains comparable.
        months = max(1.0, 120.0 * (1.0 - min(max(risk, 0.0), 1.0)))
    event_value = result.get("event")
    try:
        event = int(event_value)
    except (TypeError, ValueError):
        event = int(risk >= 0.5)
    return {"months_to_recurrence": months, "event": event, "risk": risk}, result.get("reasoning")


def main() -> None:
    args = parse_args()
    dataset = ChimeraDataset(args.data_root)
    cases = dataset.cases(args.task, labeled_only=True)
    if args.case_ids:
        selected = set(json.loads(Path(args.case_ids).read_text(encoding="utf-8")))
        cases = [case for case in cases if case.case_id in selected]
    if args.max_cases:
        cases = cases[: args.max_cases]
    agent = make_agent(args.provider, args.model, args.api_key, args.temperature)

    predictions = []
    failures = []
    valid_decisions = {"yes", "no"} if args.task == 1 else {
        "active_surveillance",
        "active_treatment",
        "continued_surveillance",
        "watchful_waiting",
    }
    start = time.time()
    for case in cases:
        try:
            result = agent.predict(case)
            decision, reasoning = normalize_output(args.task, result)
            if args.task in (1, 2) and decision not in valid_decisions:
                failures.append({
                    "case_id": case.case_id,
                    "error": f"invalid or missing decision: {decision!r}",
                })
                continue
            predictions.append({
                "case_id": case.case_id,
                "decision": decision,
                "reasoning": reasoning,
                "extracted_evidence": result.get("extracted_evidence"),
            })
        except Exception as exc:
            failures.append({"case_id": case.case_id, "error": f"{type(exc).__name__}: {exc}"})
            if args.fail_fast:
                raise
    elapsed = time.time() - start

    pred_by_id = {p["case_id"]: p for p in predictions if p.get("decision") is not None}
    evaluated_cases = [c for c in cases if c.case_id in pred_by_id]
    if not evaluated_cases:
        metrics = {"error": "no successful predictions"}
    elif args.task in (1, 2):
        truth = [str(c.label) for c in evaluated_cases]
        pred = [pred_by_id[c.case_id]["decision"] for c in evaluated_cases]
        reasoning_true = [c.reasoning for c in evaluated_cases]
        reasoning_pred = [pred_by_id[c.case_id]["reasoning"] for c in evaluated_cases]
        metrics = evaluate_task(
            args.task,
            truth,
            pred,
            reasoning_true=reasoning_true,
            reasoning_pred=reasoning_pred,
            case_ids=[c.case_id for c in evaluated_cases],
        ).to_dict()
    else:
        months = [float(c.label["months_to_recurrence"]) for c in evaluated_cases]
        events = [int(c.label["event"]) for c in evaluated_cases]
        risks = [float(pred_by_id[c.case_id]["decision"]["risk"]) for c in evaluated_cases]
        metrics = evaluate_task(
            3,
            [None] * len(evaluated_cases),
            [None] * len(evaluated_cases),
            months_true=months,
            events_true=events,
            predicted_risk=risks,
            case_ids=[c.case_id for c in evaluated_cases],
        ).to_dict()

    output = {
        "configuration": {
            "provider": args.provider,
            "model_alias": args.model,
            "model": agent.agent.config.model,
            "task": args.task,
            "temperature": args.temperature,
            "n_cases_requested": len(cases),
        },
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "predictions": predictions,
        "failures": failures,
    }
    write_json(args.output, output)
    print(json.dumps({
        "model": args.model,
        "task": args.task,
        "n_predictions": len(predictions),
        "n_failures": len(failures),
        "elapsed_seconds": elapsed,
        "case_score": metrics.get("case_score"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
