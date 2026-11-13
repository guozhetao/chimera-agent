from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from chimera.data.dataset import ChimeraDataset
from chimera.evaluation.benchmark import (
    benchmark_classifier,
    benchmark_cox,
    benchmark_event_classifier,
    benchmark_our_classifier,
    benchmark_our_survival,
    benchmark_pathology,
    benchmark_rule,
    benchmark_survival_regressor,
    stratified_subset,
)
from chimera.utils.jsonio import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the proposed model against traditional ML/DL baselines.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default=ROOT / "results" / "benchmark")
    parser.add_argument("--n-per-task", type=int, default=40)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _estimators(seed: int):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    return {
        "Logistic regression": lambda: LogisticRegression(
            C=0.2, max_iter=5000, class_weight="balanced"
        ),
        "Random forest": lambda: RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, max_features=0.5,
            class_weight="balanced_subsample", random_state=seed,
        ),
        "Gradient boosting": lambda: GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.04, max_depth=2, random_state=seed,
        ),
        "MLP": lambda: MLPClassifier(
            hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=700,
            early_stopping=False, random_state=seed,
        ),
    }


def _rf_event(seed: int):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, max_features="sqrt",
        class_weight="balanced_subsample", random_state=seed,
    )


def _mlp_event(seed: int):
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(
        hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=700,
        early_stopping=False, random_state=seed,
    )


def _task_cases(dataset: ChimeraDataset, task: int, n: int, seed: int):
    cases = dataset.cases(task, labeled_only=True)
    labels = (
        [str(case.label) for case in cases]
        if task in (1, 2)
        else [str(int(case.label["event"])) for case in cases]
    )
    indices = stratified_subset(cases, labels, n=n, seed=seed)
    return [cases[index] for index in indices]


def _run_method(label: str, function, results: list[dict]) -> None:
    print(f"  {label}...", flush=True)
    try:
        results.append(function())
    except Exception as exc:
        print(f"    failed: {type(exc).__name__}: {exc}", flush=True)


def _write_table(results: list[dict], path: Path) -> None:
    columns = [
        "task", "method", "type", "n", "metric", "value", "low", "high",
        "accuracy", "f1_weighted", "f1_macro", "roc_auc", "reasoning_composite", "case_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in results:
            row = {key: record.get(key) for key in columns}
            if record["task"] == 3:
                row["low"] = record.get("bootstrap", {}).get("harrell_c_index", {}).get("low")
                row["high"] = record.get("bootstrap", {}).get("harrell_c_index", {}).get("high")
            else:
                key = record.get("metric")
                row["low"] = record.get(f"{key}_low")
                row["high"] = record.get(f"{key}_high")
            writer.writerow(row)

    lines = [
        "# Baseline and LLM benchmark",
        "",
        "All methods use the same stratified subset and repeated folds. "
        "Values are out-of-fold estimates; 95% bootstrap CIs are shown for the primary metric.",
        "",
        "| Task | Method | Type | n | Metric | Value (95% CI) | Accuracy | Weighted F1 | Reasoning | Case score |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for record in results:
        low = record.get("low")
        high = record.get("high")
        if record["task"] == 3:
            low = record.get("bootstrap", {}).get("harrell_c_index", {}).get("low")
            high = record.get("bootstrap", {}).get("harrell_c_index", {}).get("high")
        value = f"{record['value']:.3f}"
        if low is not None and high is not None:
            value += f" ({low:.3f}–{high:.3f})"
        reasoning = record.get("reasoning_composite")
        case_score = record.get("case_score")
        lines.append(
            f"| {record['task']} | {record['method']} | {record['type']} | {record['n']} | "
            f"{record['metric']} | {value} | "
            f"{'' if record.get('accuracy') is None else f'{record["accuracy"]:.3f}'} | "
            f"{'' if record.get('f1_weighted') is None else f'{record["f1_weighted"]:.3f}'} | "
            f"{'' if reasoning is None else f'{reasoning:.3f}'} | "
            f"{'' if case_score is None else f'{case_score:.3f}'} |"
        )
    lines.extend([
        "",
        "> Case score is shown only for methods that produce structured reasoning (GARF-H and LLM agents).  ",
        "> The benchmark is descriptive and supporting, not a leaderboard.",
    ])
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = ChimeraDataset(args.data_root)
    results: list[dict] = []
    case_ids: dict[str, list[str]] = {}

    for task in (1, 2):
        cases = _task_cases(dataset, task, args.n_per_task, args.seed)
        case_ids[f"task{task}"] = [case.case_id for case in cases]
        write_json(output / f"cases_task{task}.json", case_ids[f"task{task}"])
        print(f"Task {task}: n={len(cases)}", flush=True)
        results.append(benchmark_rule(task, cases, args.seed))
        for method, factory in _estimators(args.seed).items():
            print(f"  {method}...", flush=True)
            try:
                results.append(
                    benchmark_classifier(
                        task, cases, method, factory,
                        splits=args.splits, repeats=args.repeats, seed=args.seed,
                    )
                )
            except Exception as exc:
                print(f"    failed: {type(exc).__name__}: {exc}", flush=True)
        print("  Ours...", flush=True)
        results.append(
            benchmark_our_classifier(
                task, cases, splits=args.splits, repeats=args.repeats, seed=args.seed
            )
        )

    task3_cases = _task_cases(dataset, 3, args.n_per_task, args.seed)
    case_ids["task3"] = [case.case_id for case in task3_cases]
    write_json(output / "cases_task3.json", case_ids["task3"])
    months = [float(case.label["months_to_recurrence"]) for case in task3_cases]
    events = [int(case.label["event"]) for case in task3_cases]
    print(f"Task 3: n={len(task3_cases)} events={sum(events)}", flush=True)

    _run_method(
        "Pathology score",
        lambda: benchmark_pathology(task3_cases, months, events, args.seed),
        results,
    )
    _run_method(
        "Cox PH",
        lambda: benchmark_cox(task3_cases, months, events, args.splits, args.repeats, args.seed),
        results,
    )
    _run_method(
        "RF event",
        lambda: benchmark_event_classifier(
            task3_cases, months, events, "RF event",
            lambda: _rf_event(args.seed),
            use_neural=True, splits=args.splits, repeats=args.repeats, seed=args.seed,
        ),
        results,
    )
    _run_method(
        "MLP event",
        lambda: benchmark_event_classifier(
            task3_cases, months, events, "MLP event",
            lambda: _mlp_event(args.seed),
            use_neural=False, splits=args.splits, repeats=args.repeats, seed=args.seed,
        ),
        results,
    )
    _run_method(
        "ET survival",
        lambda: benchmark_survival_regressor(task3_cases, months, events, args.splits, args.repeats, args.seed),
        results,
    )
    _run_method(
        "Ours (GARF-H)",
        lambda: benchmark_our_survival(
            task3_cases, months, events, neural=True,
            splits=args.splits, repeats=args.repeats, seed=args.seed,
        ),
        results,
    )

    write_json(output / "benchmark_results.json", {
        "configuration": {
            "n_per_task": args.n_per_task,
            "splits": args.splits,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "case_ids": case_ids,
        "results": results,
    })
    _write_table(results, output / "benchmark_results.csv")
    print(f"Wrote {output / 'benchmark_results.json'}", flush=True)


if __name__ == "__main__":
    main()
