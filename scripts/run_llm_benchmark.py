from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chimera.data.dataset import ChimeraDataset
from chimera.evaluation.cv import classification_bootstrap, survival_bootstrap
from chimera.utils.jsonio import read_json, write_json


PROVIDERS = {
    "deepseek-flash": "deepseek",
    "deepseek-v4-pro": "deepseek",
    "glm-5.3": "zhipu",
    "glm-5.3-flash": "zhipu",
    "glm-flash": "zhipu",
    "glm-pro": "zhipu",
    "glm-legacy-air": "zhipu",
}

DISPLAY = {
    "deepseek-flash": "DeepSeek-Flash",
    "deepseek-v4-pro": "DeepSeek-V4-Pro",
    "glm-5.3": "GLM-5.3",
    "glm-5.3-flash": "GLM-5.3-Flash",
    "glm-flash": "GLM-5.3-Flash",
    "glm-pro": "GLM-5.3",
    "glm-legacy-air": "GLM-4.5-Air",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM-agent arm of the benchmark.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--benchmark-dir", default=ROOT / "results" / "benchmark")
    parser.add_argument("--models", nargs="+", default=["deepseek-flash", "glm-5.3"])
    parser.add_argument("--tasks", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--interactive-keys", action="store_true",
                        help="Prompt for missing API keys in the terminal; keys are never written to disk.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", default=ROOT / "results" / "benchmark" / "llm_results.json")
    return parser.parse_args()


def ensure_keys(models: list[str], interactive: bool) -> None:
    required = {PROVIDERS[model] for model in models if model in PROVIDERS}
    missing = []
    if "deepseek" in required and not os.getenv("DEEPSEEK_API_KEY"):
        missing.append("DEEPSEEK_API_KEY")
    if "zhipu" in required and not os.getenv("ZHIPU_API_KEY"):
        missing.append("ZHIPU_API_KEY")
    if not missing:
        return
    if not interactive:
        raise SystemExit(
            "Missing " + ", ".join(missing) + ". Set them in the environment or run with "
            "--interactive-keys and type them in the terminal."
        )
    if "DEEPSEEK_API_KEY" in missing:
        os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("DeepSeek API key: ").strip()
    if "ZHIPU_API_KEY" in missing:
        os.environ["ZHIPU_API_KEY"] = getpass.getpass("GLM API key: ").strip()


def _record(task: int, model: str, payload: dict, dataset: ChimeraDataset) -> dict:
    metrics = payload.get("metrics", {})
    predictions = payload.get("predictions", [])
    cases = {case.case_id: case for case in dataset.cases(task, labeled_only=True)}
    if not predictions:
        return {
            "task": task,
            "method": DISPLAY.get(model, model),
            "type": "LLM API",
            "n": 0,
            "metric": "f1_positive" if task == 1 else ("f1_weighted" if task == 2 else "harrell_c_index"),
            "value": None,
            "error": (metrics or {}).get("error", "no successful predictions"),
        }
    if task in (1, 2):
        truth = []
        predicted = []
        selected_cases = []
        for item in predictions:
            case = cases.get(item["case_id"])
            if case is None:
                continue
            truth.append(str(case.label))
            predicted.append(item.get("decision"))
            selected_cases.append(case)
        bootstrap = classification_bootstrap(truth, predicted, task=task, seed=42)
        decision_metrics = metrics.get("metrics", {})
        record = {
            "task": task,
            "method": DISPLAY.get(model, model),
            "type": "LLM API",
            "n": len(predictions),
            "metric": "f1_positive" if task == 1 else "f1_weighted",
            "value": decision_metrics.get("f1_positive" if task == 1 else "f1_weighted"),
            "accuracy": decision_metrics.get("accuracy"),
            "f1_weighted": decision_metrics.get("f1_weighted"),
            "f1_macro": decision_metrics.get("f1_macro"),
            "roc_auc": None,
            "reasoning_composite": (metrics.get("reasoning") or {}).get("composite"),
            "case_score": metrics.get("case_score"),
            "bootstrap": bootstrap,
        }
        for key in ("f1_positive", "f1_weighted", "accuracy"):
            if key in bootstrap:
                record[f"{key}_low"] = bootstrap[key]["low"]
                record[f"{key}_high"] = bootstrap[key]["high"]
        return record

    months, events, risks = [], [], []
    for item in predictions:
        case = cases.get(item["case_id"])
        if case is None:
            continue
        decision = item.get("decision", {})
        months.append(float(case.label["months_to_recurrence"]))
        events.append(int(case.label["event"]))
        risks.append(float(decision.get("risk", decision.get("event", 0.5))))
    confidence = survival_bootstrap(months, events, risks, seed=42)["harrell_c_index"]
    return {
        "task": 3,
        "method": DISPLAY.get(model, model),
        "type": "LLM API",
        "n": len(predictions),
        "metric": "harrell_c_index",
        "value": confidence.get("value"),
        "low": confidence["low"],
        "high": confidence["high"],
        "reasoning_composite": None,
        "bootstrap": {"harrell_c_index": confidence},
    }


def _write_table(records: list[dict], path: Path) -> None:
    lines = [
        "# LLM-agent benchmark",
        "",
        "The same case subset and two-stage prompt were used for every API model. "
        "Keys are read from environment variables and are never written to disk.",
        "",
        "| Task | Model | n | Primary metric | Value (95% CI) | Accuracy | Reasoning | Case score |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for record in records:
        low = record.get("low")
        high = record.get("high")
        if record["task"] in (1, 2):
            key = record["metric"]
            low = record.get(f"{key}_low")
            high = record.get(f"{key}_high")
        value = f"{record['value']:.3f}" if record.get("value") is not None else "—"
        if low is not None and high is not None:
            value += f" ({low:.3f}–{high:.3f})"
        accuracy = "" if record.get("accuracy") is None else f"{record['accuracy']:.3f}"
        reasoning = "" if record.get("reasoning_composite") is None else f"{record['reasoning_composite']:.3f}"
        case_score = "" if record.get("case_score") is None else f"{record['case_score']:.3f}"
        lines.append(
            f"| {record['task']} | {record['method']} | {record['n']} | {record['metric']} | "
            f"{value} | {accuracy} | {reasoning} | {case_score} |"
        )
    lines.extend([
        "",
        "> Case score includes the structured reasoning score.  ",
        "> The LLM arm is descriptive and not powered to rank APIs.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_keys(args.models, args.interactive_keys)
    dataset = ChimeraDataset(args.data_root)
    benchmark_dir = Path(args.benchmark_dir)
    llm_dir = benchmark_dir / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    failures: list[dict] = []

    for task in args.tasks:
        case_file = benchmark_dir / f"cases_task{task}.json"
        if not case_file.exists():
            raise SystemExit(f"Missing {case_file}; run scripts/run_benchmark.py first.")
        for model in args.models:
            provider = PROVIDERS.get(model)
            if provider is None:
                raise SystemExit(f"Unknown model alias: {model}")
            output = llm_dir / f"{model}_task{task}.json"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_direct_api.py"),
                "--data-root", args.data_root,
                "--task", str(task),
                "--provider", provider,
                "--model", model,
                "--case-ids", str(case_file),
                "--temperature", str(args.temperature),
                "--output", str(output),
            ]
            print(f"Running {DISPLAY.get(model, model)} on Task {task}...", flush=True)
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0 or not output.exists():
                failures.append({"task": task, "model": model, "error": "API run failed"})
                continue
            payload = read_json(output)
            records.append(_record(task, model, payload, dataset))
            print(f"  n={len(payload.get('predictions', []))} failures={len(payload.get('failures', []))}", flush=True)

    write_json(args.output, {
        "configuration": {
            "models": args.models,
            "tasks": args.tasks,
            "temperature": args.temperature,
            "case_source": str(benchmark_dir),
        },
        "results": records,
        "failures": failures,
    })
    _write_table(records, Path(args.output).with_suffix(".md"))
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
