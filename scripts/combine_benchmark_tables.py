from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chimera.utils.jsonio import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=ROOT / "results" / "benchmark" / "benchmark_results.json")
    parser.add_argument("--llm", default=ROOT / "results" / "benchmark" / "llm_results.json")
    parser.add_argument("--output", default=ROOT / "results" / "benchmark" / "benchmark_combined")
    return parser.parse_args()


def _interval(record: dict):
    task = int(record["task"])
    if task == 3:
        low = record.get("low")
        high = record.get("high")
        if low is None or high is None:
            bootstrap = record.get("bootstrap", {}).get("harrell_c_index", {})
            low, high = bootstrap.get("low"), bootstrap.get("high")
    else:
        key = record.get("metric", "")
        low = record.get(f"{key}_low")
        high = record.get(f"{key}_high")
        if low is None or high is None:
            bootstrap = record.get("bootstrap", {}).get(key, {})
            low, high = bootstrap.get("low"), bootstrap.get("high")
    return low, high


def main() -> None:
    args = parse_args()
    records = list(read_json(args.benchmark).get("results", []))
    if Path(args.llm).exists():
        records.extend(read_json(args.llm).get("results", []))
    # Keep one record per (task, method), preferring the LLM record with data.
    merged: dict[tuple[int, str], dict] = {}
    for record in records:
        record["method"] = str(record["method"]).replace("Ours (GARF-H survival)", "Ours (GARF-H)")
        key = (int(record["task"]), record["method"])
        if key not in merged or (record.get("value") is not None and merged[key].get("value") is None):
            merged[key] = record
    def _sort_key(item):
        value = item.get("value")
        return (int(item["task"]), 0 if value is not None else 1, -(float(value) if value is not None else 0.0))

    records = sorted(merged.values(), key=_sort_key)

    rows = []
    for record in records:
        low, high = _interval(record)
        value = record.get("value")
        value_text = "" if value is None else f"{float(value):.3f}"
        if low is not None and high is not None:
            value_text += f" ({float(low):.3f}-{float(high):.3f})"
        note = ""
        if record.get("error"):
            note = str(record["error"])[:120]
        elif int(record.get("n", 0)) and int(record.get("n", 0)) < 40:
            note = f"partial n={record['n']}"
        rows.append({
            "task": record["task"],
            "method": record["method"],
            "type": record.get("type", ""),
            "n": record.get("n", 0),
            "metric": record.get("metric", ""),
            "value": value_text,
            "accuracy": "" if record.get("accuracy") is None else f"{record['accuracy']:.3f}",
            "weighted_f1": "" if record.get("f1_weighted") is None else f"{record['f1_weighted']:.3f}",
            "reasoning": "" if record.get("reasoning_composite") is None else f"{record['reasoning_composite']:.3f}",
            "case_score": "" if record.get("case_score") is None else f"{record['case_score']:.3f}",
            "note": note,
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Combined benchmark table",
        "",
        "Local models: 5-fold × 4-repeat stratified CV on the shared subset. LLM agents: one pass over the same subset with the same two-stage prompt. Values are out-of-fold estimates; 95% bootstrap CIs are shown where available.",
        "",
        "| Task | Method | Type | n | Metric | Value (95% CI) | Accuracy | Weighted F1 | Reasoning | Case score | Note |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['method']} | {row['type']} | {row['n']} | {row['metric']} | "
            f"{row['value']} | {row['accuracy']} | {row['weighted_f1']} | {row['reasoning']} | "
            f"{row['case_score']} | {row['note']} |"
        )
    lines.extend([
        "",
        "> Case score is shown only for methods that produce structured reasoning.  ",
        "> The benchmark is supporting evidence, not a leaderboard; partial LLM rows are marked.",
    ])
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output.with_suffix('.md')} and {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
