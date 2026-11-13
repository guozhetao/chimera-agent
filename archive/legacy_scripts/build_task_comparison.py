from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.metrics import f1_score

from chimera.data.dataset import ChimeraDataset
from chimera.evaluation.survival_metrics import concordance_index
from chimera.utils.jsonio import read_json


def api_record(api_dir: Path, alias: str, task: int):
    matches = list(api_dir.glob(f"*_{alias}_task{task}.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one API result for {alias} task {task}, found {matches}")
    return read_json(matches[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--local", required=True)
    parser.add_argument("--local-neural", required=True)
    parser.add_argument("--api-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    local = read_json(args.local)
    neural = read_json(args.local_neural)
    api_dir = Path(args.api_dir)
    rows = []

    # Task 1.
    records = local["task1"]["predictions"]
    neural_records = neural["task1"]["predictions"]
    y = [record["true"] for record in records]
    ybin = [value == "yes" for value in y]
    # The local model currently collapses to the PI-RADS 4/5 guideline anchor.
    dataset = ChimeraDataset(args.data_root)
    cases = dataset.cases(1, labeled_only=True)
    rule = []
    for case in cases:
        try:
            pirads = int(case.prompt.get("pirads"))
        except (TypeError, ValueError):
            pirads = 0
        rule.append("yes" if pirads >= 4 else "no")
    for model, pred in [
        ("PI-RADS guideline rule", rule),
        ("GARF-M3", [record["predicted"] for record in records]),
        ("GARF-M3 + neural/semi", [record["predicted"] for record in neural_records]),
    ]:
        rows.append({
            "task": 1, "model": model, "n": len(y), "metric": "positive_f1",
            "value": f1_score(ybin, [value == "yes" for value in pred]),
            "scope": "full labeled OOF", "interpretable": True,
        })
    for alias in ["deepseek-flash", "deepseek-v4-pro", "glm-legacy-air"]:
        record = api_record(api_dir, alias, 1)
        rows.append({
            "task": 1, "model": alias, "n": len(record["predictions"]), "metric": "positive_f1",
            "value": record["metrics"]["metrics"]["f1_positive"], "scope": "API pilot", "interpretable": True,
        })

    # Task 2.
    records = local["task2"]["predictions"]
    neural_records = neural["task2"]["predictions"]
    y = [record["true"] for record in records]
    cases = dataset.cases(2, labeled_only=True)
    rule = []
    for case in cases:
        isup = int(case.prompt.get("bx_isup", 0) or 0)
        rule.append("continued_surveillance" if isup <= 0 else "active_surveillance" if isup == 1 else "active_treatment")
    for model, pred in [
        ("ISUP guideline rule", rule),
        ("GARF-M3", [record["predicted"] for record in records]),
        ("GARF-M3 + neural/semi", [record["predicted"] for record in neural_records]),
    ]:
        rows.append({
            "task": 2, "model": model, "n": len(y), "metric": "weighted_f1",
            "value": f1_score(y, pred, average="weighted"), "scope": "full labeled OOF", "interpretable": True,
        })
    for alias in ["deepseek-flash", "deepseek-v4-pro", "glm-legacy-air"]:
        record = api_record(api_dir, alias, 2)
        rows.append({
            "task": 2, "model": alias, "n": len(record["predictions"]), "metric": "weighted_f1",
            "value": record["metrics"]["metrics"]["f1_weighted"],
            "scope": "API pilot; missing Cont./WW", "interpretable": False,
        })

    # Task 3.
    for model, obj in [("GARF-M3 clinical/pathology", local), ("GARF-M3 + neural/semi", neural)]:
        records = obj["task3"]["predictions"]
        rows.append({
            "task": 3, "model": model, "n": len(records), "metric": "harrell_c_index",
            "value": concordance_index(
                [r["true_months"] for r in records],
                [r["true_event"] for r in records],
                [r["risk"] for r in records],
            ),
            "scope": "full labeled OOF", "interpretable": True,
        })
    for alias in ["deepseek-flash", "deepseek-v4-pro", "glm-legacy-air"]:
        record = api_record(api_dir, alias, 3)
        rows.append({
            "task": 3, "model": alias, "n": len(record["predictions"]), "metric": "harrell_c_index",
            "value": record["metrics"]["metrics"]["harrell_c_index"],
            "scope": "API pilot; one event", "interpretable": False,
        })

    frame = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
