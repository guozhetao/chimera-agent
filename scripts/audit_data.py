from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from chimera.data.dataset import ChimeraDataset
from chimera.utils.jsonio import write_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", default=root / "results" / "audit.json")
    args = p.parse_args()
    dataset = ChimeraDataset(args.data_root)
    report = {"counts": dataset.counts()}
    for task in (1, 2, 3):
        cases = dataset.cases(task, labeled_only=True)
        entry = {"n_labeled": len(cases)}
        if task in (1, 2):
            entry["labels"] = dict(Counter(str(c.label) for c in cases))
            entry["confidence"] = dict(Counter((c.reasoning or {}).get("confidence", "missing") for c in cases))
            entry["variable_keys"] = dict(Counter(k for c in cases for k in (c.reasoning or {}).get("variable_weights", {})))
            entry["tool_usage"] = dict(Counter(x for c in cases for x in (c.reasoning or {}).get("reveal_sequence", [])))
        else:
            entry["events"] = dict(Counter(int(c.label["event"]) for c in cases))
            entry["months_range"] = [min(float(c.label["months_to_recurrence"]) for c in cases), max(float(c.label["months_to_recurrence"]) for c in cases)]
        report[f"task{task}"] = entry
    write_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
