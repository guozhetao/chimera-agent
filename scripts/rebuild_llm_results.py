from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chimera.data.dataset import ChimeraDataset
from chimera.utils.jsonio import read_json, write_json
from scripts.run_llm_benchmark import _record, _write_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild llm_results.json from per-model/per-task outputs.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--llm-dir", default=ROOT / "results" / "benchmark" / "llm")
    parser.add_argument("--output", default=ROOT / "results" / "benchmark" / "llm_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ChimeraDataset(args.data_root)
    records = []
    for path in sorted(Path(args.llm_dir).glob("*_task*.json")):
        match = re.match(r"(.+)_task(\d+)\.json$", path.name)
        if not match:
            continue
        model, task = match.group(1), int(match.group(2))
        if model not in {"deepseek-flash", "deepseek-v4-pro", "glm-5.3", "glm-5.3-flash"}:
            continue
        payload = read_json(path)
        if task in (1, 2):
            try:
                record = _record(task, model, payload, dataset)
            except Exception as exc:
                print(f"skip {path.name}: {type(exc).__name__}: {exc}")
                continue
        else:
            record = _record(task, model, payload, dataset)
        records.append(record)
        print(f"{path.name}: n={record.get('n')} value={record.get('value')}")
    write_json(args.output, {
        "configuration": {"rebuilt_from": str(args.llm_dir)},
        "results": records,
        "failures": [],
    })
    _write_table(records, Path(args.output).with_suffix(".md"))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
