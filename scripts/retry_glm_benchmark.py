"""Re-run the GLM benchmark tasks with the hardened client.

The API key is read from ``ZHIPU_API_KEY`` or prompted with ``getpass``; it is
passed to the child process through the environment and never written to disk.
Valid cached responses are reused, while truncated cache entries are refetched.
"""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--provider", default="zhipu")
    parser.add_argument("--model", default="glm-5.3")
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--llm-dir", default=ROOT / "results" / "benchmark" / "llm")
    parser.add_argument("--benchmark-dir", default=ROOT / "results" / "benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    key_env = "ZHIPU_API_KEY" if args.provider == "zhipu" else "DEEPSEEK_API_KEY"
    key = os.getenv(key_env)
    if not key:
        key = getpass.getpass(f"{key_env}: ")
    if not key.strip():
        raise SystemExit("empty API key")
    env = dict(os.environ)
    env[key_env] = key.strip()

    llm_dir = Path(args.llm_dir)
    llm_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        output = llm_dir / f"{args.model}_task{task}.json"
        case_file = Path(args.benchmark_dir) / f"cases_task{task}.json"
        if not case_file.exists():
            raise SystemExit(f"Missing {case_file}; run scripts/run_benchmark.py first.")
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_direct_api.py"),
            "--data-root",
            args.data_root,
            "--task",
            str(task),
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--case-ids",
            str(case_file),
            "--output",
            str(output),
        ]
        print(f"Running {args.model} task {task} -> {output}", flush=True)
        subprocess.run(command, check=True, env=env)
        print(f"Done task {task}", flush=True)


if __name__ == "__main__":
    main()
