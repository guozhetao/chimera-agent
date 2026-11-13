from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation, submission generation, and figures end to end.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default=ROOT / "results" / "pipeline")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-submission", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("$", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True, cwd=ROOT)


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evaluation = output / "local_eval.json"
    repeats = 2 if args.quick else args.repeats

    run(
        [
            sys.executable,
            ROOT / "scripts" / "run_local_eval.py",
            "--data-root",
            args.data_root,
            "--output",
            evaluation,
            "--neural",
            "--splits",
            str(args.splits),
            "--repeats",
            str(repeats),
        ]
        + (["--quick"] if args.quick else [])
    )

    if not args.skip_submission:
        run(
            [
                sys.executable,
                ROOT / "scripts" / "create_submission.py",
                "--data-root",
                args.data_root,
                "--output",
                output / "submission",
                "--all-cases",
            ]
        )

    if not args.skip_figures:
        run(
            [
                sys.executable,
                ROOT / "scripts" / "make_report_figures.py",
                "--data-root",
                args.data_root,
                "--eval",
                evaluation,
                "--output-dir",
                output / "figures",
            ]
        )
    print(f"Pipeline complete: {output}")


if __name__ == "__main__":
    main()
