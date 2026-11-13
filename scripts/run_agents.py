from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from chimera.agents.api_agents import AGENT_REGISTRY
from chimera.agents.comparison import compare_agents
from chimera.data.dataset import ChimeraDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--task", type=int, choices=(1, 2, 3), required=True)
    p.add_argument("--models", nargs="+", default=["deepseek-flash"], help="Options: deepseek-flash deepseek-v4-pro glm-flash glm-pro")
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--output", default=root / "results" / "agent_comparison.json")
    p.add_argument("--temperature", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    agents = {}
    for name in args.models:
        if name not in AGENT_REGISTRY:
            raise ValueError(f"Unknown model {name}; choices are {list(AGENT_REGISTRY)}")
        agents[name] = AGENT_REGISTRY[name](temperature=args.temperature)
    dataset = ChimeraDataset(args.data_root)
    results = compare_agents(dataset, args.task, agents, args.max_cases, args.output)
    for result in results:
        print(result.name, "n=", result.n, "case_score=", result.case_score, "failures=", len(result.failures))
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
