from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import numpy as np

from chimera.data.dataset import ChimeraDataset
from chimera.evaluation.validation import validate_outputs
from chimera.models.residual import GuidelineAnchoredResidualModel
from chimera.models.survival import HybridSurvivalModel
from chimera.reasoning.generator import SurvivalReasoningGenerator
from chimera.reasoning.supervised import SupervisedReasoningGenerator
from chimera.utils.jsonio import write_json


TASK1_LABELS = ["no", "yes"]
TASK2_LABELS = ["active_surveillance", "active_treatment", "continued_surveillance", "watchful_waiting"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--semi", action="store_true", help="Use conservative self-training for Tasks 1/2.")
    parser.add_argument("--no-neural", action="store_true", help="Disable MRI/WSI statistics for Task 3.")
    parser.add_argument("--all-cases", action="store_true", help="Write outputs for labeled and unlabeled cases.")
    return parser.parse_args()


def write_task12(
    dataset: ChimeraDataset,
    task: int,
    output: Path,
    semi: bool,
    all_cases: bool,
) -> None:
    labeled = dataset.cases(task, labeled_only=True)
    labels = TASK1_LABELS if task == 1 else TASK2_LABELS
    truth = [str(case.label) for case in labeled]

    model = GuidelineAnchoredResidualModel(
        labels,
        task=task,
        neural=False,
    ).fit(labeled, truth)
    if semi:
        from chimera.models.semisupervised import ConservativeSelfTraining

        unlabeled = [case for case in dataset.cases(task) if not case.has_label]
        ConservativeSelfTraining().fit(model, labeled, truth, unlabeled)

    generator = SupervisedReasoningGenerator(task=task).fit(labeled)
    cases = dataset.cases(task) if all_cases else labeled
    probabilities = model.predict_proba(cases)
    decisions = model.predict(cases)

    for case, decision, probability in zip(cases, decisions, probabilities):
        confidence = float(np.max(probability))
        reasoning = generator.generate(
            case,
            decision,
            confidence,
            exclude_case_id=case.case_id,
        )
        case_dir = output / f"task{task}" / case.case_id
        if task == 1:
            write_json(case_dir / "prostate-biopsy-decision.json", decision)
            write_json(case_dir / "prostate-biopsy-decision-reasoning.json", reasoning)
        else:
            write_json(case_dir / "prostate-treatment-decision.json", decision)
            write_json(case_dir / "prostate-treatment-decision-reasoning.json", reasoning)


def write_task3(dataset: ChimeraDataset, output: Path, neural: bool) -> None:
    labeled = dataset.cases(3, labeled_only=True)
    months = [float(case.label["months_to_recurrence"]) for case in labeled]
    events = [int(case.label["event"]) for case in labeled]
    model = HybridSurvivalModel(neural=neural).fit(labeled, months, events)
    cases = dataset.cases(3)
    predictions = model.predict(cases)
    generator = SurvivalReasoningGenerator()

    for case, prediction in zip(cases, predictions):
        case_dir = output / "task3" / case.case_id
        write_json(
            case_dir / "prostate-time-to-recurrence-or-last-follow-up.json",
            {
                "months_to_recurrence": prediction.predicted_months,
                "event": prediction.event,
            },
        )
        write_json(
            case_dir / "prostate-time-to-recurrence-or-last-follow-up-reasoning.json",
            generator.generate(
                case,
                prediction.event_probability,
                prediction.predicted_months,
                prediction.event,
            ),
        )


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    dataset = ChimeraDataset(args.data_root)
    write_task12(dataset, 1, output, args.semi, args.all_cases)
    write_task12(dataset, 2, output, args.semi, args.all_cases)
    write_task3(dataset, output, neural=not args.no_neural)
    valid, errors = validate_outputs(output)
    print("Output valid:", valid)
    if errors:
        print("\n".join(errors[:50]))
        raise SystemExit(1)
    print("Wrote", output)


if __name__ == "__main__":
    main()
