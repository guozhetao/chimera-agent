from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chimera.data.dataset import Case
from chimera.evaluation.benchmark import stratified_subset
from chimera.evaluation.cv import stratified_repeated_folds
from chimera.models.cox import CoxPHModel
from chimera.models.residual import GuidelineAnchoredResidualModel
from chimera.models.survival import HybridSurvivalModel, concordance_index
from chimera.reasoning.supervised import SupervisedReasoningGenerator


def _case(case_id, task, prompt, clinical=None, neural=None, reasoning=None, label=None):
    return Case(
        task=task,
        case_id=case_id,
        directory=Path("."),
        prompt=prompt,
        clinical=clinical or {},
        neural=neural or {},
        label=label,
        reasoning=reasoning,
    )


def test_stratified_folds_keep_every_class_in_training():
    y = np.asarray(["a"] * 15 + ["b"] * 20 + ["c"] * 5)
    folds = list(stratified_repeated_folds(y, n_splits=5, n_repeats=2, seed=7))
    assert len(folds) == 10
    for fold in folds:
        assert set(y[fold.train]) == {"a", "b", "c"}
        assert len(set(fold.train) & set(fold.test)) == 0


def test_stratified_subset_keeps_rare_classes():
    labels = ["common"] * 20 + ["rare"] * 2
    indices = stratified_subset(list(range(len(labels))), labels, n=8, seed=3)
    selected = [labels[index] for index in indices]
    assert len(indices) == 8
    assert "rare" in selected
    assert "common" in selected


def test_rule_anchored_exception_layer_learns_prior_biopsy_override():
    cases = []
    truth = []
    for index in range(60):
        prior = "Positive" if index % 2 == 0 else "None"
        label = "no" if prior == "Positive" else "yes"
        prompt = {
            "pirads": "4" if index % 3 else "5",
            "psa": 6.0 + index * 0.1,
            "psad": 0.15,
            "vol": 45.0,
            "age": 66,
            "bx": prior,
            "dre": "Normal",
            "cspca": 0.8,
            "psav": 1.2,
            "psap": 6.0,
        }
        clinical = {
            "radiology_report": "PI-RADS 4 lesion. PSA density 0.15.",
            "previous_notes": [],
            "psa_trend": [],
            "laboratory_results": [],
        }
        cases.append(_case(f"case{index}", 1, prompt, clinical))
        truth.append(label)

    model = GuidelineAnchoredResidualModel(["no", "yes"], task=1, random_state=0).fit(cases, truth)
    predicted = model.predict(cases)
    probabilities = model.predict_proba(cases)
    assert accuracy_score(truth, predicted) > 0.80
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_regularised_cox_recovers_risk_direction():
    rng = np.random.default_rng(3)
    cases, months, events = [], [], []
    for index in range(80):
        risk = index / 79
        prompt = {"age": 60, "psa": float(np.exp(1.0 + 3.5 * risk))}
        report = (
            "Gleason 4+3 (ISUP grade group 3), pT3a. Extraprostatic extension was present; "
            "surgical margins were positive; no lymph node metastasis."
            if risk > 0.5
            else "Gleason 3+3 (ISUP grade group 1), pT2c. No extraprostatic extension; margins were negative."
        )
        cases.append(_case(f"cox{index}", 3, prompt, {"surgical_pathology_report": report}))
        months.append(float(np.clip(70 - 55 * risk + rng.normal(0, 4), 2, 120)))
        events.append(int(rng.random() < 0.65))

    model = CoxPHModel(l2=20).fit(cases, months, events)
    risk = model.predict_risk(cases)
    assert concordance_index(np.asarray(months), np.asarray(events), risk) > 0.75


def test_hybrid_survival_outputs_are_valid():
    rng = np.random.default_rng(5)
    cases, months, events = [], [], []
    for index in range(50):
        risk = index / 49
        prompt = {"age": 60 + index % 10, "psa": float(np.exp(1.0 + 3.0 * risk))}
        clinical = {
            "radiology_report": "PI-RADS 4. PSA density 0.2.",
            "pathology_report": "Gleason 4+3 (ISUP grade group 3).",
            "surgical_pathology_report": (
                "Gleason 4+3 (ISUP grade group 3), pT3a. Extraprostatic extension was present; margins were positive."
                if risk > 0.5
                else "Gleason 3+3 (ISUP grade group 1), pT2c. Margins were negative."
            ),
        }
        cases.append(_case(f"surv{index}", 3, prompt, clinical))
        months.append(float(np.clip(70 - 50 * risk + rng.normal(0, 5), 2, 120)))
        events.append(int(rng.random() < 0.4))

    model = HybridSurvivalModel(neural=False, random_state=0).fit(cases, months, events)
    predictions = model.predict(cases)
    assert len(predictions) == len(cases)
    assert all(0.0 <= item.risk <= 1.0 for item in predictions)
    assert all(1.0 <= item.predicted_months <= 120.0 for item in predictions)
    assert all(item.event in (0, 1) for item in predictions)


def test_supervised_reasoning_profile_returns_structured_output():
    cases = []
    levels = ["not_used", "noted", "important", "decisive"]
    for index in range(12):
        label = "yes" if index % 2 else "no"
        reasoning = {
            "confidence": "clear",
            "variable_weights": {
                "pirads": levels[(index % 4)],
                "psa": "important",
                "bx": "noted",
            },
            "reveal_sequence": ["previous_notes", "psa_trend", "radiology_report"],
            "free_text": "PI-RADS lesion with PSA density considered before recommending biopsy.",
        }
        prompt = {
            "pirads": "4" if label == "yes" else "2",
            "psa": 6.0,
            "psad": 0.15,
            "vol": 40.0,
            "age": 65,
            "bx": "None",
            "dre": "Normal",
            "cspca": 0.8,
        }
        cases.append(
            _case(
                f"reason{index}",
                1,
                prompt,
                {"radiology_report": "PI-RADS lesion.", "previous_notes": [], "psa_trend": [], "laboratory_results": []},
                reasoning=reasoning,
                label=label,
            )
        )

    generator = SupervisedReasoningGenerator(task=1).fit(cases)
    output = generator.generate(cases[0], "yes", 0.85, exclude_case_id=cases[0].case_id)
    assert output["confidence"] in {"clear", "borderline", "uncertain"}
    assert set(output) == {"confidence", "variable_weights", "reveal_sequence", "free_text"}
    assert output["free_text"]
    assert isinstance(output["variable_weights"], dict)
