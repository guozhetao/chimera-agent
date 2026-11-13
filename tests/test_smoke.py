from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chimera.evaluation.local_evaluator import evaluate_task
from chimera.evaluation.survival_metrics import concordance_index
from chimera.features.pathology_parser import parse_pathology


def test_pathology_parser():
    text = (
        "Radical prostatectomy showed Gleason 4+3, ISUP grade group 3, pT3b. "
        "Surgical margins were positive; seminal vesicles were invaded; lymph node metastasis was present."
    )
    findings = parse_pathology(text, surgical=True)
    assert findings.gleason_primary == 4
    assert findings.gleason_secondary == 3
    assert findings.isup_grade_group == 3
    assert findings.positive_margin
    assert findings.seminal_vesicle_invasion
    assert findings.lymph_node_metastasis


def test_c_index():
    time = [1.0, 2.0, 3.0]
    event = [1, 0, 0]
    assert concordance_index(time, event, [3.0, 2.0, 1.0]) == 1.0
    assert concordance_index(time, event, [1.0, 2.0, 3.0]) == 0.0


def test_task1_evaluator():
    result = evaluate_task(1, ["yes", "no"], ["yes", "yes"])
    assert result.metrics["accuracy"] == 0.5
    assert result.case_score == result.metrics["f1_positive"]

