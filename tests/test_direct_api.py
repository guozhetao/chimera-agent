from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chimera.agents.direct_api import _payload_is_valid, normalize_decision, parse_json


TRUNCATED = (
    '{\n  "decision": "yes",\n  "reasoning": {\n    "confidence": "clear",\n'
    '    "variable_weights": {\n      "pirads": "decisive",\n      "psad": "important"\n    },\n'
    '    "free_text": "The patient has a PI-RADS 5 lesion and a rising PSA'
)


def test_truncated_response_is_not_mistaken_for_nested_dict():
    with pytest.raises(ValueError):
        parse_json(TRUNCATED, {"decision"})
    assert not _payload_is_valid(TRUNCATED, {"decision"})


def test_fenced_json_is_parsed():
    raw = '```json\n{"decision": "no", "reasoning": {}}\n```'
    assert parse_json(raw, {"decision"})["decision"] == "no"


def test_decision_aliases_are_normalised():
    assert normalize_decision(1, "Biopsy") == "yes"
    assert normalize_decision(1, "no biopsy") == "no"
    assert normalize_decision(2, "AS") == "active_surveillance"
    assert normalize_decision(2, "watchful-waiting") == "watchful_waiting"
    assert normalize_decision(2, "maybe") is None
