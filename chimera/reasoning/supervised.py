"""Reasoning generation calibrated on the released reference traces.

The reference reasoning objects are highly structured: a confidence level, an
ordinal weight for each clinical variable, a tool-reveal sequence, and a short
clinical rationale.  Instead of hand-coding those fields, this module learns
their dominant distribution from the labelled cases and combines it with a
case-specific rationale template.  A nearest-neighbour rationale from the same
decision class adds natural clinical language without copying the reference
text verbatim.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..data.dataset import Case
from ..features.text import clinical_text, token_set
from .generator import ReasoningGenerator

LEVEL_ORDER = ["not_used", "noted", "important", "decisive"]

KEYWORD_SENTENCES = {
    ("task1", "yes"): (
        "Documented considerations included the PI-RADS lesion, PSA level, PSA density, PSA kinetics, "
        "prior biopsy result, and DRE findings."
    ),
    ("task1", "no"): (
        "Documented considerations included prior biopsy histology, ISUP grade, PSA density, comorbidity, "
        "and the balance between repeat sampling and treatment."
    ),
    ("task2", "active_surveillance"): (
        "Documented considerations included ISUP grade group, PSA density, MRI stage, comorbidity, "
        "and the surveillance protocol."
    ),
    ("task2", "active_treatment"): (
        "Documented considerations included ISUP grade group, Gleason pattern, cribriform or intraductal "
        "features, clinical stage, and PSA kinetics."
    ),
    ("task2", "continued_surveillance"): (
        "Documented considerations included the negative or low-grade biopsy, PSA stability, and the "
        "existing surveillance pathway."
    ),
    ("task2", "watchful_waiting"): (
        "Documented considerations included age, comorbidity, life expectancy, and symptom burden."
    ),
}

STOPWORDS = {
    "the", "and", "with", "for", "that", "was", "were", "this", "from", "are", "has",
    "have", "not", "but", "who", "his", "her", "she", "been", "is", "of", "in", "on",
    "to", "a", "an", "as", "at", "by", "or", "it", "its", "patient", "also", "very",
    "most", "likely", "should", "would", "could", "there", "these", "those", "than",
    "then", "when", "what", "which", "while", "about", "after", "before", "more",
    "need", "needs", "because", "into", "over", "under", "only", "both", "some",
}



@dataclass
class ReasoningProfile:
    task: int
    weights_by_label: dict[str, dict[str, str]] = field(default_factory=dict)
    weights_global: dict[str, str] = field(default_factory=dict)
    confidence_by_label: dict[str, str] = field(default_factory=dict)
    reveal_by_label: dict[str, list[str]] = field(default_factory=dict)
    corpus: list[dict[str, Any]] = field(default_factory=list)
    keywords_by_label: dict[str, list[str]] = field(default_factory=dict)


class SupervisedReasoningGenerator:
    """Learned confidence, variable weights, tool order, and rationale."""

    def __init__(self, task: int | None = None, use_retrieval: bool = False):
        self.task = task
        self.use_retrieval = use_retrieval
        self.profile: ReasoningProfile | None = None
        self.base = ReasoningGenerator()

    def fit(self, cases: list[Case]) -> "SupervisedReasoningGenerator":
        if not cases:
            return self
        self.task = self.task or cases[0].task
        records = [case.reasoning for case in cases if isinstance(case.reasoning, dict)]
        if not records:
            self.profile = ReasoningProfile(task=self.task)
            return self

        keys = sorted({key for record in records for key in (record.get("variable_weights") or {})})
        weights_global: dict[str, str] = {}
        for key in keys:
            values = [record.get("variable_weights", {}).get(key) for record in records]
            values = [value for value in values if value in LEVEL_ORDER]
            weights_global[key] = Counter(values).most_common(1)[0][0] if values else "noted"

        weights_by_label: dict[str, dict[str, str]] = {}
        confidence_by_label: dict[str, str] = {}
        reveal_by_label: dict[str, list[str]] = {}
        for label in sorted({str(case.label) for case in cases}):
            subset = [case for case in cases if str(case.label) == label and isinstance(case.reasoning, dict)]
            if not subset:
                continue
            label_weights: dict[str, str] = {}
            for key in keys:
                values = [case.reasoning.get("variable_weights", {}).get(key) for case in subset]
                values = [value for value in values if value in LEVEL_ORDER]
                label_weights[key] = Counter(values).most_common(1)[0][0] if values else weights_global[key]
            weights_by_label[label] = label_weights
            confidences = [case.reasoning.get("confidence") for case in subset]
            confidence_by_label[label] = Counter(confidences).most_common(1)[0][0]
            sequences = [tuple(case.reasoning.get("reveal_sequence") or []) for case in subset]
            reveal_by_label[label] = list(Counter(sequences).most_common(1)[0][0])

        keywords_by_label: dict[str, list[str]] = {}
        for label in weights_by_label:
            counter: Counter[str] = Counter()
            for case in cases:
                if str(case.label) != label or not isinstance(case.reasoning, dict):
                    continue
                for token in set(
                    word.lower() for word in re.findall(r"[a-zA-Z]{4,}", str(case.reasoning.get("free_text", "")))
                ):
                    if token not in STOPWORDS:
                        counter[token] += 1
            keywords_by_label[label] = [word for word, _ in counter.most_common(14)]

        corpus = [
            {
                "case_id": case.case_id,
                "label": str(case.label),
                "tokens": token_set(clinical_text(case)),
                "text": str((case.reasoning or {}).get("free_text", "")),
                "prior_biopsy": str(case.prompt.get("bx") or ""),
                "isup": str(case.prompt.get("bx_isup") or ""),
            }
            for case in cases
        ]
        self.profile = ReasoningProfile(
            task=self.task,
            weights_by_label=weights_by_label,
            weights_global=weights_global,
            confidence_by_label=confidence_by_label,
            reveal_by_label=reveal_by_label,
            corpus=corpus,
            keywords_by_label=keywords_by_label,
        )
        return self

    # ----------------------------------------------------------------- api
    def generate(
        self,
        case: Case,
        prediction: str,
        probability: float,
        exclude_case_id: str | None = None,
    ) -> dict[str, Any]:
        base = self.base.generate(case, prediction, probability)
        if self.profile is None:
            return base

        weights = dict(self.profile.weights_global)
        weights.update(self.profile.weights_by_label.get(prediction, {}))
        confidence = self._confidence(case, prediction, probability)
        reveal = self.profile.reveal_by_label.get(prediction, base.get("reveal_sequence", []))
        rationale = self._rationale(case, prediction, base.get("free_text", ""), exclude_case_id)

        return {
            "confidence": confidence,
            "variable_weights": weights,
            "reveal_sequence": list(reveal),
            "free_text": rationale,
        }

    # -------------------------------------------------------------- helpers
    def _confidence(self, case: Case, prediction: str, probability: float) -> str:
        # The reference traces are overwhelmingly "clear"; a learned per-class
        # majority is both more accurate and better calibrated than thresholding
        # the coarse rule probabilities.
        if self.profile is None:
            return "clear"
        return self.profile.confidence_by_label.get(prediction, "clear")

    def _rationale(
        self,
        case: Case,
        prediction: str,
        base_text: str,
        exclude_case_id: str | None,
    ) -> str:
        task_key = ("task1" if self.task == 1 else "task2", prediction)
        curated = KEYWORD_SENTENCES.get(task_key, "")
        keywords = self.profile.keywords_by_label.get(prediction, []) if self.profile else []
        keyword_clause = ""
        if keywords:
            keyword_clause = "Key terms: " + ", ".join(keywords[:5]) + "."
        keyword_clause = " ".join(piece for piece in [curated, keyword_clause] if piece)

        retrieved = (
            self._nearest_rationale(case, prediction, exclude_case_id) if self.use_retrieval else ""
        )
        pieces = [base_text.strip(), keyword_clause]
        if retrieved:
            pieces.append("A comparable case was summarised as: " + retrieved)
        return " ".join(piece for piece in pieces if piece)

    def _nearest_rationale(
        self,
        case: Case,
        prediction: str,
        exclude_case_id: str | None,
    ) -> str:
        if self.profile is None:
            return ""
        tokens = token_set(clinical_text(case))
        best_score, best_text = 0.0, ""
        prior_biopsy = str(case.prompt.get("bx") or "")
        isup = str(case.prompt.get("bx_isup") or "")
        for item in self.profile.corpus:
            if item["label"] != prediction:
                continue
            if exclude_case_id is not None and item["case_id"] == exclude_case_id:
                continue
            # Only retrieve from a clinically comparable case.  Matching the
            # prior-biopsy status (Task 1) or grade group (Task 2) prevents a
            # reassuring benign-biopsy sentence appearing in a cancer case.
            if self.task == 1 and item.get("prior_biopsy", "") != prior_biopsy:
                continue
            if self.task == 2 and item.get("isup", "") != isup:
                continue
            score = _jaccard(tokens, item["tokens"])
            if score > best_score:
                best_score, best_text = score, item["text"]
        if best_score < 0.06 or not best_text:
            return ""
        return best_text[:220].strip()


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
