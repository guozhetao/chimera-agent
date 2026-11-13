"""Small regularised Cox proportional-hazards model.

No third-party survival package is required.  The Breslow partial likelihood is
optimised directly with an analytic gradient, which is fast enough for the
75-case release and keeps the model fully auditable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..data.dataset import Case
from .features import survival_feature_dict


def _cox_objective(beta, design, time, event, l2):
    order = np.argsort(time, kind="mergesort")
    design = design[order]
    event = event[order]
    n = len(event)

    eta = design @ beta
    eta -= eta.max()
    weights = np.exp(eta)
    risk_set = np.cumsum(weights[::-1])[::-1]
    is_event = event == 1

    log_likelihood = np.sum(eta[is_event] - np.log(risk_set[is_event] + 1e-12)) / n
    cumulative_inverse = np.cumsum(event / (risk_set + 1e-12))
    gradient = -(design[is_event].sum(axis=0)) + (design * (weights * cumulative_inverse)[:, None]).sum(axis=0)
    gradient = gradient / n + 2.0 * l2 * beta / n
    negative_log_likelihood = -log_likelihood + l2 * np.sum(beta * beta) / n
    return negative_log_likelihood, gradient


class CoxPHModel:
    """Ridge-regularised Cox model over a curated clinical/pathology feature set."""

    def __init__(self, l2: float = 50.0, random_state: int = 42):
        self.l2 = l2
        self.random_state = random_state
        self.beta_: np.ndarray | None = None
        self.columns_: list[str] = []
        self.medians_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.scales_: np.ndarray | None = None

    @staticmethod
    def _raw_frame(cases: list[Case]) -> pd.DataFrame:
        return pd.DataFrame([survival_feature_dict(case) for case in cases])

    def _prepare(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = frame.reindex(columns=self.columns_, fill_value=np.nan).to_numpy(dtype=float)
        matrix = np.where(np.isfinite(matrix), matrix, self.medians_)
        return (matrix - self.means_) / self.scales_

    def fit(self, cases: list[Case], months: list[float], events: list[int]):
        frame = self._raw_frame(cases)
        usable = [column for column in frame.columns if frame[column].notna().any()]
        frame = frame[usable]
        matrix = frame.to_numpy(dtype=float)
        medians = np.nanmedian(matrix, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        matrix = np.where(np.isfinite(matrix), matrix, medians)
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        scales = np.where(scales > 1e-8, scales, 1.0)
        standardized = (matrix - means) / scales

        event = np.asarray(events, dtype=int)
        time = np.asarray(months, dtype=float)
        result = minimize(
            _cox_objective,
            np.zeros(standardized.shape[1]),
            args=(standardized, time, event, self.l2),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 500},
        )
        self.beta_ = result.x
        self.columns_ = list(frame.columns)
        self.medians_ = medians
        self.means_ = means
        self.scales_ = scales
        return self

    def predict_risk(self, cases: list[Case]) -> np.ndarray:
        if self.beta_ is None:
            raise RuntimeError("model has not been fitted")
        return self._prepare(self._raw_frame(cases)) @ self.beta_
