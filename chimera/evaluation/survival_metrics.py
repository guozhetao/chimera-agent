from __future__ import annotations

import numpy as np


def concordance_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's concordance index with 0.5 credit for tied risks."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    comparable = 0
    concordant = 0.0
    for i in range(len(time)):
        for j in range(i + 1, len(time)):
            a, b = None, None
            if event[i] == 1 and time[i] < time[j]:
                a, b = i, j
            elif event[j] == 1 and time[j] < time[i]:
                a, b = j, i
            if a is None:
                continue
            comparable += 1
            if risk[a] > risk[b]:
                concordant += 1.0
            elif risk[a] == risk[b]:
                concordant += 0.5
    return concordant / comparable if comparable else float("nan")
