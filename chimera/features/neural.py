from __future__ import annotations

from dataclasses import dataclass
import numpy as np


MODALITIES = {
    "mri": ("MRI image", 1024),
    "biopsy": ("Biopsy slide", 960),
    "prostatectomy": ("Prostatectomy slide", 960),
}


@dataclass
class NeuralSummary:
    mri_present: bool
    biopsy_present: bool
    prostatectomy_present: bool
    mri_count: int
    biopsy_count: int
    prostatectomy_count: int
    mri_mean_norm: float
    biopsy_mean_norm: float
    prostatectomy_mean_norm: float


def modality_arrays(neural: dict) -> dict[str, np.ndarray | None]:
    result: dict[str, np.ndarray | None] = {}
    for key, (source, _dim) in MODALITIES.items():
        items = neural.get(source, [])
        result[key] = np.asarray(items, dtype=np.float32) if items else None
    return result


def summarize_neural(neural: dict) -> NeuralSummary:
    arrays = modality_arrays(neural)
    norms = {}
    for key, arr in arrays.items():
        norms[key] = float(np.linalg.norm(arr.mean(axis=0))) if arr is not None and len(arr) else 0.0
    return NeuralSummary(
        mri_present=arrays["mri"] is not None,
        biopsy_present=arrays["biopsy"] is not None,
        prostatectomy_present=arrays["prostatectomy"] is not None,
        mri_count=0 if arrays["mri"] is None else len(arrays["mri"]),
        biopsy_count=0 if arrays["biopsy"] is None else len(arrays["biopsy"]),
        prostatectomy_count=0 if arrays["prostatectomy"] is None else len(arrays["prostatectomy"]),
        mri_mean_norm=norms["mri"],
        biopsy_mean_norm=norms["biopsy"],
        prostatectomy_mean_norm=norms["prostatectomy"],
    )


class SetStatisticsEncoder:
    """Converts variable-length modality embeddings to fixed set statistics.

    ``stats`` keeps mean/std/max/count, while ``compact`` drops the noisy max
    statistic.  Compact representations are preferable for the small cohorts in
    this release and are substantially faster to fit.
    """

    def __init__(
        self,
        modalities: tuple[str, ...] = ("mri", "biopsy", "prostatectomy"),
        mode: str = "stats",
    ):
        if mode not in {"stats", "compact"}:
            raise ValueError("mode must be 'stats' or 'compact'")
        self.modalities = modalities
        self.mode = mode

    def transform(self, neural_records: list[dict]) -> np.ndarray:
        arrays = modality_arrays_for_many(neural_records)
        blocks = []
        for modality in self.modalities:
            expected_dim = MODALITIES[modality][1]
            width = expected_dim * (3 if self.mode == "stats" else 2) + 1
            stats = []
            for arr in arrays[modality]:
                if arr is None or len(arr) == 0:
                    stats.append(np.zeros(width, dtype=np.float32))
                elif self.mode == "stats":
                    stats.append(np.r_[arr.mean(0), arr.std(0), arr.max(0), len(arr)].astype(np.float32))
                else:
                    stats.append(np.r_[arr.mean(0), arr.std(0), len(arr)].astype(np.float32))
            blocks.append(np.asarray(stats, dtype=np.float32))
        return np.concatenate(blocks, axis=1)

    def __call__(self, records: list[dict]) -> np.ndarray:
        return self.transform(records)


def modality_arrays_for_many(records: list[dict]) -> dict[str, list[np.ndarray | None]]:
    result: dict[str, list[np.ndarray | None]] = {key: [] for key in MODALITIES}
    for record in records:
        arrays = modality_arrays(record)
        for key in result:
            result[key].append(arrays[key])
    return result
