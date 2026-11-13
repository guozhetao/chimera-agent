"""Publication-quality figures for the optimized CHIMERA-Agent pipeline.

The figures are built from the rich OOF evaluation JSON produced by
``scripts/run_local_eval.py``.  They emphasise uncertainty (bootstrap bands and
confidence intervals), clinically meaningful panels (decision curves,
calibration, Kaplan-Meier risk strata, time-dependent discrimination), and a
consistent colour-blind-safe visual system.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.text as mpl_text
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from scipy.stats import chi2

from chimera.data.dataset import ChimeraDataset
from chimera.evaluation.local_evaluator import (
    confidence_score,
    rationale_score,
    tool_sequence_score,
    variable_weight_score,
)
from chimera.models.guideline import task1_rule, task2_rule
from chimera.utils.jsonio import read_json


# Okabe-Ito colour-blind-safe palette.
C = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "grey": "#7F7F7F",
    "light_grey": "#D9D9D9",
    "dark": "#1A1A1A",
}

TASK2_ORDER = [
    "active_surveillance",
    "active_treatment",
    "continued_surveillance",
    "watchful_waiting",
]
TASK2_SHORT = ["AS", "AT", "CS", "WW"]
TASK2_COLORS = [C["blue"], C["vermillion"], C["green"], C["orange"]]

LAYOUT_QC: dict[str, dict] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--eval", required=True, help="Optimized OOF evaluation JSON.")
    parser.add_argument("--output-dir", default=ROOT / "results" / "report_figures")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 8.8,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.45,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def panel_label(ax, text: str, x: float = -0.17, y: float = 1.10) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=C["dark"],
    )


def save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    items = []
    for artist in fig.findobj(mpl_text.Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        try:
            bbox = artist.get_window_extent(renderer=renderer)
        except Exception:
            continue
        items.append((artist.get_text(), bbox))
    collisions = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if len(items[i][0].strip()) <= 1 or len(items[j][0].strip()) <= 1:
                # Single-character panel labels often touch adjacent tick
                # labels in the renderer even when they are visually separate.
                continue
            a, b = items[i][1], items[j][1]
            overlap_x = min(a.x1, b.x1) - max(a.x0, b.x0)
            overlap_y = min(a.y1, b.y1) - max(a.y0, b.y0)
            if overlap_x <= 0 or overlap_y <= 0:
                continue
            score = overlap_x * overlap_y / max(min(a.width * a.height, b.width * b.height), 1e-9)
            if score > 0.25:
                collisions.append((items[i][0], items[j][0]))
    LAYOUT_QC[name] = {"text_overlap_count": len(collisions), "overlaps": collisions}
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"{name}.{suffix}")
    plt.close(fig)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _bootstrap_auc(y: np.ndarray, score: np.ndarray, n_boot: int = 500, seed: int = 42) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) < 2:
            continue
        values.append(roc_auc_score(y[index], score[index]))
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _roc_band(y: np.ndarray, score: np.ndarray, n_boot: int = 400, seed: int = 42):
    from sklearn.metrics import roc_curve

    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 101)
    curves = []
    for _ in range(n_boot):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[index], score[index])
        curves.append(np.interp(grid, fpr, tpr))
    if not curves:
        return grid, np.zeros_like(grid), np.zeros_like(grid)
    curves = np.asarray(curves)
    return grid, np.percentile(curves, 2.5, axis=0), np.percentile(curves, 97.5, axis=0)


def _km(times: np.ndarray, events: np.ndarray):
    order = np.argsort(times)
    times = times[order]
    events = events[order]
    unique = np.unique(times)
    x, y = [0.0], [1.0]
    survival = 1.0
    for time in unique:
        at_risk = np.sum(times >= time)
        deaths = np.sum((times == time) & (events == 1))
        if deaths:
            survival *= 1.0 - deaths / at_risk
        x.append(float(time))
        y.append(float(survival))
    return np.asarray(x), np.asarray(y)


def _logrank_p(t1, e1, t2, e2) -> float:
    times = np.unique(np.r_[t1, t2])
    observed = expected = variance = 0.0
    for time in times:
        r1 = np.sum(t1 >= time)
        r2 = np.sum(t2 >= time)
        n = r1 + r2
        if n <= 1:
            continue
        d1 = np.sum((t1 == time) & (e1 == 1))
        d2 = np.sum((t2 == time) & (e2 == 1))
        d = d1 + d2
        observed += d1
        expected += d * r1 / n
        variance += d * r1 * r2 * (n - d) / (n * n * (n - 1))
    if variance <= 0:
        return float("nan")
    z = (observed - expected) ** 2 / variance
    return float(chi2.sf(z, 1))


def _per_class_f1_ci(y_true, y_pred, labels, n_boot: int = 500, seed: int = 42):
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    values, lows, highs = [], [], []
    for label in labels:
        point = f1_score(y_true, y_pred, labels=[label], average="macro", zero_division=0)
        samples = []
        for _ in range(n_boot):
            index = rng.integers(0, len(y_true), len(y_true))
            if label not in y_true[index]:
                continue
            samples.append(f1_score(y_true[index], y_pred[index], labels=[label], average="macro", zero_division=0))
        values.append(float(point))
        lows.append(float(np.percentile(samples, 2.5)) if samples else 0.0)
        highs.append(float(np.percentile(samples, 97.5)) if samples else 1.0)
    return values, lows, highs


def _cumulative_dynamic_auc(times: np.ndarray, events: np.ndarray, risk: np.ndarray, horizon: float) -> float:
    cases = (times <= horizon) & (events == 1)
    controls = times > horizon
    if cases.sum() < 3 or controls.sum() < 3:
        return float("nan")
    case_scores = risk[cases][:, None]
    control_scores = risk[controls][None, :]
    wins = np.sum(case_scores > control_scores) + 0.5 * np.sum(case_scores == control_scores)
    return float(wins / (cases.sum() * controls.sum()))


def _cd_auc_ci(times, events, risk, horizon, n_boot=400, seed=42):
    rng = np.random.default_rng(seed)
    values = []
    n = len(times)
    for _ in range(n_boot):
        index = rng.integers(0, n, n)
        value = _cumulative_dynamic_auc(times[index], events[index], risk[index], horizon)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _load(data_root: str, eval_path: str):
    dataset = ChimeraDataset(data_root)
    evaluation = read_json(eval_path)
    return dataset, evaluation


def _prediction_arrays(task_record: dict, task: int):
    records = sorted(task_record["predictions"], key=lambda row: row["case_id"])
    y_true = np.asarray([row["true"] for row in records])
    y_pred = np.asarray([row["predicted"] for row in records])
    probabilities = np.asarray([row["probability"] for row in records], dtype=float)
    return records, y_true, y_pred, probabilities


def _reasoning_components(task_record: dict, dataset: ChimeraDataset, task: int):
    cases = {case.case_id: case for case in dataset.cases(task, labeled_only=True)}
    output = {"baseline": [], "optimized": []}
    for record in task_record["predictions"]:
        case = cases[record["case_id"]]
        truth = case.reasoning or {}
        for key, field in (("optimized", "reasoning"), ("baseline", "reasoning_baseline")):
            predicted = record.get(field) or {}
            confidence = confidence_score([truth.get("confidence")], [predicted.get("confidence")])
            weights = variable_weight_score(truth.get("variable_weights"), predicted.get("variable_weights"))
            tools = tool_sequence_score(truth.get("reveal_sequence"), predicted.get("reveal_sequence"))["composite"]
            rationale = rationale_score(truth.get("free_text"), predicted.get("free_text"))
            output[key].append(
                {
                    "confidence": confidence,
                    "variable_weights": weights,
                    "tool_sequence": tools,
                    "rationale": rationale,
                    "composite": 0.15 * confidence + 0.25 * weights + 0.25 * tools + 0.35 * rationale,
                }
            )
    return output


# --------------------------------------------------------------------------- fig 1
def figure_overview(dataset: ChimeraDataset, evaluation: dict, outdir: Path) -> None:
    fig = plt.figure(figsize=(7.4, 5.2))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.05, 1.0], hspace=0.55, wspace=0.42)

    # a: clinical trajectory
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.set_title("Clinical trajectory", loc="left", pad=12)
    stages = [
        ("Task 1", "Pre-biopsy\nMRI + PSA", C["blue"]),
        ("Task 2", "Post-biopsy\nMRI + pathology", C["green"]),
        ("Task 3", "Post-surgery\nWSI + follow-up", C["orange"]),
    ]
    for index, (title, text, colour) in enumerate(stages):
        y = 0.70 - index * 0.31
        ax.add_patch(
            FancyBboxPatch(
                (0.03, y),
                0.94,
                0.24,
                boxstyle="round,pad=0.018",
                facecolor="white",
                edgecolor=colour,
                linewidth=1.0,
            )
        )
        ax.text(0.5, y + 0.155, title, ha="center", va="center", fontsize=7.4, fontweight="bold", color=colour)
        ax.text(0.5, y + 0.065, text, ha="center", va="center", fontsize=6.8)
        if index < 2:
            ax.add_patch(
                FancyArrowPatch(
                    (0.5, y - 0.012),
                    (0.5, y - 0.065),
                    arrowstyle="-|>",
                    mutation_scale=7,
                    color=C["grey"],
                    linewidth=0.7,
                )
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.98)
    panel_label(ax, "a", x=-0.10, y=1.10)

    # b: cohort and labels
    ax = fig.add_subplot(gs[0, 1])
    counts = dataset.counts()
    tasks = [1, 2, 3]
    labeled = [counts[task]["labeled"] for task in tasks]
    unlabeled = [counts[task]["unlabeled"] for task in tasks]
    x = np.arange(3)
    ax.bar(x, labeled, color=C["blue"], width=0.58, label="Labeled")
    ax.bar(x, unlabeled, bottom=labeled, color=C["light_grey"], width=0.58, label="Unlabeled")
    for index in range(3):
        total = labeled[index] + unlabeled[index]
        ax.text(index, total + 4, str(total), ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(x, ["Task 1", "Task 2", "Task 3"])
    ax.set_ylabel("Cases")
    ax.set_ylim(0, max(labeled) + max(unlabeled) + 32)
    ax.set_title("Cohort composition", loc="left")
    ax.legend(frameon=False, loc="upper right", ncols=1)
    panel_label(ax, "b", x=-0.20, y=1.08)

    # c: label distribution
    ax = fig.add_subplot(gs[0, 2])
    task1_yes = sum(1 for case in dataset.cases(1, labeled_only=True) if str(case.label) == "yes")
    task1_total = counts[1]["labeled"]
    task2_total = counts[2]["labeled"]
    task2_counts = [
        sum(1 for case in dataset.cases(2, labeled_only=True) if str(case.label) == label)
        for label in TASK2_ORDER
    ]
    task3_events = sum(1 for case in dataset.cases(3, labeled_only=True) if int(case.label["event"]) == 1)
    task3_total = counts[3]["labeled"]
    ax.barh([0], [task1_total - task1_yes], color=C["light_grey"], height=0.55)
    ax.barh([0], [task1_yes], left=[task1_total - task1_yes], color=C["blue"], height=0.55)
    left = 0
    task2_centres = []
    for value, colour in zip(task2_counts, TASK2_COLORS):
        ax.barh([1], [value], left=[left], color=colour, height=0.55)
        task2_centres.append(left + value / 2)
        left += value
    ax.barh([2], [task3_total - task3_events], color=C["light_grey"], height=0.55)
    ax.barh([2], [task3_events], left=[task3_total - task3_events], color=C["vermillion"], height=0.55)
    ax.set_yticks([0, 1, 2], ["Task 1", "Task 2", "Task 3"])
    ax.invert_yaxis()
    ax.set_xlabel("Labeled cases")
    ax.set_title("Endpoint distribution", loc="left")
    ax.text(task1_total - task1_yes + 2, 0, f"Yes {task1_yes}", va="center", fontsize=6.0)
    for centre, value in zip(task2_centres[:3], task2_counts[:3]):
        ax.text(centre, 1, str(value), ha="center", va="center", fontsize=6.4, color="white", fontweight="bold")
    ax.annotate(
        f"WW {task2_counts[3]}",
        xy=(task2_centres[3], 1),
        xytext=(task2_centres[3] + 8, 0.62),
        fontsize=5.8,
        arrowprops={"arrowstyle": "-", "color": C["grey"], "linewidth": 0.5},
    )
    ax.text(task3_total - task3_events + 2, 2, f"Event {task3_events}", va="center", fontsize=6.0)
    ax.set_xlim(0, max(task1_total, task2_total, task3_total) + 20)
    ax.text(0.0, -0.28, "Task 2: AS / AT / CS / WW", transform=ax.transAxes, fontsize=5.8, color=C["grey"])
    panel_label(ax, "c", x=-0.20, y=1.10)

    # d: modality availability
    ax = fig.add_subplot(gs[1, 0])
    modalities = [("MRI image", "MRI"), ("Biopsy slide", "Biopsy WSI"), ("Prostatectomy slide", "Surgery WSI")]
    matrix = []
    for task in tasks:
        cases = dataset.cases(task, labeled_only=True)
        matrix.append([sum(bool(case.neural.get(key)) for case in cases) for key, _ in modalities])
    matrix = np.asarray(matrix)
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(labeled))
    for row in range(3):
        for column in range(3):
            ax.text(column, row, f"{matrix[row, column]}", ha="center", va="center", fontsize=6.6,
                    color="white" if matrix[row, column] > max(labeled) * 0.55 else C["dark"])
    ax.set_xticks(range(3), ["MRI", "Biopsy\nWSI", "Surgery\nWSI"])
    ax.set_yticks(range(3), ["Task 1", "Task 2", "Task 3"])
    ax.set_title("Available neural modalities", loc="left", pad=8)
    ax.grid(False)
    panel_label(ax, "d", x=-0.12, y=1.18)

    # e: architecture
    ax = fig.add_subplot(gs[1, 1:])
    ax.axis("off")
    ax.set_title("Rule-anchored multimodal architecture (GARF-H)", loc="left", pad=8)
    boxes = [
        (0.02, "Inputs", "structured prompt\nclinical notes\nMRI / WSI sets", C["sky"]),
        (0.27, "Feature layer", "PSA kinetics\npathology concepts\ntext signals\nneural set stats", C["blue"]),
        (0.52, "Rule anchor", "PI-RADS / ISUP\n+ exception model\nP(exception)", C["green"]),
        (0.77, "Outputs", "decision\nsurvival risk\nreasoning", C["vermillion"]),
    ]
    for x, title, text, colour in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.50),
                0.19,
                0.34,
                boxstyle="round,pad=0.014",
                facecolor="white",
                edgecolor=colour,
                linewidth=1.0,
            )
        )
        ax.text(x + 0.095, 0.78, title, ha="center", va="center", fontsize=7.0, fontweight="bold", color=colour)
        ax.text(x + 0.095, 0.62, text, ha="center", va="center", fontsize=6.1)
    for start, end in [((0.21, 0.67), (0.27, 0.67)), ((0.46, 0.67), (0.52, 0.67)), ((0.71, 0.67), (0.77, 0.67))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7, color=C["grey"], linewidth=0.8))
    ax.text(
        0.50,
        0.34,
        "Task 3: rank ensemble of pathology anchor, neural event model, and hazard trajectory",
        ha="center",
        fontsize=6.3,
        color=C["purple"],
    )
    ax.text(
        0.50,
        0.23,
        "Every output includes a calibrated probability, structured weights, and a clinical rationale",
        ha="center",
        fontsize=6.3,
        color=C["grey"],
    )
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.15, 0.92)
    panel_label(ax, "e", x=-0.06, y=1.18)

    save(fig, outdir, "fig1_cohort_and_architecture")


# --------------------------------------------------------------------------- fig 2
def figure_task1(dataset: ChimeraDataset, evaluation: dict, outdir: Path) -> None:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    record = evaluation["task1"]
    records, y_true, y_pred, probabilities = _prediction_arrays(record, 1)
    y = (y_true == "yes").astype(int)
    score = probabilities[:, 1]
    cases = dataset.cases(1, labeled_only=True)
    rule = np.asarray([task1_rule(case) for case in cases])
    rule_y = (rule == "yes").astype(int)

    fig = plt.figure(figsize=(7.4, 4.9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.38)

    # a: ROC
    ax = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(y, score)
    grid, low, high = _roc_band(y, score)
    auc_value = roc_auc_score(y, score)
    auc_low, auc_high = _bootstrap_auc(y, score)
    ax.fill_between(grid, low, high, color=C["blue"], alpha=0.16, linewidth=0)
    ax.plot(fpr, tpr, color=C["blue"], linewidth=1.5)
    ax.plot([0, 1], [0, 1], linestyle="--", color=C["grey"], linewidth=0.7)
    tn, fp, fn, tp = confusion_matrix(y, rule_y).ravel()
    ax.scatter(fp / (fp + tn), tp / (tp + fn), s=26, marker="D", color=C["orange"], zorder=5, label="Guideline rule")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("1 − specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Discrimination", loc="left")
    ax.text(0.36, 0.10, f"AUC {auc_value:.3f}\n95% CI {auc_low:.3f}–{auc_high:.3f}", fontsize=6.6)
    ax.legend(frameon=False, loc="center right")
    panel_label(ax, "a")

    # b: precision-recall
    ax = fig.add_subplot(gs[0, 1])
    precision, recall, _ = precision_recall_curve(y, score)
    ap_value = average_precision_score(y, score)
    ax.plot(recall, precision, color=C["green"], linewidth=1.5)
    ax.axhline(y.mean(), linestyle="--", color=C["grey"], linewidth=0.7, label=f"Prevalence {y.mean():.2f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall", loc="left")
    ax.text(0.04, 0.06, f"AP {ap_value:.3f}", fontsize=6.8)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.10))
    panel_label(ax, "b")

    # c: calibration
    ax = fig.add_subplot(gs[0, 2])
    fraction, mean_predicted = calibration_curve(y, score, n_bins=5, strategy="quantile")
    brier = brier_score_loss(y, score)
    ax.plot([0, 1], [0, 1], linestyle="--", color=C["grey"], linewidth=0.7)
    ax.plot(mean_predicted, fraction, marker="o", markersize=4, color=C["orange"], linewidth=1.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(biopsy)")
    ax.set_ylabel("Observed proportion")
    ax.set_title("Calibration", loc="left")
    ax.text(0.04, 0.88, f"Brier {brier:.3f}", fontsize=6.8)
    panel_label(ax, "c")

    # d: confusion matrix
    ax = fig.add_subplot(gs[1, 0])
    matrix = confusion_matrix(y_true, y_pred, labels=["no", "yes"])
    percentage = matrix / matrix.sum()
    ax.imshow(percentage, cmap="Blues", vmin=0, vmax=0.60)
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                f"{matrix[row, column]}\n({percentage[row, column] * 100:.1f}%)",
                ha="center",
                va="center",
                fontsize=8.2,
                color="white" if percentage[row, column] > 0.50 else C["dark"],
            )
    ax.set_xticks([0, 1], ["No biopsy", "Biopsy"])
    ax.set_yticks([0, 1], ["No biopsy", "Biopsy"])
    ax.set_ylabel("Observed")
    ax.set_title("Confusion matrix", loc="left")
    ax.set_xlabel(f"Predicted (overall %, n={len(y_true)})")
    ax.grid(False)
    panel_label(ax, "d")

    # e: threshold curves
    ax = fig.add_subplot(gs[1, 1])
    thresholds = np.linspace(0.05, 0.95, 91)
    f1_values, accuracy_values, precision_values, recall_values = [], [], [], []
    for threshold in thresholds:
        predicted = (score >= threshold).astype(int)
        f1_values.append(f1_score(y, predicted, zero_division=0))
        accuracy_values.append(float(np.mean(predicted == y)))
        tp_t = np.sum((predicted == 1) & (y == 1))
        fp_t = np.sum((predicted == 1) & (y == 0))
        fn_t = np.sum((predicted == 0) & (y == 1))
        precision_values.append(tp_t / (tp_t + fp_t) if tp_t + fp_t else 1.0)
        recall_values.append(tp_t / (tp_t + fn_t) if tp_t + fn_t else 0.0)
    best_index = int(np.argmax(f1_values))
    ax.plot(thresholds, f1_values, color=C["blue"], linewidth=1.4, label="F1 positive")
    ax.plot(thresholds, accuracy_values, color=C["grey"], linewidth=1.1, label="Accuracy")
    ax.plot(thresholds, precision_values, color=C["orange"], linewidth=1.0, linestyle="--", label="Precision")
    ax.plot(thresholds, recall_values, color=C["green"], linewidth=1.0, linestyle=":", label="Recall")
    ax.scatter([thresholds[best_index]], [f1_values[best_index]], s=26, color=C["blue"], zorder=4)
    ax.annotate(
        f"max F1 {f1_values[best_index]:.2f}",
        (thresholds[best_index], f1_values[best_index]),
        xytext=(16, -20),
        textcoords="offset points",
        fontsize=6.2,
        arrowprops={"arrowstyle": "-", "color": C["dark"], "linewidth": 0.5},
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Operating characteristics", loc="left")
    ax.legend(frameon=False, loc="lower left", ncols=2)
    panel_label(ax, "e")

    # f: case landscape
    ax = fig.add_subplot(gs[1, 2])
    order = np.argsort(score)
    colours = np.where(y[order] == 1, C["blue"], C["grey"])
    errors = y_pred[order] != y_true[order]
    ax.scatter(np.arange(len(score)), score[order], c=colours, s=16, linewidth=0)
    ax.scatter(
        np.arange(len(score))[errors],
        score[order][errors],
        facecolors="none",
        edgecolors=C["vermillion"],
        s=42,
        linewidth=1.1,
        label="Error",
    )
    ax.axhline(thresholds[best_index], linestyle="--", color=C["dark"], linewidth=0.7)
    ax.set_xlabel("Cases ranked by P(biopsy)")
    ax.set_ylabel("P(biopsy)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Case-level separation", loc="left")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=C["blue"], label="Observed biopsy"),
        plt.Line2D([], [], marker="o", linestyle="", color=C["grey"], label="Observed no biopsy"),
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=C["vermillion"], label="Error"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=6.0)
    panel_label(ax, "f")

    save(fig, outdir, "fig2_task1_biopsy")


# --------------------------------------------------------------------------- fig 3
def figure_task2(dataset: ChimeraDataset, evaluation: dict, outdir: Path) -> None:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    record = evaluation["task2"]
    records, y_true, y_pred, probabilities = _prediction_arrays(record, 2)

    fig = plt.figure(figsize=(7.4, 4.9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.38)

    # a: class distribution
    ax = fig.add_subplot(gs[0, 0])
    counts = [int(np.sum(y_true == label)) for label in TASK2_ORDER]
    bars = ax.bar(np.arange(4), counts, color=TASK2_COLORS, width=0.62)
    for index, value in enumerate(counts):
        ax.text(index, value + 0.8, str(value), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(4), TASK2_SHORT)
    ax.set_ylabel("Cases")
    ax.set_ylim(0, max(counts) + 6)
    ax.set_title("Management classes", loc="left")
    ax.text(2.0, max(counts) + 4.0, "AS / AT / CS / WW", ha="center", fontsize=6.0, color=C["grey"])
    panel_label(ax, "a")

    # b: confusion matrix
    ax = fig.add_subplot(gs[0, 1])
    matrix = confusion_matrix(y_true, y_pred, labels=TASK2_ORDER)
    row_sum = matrix.sum(axis=1, keepdims=True)
    percentage = np.divide(matrix, row_sum, out=np.zeros_like(matrix, dtype=float), where=row_sum > 0)
    ax.imshow(percentage, cmap="Blues", vmin=0, vmax=1)
    for row in range(4):
        for column in range(4):
            ax.text(
                column,
                row,
                f"{matrix[row, column]}",
                ha="center",
                va="center",
                fontsize=7.4,
                color="white" if percentage[row, column] > 0.55 else C["dark"],
            )
    ax.set_xticks(range(4), TASK2_SHORT)
    ax.set_yticks(range(4), TASK2_SHORT)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title("Confusion matrix", loc="left")
    ax.grid(False)
    panel_label(ax, "b")

    # c: per-class metrics
    ax = fig.add_subplot(gs[0, 2])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=TASK2_ORDER, zero_division=0
    )
    x = np.arange(4)
    width = 0.26
    ax.bar(x - width, precision, width=width, color=C["blue"], label="Precision")
    ax.bar(x, recall, width=width, color=C["green"], label="Recall")
    ax.bar(x + width, f1, width=width, color=C["orange"], label="F1")
    ax.set_xticks(x, TASK2_SHORT)
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("Score")
    ax.set_title("Per-class performance", loc="left")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=6.0)
    panel_label(ax, "c")

    # d: row-normalised error composition
    ax = fig.add_subplot(gs[1, 0])
    row_support = matrix.sum(axis=1, keepdims=True)
    row_fraction = np.divide(matrix, row_support, out=np.zeros_like(matrix, dtype=float), where=row_support > 0)
    left = np.zeros(4)
    for predicted_index, (label, colour) in enumerate(zip(TASK2_SHORT, TASK2_COLORS)):
        values = row_fraction[:, predicted_index]
        ax.barh(np.arange(4), values, left=left, color=colour, height=0.55, label=label)
        left += values
    ax.set_yticks(np.arange(4), TASK2_SHORT)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion of true class")
    ax.set_title("Predicted label by true class", loc="left")
    panel_label(ax, "d")

    # e: per-class F1 with bootstrap confidence intervals
    ax = fig.add_subplot(gs[1, 1])
    f1_values, f1_lows, f1_highs = _per_class_f1_ci(y_true, y_pred, TASK2_ORDER)
    x = np.arange(4)
    ax.errorbar(
        x,
        f1_values,
        yerr=[np.asarray(f1_values) - np.asarray(f1_lows), np.asarray(f1_highs) - np.asarray(f1_values)],
        fmt="o",
        color=C["dark"],
        ecolor=C["grey"],
        elinewidth=0.9,
        capsize=2,
        markersize=5,
    )
    for index, value in enumerate(f1_values):
        ax.text(index + 0.10, value, f"{value:.2f}", va="center", fontsize=6.2)
    ax.set_xticks(x, TASK2_SHORT)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score (95% CI)")
    ax.set_title("Per-class uncertainty", loc="left")
    panel_label(ax, "e")

    # f: error table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.set_title("Misclassified cases", loc="left")
    errors = [
        (row["case_id"], row["true"], row["predicted"], max(row["probability"]))
        for row in records
        if row["true"] != row["predicted"]
    ]
    if errors:
        lines = [
            f"{case_id:<10} {TASK2_SHORT[TASK2_ORDER.index(true)] + ' → ' + TASK2_SHORT[TASK2_ORDER.index(predicted)]:<12} "
            f"{confidence_value:>5.2f}"
            for case_id, true, predicted, confidence_value in errors[:10]
        ]
        ax.text(
            0.0,
            0.98,
            f"{'Case':<10} {'True → Pred':<12} {'Conf':>5}",
            family="monospace",
            fontsize=5.9,
            fontweight="bold",
            va="top",
        )
        ax.text(
            0.0,
            0.90,
            "\n".join(lines),
            family="monospace",
            fontsize=5.9,
            va="top",
            linespacing=1.45,
        )
        ax.text(0.0, 0.02, f"{len(errors)} / {len(records)} cases", fontsize=6.2, color=C["grey"])
    else:
        ax.text(0.0, 0.5, "No misclassifications", fontsize=7)
    panel_label(ax, "f", x=-0.10)

    save(fig, outdir, "fig3_task2_management")


# --------------------------------------------------------------------------- fig 4
def figure_task3(dataset: ChimeraDataset, evaluation: dict, outdir: Path) -> None:
    record = evaluation["task3"]
    predictions = sorted(record["predictions"], key=lambda row: row["case_id"])
    months = np.asarray([row["true_months"] for row in predictions], dtype=float)
    events = np.asarray([row["true_event"] for row in predictions], dtype=int)
    risk = np.asarray([row["risk"] for row in predictions], dtype=float)
    event_probability = np.asarray([row["event_probability"] for row in predictions], dtype=float)
    components = record.get("components", {})
    hazard = np.asarray(components.get("hazard_by_repeat", []), dtype=float)
    time_bins = np.asarray([3, 6, 12, 24, 36, 60, 120], dtype=float)

    fig = plt.figure(figsize=(7.4, 5.4))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.12, 0.88], hspace=0.62, wspace=0.42)

    # a: KM by risk tertile
    ax = fig.add_subplot(gs[0, 0])
    tertiles = np.quantile(risk, [1 / 3, 2 / 3])
    groups = [risk <= tertiles[0], (risk > tertiles[0]) & (risk <= tertiles[1]), risk > tertiles[1]]
    labels = ["Low risk", "Intermediate", "High risk"]
    colours = [C["blue"], C["orange"], C["vermillion"]]
    for mask, label, colour in zip(groups, labels, colours):
        curve_x, curve_y = _km(months[mask], events[mask])
        ax.step(curve_x, curve_y * 100, where="post", color=colour, linewidth=1.3, label=label)
        censored = (events[mask] == 0)
        if censored.any():
            ax.scatter(
                months[mask][censored],
                np.interp(months[mask][censored], curve_x, curve_y) * 100,
                marker="|",
                s=14,
                color=colour,
                linewidth=0.8,
            )
    p_value = _logrank_p(months[groups[0]], events[groups[0]], months[groups[2]], events[groups[2]])
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Months after surgery")
    ax.set_ylabel("Recurrence-free survival (%)")
    ax.set_title("Risk-stratified survival", loc="left")
    ax.text(0.98, 0.04, f"Low vs high log-rank P={p_value:.1e}", transform=ax.transAxes, fontsize=6.3, ha="right")
    ax.legend(frameon=False, loc="center right")
    panel_label(ax, "a")

    # b: ablation C-index
    ax = fig.add_subplot(gs[0, 1])
    ablation = dict(record["summary"]["ablation"])
    ablation["final_hybrid"] = record["summary"]["pooled"]["harrell_c_index"]
    pretty = {
        "final_hybrid": "Hybrid (final)",
        "pathology_only": "Pathology",
        "event_only": "Neural event",
        "hazard_only": "Hazard",
        "pathology_event": "Pathology + event",
        "pathology_event_hazard": "Full hybrid",
        "cox_ph": "Cox PH",
        "clinical_hybrid": "Clinical hybrid",
    }
    names = list(pretty)
    values = [ablation[name]["value"] for name in names]
    lows = [ablation[name]["low"] for name in names]
    highs = [ablation[name]["high"] for name in names]
    order = np.argsort(values)
    names = [names[index] for index in order]
    values = [values[index] for index in order]
    lows = [lows[index] for index in order]
    highs = [highs[index] for index in order]
    y = np.arange(len(names))
    for index, (value, low, high) in enumerate(zip(values, lows, highs)):
        colour = C["vermillion"] if names[index] == "final_hybrid" else C["grey"]
        ax.plot([low, high], [index, index], color=colour, linewidth=1.1, alpha=0.7)
        ax.scatter([value], [index], s=26, color=colour, zorder=4)
        ax.text(high + 0.005, index, f"{value:.3f}", va="center", fontsize=6.2)
    ax.set_yticks(y, [pretty[name] for name in names])
    ax.set_xlim(0.65, 0.97)
    ax.set_xticks([0.7, 0.8, 0.9])
    ax.set_xlabel("Harrell C-index (95% CI)")
    ax.set_title("Component ablation", loc="left")
    ax.grid(axis="y", visible=False)
    panel_label(ax, "b")

    # c: risk distribution
    ax = fig.add_subplot(gs[0, 2])
    rng = np.random.default_rng(7)
    for index, (value, colour, label) in enumerate(
        ((0, C["grey"], "No recurrence"), (1, C["vermillion"], "Recurrence"))
    ):
        mask = events == value
        jitter = rng.normal(0, 0.045, mask.sum())
        ax.scatter(index + jitter, risk[mask], s=16, color=colour, alpha=0.8, linewidth=0, label=label)
        ax.boxplot(
            [risk[mask]],
            positions=[index],
            widths=0.30,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "none", "edgecolor": colour, "linewidth": 0.8},
            medianprops={"color": colour, "linewidth": 1.0},
            whiskerprops={"color": colour, "linewidth": 0.7},
            capprops={"color": colour, "linewidth": 0.7},
        )
    ax.set_xticks([0, 1], ["No recurrence", "Recurrence"])
    ax.set_ylabel("Predicted risk percentile")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Risk separation", loc="left")
    ax.legend(frameon=False, loc="lower right")
    panel_label(ax, "c")

    # d: event probability calibration
    ax = fig.add_subplot(gs[1, 0])
    quartiles = np.quantile(event_probability, [0.25, 0.5, 0.75])
    bins = [-0.01, quartiles[0], quartiles[1], quartiles[2], 1.01]
    centers, observed, lows, highs = [], [], [], []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (event_probability > left) & (event_probability <= right)
        if mask.sum() == 0:
            continue
        k = int(events[mask].sum())
        n = int(mask.sum())
        centers.append(float(event_probability[mask].mean()))
        observed.append(k / n)
        low, high = _wilson(k, n)
        lows.append(low)
        highs.append(high)
    ax.plot([0, 1], [0, 1], linestyle="--", color=C["grey"], linewidth=0.7)
    ax.errorbar(
        centers,
        observed,
        yerr=[np.asarray(observed) - np.asarray(lows), np.asarray(highs) - np.asarray(observed)],
        fmt="o-",
        color=C["purple"],
        markersize=4,
        linewidth=1.2,
        capsize=2,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted recurrence probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Event calibration", loc="left")
    panel_label(ax, "d")

    # e: cumulative/dynamic AUC
    ax = fig.add_subplot(gs[1, 1])
    horizons = [12, 24, 36, 48, 60]
    auc_values, auc_lows, auc_highs = [], [], []
    for horizon in horizons:
        value = _cumulative_dynamic_auc(months, events, risk, horizon)
        low, high = _cd_auc_ci(months, events, risk, horizon)
        auc_values.append(value)
        auc_lows.append(low)
        auc_highs.append(high)
    ax.plot(horizons, auc_values, marker="o", color=C["blue"], linewidth=1.3, markersize=4)
    ax.fill_between(horizons, auc_lows, auc_highs, color=C["blue"], alpha=0.15, linewidth=0)
    ax.axhline(0.5, linestyle="--", color=C["grey"], linewidth=0.7)
    ax.set_ylim(0.5, 1.02)
    ax.set_xlabel("Time horizon (months)")
    ax.set_ylabel("Cumulative/dynamic AUC")
    ax.set_title("Time-dependent discrimination", loc="left")
    panel_label(ax, "e")

    # f: model-implied cumulative incidence by risk group
    ax = fig.add_subplot(gs[1, 2])
    if hazard.size:
        mean_hazard = hazard.mean(axis=0)
        for mask, label, colour in zip(groups, labels, colours):
            curve = mean_hazard[mask].mean(axis=0) * 100
            ax.step(np.r_[0, time_bins], np.r_[0, curve], where="post", color=colour, linewidth=1.2, label=label)
        ax.set_xlim(0, 125)
        ax.set_ylim(0, 105)
        ax.set_xticks([0, 25, 50, 75, 100, 120])
        ax.set_xlabel("Months after surgery")
        ax.set_ylabel("Cumulative incidence (%)")
        ax.set_title("Model-implied incidence", loc="left")
        ax.legend(frameon=False, loc="upper left")
    else:
        ax.axis("off")
    panel_label(ax, "f")

    save(fig, outdir, "fig4_task3_recurrence")


# --------------------------------------------------------------------------- fig 5
def figure_summary(dataset: ChimeraDataset, evaluation: dict, outdir: Path) -> None:
    task1 = evaluation["task1"]["summary"]
    task2 = evaluation["task2"]["summary"]
    task3 = evaluation["task3"]["summary"]
    task1_reasoning = _reasoning_components(evaluation["task1"], dataset, 1)
    task2_reasoning = _reasoning_components(evaluation["task2"], dataset, 2)

    fig = plt.figure(figsize=(7.4, 5.2))
    gs = GridSpec(2, 3, figure=fig, hspace=0.62, wspace=0.44, width_ratios=[1.0, 1.0, 1.22])

    # a: GARF-H vs clinical anchor
    ax = fig.add_subplot(gs[0, 0])
    t1 = task1["pooled"]["bootstrap"]["f1_positive"]
    t2 = task2["pooled"]["bootstrap"]["f1_weighted"]
    t3 = task3["pooled"]["harrell_c_index"]
    anchor = [
        task1["baseline_rule"]["case_score"],
        task2["baseline_rule"]["case_score"],
        task3["ablation"]["pathology_only"]["value"],
    ]
    garf = [t1["value"], t2["value"], t3["value"]]
    garf_low = [t1["low"], t2["low"], t3["low"]]
    garf_high = [t1["high"], t2["high"], t3["high"]]
    colours = [C["blue"], C["green"], C["orange"]]
    y = np.array([2.0, 1.0, 0.0])
    for index, (base, value, low, high, colour) in enumerate(
        zip(anchor, garf, garf_low, garf_high, colours)
    ):
        ax.plot([base], [y[index] + 0.15], marker="o", markersize=5, color=C["grey"], linestyle="")
        ax.plot([low, high], [y[index] - 0.15, y[index] - 0.15], color=colour, linewidth=1.2)
        ax.plot([value], [y[index] - 0.15], marker="o", markersize=6, color=colour, linestyle="")
        ax.text(base - 0.012, y[index] + 0.15, f"{base:.3f}", ha="right", va="center",
                fontsize=6.2, color=C["grey"])
        ax.text(high + 0.008, y[index] - 0.15, f"{value:.3f}", ha="left", va="center",
                fontsize=6.2, color=colour)
    ax.set_yticks(y, ["Task 1\nF1 positive", "Task 2\nWeighted F1", "Task 3\nC-index"])
    ax.set_xlim(0.60, 1.04)
    ax.set_ylim(-0.55, 2.92)
    ax.set_xlabel("Primary metric")
    ax.set_title("GARF-H vs clinical anchor", loc="left")
    ax.plot([0.64], [2.72], marker="o", markersize=5, color=C["grey"], linestyle="")
    ax.text(0.66, 2.72, "Anchor", va="center", fontsize=6.2, color=C["grey"])
    ax.plot([0.80], [2.72], marker="o", markersize=6, color=C["dark"], linestyle="")
    ax.text(0.82, 2.72, "GARF-H", va="center", fontsize=6.2, color=C["dark"])
    panel_label(ax, "a")

    # b/c: current GARF-H reasoning components
    for column, (components, summary, title) in enumerate(
        (
            (task1_reasoning, task1, "Task 1 reasoning"),
            (task2_reasoning, task2, "Task 2 reasoning"),
        )
    ):
        ax = fig.add_subplot(gs[0, 1 + column])
        keys = ["confidence", "variable_weights", "tool_sequence", "rationale", "composite"]
        pretty = ["Conf.", "Weights", "Tools", "Rationale", "Composite"]
        values = [summary["reasoning"][key]["mean"] for key in keys]
        y = np.arange(len(keys))[::-1]
        ax.barh(y, values, height=0.45, color=C["blue"] if column == 0 else C["green"])
        for position, value in zip(y, values):
            ax.text(value + 0.015, position, f"{value:.2f}", va="center", fontsize=5.9)
        ax.set_yticks(y, pretty)
        ax.set_xlim(0, 1.10)
        ax.set_ylim(-0.6, 4.9)
        ax.set_xlabel("Score")
        ax.set_title(f"{title} (GARF-H)", loc="left")
        panel_label(ax, "b" if column == 0 else "c")

    # d: current case-score composition
    ax = fig.add_subplot(gs[1, 0])
    rows = [
        ("Task 1", task1["pooled"]["case_score"], task1["pooled"]["reasoning"]["composite"]),
        ("Task 2", task2["pooled"]["case_score"], task2["pooled"]["reasoning"]["composite"]),
    ]
    labels = [row[0] for row in rows]
    case_scores = [row[1] for row in rows]
    reasoning_scores = [row[2] for row in rows]
    decision_parts = [0.75 * score if index < 2 else score for index, score in enumerate(case_scores)]
    reasoning_parts = [0.25 * reasoning for reasoning in reasoning_scores]
    x = np.arange(len(rows))
    ax.bar(x, decision_parts, color=C["blue"], width=0.55, label="Decision")
    ax.bar(x, reasoning_parts, bottom=decision_parts, color=C["orange"], width=0.55, label="Reasoning")
    for index in range(len(rows)):
        ax.text(index, decision_parts[index] + reasoning_parts[index] + 0.015,
                f"{case_scores[index]:.3f}", ha="center", fontsize=6.2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Case score")
    ax.set_title("Case-score composition", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncols=2, fontsize=6.0)
    panel_label(ax, "d")

    # e: per-case reasoning quality distribution
    ax = fig.add_subplot(gs[1, 1])
    rng = np.random.default_rng(42)
    for index, (components, colour, label) in enumerate(
        ((task1_reasoning, C["blue"], "Task 1"), (task2_reasoning, C["green"], "Task 2"))
    ):
        values = np.asarray([row["composite"] for row in components["optimized"]])
        jitter = rng.normal(0, 0.035, len(values))
        ax.scatter(index + jitter, values, s=12, color=colour, alpha=0.55, linewidth=0)
        ax.boxplot(
            [values], positions=[index], widths=0.30, showfliers=False, patch_artist=True,
            boxprops={"facecolor": "none", "edgecolor": colour, "linewidth": 0.8},
            medianprops={"color": colour, "linewidth": 1.0},
            whiskerprops={"color": colour, "linewidth": 0.7},
            capprops={"color": colour, "linewidth": 0.7},
        )
        ax.scatter([index], [values.mean()], s=36, color=colour, edgecolor="white",
                   linewidth=0.6, zorder=5)
    ax.set_xticks([0, 1], ["Task 1", "Task 2"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Per-case reasoning composite")
    ax.set_title("Per-case reasoning quality", loc="left")
    panel_label(ax, "e")

    # f: task-level summary table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    t1_reason = task1["reasoning"]["composite"]["mean"]
    t2_reason = task2["reasoning"]["composite"]["mean"]
    t1_acc = task1["pooled"]["decision"]["accuracy"]
    t2_acc = task2["pooled"]["decision"]["accuracy"]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Task-level summary", loc="left")
    for text, x, alignment in (
        ("Metric", 0.02, "left"),
        ("Task 1", 0.44, "center"),
        ("Task 2", 0.66, "center"),
        ("Task 3", 0.88, "center"),
    ):
        ax.text(x, 0.95, text, transform=ax.transAxes, fontsize=6.0, fontweight="bold",
                ha=alignment, va="center")
    rows = [
        ("Decision", ["F1+", "wF1", "C-index"]),
        ("Anchor", [
            f"{task1['baseline_rule']['case_score']:.3f}",
            f"{task2['baseline_rule']['case_score']:.3f}",
            f"{task3['ablation']['pathology_only']['value']:.3f}",
        ]),
        ("GARF-H (CI)", [
            f"{t1['value']:.3f}\n[{t1['low']:.2f},{t1['high']:.2f}]",
            f"{t2['value']:.3f}\n[{t2['low']:.2f},{t2['high']:.2f}]",
            f"{t3['value']:.3f}\n[{t3['low']:.2f},{t3['high']:.2f}]",
        ]),
        ("Accuracy", [f"{t1_acc:.3f}", f"{t2_acc:.3f}", "—"]),
        ("Reasoning", [f"{t1_reason:.3f}", f"{t2_reason:.3f}", "—"]),
        ("Case score", [
            f"{task1['pooled']['case_score']:.3f}",
            f"{task2['pooled']['case_score']:.3f}",
            f"{t3['value']:.3f}",
        ]),
    ]
    y = 0.82
    for label, values in rows:
        ax.text(0.02, y, label, transform=ax.transAxes, fontsize=5.4, ha="left", va="center")
        value_font = 5.1 if label == "GARF-H (CI)" else (4.9 if label == "Reasoning" else 5.2)
        for x, value in zip((0.44, 0.66, 0.88), values):
            ax.text(x, y, value, transform=ax.transAxes, fontsize=value_font, ha="center",
                    va="center", linespacing=1.0)
        y -= 0.13
    for line_y in (0.89, 0.76, 0.63, 0.50, 0.37, 0.24, 0.11):
        ax.plot([0, 1], [line_y, line_y], transform=ax.transAxes, color=C["light_grey"],
                linewidth=0.4, clip_on=False)
    ax.text(0.0, 0.065, "Anchor: guideline rule (Tasks 1/2), pathology score (Task 3).",
            fontsize=5.0, color=C["grey"], transform=ax.transAxes)
    ax.text(0.0, 0.015, "GARF-H: 95% bootstrap CI; case = 0.75×decision + 0.25×reasoning.",
            fontsize=5.0, color=C["grey"], transform=ax.transAxes)
    panel_label(ax, "f", x=-0.10)

    save(fig, outdir, "fig5_model_comparison")


def main() -> None:
    args = parse_args()
    style()
    mpl.rcParams["savefig.dpi"] = args.dpi
    dataset, evaluation = _load(args.data_root, args.eval)
    outdir = Path(args.output_dir)
    figure_overview(dataset, evaluation, outdir)
    figure_task1(dataset, evaluation, outdir)
    figure_task2(dataset, evaluation, outdir)
    figure_task3(dataset, evaluation, outdir)
    figure_summary(dataset, evaluation, outdir)
    (outdir / "layout_qc.json").write_text(json.dumps(LAYOUT_QC, indent=2), encoding="utf-8")
    print(json.dumps(LAYOUT_QC, indent=2))


if __name__ == "__main__":
    main()
