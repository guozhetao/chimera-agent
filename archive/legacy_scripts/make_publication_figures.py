from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

from chimera.data.dataset import ChimeraDataset
from chimera.evaluation.survival_metrics import concordance_index
from chimera.utils.jsonio import read_json


# Okabe-Ito colorblind-safe palette.
C = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#111111",
    "grey": "#6F7782",
    "light_grey": "#E6E9ED",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--local", required=True)
    p.add_argument("--local-neural", default=None)
    p.add_argument("--api-glob", default=None)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def set_publication_style() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 6.2,
        "axes.titlesize": 7.0,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 5.5,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "svg.fonttype": "none",
        "pdf.compression": 6,
    })


def panel_label(ax, label: str, x=-0.12, y=1.05):
    ax.text(
        x, y, label, transform=ax.transAxes,
        fontsize=8.5, fontweight="bold", va="bottom", ha="left", color=C["black"],
    )


def export(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def load_local_predictions(path: str) -> dict[int, list[dict[str, Any]]]:
    data = read_json(path)
    return {task: data.get(f"task{task}", {}).get("predictions", []) for task in (1, 2, 3)}


def bootstrap_roc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = auc(*roc_curve(y_true, y_score)[:2])
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        sample_y, sample_s = y_true[idx], y_score[idx]
        if sample_y.sum() in (0, len(sample_y)):
            continue
        values.append(auc(*roc_curve(sample_y, sample_s)[:2]))
    return float(point), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def bootstrap_c_index_ci(
    times: np.ndarray,
    events: np.ndarray,
    risks: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = concordance_index(times, events, risks)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(times), len(times))
        sample_t, sample_e, sample_r = times[idx], events[idx], risks[idx]
        if sample_e.sum() == 0 or len(np.unique(sample_t)) < 2:
            continue
        values.append(concordance_index(sample_t, sample_e, sample_r))
    return float(point), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def net_benefit(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> float:
    pred = prob >= threshold
    tp = np.sum(pred & (y_true == 1))
    fp = np.sum(pred & (y_true == 0))
    return tp / len(y_true) - fp / len(y_true) * threshold / (1 - threshold)


def kaplan_meier(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times)
    t, e = times[order], events[order]
    unique = np.unique(t)
    surv, n = 1.0, len(t)
    out_t, out_s = [0], [1]
    for time in unique:
        risk = np.sum(t >= time)
        d = np.sum((t == time) & (e == 1))
        if risk > 0 and d > 0:
            surv *= 1 - d / risk
        out_t.append(time)
        out_s.append(surv)
    return np.asarray(out_t), np.asarray(out_s)


def logrank_p(time1, event1, time2, event2) -> float:
    # Two-group Mantel-Haenszel log-rank test.
    all_times = np.unique(np.r_[time1, time2])
    observed_1 = expected_1 = variance = 0.0
    n1, n2 = len(time1), len(time2)
    for t in all_times:
        risk1 = np.sum(time1 >= t)
        risk2 = np.sum(time2 >= t)
        n = risk1 + risk2
        if n <= 1:
            continue
        d1 = np.sum((time1 == t) & (event1 == 1))
        d2 = np.sum((time2 == t) & (event2 == 1))
        d = d1 + d2
        observed_1 += d1
        expected_1 += d * risk1 / n
        variance += d * risk1 * risk2 * (n - d) / (n * n * (n - 1)) if n > 1 else 0
    z = (observed_1 - expected_1) ** 2 / variance if variance > 0 else 0
    return float(chi2.sf(z, 1))


def collect_api_records(pattern: str | None) -> list[dict[str, Any]]:
    records = []
    if not pattern:
        return records
    for path in sorted(Path().glob(pattern)):
        data = read_json(path)
        if not data:
            continue
        cfg = data.get("configuration", {})
        records.append({
            "name": f"{cfg.get('provider')}/{cfg.get('model_alias')}",
            "task": int(cfg.get("task")),
            "data": data,
        })
    return records


def complete_api_models(api_records: list[dict[str, Any]]) -> list[str]:
    """Models eligible for performance comparison: 5/5 successful calls on all tasks."""
    names = sorted({r["name"] for r in api_records})
    complete = []
    for name in names:
        okay = True
        for task in (1, 2, 3):
            rec = next((r for r in api_records if r["name"] == name and r["task"] == task), None)
            if not rec or len(rec["data"].get("predictions", [])) != rec["data"].get("configuration", {}).get("n_cases_requested"):
                okay = False
                break
        if okay:
            complete.append(name)
    return complete


def api_primary_metric(data: dict[str, Any], task: int) -> float:
    metrics = data.get("metrics", {}).get("metrics", data.get("metrics", {}))
    return float(metrics.get("f1_positive" if task == 1 else "f1_weighted" if task == 2 else "harrell_c_index"))


def weighted_f1(y_true: list[str], y_pred: list[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    f1s, weights = [], []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        weights.append(y_true.count(label))
    return float(np.average(f1s, weights=weights))


def draw_box(ax, xy, width, height, text, face="#F7FAFC", edge="#285F8F", fontsize=5.6):
    box = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=0.6, edgecolor=edge, facecolor=face, mutation_aspect=1.0,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize, linespacing=1.25)


def figure1(args, dataset: ChimeraDataset, local_preds) -> Path:
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.05, 1.18], hspace=0.44, wspace=0.34)

    # A. Cohort and task structure.
    ax = fig.add_subplot(gs[0, 0])
    tasks = ["Task 1\nBiopsy", "Task 2\nTreatment", "Task 3\nBCR"]
    counts = [dataset.counts()[i]["total"] for i in (1, 2, 3)]
    labeled = [dataset.counts()[i]["labeled"] for i in (1, 2, 3)]
    unlabeled = [total - lab for total, lab in zip(counts, labeled)]
    x = np.arange(3)
    ax.bar(x, labeled, color=[C["blue"], C["green"], C["orange"]], label="Labeled")
    ax.bar(x, unlabeled, bottom=labeled, color=C["light_grey"], label="Unlabeled")
    for i, (lab, total) in enumerate(zip(labeled, counts)):
        ax.text(i, total + 5, f"{lab}/{total}", ha="center", fontsize=5.5)
    ax.set_xticks(x, tasks)
    ax.set_ylabel("Cases")
    ax.set_ylim(0, max(counts) * 1.22)
    ax.legend(frameon=False, ncols=2, loc="upper right")
    ax.set_title("Cohort and endpoint structure")
    panel_label(ax, "a")

    # B. Label distributions.
    ax = fig.add_subplot(gs[0, 1])
    distributions = [
        ["csPCa-\nnegative", "csPCa-\npositive"],
        ["AS", "Treat", "Cont.", "WW"],
        ["No BCR", "BCR"],
    ]
    values = [
        [35, 56],
        [25, 31, 14, 2],
        [56, 19],
    ]
    colors = [[C["grey"], C["vermillion"]], [C["green"], C["vermillion"], C["blue"], C["purple"]], [C["grey"], C["orange"]]]
    left = np.zeros(3)
    for j in range(4):
        vals = np.array([v[j] if j < len(v) else 0 for v in values], dtype=float)
        ax.barh(np.arange(3), vals, left=left, color=[colors[i][j] if j < len(colors[i]) else "white" for i in range(3)], height=.55)
        left += vals
    ax.set_yticks([0, 1, 2], ["Task 1", "Task 2", "Task 3"])
    ax.invert_yaxis()
    ax.set_xlabel("Labeled cases")
    ax.set_title("Class balance and rare decisions")
    panel_label(ax, "b")

    # C. Modality availability.
    ax = fig.add_subplot(gs[0, 2])
    modalities = ["MRI", "Biopsy WSI", "Prostatectomy WSI"]
    matrix = []
    for task in (1, 2, 3):
        row = []
        for source in ["MRI image", "Biopsy slide", "Prostatectomy slide"]:
            available = 0
            for case in dataset.cases(task, labeled_only=True):
                neural_file = case.directory / "prostate-modality-level-neural-representations.json"
                neural = read_json(neural_file, {}) or {}
                if neural.get(source, []):
                    available += 1
            row.append(available)
        matrix.append(row)
    matrix = np.asarray(matrix)
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(labeled))
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{matrix[i,j]}/{labeled[i]}", ha="center", va="center", fontsize=5.1)
    ax.set_xticks(range(3), modalities)
    ax.set_yticks(range(3), ["Task 1", "Task 2", "Task 3"])
    ax.set_title("Multimodal availability")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=.72, pad=.03)
    cb.ax.tick_params(labelsize=4.8)
    panel_label(ax, "c")

    # D. Architecture and information route.
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    ax.set_title("GARF-M3: guideline-anchored residual multimodal decision agent", y=1.04)
    # Clean four-column architecture: inputs -> representation -> constrained learning -> outputs.
    boxes = {
        "prompt": ((.03, .67), "Structured prompt\nPSA · age · DRE\nbiopsy history"),
        "reports": ((.03, .43), "Clinical reports\nradiology · pathology\nsurgical notes"),
        "neural": ((.03, .19), "Neural representations\nMRI 1024-d\nWSI 960-d sets"),
        "canonical": ((.25, .43), "Canonical feature layer\nmissingness masks\nPSA trajectory\npathology concepts"),
        "guideline": ((.47, .66), "EAU/NCCN\nprior\nπ_rule"),
        "residual": ((.47, .37), "Gated residual\nlearner\nΔ(x)"),
        "fusion": ((.66, .51), "Logit fusion\nsoftmax(logit π_rule\n+ g(x)Δ(x))"),
        "decision": ((.83, .66), "Decision\nbiopsy /\nmanagement /\nBCR risk"),
        "reasoning": ((.83, .31), "Auditable reasoning\nconfidence\nvariable weights\ntool trace"),
    }
    for box in boxes.values():
        draw_box(ax, box[0], .16, .17, box[1], fontsize=5.25)
    arrows = [
        ((.11, .76), (.25, .56)), ((.11, .52), (.25, .52)), ((.11, .28), (.25, .47)),
        ((.41, .52), (.47, .73)), ((.41, .52), (.47, .45)),
        ((.63, .75), (.66, .60)), ((.63, .45), (.66, .56)),
        ((.82, .60), (.83, .72)), ((.82, .60), (.83, .42)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=6, linewidth=.6, color=C["grey"], shrinkA=2, shrinkB=2))
    ax.text(.60, .09, "Unlabeled cases → masked-modality consistency → conservative pseudo-labels", ha="center", fontsize=5.6, color=C["blue"])
    ax.add_patch(FancyArrowPatch((.60, .13), (.60, .25), arrowstyle="-|>", mutation_scale=6, linewidth=.6, color=C["blue"]))
    ax.text(.03, .04, "Solid boxes: supervised clinical pathway; blue arrow: semi-supervised consistency branch", fontsize=5.2, color=C["grey"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_label(ax, "d", x=.01, y=.97)
    return export_helper(fig, args.output_dir, "fig1_cohort_architecture")


def export_helper(fig, output_dir, name):
    path = Path(output_dir) / name
    export(fig, path)
    return path


def figure2(args, local_preds, neural_preds) -> Path:
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = GridSpec(3, 4, figure=fig, hspace=.73, wspace=.58)

    # Task 1 ROC.
    p1 = local_preds[1]
    y = np.array([1 if x["true"] == "yes" else 0 for x in p1])
    prob = np.array([x["probability"][1] for x in p1])
    fpr, tpr, _ = roc_curve(y, prob)
    auc_value, auc_lo, auc_hi = bootstrap_roc_ci(y, prob, n_boot=1000)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(fpr, tpr, color=C["blue"], lw=1.1)
    ax.plot([0, 1], [0, 1], "--", color=C["grey"], lw=.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("MRI-only biopsy decision (OOF)")
    ax.text(.35, .12, f"AUC {auc_value:.2f}\n({auc_lo:.2f}–{auc_hi:.2f})", fontsize=5.5)
    panel_label(ax, "a", x=-.22)

    # Task 1 precision-recall.
    precision, recall, _ = precision_recall_curve(y, prob)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(recall, precision, color=C["green"], lw=1.1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.axhline(y.mean(), color=C["grey"], ls="--", lw=.6)
    ax.set_title("Precision–recall")
    panel_label(ax, "b", x=-.22)

    # Task 1 calibration.
    ax = fig.add_subplot(gs[0, 2])
    frac_pos, mean_pred = calibration_curve(y, prob, n_bins=5, strategy="quantile")
    ax.plot(mean_pred, frac_pos, "o-", color=C["orange"], lw=1, ms=2.6)
    ax.plot([0, 1], [0, 1], "--", color=C["grey"], lw=.6)
    ax.set_xlabel("Predicted P(biopsy)"); ax.set_ylabel("Observed fraction")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Probability calibration")
    panel_label(ax, "c", x=-.22)

    # Task 1 decision curve.
    ax = fig.add_subplot(gs[0, 3])
    thresholds = np.linspace(.05, .95, 50)
    model_nb = [net_benefit(y, prob, t) for t in thresholds]
    all_nb = [y.mean() - (1 - y.mean()) * t / (1 - t) for t in thresholds]
    ax.plot(thresholds, model_nb, color=C["blue"], lw=1.1, label="GARF-M3")
    ax.plot(thresholds, all_nb, color=C["grey"], ls="--", lw=.7, label="Biopsy all")
    ax.axhline(0, color=C["black"], ls=":", lw=.5, label="Biopsy none")
    ax.set_xlim(.05, .95); ax.set_ylim(-.15, .55)
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Net benefit")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Clinical utility")
    panel_label(ax, "d", x=-.22)

    # Task 2 confusion.
    p2 = local_preds[2]
    labels2 = ["active_surveillance", "active_treatment", "continued_surveillance", "watchful_waiting"]
    short2 = ["AS", "Treat", "Cont.", "WW"]
    cm = confusion_matrix([x["true"] for x in p2], [x["predicted"] for x in p2], labels=labels2)
    ax = fig.add_subplot(gs[1, :2])
    im = ax.imshow(cm, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=5.2,
                    color="white" if cm[i, j] > cm.max() * .6 else C["black"])
    ax.set_xticks(range(4), short2); ax.set_yticks(range(4), short2)
    ax.set_xlabel("Predicted management"); ax.set_ylabel("Guideline management")
    ax.set_title("Post-biopsy risk stratification (OOF)")
    ax.grid(False)
    panel_label(ax, "e", x=-.18)

    # Task 2 per-class F1.
    ax = fig.add_subplot(gs[1, 2:])
    tp = np.diag(cm); fp = cm.sum(0)-tp; fn = cm.sum(1)-tp
    f1 = 2*tp/(2*tp+fp+fn)
    ax.bar(range(4), f1, color=[C["green"], C["vermillion"], C["blue"], C["purple"]], width=.58)
    ax.set_xticks(range(4), short2)
    ax.set_ylim(0, 1.06); ax.set_ylabel("F1 score")
    ax.set_title("Class-level safety and rare-class behavior")
    for i, v in enumerate(f1):
        ax.text(i, v+.02, f"{v:.2f}", ha="center", fontsize=5.2)
        ax.text(i, .05, f"n={int(cm[i].sum())}", ha="center", fontsize=4.8, color=C["grey"])
    panel_label(ax, "f", x=-.12)

    # Task 3 Kaplan-Meier by model risk.
    p3 = local_preds[3]
    frame = pd.DataFrame(p3)
    frame["risk_group"] = pd.qcut(frame["risk"], 3, labels=["Low", "Intermediate", "High"])
    ax = fig.add_subplot(gs[2, :2])
    colors = [C["blue"], C["orange"], C["vermillion"]]
    for group, color in zip(["Low", "Intermediate", "High"], colors):
        sub = frame[frame["risk_group"] == group]
        t, s = kaplan_meier(sub["true_months"].to_numpy(), sub["true_event"].to_numpy())
        ax.step(t, s*100, where="post", color=color, lw=1.1, label=group)
    low = frame[frame["risk_group"] == "Low"]; high = frame[frame["risk_group"] == "High"]
    pval = logrank_p(low["true_months"], low["true_event"], high["true_months"], high["true_event"])
    ax.set_xlabel("Months after radical prostatectomy"); ax.set_ylabel("Recurrence-free survival (%)")
    ax.set_ylim(0, 103); ax.set_xlim(0, 115)
    ax.legend(frameon=False, title="Predicted risk", ncols=3, loc="lower left")
    ax.set_title(f"Model-defined recurrence risk (P={pval:.2g})")
    panel_label(ax, "g", x=-.18)

    # Task 3 c-index and neural gain.
    ax = fig.add_subplot(gs[2, 2])
    base_c, base_lo, base_hi = bootstrap_c_index_ci(
        frame["true_months"].to_numpy(),
        frame["true_event"].to_numpy(),
        frame["risk"].to_numpy(),
    )
    neural_frame = pd.DataFrame(neural_preds[3]) if neural_preds else None
    if neural_frame is not None:
        neural_c, neural_lo, neural_hi = bootstrap_c_index_ci(
            neural_frame["true_months"].to_numpy(),
            neural_frame["true_event"].to_numpy(),
            neural_frame["risk"].to_numpy(),
            seed=43,
        )
    else:
        neural_c, neural_lo, neural_hi = np.nan, np.nan, np.nan
    bars = ax.bar([0, 1], [base_c, neural_c], color=[C["grey"], C["purple"]], width=.55)
    ax.errorbar(
        [0, 1], [base_c, neural_c],
        yerr=[[base_c-base_lo, neural_c-neural_lo], [base_hi-base_c, neural_hi-neural_c]],
        fmt="none", ecolor=C["black"], elinewidth=.55, capsize=1.6,
    )
    ax.set_xticks([0, 1], ["Clinical\n+ pathology", "+ Neural\n+ semi"], fontsize=5.6)
    ax.set_ylim(.45, .95); ax.set_ylabel("Harrell C-index")
    ax.set_title("Multimodal gain")
    for bar, value in zip(bars, [base_c, neural_c]):
        ax.text(bar.get_x()+bar.get_width()/2, value+.028, f"{value:.3f}", ha="center", fontsize=5.3)
    panel_label(ax, "h", x=-.22)

    # Task 3 event rate by risk quartile.
    ax = fig.add_subplot(gs[2, 3])
    frame["quartile"] = pd.qcut(frame["risk"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    rates = frame.groupby("quartile", observed=True)["true_event"].mean()*100
    ax.bar(rates.index.astype(str), rates.values, color=C["orange"], width=.62)
    ax.set_ylim(0, 110); ax.set_ylabel("BCR events (%)")
    ax.set_xlabel("Predicted-risk quartile")
    ax.set_title("Risk stratification")
    panel_label(ax, "i", x=-.22)
    return export_helper(fig, args.output_dir, "fig2_performance_clinical_utility")


def figure3(args, local_preds, api_records) -> Path:
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = GridSpec(3, 3, figure=fig, hspace=.82, wspace=.52)

    complete_names = complete_api_models(api_records)
    model_colors = {
        "deepseek/deepseek-flash": C["blue"],
        "deepseek/deepseek-v4-pro": C["green"],
        "zhipu/glm-legacy-air": C["orange"],
    }

    # A. Explicit availability/status matrix. Missing/unavailable models are shown rather than hidden.
    ax = fig.add_subplot(gs[0, :2])
    display_rows = [
        "DeepSeek Flash",
        "DeepSeek V4 Pro",
        "GLM 4.5 Air",
        "GLM 4 Flash",
        "GLM 4 Plus",
        "GLM 5.x family",
    ]
    lookup = {
        ("DeepSeek Flash", 1): "5/5", ("DeepSeek Flash", 2): "5/5", ("DeepSeek Flash", 3): "5/5",
        ("DeepSeek V4 Pro", 1): "5/5", ("DeepSeek V4 Pro", 2): "5/5", ("DeepSeek V4 Pro", 3): "5/5",
        ("GLM 4.5 Air", 1): "5/5", ("GLM 4.5 Air", 2): "5/5", ("GLM 4.5 Air", 3): "5/5",
        ("GLM 4 Flash", 1): "4/5*", ("GLM 4 Flash", 2): "3/5", ("GLM 4 Flash", 3): "5/5",
        ("GLM 4 Plus", 1): "429", ("GLM 4 Plus", 2): "429", ("GLM 4 Plus", 3): "429",
        ("GLM 5.x family", 1): "429", ("GLM 5.x family", 2): "429", ("GLM 5.x family", 3): "429",
    }
    status_numeric = np.zeros((len(display_rows), 3))
    for i, row in enumerate(display_rows):
        for j, task in enumerate((1, 2, 3)):
            text = lookup[(row, task)]
            if text.startswith("5/5"):
                status_numeric[i,j] = 2
            elif text == "429":
                status_numeric[i,j] = 0
            else:
                status_numeric[i,j] = 1
    cmap = mpl.colors.ListedColormap([C["light_grey"], C["yellow"], C["green"]])
    im = ax.imshow(status_numeric, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    for i in range(len(display_rows)):
        for j in range(3):
            ax.text(j, i, lookup[(display_rows[i], j+1)], ha="center", va="center", fontsize=5.2)
    ax.set_xticks(range(3), ["Task 1", "Task 2", "Task 3"])
    ax.set_yticks(range(len(display_rows)), display_rows)
    ax.set_title("API feasibility before performance comparison")
    ax.grid(False)
    ax.text(.5, -.28, "Green = complete pilot (5/5) · yellow = partial · grey = HTTP 429/quota · *includes timeouts",
            transform=ax.transAxes, ha="center", fontsize=5.0, color=C["grey"])
    panel_label(ax, "a", x=-.20)

    # B. Performance only for complete 5/5 models.
    ax = fig.add_subplot(gs[0, 2])
    task_markers = {1:"o", 2:"s", 3:"^"}
    task_colors = {1:C["blue"], 2:C["green"], 3:C["orange"]}
    for yi, name in enumerate(complete_names):
        for task in (1, 2, 3):
            rec = next(r for r in api_records if r["name"] == name and r["task"] == task)
            metric = api_primary_metric(rec["data"], task)
            ax.scatter(metric, yi + (task-2)*.16, marker=task_markers[task], s=13,
                       color=task_colors[task], edgecolor="white", linewidth=.3, zorder=3)
    ax.set_yticks(range(len(complete_names)), [n.split("/")[-1] for n in complete_names])
    ax.set_xlim(-.02, 1.06)
    ax.set_xlabel("Primary metric")
    ax.set_title("Complete-model pilot results")
    ax.grid(axis="x", alpha=.2)
    handles = [plt.Line2D([],[], marker=task_markers[t], ls="none", color=task_colors[t], ms=3.2, label=f"Task {t}") for t in (1,2,3)]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=4.6)
    ax.text(.5, -.42, "Only models with 5/5 calls on all tasks\nT2 subset lacks Cont./WW; T3 has 1 event", transform=ax.transAxes, ha="center", va="top", fontsize=4.8, color=C["grey"])
    panel_label(ax, "b", x=-.34)

    # C. Paired local-API difference, complete models only.
    ax = fig.add_subplot(gs[1, 0])
    labels, differences = [], []
    for name in complete_names:
        for task in (1,2,3):
            rec = next(r for r in api_records if r["name"] == name and r["task"] == task)
            ids = {p["case_id"] for p in rec["data"]["predictions"]}
            local = [p for p in local_preds[task] if p["case_id"] in ids]
            if task == 1:
                yt = np.array([p["true"] == "yes" for p in local], int)
                yp = np.array([p["predicted"] == "yes" for p in local], int)
                local_metric = 2*np.sum((yt==1)&(yp==1))/max(2*np.sum((yt==1)&(yp==1))+np.sum((yt==0)&(yp==1))+np.sum((yt==1)&(yp==0)),1)
            elif task == 2:
                local_metric = weighted_f1([p["true"] for p in local], [p["predicted"] for p in local])
            else:
                local_metric = concordance_index([p["true_months"] for p in local], [p["true_event"] for p in local], [p["risk"] for p in local])
            differences.append(local_metric-api_primary_metric(rec["data"], task))
            labels.append(f"T{task}\n{name.split('/')[-1].replace('deepseek-','DS-').replace('glm-legacy-air','GLM-Air')}")
    ax.bar(range(len(differences)), differences, color=[C["blue"] if d>=0 else C["vermillion"] for d in differences], width=.62)
    ax.axhline(0, color=C["black"], lw=.5)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=4.5)
    ax.set_ylabel("Local − API")
    ax.set_title("Paired pilot difference")
    panel_label(ax, "c", x=-.26)

    # D. Reasoning trace quality for Tasks 1/2 only; Task 3 has no official reasoning trace.
    ax = fig.add_subplot(gs[1, 1:])
    reasoning_labels = ["Confidence", "Variable weights", "Tool sequence", "Rationale"]
    reasoning_keys = ["confidence", "variable_weights", "tool_sequence", "rationale"]
    matrix, row_labels = [], []
    for name in complete_names:
        for task in (1,2):
            rec = next(r for r in api_records if r["name"] == name and r["task"] == task)
            reasoning = rec["data"].get("metrics", {}).get("reasoning", {})
            matrix.append([reasoning.get(k,0) for k in reasoning_keys])
            row_labels.append(f"{name.split('/')[-1]}\nTask {task}")
    matrix = np.asarray(matrix)
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=4.7)
    ax.set_xticks(range(len(reasoning_labels)), reasoning_labels)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=4.7)
    ax.set_title("Reasoning trace quality (Tasks 1–2 only)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=.72, pad=.02)
    cb.ax.tick_params(labelsize=4.5)
    panel_label(ax, "d", x=-.16)

    # E. Operational efficiency for complete models.
    ax = fig.add_subplot(gs[2, 0])
    model_names = complete_names
    mean_latency = []
    for name in model_names:
        vals = []
        for task in (1,2,3):
            rec = next(r for r in api_records if r["name"] == name and r["task"] == task)
            n_success = max(len(rec["data"].get("predictions", [])), 1)
            vals.append(rec["data"].get("elapsed_seconds", np.nan)/n_success)
        mean_latency.append(np.nanmean(vals))
    ax.barh(np.arange(len(model_names)), mean_latency, color=[model_colors.get(n,C["grey"]) for n in model_names], height=.55)
    ax.set_yticks(np.arange(len(model_names)), [n.split("/")[-1] for n in model_names])
    ax.invert_yaxis()
    ax.set_xlabel("Seconds / successful case")
    ax.set_title("Operational efficiency")
    for i,v in enumerate(mean_latency):
        ax.text(v+.3,i,f"{v:.1f}s",va="center",fontsize=4.8)
    panel_label(ax, "e", x=-.28)

    # F. Clinical translation route.
    ax = fig.add_subplot(gs[2, 1:])
    ax.axis("off")
    ax.set_title("Proposed clinical deployment")
    stages = [
        ("1. Pre-biopsy", "Avoid unnecessary biopsy; explain PI-RADS/PSA trade-off", C["blue"]),
        ("2. Post-biopsy", "AS vs treatment vs continued surveillance vs WW", C["green"]),
        ("3. Post-surgery", "Risk-tiered PSA follow-up and earlier salvage consideration", C["orange"]),
    ]
    for i, (title, text, color) in enumerate(stages):
        y = .74-i*.31
        ax.add_patch(FancyBboxPatch((.04,y),.90,.18, boxstyle="round,pad=.02", facecolor="white", edgecolor=color, linewidth=.8))
        ax.text(.49,y+.115,title,ha="center",va="center",fontsize=6,fontweight="bold")
        ax.text(.49,y+.045,text,ha="center",va="center",fontsize=5.2)
        if i < 2:
            ax.add_patch(FancyArrowPatch((.49,y-.02),(.49,y-.10), arrowstyle="-|>", mutation_scale=7, color=C["grey"]))
    panel_label(ax, "f", x=-.10)
    return export_helper(fig, args.output_dir, "fig3_api_reasoning_translation")


def main():
    args = parse_args()
    set_publication_style()
    dataset = ChimeraDataset(args.data_root)
    local = load_local_predictions(args.local)
    neural = load_local_predictions(args.local_neural) if args.local_neural else {}
    api = collect_api_records(args.api_glob)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    f1 = figure1(args, dataset, local)
    f2 = figure2(args, local, neural)
    f3 = figure3(args, local, api)
    print(f1)
    print(f2)
    print(f3)


if __name__ == "__main__":
    main()
