from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.text as mpl_text
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    roc_curve,
)

from chimera.data.dataset import ChimeraDataset
from chimera.evaluation.survival_metrics import concordance_index
from chimera.features.pathology_parser import pathology_risk_score
from chimera.utils.jsonio import read_json


C = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "grey": "#6F7782",
    "light": "#E6E9ED",
    "black": "#111111",
}

LAYOUT_REPORT = {}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--local", required=True)
    p.add_argument("--local-neural", required=True)
    p.add_argument("--api-dir", required=True)
    p.add_argument("--comparison", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def style():
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.4,
            "axes.titlesize": 7.1,
            "axes.labelsize": 6.6,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 5.6,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "svg.fonttype": "none",
        }
    )


def label(ax, text, x=-0.14, y=1.06):
    ax.text(x, y + 0.08, text, transform=ax.transAxes, fontsize=8.6, fontweight="bold", va="bottom", ha="left")


def save(fig, outdir: Path, name: str):
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
            a, b = items[i][1], items[j][1]
            dx = min(a.x1, b.x1) - max(a.x0, b.x0)
            dy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if dx <= 0 or dy <= 0:
                continue
            score = dx * dy / max(min(a.width * a.height, b.width * b.height), 1e-9)
            if score > 0.08:
                collisions.append({"text_1": items[i][0], "text_2": items[j][0], "score": round(score, 3)})
    canvas = fig.bbox
    outside = [text for text, bbox in items if bbox.x0 < canvas.x0 - 1 or bbox.y0 < canvas.y0 - 1 or bbox.x1 > canvas.x1 + 1 or bbox.y1 > canvas.y1 + 1]
    LAYOUT_REPORT[name] = {
        "text_elements": len(items),
        "text_overlap_count": len(collisions),
        "text_overlaps": collisions,
        "text_outside_canvas": outside,
        # savefig uses bbox='tight', so artists slightly outside the initial canvas are included
        # in the exported file. Text-text collisions are the hard failure criterion.
        "status": "PASS" if not collisions else "FAIL",
    }
    fig.savefig(outdir / f"{name}.png")
    fig.savefig(outdir / f"{name}.pdf")
    fig.savefig(outdir / f"{name}.svg")
    plt.close(fig)


def write_layout_report(outdir: Path):
    import json
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "layout_qc.json").write_text(json.dumps(LAYOUT_REPORT, ensure_ascii=False, indent=2), encoding="utf-8")


def preds(data, task):
    return data.get(f"task{task}", {}).get("predictions", [])


def load_api(api_dir: str):
    records = {}
    for path in sorted(Path(api_dir).glob("*.json")):
        d = read_json(path)
        if not d or d.get("metrics", {}).get("error"):
            continue
        cfg = d.get("configuration", {})
        alias = cfg.get("model_alias")
        task = cfg.get("task")
        requested = cfg.get("n_cases_requested", 0)
        successful = len(d.get("predictions", []))
        if successful != requested or successful == 0:
            continue
        records[(alias, task)] = d
    return records


def task1_rule(case):
    try:
        return "yes" if int(case.prompt.get("pirads")) >= 4 else "no"
    except (TypeError, ValueError):
        return "no"


def task2_rule(case):
    try:
        isup = int(case.prompt.get("bx_isup"))
    except (TypeError, ValueError):
        isup = 0
    if isup <= 0:
        return "continued_surveillance"
    if isup == 1:
        return "active_surveillance"
    return "active_treatment"


def metric_task1(y, p):
    return {
        "accuracy": accuracy_score(y, p),
        "positive_f1": f1_score(y, p, pos_label="yes"),
    }


def metric_task2(y, p):
    return {
        "accuracy": accuracy_score(y, p),
        "weighted_f1": f1_score(y, p, average="weighted"),
        "macro_f1": f1_score(y, p, average="macro"),
    }


def bootstrap_ci(y, score, metric="auc", n=1000):
    rng = np.random.default_rng(42)
    if metric == "auc":
        vals = []
        for _ in range(n):
            idx = rng.integers(0, len(y), len(y))
            yy, ss = np.asarray(y)[idx], np.asarray(score)[idx]
            if yy.sum() in (0, len(yy)):
                continue
            vals.append(auc(*roc_curve(yy, ss)[:2]))
        point = auc(*roc_curve(y, score)[:2])
    else:
        vals = []
        for _ in range(n):
            idx = rng.integers(0, len(y), len(y))
            raw_y = np.asarray(y).tolist()
            raw_s = np.asarray(score).tolist()
            yy = np.asarray([str(v).lower() == "yes" or v == 1 for v in np.asarray(raw_y)[idx]], int)
            ss = np.asarray([str(v).lower() == "yes" or v == 1 for v in np.asarray(raw_s)[idx]], int)
            vals.append(f1_score(yy, ss))
        raw_y = np.asarray(y).tolist(); raw_s = np.asarray(score).tolist()
        yy = np.asarray([str(v).lower() == "yes" or v == 1 for v in raw_y], int)
        ss = np.asarray([str(v).lower() == "yes" or v == 1 for v in raw_s], int)
        point = f1_score(yy, ss)
    return point, np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def cindex_bootstrap(times, events, risks, n=1000):
    rng = np.random.default_rng(42)
    vals = []
    times, events, risks = map(np.asarray, (times, events, risks))
    for _ in range(n):
        idx = rng.integers(0, len(times), len(times))
        t, e, r = times[idx], events[idx], risks[idx]
        if e.sum() == 0 or len(np.unique(t)) < 2:
            continue
        vals.append(concordance_index(t, e, r))
    point = concordance_index(times, events, risks)
    return point, np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def km(times, events):
    t, e = np.asarray(times), np.asarray(events)
    unique = np.unique(t)
    x, y = [0], [1]
    surv = 1
    for time in unique:
        risk = np.sum(t >= time)
        d = np.sum((t == time) & (e == 1))
        if risk and d:
            surv *= 1 - d / risk
        x.append(time)
        y.append(surv)
    return np.asarray(x), np.asarray(y)


def logrank(t1, e1, t2, e2):
    times = np.unique(np.r_[t1, t2])
    o1 = exp1 = var = 0.0
    for t in times:
        r1, r2 = np.sum(t1 >= t), np.sum(t2 >= t)
        n = r1 + r2
        if n <= 1:
            continue
        d1 = np.sum((t1 == t) & (e1 == 1))
        d2 = np.sum((t2 == t) & (e2 == 1))
        d = d1 + d2
        o1 += d1
        exp1 += d * r1 / n
        var += d * r1 * r2 * (n - d) / (n * n * (n - 1))
    z = (o1 - exp1) ** 2 / var if var else 0
    return chi2.sf(z, 1)


def net_benefit(y, prob, threshold):
    y, prob = np.asarray(y), np.asarray(prob)
    pred = prob >= threshold
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    return tp / len(y) - fp / len(y) * threshold / (1 - threshold)


def api_metric(record, task):
    m = record["metrics"]["metrics"]
    if task == 1:
        return m["f1_positive"]
    if task == 2:
        return m["f1_weighted"]
    return m["harrell_c_index"]


def api_display_name(alias):
    return {
        "deepseek-flash": "DeepSeek Flash",
        "deepseek-v4-pro": "DeepSeek V4 Pro",
        "glm-legacy-air": "GLM 4.5 Air",
    }.get(alias, alias)


def overview_figure(dataset, outdir):
    fig = plt.figure(figsize=(7.2, 4.3), constrained_layout=True)
    gs = GridSpec(2, 4, figure=fig)

    ax = fig.add_subplot(gs[:, 0])
    ax.axis("off")
    ax.set_title("Clinical trajectory", pad=12)
    stages = [
        ("Pre-biopsy", "MRI + PSA\nBiopsy decision", C["blue"]),
        ("Post-biopsy", "MRI + pathology\nManagement", C["green"]),
        ("Post-surgery", "WSI + follow-up\nRecurrence risk", C["orange"]),
    ]
    for i, (title, text, color) in enumerate(stages):
        y = 0.75 - i * 0.33
        ax.add_patch(FancyBboxPatch((0.04, y), 0.92, 0.20, boxstyle="round,pad=0.02", facecolor="white", edgecolor=color, lw=0.8))
        ax.text(0.5, y + 0.135, title, ha="center", va="center", fontweight="bold", fontsize=6.3)
        ax.text(0.5, y + 0.065, text, ha="center", va="center", fontsize=5.6)
        if i < 2:
            ax.add_patch(FancyArrowPatch((0.5, y - 0.025), (0.5, y - 0.10), arrowstyle="-|>", mutation_scale=7, color=C["grey"]))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    label(ax, "a", x=-0.08, y=1.03)

    ax = fig.add_subplot(gs[0, 1])
    totals = [91, 72, 75]
    yes = [56, 31, 19]
    no = [35, 41, 56]
    x = np.arange(3)
    ax.bar(x, yes, color=C["red"], label="Positive/action")
    ax.bar(x, no, bottom=yes, color=C["grey"], label="Negative/other")
    ax.set_xticks(x, ["Task 1", "Task 2", "Task 3"])
    ax.set_ylabel("Labeled cases")
    ax.set_title("Endpoint distribution")
    ax.legend(frameon=False, ncols=2, loc="lower right")
    for i, n in enumerate(totals):
        ax.text(i, n + 2, str(n), ha="center", fontsize=5.5)
    ax.set_ylim(0, 105)
    label(ax, "b", x=-0.18, y=1.08)

    ax = fig.add_subplot(gs[1, 1])
    matrix = []
    labels = ["MRI", "Biopsy", "Prostatectomy"]
    for task in (1, 2, 3):
        cases = dataset.cases(task, labeled_only=True)
        matrix.append([
            sum(bool(c.neural.get("MRI image")) for c in cases),
            sum(bool(c.neural.get("Biopsy slide")) for c in cases),
            sum(bool(c.neural.get("Prostatectomy slide")) for c in cases),
        ])
    matrix = np.asarray(matrix)
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=91)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{matrix[i,j]}/{totals[i]}", ha="center", va="center", fontsize=5.1)
    ax.set_xticks(range(3), ["MRI", "Bx WSI", "Surgery"])
    ax.set_yticks(range(3), ["Task 1", "Task 2", "Task 3"])
    ax.set_title("Available modalities")
    ax.grid(False)
    label(ax, "c", x=-0.18, y=1.08)

    ax = fig.add_subplot(gs[:, 2:])
    ax.axis("off")
    ax.set_title("GARF-M3: guideline-anchored residual multimodal agent", pad=16)
    boxes = [
        (0.04, 0.63, "Inputs", "Structured prompt\nClinical reports\nMRI/WSI sets", C["sky"]),
        (0.28, 0.63, "Canonical layer", "Missingness masks\nPSA trajectory\nPathology concepts", C["sky"]),
        (0.52, 0.68, "Guideline prior", "EAU/NCCN π_rule", C["green"]),
        (0.52, 0.46, "Residual learner", "Gated Δ(x)", C["orange"]),
        (0.74, 0.57, "Fusion", "logit π_rule\n+ g(x)Δ(x)", C["purple"]),
        (0.89, 0.68, "Decision", "Biopsy / management\nBCR risk", C["blue"]),
        (0.89, 0.42, "Reasoning", "Confidence\nWeights / tools\nExplanation", C["blue"]),
    ]
    for x, y, title, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 0.14, 0.17, boxstyle="round,pad=0.008", facecolor="white", edgecolor=color, lw=0.8))
        ax.text(x + 0.07, y + 0.115, title, ha="center", fontsize=6.0, fontweight="bold")
        ax.text(x + 0.07, y + 0.055, text, ha="center", fontsize=5.2)
    arrows = [((0.18,0.71),(0.28,0.71)), ((0.42,0.71),(0.52,0.75)), ((0.42,0.60),(0.52,0.54)), ((0.66,0.75),(0.74,0.65)), ((0.66,0.54),(0.74,0.60)), ((0.88,0.65),(0.89,0.72)), ((0.88,0.63),(0.89,0.52))]
    for a,b in arrows:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=6,color=C["grey"],lw=.6))
    ax.text(0.5, 0.22, "Unlabeled cases:\nmasked-modality consistency\n+ conservative pseudo-labels", ha="center", va="top", fontsize=5.4, color=C["blue"])
    ax.add_patch(FancyArrowPatch((0.5,0.40),(0.5,0.46),arrowstyle="-|>",mutation_scale=6,color=C["blue"],lw=.6))
    ax.set_xlim(0,1.05); ax.set_ylim(0.12,0.90)
    label(ax, "d", x=-0.04, y=1.02)
    save(fig, Path(outdir), "fig0_overview_architecture")


def task1_figure(dataset, local, neural, api, outdir):
    cases = dataset.cases(1, labeled_only=True)
    p = preds(local, 1)
    pn = preds(neural, 1)
    y = [x["true"] for x in p]
    ybin = np.array([v == "yes" for v in y], int)
    prob = np.array([x["probability"][1] for x in p])
    pred = [x["predicted"] for x in p]
    base = [task1_rule(c) for c in cases]

    fig = plt.figure(figsize=(7.2, 5.1), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    values = [y.count("no"), y.count("yes")]
    ax.bar(["No biopsy", "Biopsy"], values, color=[C["grey"], C["red"]], width=.58)
    for i,v in enumerate(values): ax.text(i,v+1,str(v),ha="center")
    ax.set_ylabel("Cases"); ax.set_ylim(0,68); ax.set_title("Task 1 endpoint")
    label(ax,"a",x=-.18,y=1.10)

    ax = fig.add_subplot(gs[0, 1])
    fpr,tpr,_=roc_curve(ybin,prob)
    aucv,lo,hi=bootstrap_ci(ybin,prob,"auc",1000)
    ax.plot(fpr,tpr,color=C["blue"],lw=1.1)
    ax.plot([0,1],[0,1],"--",color=C["grey"],lw=.6)
    ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity"); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    ax.set_title("Discrimination")
    ax.text(.32,.10,f"AUC {aucv:.2f}\n95% CI {lo:.2f}–{hi:.2f}",fontsize=5.5)
    label(ax,"b",x=-.20,y=1.10)

    ax = fig.add_subplot(gs[0, 2])
    pt, plo, phi = bootstrap_ci(ybin, pred, "f1", 1000)
    frac, mean = calibration_curve(ybin, prob, n_bins=5, strategy="quantile")
    ax.plot(mean,frac,"o-",color=C["orange"],lw=1,ms=2.5)
    ax.plot([0,1],[0,1],"--",color=C["grey"],lw=.6)
    ax.set_xlabel("Predicted P(biopsy)"); ax.set_ylabel("Observed rate"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title("Probability calibration")
    label(ax,"c",x=-.20,y=1.10)

    ax = fig.add_subplot(gs[1, 0])
    thresholds=np.linspace(.05,.95,50)
    model=[net_benefit(ybin,prob,t) for t in thresholds]
    allv=[ybin.mean()-(1-ybin.mean())*t/(1-t) for t in thresholds]
    ax.plot(thresholds,model,color=C["blue"],label="GARF-M3")
    ax.plot(thresholds,allv,"--",color=C["grey"],label="Biopsy all")
    ax.axhline(0,ls=":",color=C["black"],label="Biopsy none")
    ax.set_xlim(.05,.95); ax.set_ylim(-.12,.45); ax.set_xlabel("Threshold"); ax.set_ylabel("Net benefit"); ax.set_title("Clinical utility")
    ax.legend(frameon=False,fontsize=5.0)
    label(ax,"d",x=-.20,y=1.10)

    ax=fig.add_subplot(gs[1,1])
    yt=np.asarray([v=="yes" for v in y],int); yp=np.asarray([v=="yes" for v in pred],int)
    counts=[
        int(np.sum((yt==0)&(yp==0))),
        int(np.sum((yt==0)&(yp==1))),
        int(np.sum((yt==1)&(yp==0))),
        int(np.sum((yt==1)&(yp==1))),
    ]
    names=["True no","False positive","False negative","True yes"]
    colors=[C["grey"],C["orange"],C["red"],C["blue"]]
    ax.barh(np.arange(4),counts,color=colors,height=.60)
    ax.set_yticks(np.arange(4),names); ax.invert_yaxis(); ax.set_xlim(0,60); ax.set_xlabel("Cases")
    ax.set_title("Clinical error profile")
    for i,v in enumerate(counts): ax.text(v+1,i,str(v),va="center",fontsize=5.4)
    label(ax,"e",x=-.24,y=1.10)

    ax=fig.add_subplot(gs[1,2])
    aliases=["deepseek-flash","deepseek-v4-pro","glm-legacy-air"]
    display=[api_display_name(a) for a in aliases]
    vals=[]; local_vals=[]
    for a in aliases:
        rec=api[(a,1)]
        ids={x["case_id"] for x in rec["predictions"]}
        lp=[x for x in p if x["case_id"] in ids]
        vals.append(api_metric(rec,1)); local_vals.append(metric_task1([x["true"] for x in lp],[x["predicted"] for x in lp])["positive_f1"])
    x=np.arange(3); w=.34
    ax.bar(x-w/2,local_vals,w,color=C["blue"],label="Local paired")
    ax.bar(x+w/2,vals,w,color=C["sky"],label="API pilot")
    ax.set_xticks(x,display,rotation=20,ha="right"); ax.set_ylim(0,1.05); ax.set_ylabel("Positive F1"); ax.set_title("Pilot comparison (n=5)")
    ax.legend(frameon=False,fontsize=5.0)
    label(ax,"f",x=-.22,y=1.10)
    save(fig,Path(outdir),"fig1_task1_biopsy_decision")


def task2_figure(dataset, local, neural, api, outdir):
    cases=dataset.cases(2,labeled_only=True)
    p=preds(local,2); pn=preds(neural,2)
    y=[x["true"] for x in p]; pred=[x["predicted"] for x in p]; base=[task2_rule(c) for c in cases]
    labels=["active_surveillance","active_treatment","continued_surveillance","watchful_waiting"]
    short=["AS","Treat","Cont.","WW"]
    fig=plt.figure(figsize=(7.2,5.1),constrained_layout=True); gs=GridSpec(2,3,figure=fig)

    ax=fig.add_subplot(gs[0,0]); counts=[y.count(v) for v in labels]
    ax.bar(short,counts,color=[C["green"],C["red"],C["blue"],C["purple"]],width=.62)
    for i,v in enumerate(counts): ax.text(i,v+.5,str(v),ha="center")
    ax.set_ylabel("Cases"); ax.set_ylim(0,37); ax.set_title("Management classes"); label(ax,"a",x=-.20,y=1.10)

    ax=fig.add_subplot(gs[0,1]); cm=confusion_matrix(y,pred,labels=labels); im=ax.imshow(cm,cmap="Blues")
    for i in range(4):
        for j in range(4): ax.text(j,i,cm[i,j],ha="center",va="center",fontsize=5.2,color="white" if cm[i,j]>cm.max()*.6 else C["black"])
    ax.set_xticks(range(4),short); ax.set_yticks(range(4),short); ax.set_xlabel("Predicted"); ax.set_ylabel("Guideline"); ax.set_title("Confusion matrix"); ax.grid(False); label(ax,"b",x=-.20,y=1.10)

    ax=fig.add_subplot(gs[0,2]); tp=np.diag(cm); fp=cm.sum(0)-tp; fn=cm.sum(1)-tp; f1=2*tp/(2*tp+fp+fn)
    ax.bar(short,f1,color=[C["green"],C["red"],C["blue"],C["purple"]],width=.62); ax.set_ylim(0,1.12); ax.set_ylabel("F1")
    for i,(v,s) in enumerate(zip(f1,[int(x) for x in cm.sum(1)])): ax.text(i,v+.02,f"{v:.2f}\nn={s}",ha="center",fontsize=5.0)
    ax.set_title("Class-level safety"); label(ax,"c",x=-.20,y=1.10)

    ax=fig.add_subplot(gs[1,0]); metrics=["accuracy","weighted_f1","macro_f1"]; metric_names=["Accuracy","Weighted F1","Macro F1"]
    rows=[[metric_task2(y,base)[m] for m in metrics],[metric_task2(y,pred)[m] for m in metrics],[metric_task2(y,[x["predicted"] for x in pn])[m] for m in metrics]]
    x=np.arange(3); w=.25
    ax.bar(x-w,rows[0],w,color=C["grey"],label="ISUP rule")
    ax.bar(x,rows[1],w,color=C["blue"],label="GARF-M3")
    ax.bar(x+w,rows[2],w,color=C["purple"],label="+ neural/semi")
    ax.set_xticks(x,metric_names); ax.set_ylim(0,1.05); ax.set_ylabel("Score"); ax.set_title("Local OOF comparison"); ax.legend(frameon=False,fontsize=5.0); label(ax,"d",x=-.20,y=1.10)

    # Correctness matrix on the API pilot subset. No aggregate is shown because the 5-case subset lacks two classes.
    ax=fig.add_subplot(gs[1,1:]); aliases=["deepseek-flash","deepseek-v4-pro","glm-legacy-air"]
    first=api[(aliases[0],2)]["predictions"]
    case_ids=[x["case_id"] for x in first]; truth={x["case_id"]:x["decision"] for x in first}
    rows=[]; rowlabels=[]
    local_by={x["case_id"]:x for x in p}
    rows.append([local_by[c]["predicted"]==truth[c] for c in case_ids]); rowlabels.append("Local GARF-M3")
    for a in aliases:
        by={x["case_id"]:x["decision"] for x in api[(a,2)]["predictions"]}
        rows.append([by[c]==truth[c] for c in case_ids]); rowlabels.append(api_display_name(a))
    matrix=np.asarray(rows,dtype=int)
    ax.imshow(matrix,cmap=mpl.colors.ListedColormap([C["light"],C["green"]]),aspect="auto",vmin=0,vmax=1)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j,i,"1" if matrix[i,j] else "0",ha="center",va="center",fontsize=6,color=C["black"] if matrix[i,j] else C["red"])
    short_truth={"active_surveillance":"AS","active_treatment":"Treat","continued_surveillance":"Cont.","watchful_waiting":"WW"}
    ax.set_xticks(range(len(case_ids)),[f"Case {i+1}\n{short_truth[truth[c]]}" for i,c in enumerate(case_ids)],fontsize=5.4)
    ax.set_yticks(range(len(rowlabels)),rowlabels); ax.set_title("Pilot correctness (n=5; no Cont./WW)"); ax.grid(False); label(ax,"e",x=-.12,y=1.10)
    save(fig,Path(outdir),"fig2_task2_management")


def task3_figure(local,neural,api,outdir):
    p=preds(local,3); pn=preds(neural,3); df=pd.DataFrame(p); dfn=pd.DataFrame(pn)
    fig=plt.figure(figsize=(7.2,5.1),constrained_layout=True); gs=GridSpec(2,3,figure=fig)
    ax=fig.add_subplot(gs[0,0])
    for ev,color in [(0,C["grey"]),(1,C["orange"])]:
        sub=df[df.true_event==ev]; ax.scatter(sub.true_months,np.full(len(sub),ev),s=12,color=color,label=f"event={ev}")
    ax.set_xlabel("Follow-up months"); ax.set_ylabel("BCR event"); ax.set_yticks([0,1],["No","Yes"]); ax.set_title("Outcome distribution"); ax.legend(frameon=False); label(ax,"a",x=-.20,y=1.10)

    ax=fig.add_subplot(gs[0,1]); df["group"]=pd.qcut(df.risk,3,labels=["Low","Int.","High"])
    colors=[C["blue"],C["orange"],C["red"]]
    for gr,color in zip(["Low","Int.","High"],colors):
        sub=df[df.group==gr]; x,y=km(sub.true_months,sub.true_event); ax.step(x,y*100,where="post",color=color,lw=1.0,label=gr)
    low=df[df.group=="Low"]; high=df[df.group=="High"]; pv=logrank(low.true_months.to_numpy(),low.true_event.to_numpy(),high.true_months.to_numpy(),high.true_event.to_numpy())
    ax.set_xlabel("Months after surgery"); ax.set_ylabel("Recurrence-free survival (%)"); ax.set_ylim(0,105); ax.set_xlim(0,115); ax.legend(frameon=False,title="Risk",ncols=3,loc="lower left"); ax.set_title(f"Risk tertiles; P={pv:.1g}"); label(ax,"b",x=-.20,y=1.10)

    ax=fig.add_subplot(gs[0,2]); base_c,base_lo,base_hi=cindex_bootstrap(df.true_months,df.true_event,df.risk); neuc,neulo,neuhi=cindex_bootstrap(dfn.true_months,dfn.true_event,dfn.risk)
    bars=ax.bar([0,1],[base_c,neuc],color=[C["grey"],C["purple"]],width=.55)
    ax.errorbar([0,1],[base_c,neuc],yerr=[[base_c-base_lo,neuc-neulo],[base_hi-base_c,neuhi-neuc]],fmt="none",color=C["black"],capsize=1.6,lw=.5)
    ax.set_xticks([0,1],["Clinical\n+ pathology","+ neural\n+ semi"]); ax.set_ylim(.45,.95); ax.set_ylabel("Harrell C-index"); ax.set_title("Local OOF comparison")
    for i,v in enumerate([base_c,neuc]): ax.text(i,v+.025,f"{v:.3f}",ha="center",fontsize=5.4)
    label(ax,"c",x=-.20,y=1.10)

    ax=fig.add_subplot(gs[1,0]); df["q"]=pd.qcut(df.risk,4,labels=["Q1","Q2","Q3","Q4"]); rates=df.groupby("q",observed=True).true_event.mean()*100
    ax.bar(rates.index.astype(str),rates.values,color=C["orange"],width=.62); ax.set_ylim(0,58); ax.set_xlabel("Risk quartile"); ax.set_ylabel("Events (%)"); ax.set_title("Observed BCR rate")
    for i,v in enumerate(rates.values): ax.text(i,v+1,f"{v:.0f}%",ha="center",fontsize=5.3)
    label(ax,"d",x=-.20,y=1.10)

    ax=fig.add_subplot(gs[1,1:]); aliases=["deepseek-flash","deepseek-v4-pro","glm-legacy-air"]
    first=api[(aliases[0],3)]["predictions"]; case_ids=[x["case_id"] for x in first]
    truthmap={x["case_id"]:(x["decision"]["months_to_recurrence"],x["decision"]["event"]) for x in first}
    localmap={x["case_id"]:x for x in p}
    offsets={"Observed":0,"Local":.16,"DeepSeek Flash":.32,"DeepSeek V4 Pro":.48,"GLM 4.5 Air":.64}
    styles={
        "Observed":{"color":C["black"],"marker":"|","linewidth":.9},
        "Local":{"color":C["blue"],"marker":"o"},
        "DeepSeek Flash":{"color":C["green"],"marker":"s"},
        "DeepSeek V4 Pro":{"color":C["orange"],"marker":"^"},
        "GLM 4.5 Air":{"color":C["purple"],"marker":"D"},
    }
    for j,cid in enumerate(case_ids):
        items=[("Observed",truthmap[cid][1]),("Local",localmap[cid]["risk"])]
        for a in aliases:
            rec=next(x for x in api[(a,3)]["predictions"] if x["case_id"]==cid); items.append((api_display_name(a),rec["decision"]["risk"]))
        for name,val in items:
            s=styles[name]
            if name == "Observed":
                ax.axhline(j+offsets[name],color=C["light"],lw=.35,zorder=0)
                ax.plot([val],[j+offsets[name]],marker=s["marker"],color=s["color"],markersize=6,markeredgewidth=s["linewidth"],markerfacecolor="none")
            else:
                ax.scatter([val],[j+offsets[name]],s=13,marker=s["marker"],color=s["color"],edgecolor="white",linewidth=.25,zorder=3)
    ax.set_yticks(np.arange(len(case_ids)),[f"Case {i+1}\nevent={truthmap[case_ids[i]][1]}" for i in range(len(case_ids))],fontsize=5.2)
    ax.set_xlabel("Reported risk score"); ax.set_title("Pilot risk profiles (n=5; one event)")
    ax.grid(axis="x",alpha=.18)
    ax.set_xlim(-0.65, 1.85)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5])
    handles=[plt.Line2D([],[],ls="none",marker=styles[n]["marker"],color=styles[n]["color"],ms=3.2,label=n) for n in styles]
    ax.legend(handles=handles,frameon=False,ncols=3,loc="upper right",fontsize=4.6)
    label(ax,"e",x=-.12,y=1.10)
    save(fig,Path(outdir),"fig3_task3_recurrence")


def comparison_figure(comp_path, outdir):
    frame = pd.read_csv(comp_path)
    fig = plt.figure(figsize=(7.2, 2.7), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig)
    specs = [
        (1, "Task 1: biopsy decision", "Positive F1"),
        (2, "Task 2: management", "Weighted F1"),
        (3, "Task 3: recurrence", "Harrell C-index"),
    ]
    for col, (task, title, xlabel) in enumerate(specs):
        ax = fig.add_subplot(gs[col])
        sub = frame[(frame.task == task) & (frame.scope != "excluded")].copy()
        # Local full-cohort results first, followed by API pilot results.
        sub["is_local"] = sub.scope.str.startswith("full")
        sub = sub.sort_values(["is_local", "model"], ascending=[False, True]).iloc[::-1]
        y = np.arange(len(sub))
        colors = [C["blue"] if local else C["sky"] for local in sub.is_local]
        bars = ax.barh(y, sub.value, color=colors, height=.62)
        for bar, (_, row) in zip(bars, sub.iterrows()):
            if not row.is_local:
                bar.set_hatch("//")
                bar.set_edgecolor("white")
                if not row.interpretable:
                    bar.set_alpha(.45)
            ax.text(min(row.value + .015, .88), bar.get_y() + bar.get_height()/2, f"{row.value:.3f} (n={row.n})", va="center", fontsize=4.7)
        ax.set_yticks(y, sub.model, fontsize=5.0)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="x", alpha=.18)
        label(ax, "abc"[col], x=-.30, y=1.14)
    handles = [
        plt.Rectangle((0,0),1,1,color=C["blue"],label="Local full OOF"),
        plt.Rectangle((0,0),1,1,facecolor=C["sky"],hatch="//",edgecolor="white",label="API pilot (n=5)"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=5.0, loc="outside lower center", ncols=2)
    save(fig, Path(outdir), "fig4_task_model_comparison")


def main():
    args=parse_args(); style()
    dataset=ChimeraDataset(args.data_root); local=read_json(args.local); neural=read_json(args.local_neural); api=load_api(args.api_dir)
    out=Path(args.output_dir)
    overview_figure(dataset,out)
    task1_figure(dataset,local,neural,api,out)
    task2_figure(dataset,local,neural,api,out)
    task3_figure(local,neural,api,out)
    comparison_figure(args.comparison,out)
    write_layout_report(out)
    print("Wrote task figures to",out)


if __name__=="__main__": main()
