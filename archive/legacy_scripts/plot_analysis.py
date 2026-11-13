from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chimera.utils.jsonio import read_json, write_json


TASK_LABELS = {
    1: ["no", "yes"],
    2: ["active_surveillance", "active_treatment", "continued_surveillance", "watchful_waiting"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--local", required=True, help="local_eval.json")
    p.add_argument("--local-neural", default=None, help="local_eval_neural_semi.json")
    p.add_argument("--api-glob", default=None, help="Glob for api_*_task*.json files")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def collect_results(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = []
    local = read_json(args.local)
    if local:
        records.append({"name": "Local GARF-M3", "kind": "local", "task1": local.get("task1"), "task2": local.get("task2"), "task3": local.get("task3")})
    if args.local_neural:
        neural = read_json(args.local_neural)
        if neural:
            records.append({"name": "Local GARF-M3 + neural/semi", "kind": "local_neural", "task1": neural.get("task1"), "task2": neural.get("task2"), "task3": neural.get("task3")})
    if args.api_glob:
        for path in Path().glob(args.api_glob):
            data = read_json(path)
            if not data:
                continue
            model = data.get("configuration", {}).get("model_alias", path.stem)
            provider = data.get("configuration", {}).get("provider", "api")
            task = data.get("configuration", {}).get("task")
            if data.get("metrics", {}).get("error"):
                continue
            records.append({
                "name": f"{provider}/{model}",
                "kind": "api",
                "task": task,
                "task1": data if task == 1 else None,
                "task2": data if task == 2 else None,
                "task3": data if task == 3 else None,
                "api_data": data,
            })
    return records


def primary_metric_from_run(run: dict[str, Any], task: int) -> float | None:
    metrics = run.get("metrics", {})
    if isinstance(metrics, dict) and isinstance(metrics.get("metrics"), dict):
        metrics = metrics["metrics"]
    if task == 1:
        return metrics.get("f1_positive")
    if task == 2:
        return metrics.get("f1_weighted")
    return metrics.get("harrell_c_index")


def metric_for(record: dict[str, Any], task: int) -> dict[str, Any] | None:
    value = record.get(f"task{task}")
    if record.get("kind") == "api":
        return record.get("api_data") if record.get("task") == task else None
    return value


def extract_run(value: dict[str, Any]) -> dict[str, Any]:
    if "model_runs" in value and value["model_runs"]:
        return value["model_runs"][0]
    if "metrics" in value and isinstance(value["metrics"], dict) and "metrics" in value["metrics"]:
        return value["metrics"]
    return value


def plot_confusion(ax, matrix: list[list[int]], labels: list[str], title: str) -> None:
    arr = np.asarray(matrix, dtype=int)
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", color="white" if arr[i, j] > arr.max() / 2 else "black")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def flowchart(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    boxes = [
        ("Dataset\n423 cases", 0.05, 0.68),
        ("Guideline rules\nTask 1/2/3", 0.26, 0.68),
        ("Clinical/pathology\nfeature extraction", 0.47, 0.68),
        ("Neural representation\nset statistics", 0.68, 0.68),
        ("Residual fusion\nGARF-M3", 0.47, 0.37),
        ("Reasoning generator\nconfidence/weights/trace", 0.68, 0.37),
        ("Local evaluator\nF1 / weighted F1 / C-index", 0.47, 0.11),
        ("API agents\nDeepSeek / GLM", 0.05, 0.37),
        ("Comparison & plots", 0.68, 0.11),
    ]
    for text, x, y in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.21, 0.15, fill=True, facecolor="#eef5ff", edgecolor="#3574b5"))
        ax.text(x + 0.105, y + 0.075, text, ha="center", va="center", fontsize=9)
    arrows = [
        ((0.26, 0.755), (0.26, 0.755)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "color": "#555"})
    ax.annotate("", xy=(0.26, 0.755), xytext=(0.26, 0.755))
    ax.annotate("", xy=(0.26, 0.755), xytext=(0.26, 0.755))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.92)
    ax.set_title("CHIMERA-Agent analysis architecture")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def analysis_flowchart(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    steps = [
        ("1. Data audit"),
        ("2. Rule baseline"),
        ("3. GARF-M3 training"),
        ("4. Neural + semi-supervised"),
        ("5. API agent benchmark"),
        ("6. Reasoning scoring"),
        ("7. Error analysis"),
        ("8. Visualization"),
    ]
    for i, step in enumerate(steps):
        x = 0.04 + i * 0.115
        ax.add_patch(plt.Rectangle((x, 0.42), 0.10, 0.20, facecolor="#f7f7f7", edgecolor="#666"))
        ax.text(x + 0.05, 0.52, step, ha="center", va="center", fontsize=9)
        if i:
            ax.annotate("", xy=(x, 0.52), xytext=(x - 0.015, 0.52), arrowprops={"arrowstyle": "->"})
    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, 0.75)
    ax.set_title("End-to-end analysis workflow")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = collect_results(args)
    if not records:
        raise ValueError("No result files found")

    flowchart(output_dir / "architecture.png")
    analysis_flowchart(output_dir / "analysis_workflow.png")

    summary_rows = []
    for record in records:
        for task in (1, 2, 3):
            value = metric_for(record, task)
            if not value:
                continue
            run = extract_run(value)
            metrics = run.get("metrics", {})
            if task == 1:
                summary_rows.append({
                    "model": record["name"], "task": task, "accuracy": metrics.get("accuracy"),
                    "primary_metric": metrics.get("f1_positive"), "metric_name": "positive_f1",
                    "case_score": run.get("case_score"),
                })
            elif task == 2:
                summary_rows.append({
                    "model": record["name"], "task": task, "accuracy": metrics.get("accuracy"),
                    "primary_metric": metrics.get("f1_weighted"), "metric_name": "weighted_f1",
                    "case_score": run.get("case_score"),
                })
            else:
                summary_rows.append({
                    "model": record["name"], "task": task, "accuracy": metrics.get("event_accuracy"),
                    "primary_metric": metrics.get("harrell_c_index"), "metric_name": "c_index",
                    "case_score": run.get("case_score"),
                })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "metric_summary.csv", index=False)

    # Fair paired comparison: use only the same case IDs sampled by each API run.
    paired_rows = []
    local_record = next((r for r in records if r["kind"] in {"local", "local_neural"}), None)
    if local_record:
        for record in records:
            if record["kind"] != "api" or not record.get("api_data"):
                continue
            api_data = record["api_data"]
            task = api_data.get("configuration", {}).get("task")
            api_predictions = api_data.get("predictions", [])
            if not api_predictions or task not in (1, 2, 3):
                continue
            case_ids = {p["case_id"] for p in api_predictions}
            local_predictions = local_record.get(f"task{task}", {}).get("predictions", [])
            matched = [p for p in local_predictions if p["case_id"] in case_ids]
            if not matched:
                continue
            if task in (1, 2):
                y_true = [p["true"] for p in matched]
                y_pred = [p["predicted"] for p in matched]
                if task == 1:
                    local_metric = float(pd.Series(y_true).eq(y_pred).mean())
                    # Positive F1 is more informative than accuracy.
                    tp = sum(t == "yes" and p == "yes" for t, p in zip(y_true, y_pred))
                    fp = sum(t == "no" and p == "yes" for t, p in zip(y_true, y_pred))
                    fn = sum(t == "yes" and p == "no" for t, p in zip(y_true, y_pred))
                    local_metric = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
                    metric_name = "positive_f1"
                else:
                    # weighted F1
                    labels = sorted(set(y_true) | set(y_pred))
                    supports = {label: y_true.count(label) for label in labels}
                    f1s = []
                    weights = []
                    for label in labels:
                        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
                        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
                        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
                        prec = tp / (tp + fp) if tp + fp else 0.0
                        rec = tp / (tp + fn) if tp + fn else 0.0
                        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
                        f1s.append(f1)
                        weights.append(supports[label])
                    local_metric = float(np.average(f1s, weights=weights))
                    metric_name = "weighted_f1"
            else:
                from chimera.evaluation.survival_metrics import concordance_index
                times = [float(p["true_months"]) for p in matched]
                events = [int(p["true_event"]) for p in matched]
                risks = [float(p["risk"]) for p in matched]
                local_metric = float(concordance_index(times, events, risks))
                metric_name = "c_index"
            api_metric = primary_metric_from_run(api_data, task)
            paired_rows.append({
                "task": task,
                "metric_name": metric_name,
                "n_paired_cases": len(matched),
                "local_primary_metric": local_metric,
                "api_primary_metric": api_metric,
                "api_model": record["name"],
            })
        if paired_rows:
            paired_df = pd.DataFrame(paired_rows)
            paired_df.to_csv(output_dir / "paired_metric_summary.csv", index=False)
            fig, ax = plt.subplots(figsize=(12, 5))
            labels = [f"Task {r['task']}\n{r['api_model']}" for r in paired_rows]
            x = np.arange(len(paired_rows))
            width = 0.38
            ax.bar(x - width / 2, paired_df["local_primary_metric"], width, label="Local GARF-M3")
            ax.bar(x + width / 2, paired_df["api_primary_metric"], width, label="API agent")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("Primary metric")
            ax.set_title("Paired local/API comparison on identical cases")
            ax.grid(axis="y", alpha=.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "paired_primary_metrics.png", dpi=220)
            plt.close(fig)

    # Grouped primary metric plot.
    fig, ax = plt.subplots(figsize=(12, 5))
    if not summary_df.empty:
        pivot = summary_df.pivot(index="model", columns="task", values="primary_metric")
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Primary metric")
    ax.set_title("Primary metric by model and task")
    ax.set_xlabel("Model")
    ax.set_ylabel("Primary metric")
    ax.grid(axis="y", alpha=.25)
    ax.legend(title="Task")
    fig.tight_layout()
    fig.savefig(output_dir / "primary_metrics.png", dpi=220)
    plt.close(fig)

    # Confusion matrices for the first local record.
    local = next((r for r in records if r["kind"] in {"local", "local_neural"}), None)
    if local:
        for task in (1, 2):
            value = local.get(f"task{task}")
            if not value:
                continue
            cm = extract_run(value).get("metrics", {}).get("confusion_matrix")
            if not cm:
                continue
            fig, ax = plt.subplots(figsize=(7, 6))
            plot_confusion(ax, cm, TASK_LABELS[task], f"Task {task} confusion matrix")
            fig.tight_layout()
            fig.savefig(output_dir / f"task{task}_confusion.png", dpi=220)
            plt.close(fig)

        # ROC for Task 1 if probabilities are present.
        task1 = local.get("task1")
        if task1 and task1.get("predictions"):
            from sklearn.metrics import roc_curve, roc_auc_score
            preds = task1["predictions"]
            y_true = [1 if p["true"] == "yes" else 0 for p in preds]
            probs = [float(p["probability"][1]) for p in preds if isinstance(p.get("probability"), list) and len(p["probability"]) > 1]
            if len(probs) == len(y_true) and len(set(y_true)) == 2:
                fpr, tpr, _ = roc_curve(y_true, probs)
                auc = roc_auc_score(y_true, probs)
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
                ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
                ax.set_xlabel("False positive rate")
                ax.set_ylabel("True positive rate")
                ax.set_title("Task 1 out-of-fold ROC")
                ax.legend()
                ax.grid(alpha=.25)
                fig.tight_layout()
                fig.savefig(output_dir / "task1_roc.png", dpi=220)
                plt.close(fig)

        # Task 3 risk/time scatter.
        task3 = local.get("task3")
        if task3 and task3.get("predictions"):
            frame = pd.DataFrame(task3["predictions"])
            fig, ax = plt.subplots(figsize=(8, 6))
            for event in (0, 1):
                sub = frame[frame["true_event"] == event]
                ax.scatter(sub["true_months"], sub["risk"], label=f"event={event}", alpha=.75)
            ax.set_xlabel("Observed months")
            ax.set_ylabel("Predicted risk")
            ax.set_title("Task 3 observed time vs predicted risk")
            ax.legend()
            ax.grid(alpha=.25)
            fig.tight_layout()
            fig.savefig(output_dir / "task3_risk_vs_time.png", dpi=220)
            plt.close(fig)

            frame["risk_quartile"] = pd.qcut(frame["risk"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
            event_rate = frame.groupby("risk_quartile", observed=True)["true_event"].mean()
            fig, ax = plt.subplots(figsize=(7, 5))
            event_rate.plot(kind="bar", ax=ax)
            ax.set_xlabel("Predicted-risk quartile")
            ax.set_ylabel("Observed event rate")
            ax.set_title("Task 3 event rate by predicted-risk quartile")
            ax.grid(axis="y", alpha=.25)
            fig.tight_layout()
            fig.savefig(output_dir / "task3_event_rate_by_risk.png", dpi=220)
            plt.close(fig)

    # API/local case-score comparison.
    if not summary_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        pivot = summary_df.pivot(index="model", columns="task", values="case_score")
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("Local case score comparison")
        ax.set_xlabel("Model")
        ax.set_ylabel("Local case score")
        ax.grid(axis="y", alpha=.25)
        ax.legend(title="Task")
        fig.tight_layout()
        fig.savefig(output_dir / "case_scores.png", dpi=220)
        plt.close(fig)

    # API latency and success/failure summary.
    api_rows = []
    for record in records:
        if record["kind"] != "api" or not record.get("api_data"):
            continue
        data = record["api_data"]
        task = data.get("configuration", {}).get("task")
        api_rows.append({
            "model": record["name"],
            "task": task,
            "elapsed_seconds": data.get("elapsed_seconds"),
            "successful_predictions": len(data.get("predictions", [])),
            "failures": len(data.get("failures", [])),
        })
    if api_rows:
        api_df = pd.DataFrame(api_rows)
        api_df.to_csv(output_dir / "api_runtime_summary.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        labels = [f"Task {r['task']}\n{r['model']}" for r in api_rows]
        x = np.arange(len(api_df))
        axes[0].bar(x, api_df["elapsed_seconds"], color="#5b8ff9")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=45, ha="right")
        axes[0].set_ylabel("Seconds")
        axes[0].set_title("API latency")
        axes[0].grid(axis="y", alpha=.25)
        axes[1].bar(x, api_df["successful_predictions"], label="successful", color="#5ad8a6")
        axes[1].bar(x, api_df["failures"], bottom=api_df["successful_predictions"], label="failed", color="#e8684a")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha="right")
        axes[1].set_ylabel("Cases")
        axes[1].set_title("API successful/failed calls")
        axes[1].legend()
        axes[1].grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(output_dir / "api_latency_success.png", dpi=220)
        plt.close(fig)

    # Reasoning component radar for tasks 1/2.
    labels = ["confidence", "variable_weights", "tool_sequence", "rationale"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "polar"})
    for idx, task in enumerate((1, 2)):
        ax = axes[idx - 1]
        for record in records:
            value = metric_for(record, task)
            if not value:
                continue
            reasoning = extract_run(value).get("reasoning", {})
            vals = [reasoning.get(label, 0) for label in labels]
            if not any(vals):
                continue
            vals += vals[:1]
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            angles += angles[:1]
            ax.plot(angles, vals, label=record["name"])
            ax.fill(angles, vals, alpha=.12)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title(f"Task {task} reasoning components")
    axes[0].legend(loc="upper left", bbox_to_anchor=(-.35, 1.10))
    fig.tight_layout()
    fig.savefig(output_dir / "reasoning_radar.png", dpi=220)
    plt.close(fig)

    write_json(output_dir / "plot_manifest.json", {
        "figures": [p.name for p in output_dir.glob("*.png")],
        "tables": [p.name for p in output_dir.glob("*.csv")],
        "summary_rows": summary_rows,
    })
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
