"""Supplementary benchmark figure: proposed model vs traditional ML/DL and LLM APIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.text as mpl_text
from matplotlib.gridspec import GridSpec
import numpy as np

from chimera.utils.jsonio import read_json


C = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "grey": "#7F7F7F",
    "light_grey": "#D9D9D9",
    "dark": "#1A1A1A",
}

TYPE_COLORS = {
    "Rule": C["grey"],
    "Clinical anchor": C["grey"],
    "Traditional ML": C["blue"],
    "Traditional survival": C["blue"],
    "Deep learning": C["purple"],
    "LLM API": C["orange"],
    "Proposed": C["vermillion"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=ROOT / "results" / "benchmark" / "benchmark_results.json")
    parser.add_argument("--llm", default=ROOT / "results" / "benchmark" / "llm_results.json")
    parser.add_argument("--output-dir", default=ROOT / "results" / "benchmark" / "figures")
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
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.45,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax, text: str, x: float = -0.16, y: float = 1.10) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10.0, fontweight="bold",
            va="bottom", ha="left", color=C["dark"])


def _value(record: dict) -> float | None:
    value = record.get("value")
    return float(value) if value is not None and np.isfinite(value) else None


def _interval(record: dict) -> tuple[float, float]:
    task = record["task"]
    if task == 3:
        low = record.get("low")
        high = record.get("high")
    else:
        key = record["metric"]
        low = record.get(f"{key}_low")
        high = record.get(f"{key}_high")
    value = _value(record)
    if low is None or high is None or value is None:
        return value, value
    return float(low), float(high)


def load_records(benchmark_path: Path, llm_path: Path) -> list[dict]:
    payload = read_json(benchmark_path)
    records = list(payload.get("results", []))
    if llm_path.exists():
        llm = read_json(llm_path)
        records.extend(llm.get("results", []))
    # Keep the first occurrence if a method appears twice.
    unique: dict[tuple[int, str], dict] = {}
    for record in records:
        key = (int(record["task"]), str(record["method"]))
        unique.setdefault(key, record)
    return list(unique.values())


def _short_type(method_type: str) -> str:
    return {
        "Rule": "Rule",
        "Clinical anchor": "Anchor",
        "Traditional ML": "ML",
        "Traditional survival": "Survival",
        "Deep learning": "DL",
        "LLM API": "LLM",
        "Proposed": "Proposed",
    }.get(method_type, method_type)


def plot_task(ax, records: list[dict], task: int, title: str, xlabel: str, xlim: tuple[float, float]) -> None:
    subset = [record for record in records if int(record["task"]) == task and _value(record) is not None]
    subset.sort(key=lambda record: _value(record))
    y_positions = np.arange(len(subset))
    for position, record in zip(y_positions, subset):
        value = _value(record)
        low, high = _interval(record)
        colour = TYPE_COLORS.get(record.get("type"), C["grey"])
        ax.plot([low, high], [position, position], color=colour, linewidth=1.2, alpha=0.8)
        ax.scatter([value], [position], s=34, color=colour, edgecolor="white", linewidth=0.5, zorder=4)
        ax.text(high + (xlim[1] - xlim[0]) * 0.010, position, f"{value:.3f}",
                va="center", fontsize=6.0, color=colour)
    ax.set_yticks(y_positions, [record["method"] for record in subset])
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)


def summary_table(ax, records: list[dict]) -> None:
    ax.axis("off")
    ax.set_title("Benchmark summary", loc="left")
    methods = sorted({record["method"] for record in records})
    tasks = [1, 2, 3]

    def find(method: str, task: int):
        for record in records:
            if record["method"] == method and int(record["task"]) == task:
                return record
        return None

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    headers = [("Method", 0.01, "left"), ("Type", 0.43, "left"),
               ("T1 F1+", 0.62, "center"), ("T2 wF1", 0.78, "center"), ("T3 C-idx", 0.94, "center")]
    for text, x, alignment in headers:
        ax.text(x, 0.96, text, transform=ax.transAxes, fontsize=5.4, fontweight="bold",
                ha=alignment, va="center")
    y = 0.89
    step = 0.89 / max(len(methods), 1)
    for method in methods:
        record_any = next(record for record in records if record["method"] == method)
        colour = TYPE_COLORS.get(record_any.get("type"), C["grey"])
        ax.text(0.01, y, method, transform=ax.transAxes, fontsize=4.5, color=colour,
                ha="left", va="center")
        ax.text(0.43, y, _short_type(record_any.get("type", "")), transform=ax.transAxes,
                fontsize=4.3, color=C["grey"], ha="left", va="center")
        for task, x in zip(tasks, (0.62, 0.78, 0.94)):
            record = find(method, task)
            value = _value(record) if record else None
            ax.text(x, y, "—" if value is None else f"{value:.3f}", transform=ax.transAxes,
                    fontsize=4.5, ha="center", va="center", color=C["dark"])
        y -= step
    ax.plot([0, 1], [0.93, 0.93], transform=ax.transAxes, color=C["light_grey"], linewidth=0.4)


def save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = []
    for artist in fig.findobj(mpl_text.Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        try:
            texts.append((artist.get_text(), artist.get_window_extent(renderer=renderer)))
        except Exception:
            continue
    collisions = 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if len(texts[i][0].strip()) <= 1 or len(texts[j][0].strip()) <= 1:
                continue
            a, b = texts[i][1], texts[j][1]
            dx = min(a.x1, b.x1) - max(a.x0, b.x0)
            dy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if dx > 0 and dy > 0:
                score = dx * dy / max(min(a.width * a.height, b.width * b.height), 1e-9)
                if score > 0.25:
                    collisions += 1
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"{name}.{suffix}")
    plt.close(fig)
    (outdir / "layout_qc.json").write_text(json.dumps({"text_overlap_count": collisions}, indent=2))


def main() -> None:
    args = parse_args()
    style()
    mpl.rcParams["savefig.dpi"] = args.dpi
    records = load_records(Path(args.benchmark), Path(args.llm))
    fig = plt.figure(figsize=(7.4, 5.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.50)

    ax = fig.add_subplot(gs[0, 0])
    plot_task(ax, records, 1, "Task 1 — biopsy decision", "Positive F1 (95% CI)", (0.0, 1.10))
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    plot_task(ax, records, 2, "Task 2 — management decision", "Weighted F1 (95% CI)", (0.0, 1.10))
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    plot_task(ax, records, 3, "Task 3 — recurrence risk", "Harrell C-index (95% CI)", (0.0, 1.05))
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    summary_table(ax, records)
    panel_label(ax, "d", x=-0.10)

    save(fig, Path(args.output_dir), "fig6_benchmark_comparison")
    print(f"Wrote {Path(args.output_dir) / 'fig6_benchmark_comparison.png'}")


if __name__ == "__main__":
    main()
