"""Matplotlib figures for the training-results tab. Reads only the numbers the
AL pipeline already recorded (via demo.engine); never re-measures anything."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from demo import engine


def render(fig: Figure) -> np.ndarray:
    """Rasterise a figure to an RGB array for a static gr.Image."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()

# strategy -> (display label, line colour)
_STYLE = {
    "random":      ("S0 Random",      "#94a3b8"),
    "uncertainty": ("S1 Uncertainty", "#2563eb"),
    "coreset":     ("S2 CoreSet",     "#f59e0b"),
    "ppal":        ("S3 PPAL",        "#16a34a"),
}
_ORDER = ("random", "uncertainty", "coreset", "ppal")
_INK, _MUTED, _GRID = "#0f172a", "#64748b", "#e2e8f0"


def _style_axes(ax) -> None:
    ax.grid(True, color=_GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=9)


def _per_round(rows: list[dict], strategy: str) -> list[tuple[float, float, float]]:
    """-> sorted [(data_pct, mean_test_mAP50, std)] over seeds for one strategy."""
    buckets: dict[int, dict] = {}
    for r in rows:
        if r["strategy"] != strategy or r.get("test_mAP50") is None:
            continue
        b = buckets.setdefault(r["round"], {"frac": r["frac"] or 0.0, "vals": []})
        b["vals"].append(r["test_mAP50"])
    out = []
    for rnd in sorted(buckets):
        vals = buckets[rnd]["vals"]
        mean = sum(vals) / len(vals)
        std = float(np.std(vals)) if len(vals) > 1 else 0.0
        out.append((buckets[rnd]["frac"] * 100, mean, std))
    return out


def comparison_curve() -> Figure:
    rows, base = engine.al_results(), engine.baseline_summary()
    fig = Figure(figsize=(5.8, 3.7), dpi=100)
    fig.set_facecolor("white")
    ax = fig.add_subplot(111)
    for strat in _ORDER:
        pts = _per_round(rows, strat)
        if not pts:
            continue
        label, color = _STYLE[strat]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        lo = [p[1] - p[2] for p in pts]
        hi = [p[1] + p[2] for p in pts]
        ax.plot(xs, ys, marker="o", ms=4, lw=1.9, color=color, label=label)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.12, lw=0)
    if base:
        ax.axhline(base["test_mAP50"], ls="--", lw=1.4, color=_INK,
                   label=f"Oracle 100% ({base['test_mAP50']:.3f})")
    ax.set_xlabel("% dữ liệu gán nhãn", fontsize=10, color=_MUTED)
    ax.set_ylabel("test mAP@50", fontsize=10, color=_MUTED)
    ax.set_title("Hiệu năng theo lượng dữ liệu (trung bình ± std, 3 seed)",
                 fontsize=11, color=_INK)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def per_class_bars() -> Figure:
    rows, base = engine.al_results(), engine.baseline_summary()
    cap = max((r["round"] for r in rows), default=6)  # final round = 20% cap
    classes = engine.CLASSES
    fig = Figure(figsize=(5.8, 3.7), dpi=100)
    fig.set_facecolor("white")
    ax = fig.add_subplot(111)
    series = list(_ORDER) + (["baseline"] if base else [])
    width = 0.82 / len(series)
    x = np.arange(len(classes))
    for i, strat in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        if strat == "baseline":
            vals = [base["per_class"].get(c, 0.0) for c in classes]
            ax.bar(x + offset, vals, width, label="Oracle 100%", color=_INK)
            continue
        label, color = _STYLE[strat]
        vals = []
        for c in classes:
            v = [r[c] for r in rows
                 if r["strategy"] == strat and r["round"] == cap and r.get(c) is not None]
            vals.append(sum(v) / len(v) if v else 0.0)
        ax.bar(x + offset, vals, width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylabel("AP@50 (test)", fontsize=10, color=_MUTED)
    ax.set_title("AP@50 theo lớp tại vòng cuối (~20% dữ liệu)", fontsize=11, color=_INK)
    ax.legend(fontsize=8, ncol=2, loc="lower right", framealpha=0.9)
    _style_axes(ax)
    fig.tight_layout()
    return fig
