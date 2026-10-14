"""Shared figure and document helpers, so every model reports the same way.

Figures are written as 200 DPI PNGs sized to sit inside a Word page without
rescaling. Documents are .docx so findings can be pasted straight into the
report.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from docx import Document  # noqa: E402
from docx.shared import Inches, Pt, RGBColor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"
DOCS = ROOT / "docs"
DPI = 200

# one palette across every figure so the report looks consistent
INK = "#17201f"
MUTED = "#5d6b68"
ACCENT = "#14505c"
GOOD = "#1f7a4d"
WARN = "#c08a1f"
BAD = "#a8442e"
GRID = "#e1e5e2"


def _style(ax) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.title.set_color(INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   figure: {path.relative_to(ROOT)}")
    return path


def grouped_bars(title, ylabel, group_labels, series, name, percent=False,
                 annotate=True):
    """series: list of (label, values, colour), one value per group."""
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    n = len(series)
    width = 0.8 / n
    positions = range(len(group_labels))
    for index, (label, values, colour) in enumerate(series):
        offset = (index - (n - 1) / 2) * width
        bars = ax.bar([p + offset for p in positions], values, width * 0.92,
                      label=label, color=colour)
        if annotate:
            for bar, value in zip(bars, values):
                text = f"{value:.1%}" if percent else f"{value:.3f}"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.015 if percent else 0.012),
                        text, ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=12)
    if percent:
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    else:
        # headroom so the tallest bar's label cannot collide with the legend
        tallest = max(max(values) for _label, values, _colour in series)
        ax.set_ylim(0, tallest * 1.35)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, ncols=n, loc="upper right")
    _style(ax)
    return save(fig, name)


def horizontal_bars(title, xlabel, labels, values, name, counts=None,
                    percent=True, threshold=None):
    """Horizontal bars, longest at the top, optionally annotated with counts."""
    order = sorted(range(len(labels)), key=lambda i: values[i])
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    counts = [counts[i] for i in order] if counts else None

    fig, ax = plt.subplots(figsize=(8.0, 0.36 * len(labels) + 1.4))
    colours = [GOOD if v >= 0.95 else WARN if v >= 0.8 else BAD for v in values]
    bars = ax.barh(labels, values, color=colours, height=0.68)
    for index, (bar, value) in enumerate(zip(bars, values)):
        text = f"{value:.1%}" if percent else f"{value:.3f}"
        if counts:
            text += f"   (n={counts[index]:,})"
        ax.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2,
                text, va="center", fontsize=8.5, color=MUTED)
    if threshold is not None:
        ax.axvline(threshold, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlim(0, 1.34 if percent else max(values) * 1.4)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=12)
    if percent:
        # a precision axis stops at 100%; the extra room is only for the labels
        ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
        ax.xaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    return save(fig, name)


# ----------------------------------------------------------------- documents

def _runs(paragraph, text: str) -> None:
    for index, chunk in enumerate(text.split("**")):
        if chunk:
            run = paragraph.add_run(chunk)
            run.bold = index % 2 == 1


def write_doc(filename: str, title: str, subtitle: str, blocks) -> Path:
    """blocks: (kind, payload) where kind is h1/h2/p/bullet/num/table/figure/code."""
    DOCS.mkdir(parents=True, exist_ok=True)
    document = Document()
    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)

    document.add_heading(title, level=0)
    if subtitle:
        paragraph = document.add_paragraph()
        run = paragraph.add_run(subtitle)
        run.italic = True
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    for kind, payload in blocks:
        if kind in ("h1", "h2", "h3"):
            document.add_heading(payload, level={"h1": 1, "h2": 2, "h3": 3}[kind])
        elif kind == "p":
            _runs(document.add_paragraph(), payload)
        elif kind == "bullet":
            _runs(document.add_paragraph(style="List Bullet"), payload)
        elif kind == "num":
            _runs(document.add_paragraph(style="List Number"), payload)
        elif kind == "code":
            paragraph = document.add_paragraph()
            run = paragraph.add_run(payload)
            run.font.name = "Consolas"
            run.font.size = Pt(8.5)
        elif kind == "figure":
            path, caption = payload
            document.add_picture(str(path), width=Inches(6.2))
            paragraph = document.add_paragraph()
            run = paragraph.add_run(caption)
            run.italic = True
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        elif kind == "table":
            rows = payload
            table = document.add_table(rows=len(rows), cols=len(rows[0]))
            table.style = "Light Grid Accent 1"
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    target = table.cell(r, c)
                    target.text = ""
                    _runs(target.paragraphs[0], str(cell))
                    for run in target.paragraphs[0].runs:
                        run.font.size = Pt(9)
                        if r == 0:
                            run.bold = True
            document.add_paragraph()
        else:
            raise ValueError(f"unknown block: {kind}")

    path = DOCS / filename
    document.save(str(path))
    print(f"   document: {path.relative_to(ROOT)}")
    return path


def per_class_bars(title, labels, values, name, counts=None, baseline=None):
    """One bar per category, worst at the bottom, so weak classes stand out."""
    return horizontal_bars(title, "F1 score", labels, values, name,
                           counts=counts, percent=False, threshold=baseline)


def confusion_heatmap(title, labels, matrix, name, normalise_rows=True):
    """Confusion matrix. Rows are the true category, columns the prediction."""
    import numpy as np

    data = np.array(matrix, dtype=float)
    if normalise_rows:
        totals = data.sum(axis=1, keepdims=True)
        data = np.divide(data, totals, out=np.zeros_like(data), where=totals != 0)

    size = 0.42 * len(labels) + 2.6
    fig, ax = plt.subplots(figsize=(size, size * 0.86))
    image = ax.imshow(data, cmap="BuGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("predicted", fontsize=9.5, color=MUTED)
    ax.set_ylabel("true category", fontsize=9.5, color=MUTED)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=14)

    # annotate only cells worth reading, so the figure stays legible
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = data[row, column]
            if value >= 0.04:
                ax.text(column, row, f"{value*100:.0f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if value > 0.55 else INK)
    bar = fig.colorbar(image, ax=ax, shrink=0.72)
    bar.set_label("share of the true category's rows (%)", fontsize=8.5, color=MUTED)
    bar.ax.tick_params(labelsize=8, colors=MUTED)
    bar.ax.yaxis.set_major_formatter(lambda v, _p: f"{v*100:.0f}")
    ax.tick_params(colors=MUTED)
    ax.grid(visible=False)
    for side in ax.spines.values():
        side.set_color(GRID)
    return save(fig, name)


def reliability_diagram(title, buckets, name, ece=None, subtitle=None):
    """Stated confidence against observed accuracy, with the ideal diagonal."""
    import numpy as np

    used = [b for b in buckets if b["n"] > 0]
    centres = [(b["lower"] + b["upper"]) / 2 for b in used]
    accuracy = [b["accuracy"] for b in used]
    confidence = [b["confidence"] for b in used]
    counts = [b["n"] for b in used]

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(6.6, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})

    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1,
            label="perfectly calibrated")
    ax.bar(centres, accuracy, width=0.085, color=ACCENT, label="observed accuracy")
    ax.plot(confidence, accuracy, "o-", color=BAD, markersize=4.5,
            linewidth=1.4, label="actual")
    ax.set_ylabel("observed accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1.02)
    heading = title if ece is None else f"{title}   (ECE {ece:.4f})"
    # the subtitle sits in the gap the title pad reserves, so the two cannot
    # overlap however long either string is
    ax.set_title(heading, fontsize=12, fontweight="bold", loc="left",
                 pad=30 if subtitle else 12)
    if subtitle:
        ax.text(0.0, 1.022, subtitle, transform=ax.transAxes, fontsize=8.5,
                color=MUTED, va="bottom")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED, loc="upper left")
    _style(ax)

    ax2.bar(centres, counts, width=0.085, color=GRID, edgecolor=MUTED,
            linewidth=0.4)
    ax2.set_xlabel("confidence the model stated")
    ax2.set_ylabel("rows", fontsize=9)
    _style(ax2)
    for a in (ax, ax2):
        a.xaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    return save(fig, name)
