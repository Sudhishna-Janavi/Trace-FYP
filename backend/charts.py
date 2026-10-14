"""Charts, drawn as inline SVG on the server.

Why not a charting library
    The application has no build step and loads nothing from a CDN, so
    deployment stays a single process with no asset pipeline. Drawing here
    also means a chart renders with scripting unavailable, and — the reason
    that matters most in practice — the marks reference the same CSS custom
    properties as the rest of the page, so every chart follows the light and
    dark themes and stays sharp at any zoom when captured for a document.

Colour is never the only channel. Every segment and bar also carries a
`<title>`, which browsers surface as a tooltip and screen readers announce.
"""
from __future__ import annotations

import math

# Enough hues to separate the categories a donut shows before they collapse
# into "Other". Chosen for distinguishability in both themes rather than to
# match the accent.
WHEEL = ["#1A6B5C", "#C97B3C", "#4A6FA5", "#8E5A8E", "#5B8C5A", "#B5503F",
         "#7A7D52", "#9BA5A1"]


def cashflow(months: list[dict]) -> str:
    """Income against spending, one pair of bars per month."""
    if not months:
        return ""
    width, height = 740, 196
    left, right, top, bottom = 42, 6, 8, 24
    peak = max(max(m["income"], m["spending"]) for m in months) or 1.0
    slot = (width - left - right) / len(months)
    bar = min(13.0, slot / 3.4)

    def y(value: float) -> float:
        return top + (height - top - bottom) * (1 - value / peak)

    parts = []
    for index in range(4):
        value = peak * index / 3
        at = y(value)
        parts.append(
            f'<line x1="{left}" y1="{at:.1f}" x2="{width - right}" y2="{at:.1f}" '
            f'stroke="var(--line)"/>'
            f'<text x="{left - 7}" y="{at + 3.5:.1f}" text-anchor="end" '
            f'font-size="9.5" fill="var(--faint)">{round(value / 1000)}k</text>')
    for index, month in enumerate(months):
        centre = left + slot * index + slot / 2
        for value, colour, offset in ((month["income"], "var(--income)", -bar - 1.5),
                                      (month["spending"], "var(--spend)", 1.5)):
            parts.append(
                f'<rect x="{centre + offset:.1f}" y="{y(value):.1f}" '
                f'width="{bar:.1f}" height="{height - bottom - y(value):.1f}" '
                f'rx="2.5" fill="{colour}"><title>{month["month"]} '
                f'${value:,.0f}</title></rect>')
        parts.append(
            f'<text x="{centre:.1f}" y="{height - 8}" text-anchor="middle" '
            f'font-size="9.5" fill="var(--faint)">{month["label"]}</text>')
    return _svg(width, height, parts, "Monthly income and spending")


def donut(categories: list[dict], keep: int = 8) -> tuple[str, list[dict]]:
    """Spending as a ring, with the smallest categories gathered into Other.

    A ring shows part-to-whole well and ranks poorly: people compare angles
    badly, and fourteen segments would be unreadable. The large categories get
    their own segment, the tail is collected, and the exact figures live in the
    interaction and in the ranked list beside it rather than in the angles.

    `keep` is eight rather than six because at six the gathered Other slice was
    the third largest on the ring, which hides more than the simplification
    saves.

    Returns the markup and the segment list the page needs for its legend.
    """
    if not categories:
        return "", []

    head = categories[:keep]
    tail = categories[keep:]
    segments = [{"category": c["category"], "amount": c["amount"],
                 "share": c["share"], "count": c["count"],
                 "colour": WHEEL[index % len(WHEEL)]}
                for index, c in enumerate(head)]
    if tail:
        segments.append({
            "category": "Other", "amount": sum(c["amount"] for c in tail),
            "share": sum(c["share"] for c in tail),
            "count": sum(c["count"] for c in tail),
            "colour": WHEEL[-1],
            "members": ", ".join(c["category"] for c in tail)})

    size, thickness = 230.0, 30.0
    radius = (size - thickness) / 2
    circumference = 2 * math.pi * radius
    total = sum(s["amount"] for s in segments) or 1.0

    parts = [f'<circle cx="{size / 2}" cy="{size / 2}" r="{radius}" fill="none" '
             f'stroke="var(--surface-2)" stroke-width="{thickness}"/>']
    offset = 0.0
    for index, segment in enumerate(segments):
        length = segment["amount"] / total * circumference
        # a hairline gap keeps adjacent segments legible when hues are close
        gap = min(2.0, length * 0.08)
        parts.append(
            f'<circle class="seg" data-seg="{index}" cx="{size / 2}" '
            f'cy="{size / 2}" r="{radius}" fill="none" '
            f'stroke="{segment["colour"]}" stroke-width="{thickness}" '
            f'stroke-dasharray="{max(length - gap, 0.1):.2f} '
            f'{circumference - max(length - gap, 0.1):.2f}" '
            f'stroke-dashoffset="{-offset:.2f}" '
            f'transform="rotate(-90 {size / 2} {size / 2})">'
            f'<title>{segment["category"]} ${segment["amount"]:,.0f} '
            f'({segment["share"]:.1f}%)</title></circle>')
        offset += length
    return _svg(size, size, parts, "Spending by category",
                extra='class="donut"'), segments


def diverging(items: list[dict], value_key: str, limit: int = 9) -> str:
    """Change per category: increases to the right, decreases to the left.

    Sorted by size of movement, so the question "what changed?" is answered by
    the top of the chart rather than by reading every row.
    """
    items = [i for i in items if abs(i[value_key]) >= 0.5][:limit]
    if not items:
        return ""
    width = 700.0
    row_height, gap = 26.0, 6.0
    label = 118.0
    height = len(items) * (row_height + gap) + 20
    axis = label + (width - label) / 2
    widest = max(abs(i[value_key]) for i in items) or 1.0
    scale = ((width - label) / 2 - 56) / widest

    parts = [f'<line x1="{axis}" y1="4" x2="{axis}" y2="{height - 14}" '
             f'stroke="var(--line)"/>']
    for index, item in enumerate(items):
        value = item[value_key]
        top = 8 + index * (row_height + gap)
        extent = abs(value) * scale
        x = axis if value >= 0 else axis - extent
        colour = "var(--spend)" if value > 0 else "var(--income)"
        parts.append(
            f'<text x="{label - 10}" y="{top + row_height / 2 + 4:.1f}" '
            f'text-anchor="end" font-size="11.5" fill="var(--muted)">'
            f'{item["category"]}</text>'
            f'<rect x="{x:.1f}" y="{top:.1f}" width="{max(extent, 1.5):.1f}" '
            f'height="{row_height}" rx="3" fill="{colour}" opacity=".85">'
            f'<title>{item["category"]} {value:+,.0f}</title></rect>'
            f'<text x="{(x + extent + 7) if value >= 0 else (x - 7):.1f}" '
            f'y="{top + row_height / 2 + 4:.1f}" font-size="11" '
            f'text-anchor="{"start" if value >= 0 else "end"}" '
            f'fill="var(--faint)">{value:+,.0f}</text>')
    return _svg(width, height, parts, "Change in spending by category")


def protocols(performance: dict, labels: dict) -> str:
    """Macro-F1 and review rate under each evaluation protocol.

    Both bars matter together: the score falling and the review rate rising
    across the protocols is one finding, not two.
    """
    names = [n for n in ("in_distribution", "unseen_merchant", "strictly_novel")
             if n in performance]
    if not names:
        return ""
    width, height = 700, 220
    left, bottom, top = 46, 46, 10
    slot = (width - left - 10) / len(names)
    bar = 46.0

    def y(fraction: float) -> float:
        return top + (height - top - bottom) * (1 - fraction)

    parts = []
    for step in range(5):
        fraction = step / 4
        at = y(fraction)
        parts.append(
            f'<line x1="{left}" y1="{at:.1f}" x2="{width - 10}" y2="{at:.1f}" '
            f'stroke="var(--line)"/>'
            f'<text x="{left - 8}" y="{at + 3.5:.1f}" text-anchor="end" '
            f'font-size="9.5" fill="var(--faint)">{fraction:.1f}</text>')
    for index, name in enumerate(names):
        block = performance[name]
        centre = left + slot * index + slot / 2
        for value, colour, offset, what in (
                (block["macro_f1"], "var(--accent)", -bar - 3, "macro F1"),
                (block["review_rate"], "var(--review)", 3, "held for review")):
            parts.append(
                f'<rect x="{centre + offset:.1f}" y="{y(value):.1f}" '
                f'width="{bar:.1f}" height="{height - bottom - y(value):.1f}" '
                f'rx="3" fill="{colour}"><title>{labels[name]}: {what} '
                f'{value:.3f}</title></rect>'
                f'<text x="{centre + offset + bar / 2:.1f}" y="{y(value) - 6:.1f}" '
                f'text-anchor="middle" font-size="10.5" font-weight="600" '
                f'fill="var(--muted)">{value:.2f}</text>')
        parts.append(
            f'<text x="{centre:.1f}" y="{height - 26}" text-anchor="middle" '
            f'font-size="11.5" font-weight="500" fill="var(--ink)">'
            f'{labels[name]}</text>'
            f'<text x="{centre:.1f}" y="{height - 11}" text-anchor="middle" '
            f'font-size="10" fill="var(--faint)">n = {block["n"]:,}</text>')
    return _svg(width, height, parts, "Performance under three protocols")


def _svg(width: float, height: float, parts: list[str], label: str,
         extra: str = "") -> str:
    return (f'<svg viewBox="0 0 {width:.0f} {height:.0f}" {extra} '
            f'style="width:100%;height:auto" role="img" aria-label="{label}">'
            + "".join(parts) + "</svg>")
