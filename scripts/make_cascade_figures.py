#!/usr/bin/env python3
"""Figures for the cascade evaluation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import matplotlib.pyplot as plt  # noqa: E402
from reporting import ACCENT, BAD, GOOD, MUTED, WARN, grouped_bars, save  # noqa: E402

PRETTY = {"logreg": "Logistic regression", "svm": "Linear SVM",
          "minilm": "MiniLM embeddings", "nb": "Naive Bayes"}
COLOUR = {"logreg": GOOD, "svm": ACCENT, "minilm": BAD, "nb": WARN}


def main() -> int:
    data = json.loads((ROOT / "results" / "unit2_cascade.json").read_text())
    results = data["results"]

    # 1. what each layer handles, and how that changes with the protocol
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, protocol, title in zip(
            axes, ("stratified", "merchant"),
            ("Returning user (merchants familiar)",
             "New user (merchants unfamiliar)")):
        layers = results[protocol]["logreg"]["sweep"][0]["layers"]
        shares = [layers[n]["share"] for n in ("rule", "memory", "model")]
        accuracies = [layers[n]["accuracy"] for n in ("rule", "memory", "model")]
        bars = ax.bar(["rules", "merchant\nmemory", "model"], shares,
                      color=[GOOD, BAD, ACCENT], width=0.6)
        for bar, share, accuracy in zip(bars, shares, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, share + 0.02,
                    f"{share:.0%}\n{accuracy:.0%} right", ha="center",
                    va="bottom", fontsize=8.5, color=MUTED)
        ax.set_title(title, fontsize=10.5, color=MUTED, pad=10)
        ax.set_ylim(0, 1.0)
        ax.set_facecolor("white")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#e1e5e2", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    axes[0].set_ylabel("share of transactions handled")
    fig.suptitle("The cascade shifts work between layers as merchants become "
                 "unfamiliar", fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig1 = save(fig, "unit2_cascade_layers.png")

    # 2. the review burden: the figure that chooses the threshold
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for kind in ("logreg", "svm", "minilm", "nb"):
        sweep = results["merchant"][kind]["sweep"]
        x = [e["review_rate"] for e in sweep]
        y = [e["accepted_accuracy"] for e in sweep]
        ax.plot(x, y, "o-", color=COLOUR[kind], linewidth=1.8, markersize=4.5,
                label=PRETTY[kind])
    ax.axhline(0.90, color=MUTED, linestyle="--", linewidth=1)
    ax.text(0.005, 0.905, "90% accuracy on what is auto-accepted", fontsize=8.5,
            color=MUTED)
    ax.set_xlabel("share of transactions sent for human review")
    ax.set_ylabel("accuracy of what is accepted automatically")
    ax.set_title("Review burden: how much human effort buys how much accuracy",
                 fontsize=12, fontweight="bold", loc="left", pad=22)
    ax.text(0.0, 1.03, "merchant held out - what a new user meets. Further "
                       "left and higher is better.",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="lower right")
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color="#e1e5e2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    fig2 = save(fig, "unit2_review_burden.png")

    # 3. does assembling the cascade beat the model alone?
    gains = data["cascade_gain_over_model_alone"]
    fig3 = grouped_bars(
        "Does the cascade beat the model on its own?",
        "change in macro-F1 when the model is placed in the cascade",
        [PRETTY[k] for k in ("logreg", "svm", "minilm", "nb")],
        [("Returning user", [gains[f"stratified/{k}"]
                             for k in ("logreg", "svm", "minilm", "nb")], GOOD),
         ("New user", [gains[f"merchant/{k}"]
                       for k in ("logreg", "svm", "minilm", "nb")], BAD)],
        "unit2_cascade_gain.png", percent=False)

    print(f"[PASS] wrote 3 cascade figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
