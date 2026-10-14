#!/usr/bin/env python3
"""Unit 2, step 1: bring all six models into one comparison.

Reads the stored result files and produces the table and figures that let the
six models be seen together, which no individual model document can do.

Scores are on VALIDATION throughout. The test sets remain untouched until a
single configuration has been chosen, so that the final reported figure is not
one that was selected for.

Run:
    ~/.venvs/oct14/bin/python scripts/run_unit2_comparison.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reporting import ACCENT, BAD, GOOD, MUTED, WARN, grouped_bars, save  # noqa: E402

FILES = [
    ("1. Rule baseline", "model1_rules"),
    ("1b. Merchant memory", "model1b_merchant_memory"),
    ("2. Naive Bayes", "model2_naive_bayes"),
    ("3. Logistic regression", "model3_logistic_regression"),
    ("4. Linear SVM", "model4_linear_svm"),
    ("5. MiniLM embeddings", "model5_minilm"),
    ("6. Neural network", "model6_mlp"),
]
KEYS = ["stratified (validation)", "merchant held out (validation)",
        "  of which strictly novel"]
SHORT = ["Stratified", "Merchant held out", "Strictly novel"]

# Can the model say which words drove a prediction, and at what cost?
EXPLAINABILITY = {
    "1. Rule baseline": ("exact", "the matched phrase is the explanation"),
    "1b. Merchant memory": ("exact", "the remembered merchant is the explanation"),
    "2. Naive Bayes": ("exact", "per-token log probabilities, read directly"),
    "3. Logistic regression": ("exact", "weight times value; no extra inference"),
    "4. Linear SVM": ("partial", "an ensemble of calibrated folds, so there is no "
                                 "single weight vector to read"),
    "5. MiniLM embeddings": ("approximate", "384 abstract dimensions; needs LIME "
                                            "or SHAP, about 96 encoder passes"),
    "6. Neural network": ("approximate", "same abstract dimensions, plus "
                                         "non-linear layers above them"),
}
ABSTAINS = {"1. Rule baseline", "1b. Merchant memory"}


def main() -> int:
    loaded = {}
    for label, filename in FILES:
        path = ROOT / "results" / f"{filename}.json"
        if not path.exists():
            print(f"   [skip] {label}: not run yet")
            continue
        loaded[label] = json.loads(path.read_text())

    # ------------------------------------------------------------ tables ---
    print("=" * 94)
    print("UNIT 2 - ALL MODELS COMPARED")
    print("macro-F1 on VALIDATION. Test sets remain untouched.")
    print("=" * 94)
    print(f"{'model':<26}" + "".join(f"{s:>19}" for s in SHORT))
    for label, data in loaded.items():
        row = f"{label:<26}"
        for key in KEYS:
            row += f"{data['results'][key]['macro_f1']:>19.4f}"
        print(row + ("   *abstains" if label in ABSTAINS else ""))

    print(f"\n{'model':<26}{'merch ECE':>11}{'train s':>10}{'coverage':>11}"
          f"{'explanation':>14}")
    for label, data in loaded.items():
        ece = data.get("calibration", {}).get("merchant", {}).get("ece")
        seconds = data.get("train_seconds", data.get("head_train_seconds", 0.0))
        coverage = data["results"][KEYS[0]]["coverage"]
        kind = EXPLAINABILITY.get(label, ("", ""))[0]
        print(f"{label:<26}{(f'{ece:.4f}' if ece is not None else 'n/a'):>11}"
              f"{seconds:>10.2f}{coverage:>11.1%}{kind:>14}")
    print("\n* models that abstain are not comparable on macro-F1: every "
          "abstention counts as a miss.")
    print("=" * 94)

    # ----------------------------------------------------------- figures ---
    print("\nfigures")
    learned = {k: v for k, v in loaded.items() if k not in ABSTAINS}
    palette = [WARN, GOOD, ACCENT, BAD, MUTED]
    fig1 = grouped_bars(
        "Unit 2: every learned model, every protocol",
        "macro-F1 (validation)", SHORT,
        [(label, [data["results"][k]["macro_f1"] for k in KEYS], palette[i % 5])
         for i, (label, data) in enumerate(learned.items())],
        "unit2_all_models.png", percent=False, annotate=False)

    calibrated = {k: v for k, v in learned.items() if "calibration" in v}
    fig2 = grouped_bars(
        "Unit 2: can each model's confidence be trusted?",
        "expected calibration error (lower is better)", SHORT,
        [(label, [data["calibration"][p]["ece"]
                  for p in ("stratified", "merchant", "strictly_novel")],
          palette[i % 5])
         for i, (label, data) in enumerate(calibrated.items())],
        "unit2_calibration.png", percent=False, annotate=False)

    # coverage of the two deterministic layers, which macro-F1 cannot show
    deterministic = {k: v for k, v in loaded.items() if k in ABSTAINS}
    fig3 = grouped_bars(
        "Unit 2: the deterministic layers answer different questions",
        "coverage", SHORT,
        [(label, [data["results"][k]["coverage"] for k in KEYS],
          GOOD if "Rule" in label else BAD)
         for label, data in deterministic.items()],
        "unit2_deterministic_coverage.png", percent=True)

    # accuracy against explainability: the trade-off that decides deployment
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    marker = {"exact": "o", "partial": "s", "approximate": "^"}
    colour = {"exact": GOOD, "partial": WARN, "approximate": BAD}
    for label, data in learned.items():
        kind = EXPLAINABILITY[label][0]
        x = data["results"][KEYS[2]]["macro_f1"]
        y = data["results"][KEYS[0]]["macro_f1"]
        ax.scatter(x, y, s=120, marker=marker[kind], color=colour[kind],
                   zorder=3, edgecolor="white", linewidth=1.2)
        # place the label inside the axes: a point near the right edge gets
        # its text on the left, so nothing runs off the figure
        right = x > 0.55
        ax.annotate(label.split(". ", 1)[-1], (x, y), fontsize=8.5, color=MUTED,
                    xytext=(-9 if right else 9, -3),
                    textcoords="offset points",
                    ha="right" if right else "left")
    for kind, name in (("exact", "exact explanation"),
                       ("partial", "partial explanation"),
                       ("approximate", "approximate explanation only")):
        ax.scatter([], [], marker=marker[kind], color=colour[kind], s=90,
                   label=name)
    ax.margins(x=0.14, y=0.12)
    ax.set_xlabel("macro-F1 on strictly novel merchants (what a new user meets)")
    ax.set_ylabel("macro-F1 on familiar merchants")
    ax.set_title("The deployment trade-off: accuracy against explainability",
                 fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED, loc="lower left")
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color="#e1e5e2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    fig4 = save(fig, "unit2_tradeoff.png")

    summary = {
        "note": "validation scores only; test sets untouched",
        "models": {
            label: {
                "macro_f1": {s: data["results"][k]["macro_f1"]
                             for s, k in zip(SHORT, KEYS)},
                "merchant_ece": data.get("calibration", {})
                                    .get("merchant", {}).get("ece"),
                "train_seconds": data.get("train_seconds",
                                          data.get("head_train_seconds")),
                "stratified_coverage": data["results"][KEYS[0]]["coverage"],
                "explanation": EXPLAINABILITY.get(label, ("", ""))[0],
                "explanation_note": EXPLAINABILITY.get(label, ("", ""))[1],
                "abstains": label in ABSTAINS,
            }
            for label, data in loaded.items()
        },
        "figures": [f.name for f in (fig1, fig2, fig3, fig4)],
    }
    out = ROOT / "results" / "unit2_comparison.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[PASS] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
