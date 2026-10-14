#!/usr/bin/env python3
"""Figures and findings document for Unit 4.5."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import matplotlib.pyplot as plt  # noqa: E402
from reporting import (ACCENT, BAD, GOOD, MUTED, WARN, grouped_bars,  # noqa: E402
                       save, write_doc)


def main() -> int:
    d = json.loads((ROOT / "results" / "unit4_5_explainability.json").read_text())
    v, cost, shares, stab = (d["validation_on_linear_model"], d["cost"],
                             d["layer_shares"], d["stability_top3"])

    fig1 = grouped_bars(
        "Do LIME and SHAP recover the truth where the truth is known?",
        "agreement with exact contributions",
        ["Spearman rank", "Sign agreement", "Top-3 overlap"],
        [("LIME", [v["lime"]["spearman"], v["lime"]["sign_agreement"],
                   v["lime"]["jaccard_top3"]], WARN),
         ("SHAP", [v["shap"]["spearman"], v["shap"]["sign_agreement"],
                   v["shap"]["jaccard_top3"]], GOOD)],
        "unit45_validation.png", percent=False)

    # who explains what, and at what cost
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    layers = ["rule", "memory", "model"]
    values = [shares.get(k, 0) for k in layers]
    colours = [GOOD, GOOD, BAD]
    bars = ax.bar(["rules\n(exact)", "merchant memory\n(exact)",
                   "model\n(approximate)"], values, color=colours, width=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015,
                f"{value:.0%}", ha="center", fontsize=10, color=MUTED)
    ax.set_ylim(0, max(values) * 1.3)
    ax.set_ylabel("share of transactions")
    ax.set_title("Which layer explains", fontsize=11, fontweight="bold",
                 loc="left", pad=10)
    ax.yaxis.set_major_formatter(lambda x, _p: f"{x:.0%}")

    methods = ["rules", "memory", "LIME", "SHAP"]
    times = [0.0, 0.0, cost["lime"]["median_ms"], cost["shap"]["median_ms"]]
    bars = ax2.bar(methods, times, color=[GOOD, GOOD, WARN, ACCENT], width=0.6)
    for bar, t in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width() / 2, t + 1,
                 f"{t:.0f} ms" if t else "0 ms", ha="center", fontsize=10,
                 color=MUTED)
    ax2.set_ylim(0, max(times) * 1.35)
    ax2.set_ylabel("median time per explanation")
    ax2.set_title("What each costs", fontsize=11, fontweight="bold",
                  loc="left", pad=10)
    for a in (ax, ax2):
        a.set_facecolor("white")
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.grid(axis="y", color="#e1e5e2", linewidth=0.8)
        a.set_axisbelow(True)
        a.tick_params(colors=MUTED, labelsize=9)
    fig.suptitle("Most explanations are exact and free; only the model layer "
                 "pays", fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig2 = save(fig, "unit45_layers_and_cost.png")

    write_doc(
        "unit4_5_explainability.docx",
        "Unit 4.5 - Explaining the Cascade's Decisions",
        "Layer-aware explanations: exact where possible, validated where not.",
        [
        ("h1", "1. Why explanations differ by layer"),
        ("p", "Each layer of the cascade decides for a different reason, so each "
              "explains itself differently. Treating them uniformly would either "
              "waste effort or misrepresent what happened."),
        ("table", [["Layer", "Why it decided", "Explanation shown", "Nature"],
                   ["Rule", "a phrase matched",
                    "“The description contains SUSHI, which always "
                    "indicates Restaurants.”", "exact, free"],
                   ["Merchant memory", "this shop was seen during training",
                    "“You have 2 previous transactions at RALPH SAN, 2 of "
                    "which were Groceries.”", "exact, free"],
                   ["Model", "predicted from the text",
                    "token weights recovered by SHAP, with a note that the "
                    "explanation is an approximation", "approximate, costed"]]),
        ("p", "Only the model layer needs a post-hoc method. The deployed model "
              "is an embedding model whose 384 dimensions carry no word-level "
              "meaning, so a token-level explanation has to be reconstructed "
              "after the fact."),
        ("h1", "2. Most explanations are exact and free"),
        ("figure", (fig2, "Figure 1. Which layer explains each transaction, and "
                          "what each explanation costs.")),
        ("table", [["Layer", "Share of transactions", "Cost per explanation"],
                   ["Rule", f"{shares.get('rule', 0):.0%}", "0 ms, exact"],
                   ["Merchant memory", f"{shares.get('memory', 0):.0%}",
                    "0 ms, exact"],
                   ["Model", f"{shares.get('model', 0):.0%}",
                    f"{cost['shap']['median_ms']:.0f} ms median, approximate"]]),
        ("p", f"**{shares.get('rule', 0) + shares.get('memory', 0):.0%} of "
              "transactions receive an exact explanation at no cost.** Only the "
              "remainder pays for a post-hoc approximation. The cascade "
              "therefore improves the quality and the cost of explainability at "
              "the same time as it improves accuracy, which is a benefit of the "
              "architecture that the accuracy figures alone do not reveal."),
        ("h1", "3. Validating the post-hoc methods"),
        ("p", "For the deployed embedding model there is no exact attribution to "
              "compare against, so LIME and SHAP cannot be checked there "
              "directly. They were therefore first run on a linear model, where "
              "the contribution of every token is exactly computable as weight "
              "times value, and their agreement with that ground truth was "
              "measured. Passing this test is what earns the right to use them "
              "where truth is unavailable."),
        ("figure", (fig1, "Figure 2. Agreement with exact contributions on a "
                          "linear model, over "
                          f"{v['n']} transactions.")),
        ("table", [["Measure", "LIME", "SHAP", "What it means"],
                   ["Spearman rank correlation",
                    f"{v['lime']['spearman']:.4f}", f"{v['shap']['spearman']:.4f}",
                    "does it rank the tokens in the right order"],
                   ["Sign agreement", f"{v['lime']['sign_agreement']:.4f}",
                    f"{v['shap']['sign_agreement']:.4f}",
                    "does it agree on which tokens help and which hinder"],
                   ["Top-3 overlap", f"{v['lime']['jaccard_top3']:.4f}",
                    f"{v['shap']['jaccard_top3']:.4f}",
                    "does it identify the same leading evidence"]]),
        ("p", f"**SHAP agrees with the truth more closely than LIME on every "
              f"measure.** Its additivity residual is "
              f"{v['shap_additivity_residual']:.2e}, which confirms the Shapley "
              "values are enumerated exactly rather than sampled: descriptions "
              "in this corpus are short enough for every coalition to be "
              "evaluated. LIME's own surrogate explains only "
              f"R² {v['lime_surrogate_r2']:.4f} of the local variance it is "
              "fitted to, which is a weakness SHAP does not share."),
        ("p", "**SHAP is therefore the method served by the application**, with "
              "LIME retained for comparison rather than as the recommended "
              "explainer."),
        ("h1", "4. Cost and stability"),
        ("table", [["Method", "Median", "Maximum", "Encoder passes",
                    "Top-3 stability across runs"],
                   ["LIME", f"{cost['lime']['median_ms']:.0f} ms",
                    f"{cost['lime']['max_ms']:.0f} ms",
                    f"{cost['lime']['mean_model_calls']:.0f}",
                    f"{stab['lime']:.4f}"],
                   ["SHAP", f"{cost['shap']['median_ms']:.0f} ms",
                    f"{cost['shap']['max_ms']:.0f} ms",
                    f"{cost['shap']['mean_model_calls']:.0f}",
                    f"{stab['shap']:.4f}"]]),
        ("p", f"At a median of {cost['shap']['median_ms']:.0f} milliseconds, an "
              "explanation can be produced when a user selects a transaction "
              "without any perceptible delay, so the interface needs no loading "
              "state. Explanations are nevertheless computed on demand rather "
              "than in advance: precomputing for every transaction would mean "
              "tens of thousands of encoder passes for a view most of which is "
              "never opened."),
        ("p", "**Stability is exact.** Both methods return the same leading "
              "terms on every run, because the random seed is derived from a "
              "hash of the description rather than from the clock. Without that, "
              "a user refreshing the page could be shown different reasons for "
              "the same unchanged decision, which would rightly undermine trust "
              "in the feature. This corrects an earlier expectation that LIME "
              "would be unstable; seeded, it is not."),
        ("h1", "5. Honesty in the interface"),
        ("p", "The explanation shown for a model decision states in plain words "
              "that it is reconstructed after the fact and is an approximation "
              "of the model's reasoning. The explanations shown for the rule and "
              "memory layers make no such claim, because they are exact. A user "
              "can therefore tell how much weight to place on what they are "
              "being shown."),
        ("h1", "6. Limitations"),
        ("bullet", "The agreement figures were measured on a linear model, not "
                   "on the deployed embedding model. They establish that the "
                   "methods behave correctly where truth is knowable; they do "
                   "not prove the same fidelity holds on a different model."),
        ("bullet", "Only word-level tokens are attributed. The linear testbed "
                   "also uses character n-grams, which contribute to its "
                   "decisions and are excluded from the comparison because they "
                   "read as fragments rather than evidence."),
        ("bullet", "Exact Shapley enumeration depends on descriptions being "
                   "short. A description with more than twelve unique tokens "
                   "falls back to a sampled estimate, and the additivity "
                   "guarantee no longer holds exactly."),
        ("h1", "7. For the report"),
        ("p", "Explanations are produced per cascade layer. Rule and "
              "merchant-memory decisions explain themselves exactly and at no "
              "cost, covering "
              f"{shares.get('rule', 0) + shares.get('memory', 0):.0%} of "
              "transactions. Model decisions are explained with Shapley values, "
              "enumerated exactly for descriptions of twelve tokens or fewer, at "
              f"a median of {cost['shap']['median_ms']:.0f} milliseconds. Before "
              "being relied upon, both SHAP and LIME were validated against "
              "exact contributions on a linear model, where SHAP achieved a "
              f"Spearman correlation of {v['shap']['spearman']:.4f} and "
              f"{v['shap']['jaccard_top3']:.0%} top-3 agreement with the truth, "
              "outperforming LIME on every measure."),
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
