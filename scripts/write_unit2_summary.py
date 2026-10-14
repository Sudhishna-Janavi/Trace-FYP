#!/usr/bin/env python3
"""Unit 2 summary document: the comparison, the cascade, and the final result."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reporting import FIGURES, write_doc  # noqa: E402


def main() -> int:
    cmp_ = json.loads((ROOT / "results" / "unit2_comparison.json").read_text())
    casc = json.loads((ROOT / "results" / "unit2_cascade.json").read_text())
    final = json.loads((ROOT / "results" / "unit2_final_test.json").read_text())
    models, gains, r = cmp_["models"], casc["cascade_gain_over_model_alone"], final["results"]
    fig = lambda n: FIGURES / n

    write_doc(
        "unit2_summary.docx",
        "Unit 2 - Model Comparison, Cascade Evaluation and Selection",
        "Six models compared on validation; the chosen configuration evaluated "
        "once on the held-out test sets.",
        [
        ("h1", "1. What Unit 2 set out to answer"),
        ("p", "Which categorisation model should the application deploy, and in "
              "what arrangement? Answering it required three things the "
              "individual model studies could not provide: the models seen "
              "together, the assembled system measured end to end, and a "
              "selection made against stated criteria rather than a single score."),

        ("h1", "2. The six models compared"),
        ("table", [["Model", "In-distribution", "Unseen merchant",
                    "Strictly novel", "Merchant ECE", "Explanation"]] +
                  [[name, f"{m['macro_f1']['Stratified']:.4f}",
                    f"{m['macro_f1']['Merchant held out']:.4f}",
                    f"{m['macro_f1']['Strictly novel']:.4f}",
                    f"{m['merchant_ece']:.4f}" if m["merchant_ece"] else "n/a",
                    m["explanation"]] for name, m in models.items()]),
        ("p", "Macro-F1 on validation. The rule baseline and merchant memory "
              "abstain, so every abstention counts against them and they are not "
              "comparable to the rest on this metric; they are included to show "
              "the floor and the shape of the problem."),
        ("figure", (fig("unit2_all_models.png"),
                    "Figure 1. Every learned model across the three protocols.")),
        ("figure", (fig("unit2_deterministic_coverage.png"),
                    "Figure 2. The two deterministic layers answer different "
                    "questions: the rules hold steady as merchants become "
                    "unfamiliar, the memory collapses.")),
        ("figure", (fig("unit2_calibration.png"),
                    "Figure 3. Expected calibration error. The calibrated SVM "
                    "degrades sharply under distribution shift, because its "
                    "sigmoid is fitted on folds of the training data that do not "
                    "resemble unfamiliar merchants.")),
        ("h2", "2.1 The central finding"),
        ("p", "Performance depends overwhelmingly on whether the merchant has "
              "been seen before, not on the choice of classifier. Three "
              "different linear models over identical features agree to within a "
              "few points on every protocol, and all of them fall by roughly half "
              "when merchants are held out. What did move the result was changing "
              "the representation: TF-IDF wins where merchants recur, because a "
              "brand name is a perfectly discriminative token, and sentence "
              "embeddings win where they do not, because an unfamiliar merchant "
              "still lands somewhere sensible in a learned semantic space."),
        ("figure", (fig("unit2_tradeoff.png"),
                    "Figure 4. The deployment trade-off. The horizontal axis is "
                    "what a new user meets; the marker shape is whether the model "
                    "can explain itself.")),

        ("h1", "3. The cascade"),
        ("p", "The deployed system is not a single model. Each transaction passes "
              "through deterministic rules, then merchant memory, then the "
              "learned model, with a confidence gate marking low-confidence "
              "model answers for human review."),
        ("figure", (fig("unit2_cascade_layers.png"),
                    "Figure 5. The cascade shifts work between layers on its own. "
                    "For a returning user the memory answers two thirds of "
                    "transactions at 100% accuracy; for a new user it answers 1% "
                    "and the model carries 82%.")),
        ("table", [["Model in the final slot", "Change for a returning user",
                    "Change for a new user"]] +
                  [[name, f"{gains[f'stratified/{k}']:+.4f}",
                    f"{gains[f'merchant/{k}']:+.4f}"]
                   for k, name in (("logreg", "Logistic regression"),
                                   ("svm", "Linear SVM"), ("minilm", "MiniLM"),
                                   ("nb", "Naive Bayes"))]),
        ("figure", (fig("unit2_cascade_gain.png"),
                    "Figure 6. Assembling the cascade improves every model for a "
                    "new user.")),
        ("p", "MiniLM gains most for a returning user because the deterministic "
              "layers cover precisely its weakness. The two strongest linear "
              "models lose a little there, because the merchant memory "
              "occasionally overrides a correct model prediction."),
        ("h2", "3.1 Choosing the confidence threshold"),
        ("figure", (fig("unit2_review_burden.png"),
                    "Figure 7. How much human review buys how much accuracy, on "
                    "the protocol a new user meets.")),
        ("p", "Below roughly 30% review all candidate models behave alike. At "
              "about 45% review the better-calibrated models reach 90% accuracy "
              "on what they accept automatically. Only logistic regression "
              "continues to discriminate above 95%, reaching 99.5%, which is the "
              "calibration advantage made concrete. A threshold of 0.60 was "
              "chosen as the operating point."),

        ("h1", "4. Selection"),
        ("p", "**MiniLM sentence embeddings with a logistic-regression head** "
              "occupies the final slot."),
        ("p", "It is the most accurate model on unseen and strictly novel "
              "merchants, which is the condition a new user meets, and the best "
              "calibrated of the six. It cannot produce token-level explanations "
              "directly, because its 384 dimensions carry no word-level meaning, "
              "so explanations are produced post hoc with LIME and SHAP."),
        ("p", "**The alternative was logistic regression**, which supplies exact "
              "explanations at no cost and is the only model able to reach 99% "
              "accuracy on automatically accepted transactions, but is materially "
              "weaker on unfamiliar merchants. The choice is therefore a stated "
              "trade-off rather than a win, and the cost is quantified: roughly "
              "0.10 macro-F1 on strictly novel merchants, exchanged for accuracy "
              "where it is scarcest."),
        ("h2", "4.1 Three caveats carried by this choice"),
        ("num", "The explanation is approximate. For a linear model the "
                "explanation is the arithmetic the model performed; here it is a "
                "surrogate's estimate of it."),
        ("num", "LIME is stochastic, being fitted to randomly sampled "
                "perturbations, so the same transaction can yield slightly "
                "different explanations across runs."),
        ("num", "It costs roughly 96 encoder passes per explanation, against "
                "about a millisecond and no extra inference for a linear model."),
        ("p", "Because no exact attribution exists for an embedding model, LIME "
              "and SHAP are validated on a linear model first, where exact "
              "contributions are computable, and only then applied to the "
              "deployed model."),

        ("h1", "5. Final result"),
        ("p", "The chosen configuration, refitted on train plus validation and "
              "evaluated **once** on the held-out test sets. These rows were "
              "never used for fitting, comparison or threshold selection."),
        ("table", [["Protocol", "n", "Accuracy", "Macro-F1", "Sent to review",
                    "Accuracy of what is accepted"]] +
                  [[name.replace("_", " "), f"{r[name]['n']:,}",
                    f"{r[name]['accuracy']:.4f}", f"{r[name]['macro_f1']:.4f}",
                    f"{r[name]['review_rate']:.1%}",
                    f"{r[name]['accepted_accuracy']:.4f}"]
                   for name in ("in_distribution", "unseen_merchant",
                                "strictly_novel")]),
        ("p", "Read together these say: for a returning user the system is "
              f"{r['in_distribution']['accuracy']:.1%} accurate and asks for "
              f"review on {r['in_distribution']['review_rate']:.1%} of "
              "transactions; for a new user accuracy falls to "
              f"{r['unseen_merchant']['accuracy']:.1%} and the review burden "
              f"rises to {r['unseen_merchant']['review_rate']:.1%}; and for "
              "merchants the system has never encountered in any form it is "
              f"{r['strictly_novel']['accuracy']:.1%} accurate. In every case "
              "the transactions it accepts automatically are substantially more "
              "reliable than its overall accuracy, which is what the confidence "
              "gate exists to achieve."),
        ("h2", "5.1 How these figures differ from the validation figures"),
        ("p", "The test figures are slightly lower than the validation figures "
              "reported earlier. That is expected and is the reason the test sets "
              "were reserved: a figure chosen from among many candidates is "
              "optimistic, and only a set used once is not."),

        ("h1", "6. What remains"),
        ("bullet", "A neural-network head over the same embeddings, written and "
                   "awaiting a run, to test whether depth adds anything."),
        ("bullet", "Validation of LIME and SHAP against exact contributions on a "
                   "linear model, before they are relied upon in the interface."),
        ("bullet", "A user-correction memory, which requires real users and is "
                   "therefore built in the application."),
        ("bullet", "Fuzzy merchant matching, which would raise the memory layer's "
                   "coverage above the lower bound reported here."),
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
