#!/usr/bin/env python3
"""Unit 2, Model 3 of 5: multinomial logistic regression over TF-IDF features.

Adds calibration to the comparison, because this project routes low-confidence
predictions to human review and therefore needs confidence scores that mean
something.

Run:
    .venv/bin/python scripts/run_model3_logistic_regression.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_model import (calibration, load, print_table, score,  # noqa: E402
                            strictly_novel)
from reporting import (ACCENT, GOOD, confusion_heatmap, grouped_bars,  # noqa: E402
                       horizontal_bars, per_class_bars, reliability_diagram,
                       write_doc)
from models.logistic_regression import build_classifier  # noqa: E402
from models.tfidf_features import build_vectoriser  # noqa: E402

MODEL = "TF-IDF + Logistic Regression"


def fit(train):
    vectoriser = build_vectoriser()
    started = time.perf_counter()
    features = vectoriser.fit_transform([r["normalised_description"] for r in train])
    classifier = build_classifier()
    classifier.fit(features, [r["category"] for r in train])
    return vectoriser, classifier, time.perf_counter() - started, features.shape[1]


def predict(vectoriser, classifier, rows):
    probabilities = classifier.predict_proba(
        vectoriser.transform([r["normalised_description"] for r in rows]))
    classes = [str(c) for c in classifier.classes_]
    return [classes[i] for i in probabilities.argmax(1)], probabilities, classes


def top_tokens(vectoriser, classifier, category, k=8):
    """The word features pushing hardest toward a category.

    Only the word block is shown: character n-grams carry real signal for the
    model but read as fragments rather than evidence.
    """
    names = vectoriser.get_feature_names_out()
    is_word = np.array([str(n).startswith("tfidfvectorizer-1__") for n in names])
    clean = np.array([str(n).split("__", 1)[-1] for n in names])
    index = list(classifier.classes_).index(category)
    weights = classifier.coef_[index].copy()
    weights[~is_word] = -np.inf
    order = np.argsort(-weights)[:k]
    return [(clean[i], float(weights[i])) for i in order]


def main() -> int:
    results, calibrations = {}, {}

    s_train = load("stratified_train.csv")
    s_validation = load("stratified_validation.csv")
    vec_s, clf_s, train_seconds, n_features = fit(s_train)
    print(f"fitted on {len(s_train):,} rows in {train_seconds:.2f}s "
          f"({n_features:,} features)")

    predicted_s, prob_s, classes = predict(vec_s, clf_s, s_validation)
    gold_s = [r["category"] for r in s_validation]
    results["stratified (validation)"] = score(gold_s, predicted_s)
    calibrations["stratified"] = calibration(gold_s, prob_s, classes)

    m_train = load("merchant_train.csv")
    m_validation = load("merchant_validation.csv")
    vec_m, clf_m, _s, _f = fit(m_train)
    predicted_m, prob_m, classes_m = predict(vec_m, clf_m, m_validation)
    gold_m = [r["category"] for r in m_validation]
    results["merchant held out (validation)"] = score(gold_m, predicted_m)
    calibrations["merchant"] = calibration(gold_m, prob_m, classes_m)

    novel = strictly_novel(m_train, m_validation)
    predicted_n, prob_n, _c = predict(vec_m, clf_m, novel)
    gold_n = [r["category"] for r in novel]
    results["  of which strictly novel"] = score(gold_n, predicted_n)
    calibrations["strictly_novel"] = calibration(gold_n, prob_n, classes_m)

    print_table(f"MODEL 3 of 5 - {MODEL.upper()}", results)

    print("\ncalibration (does its confidence mean anything?)")
    print(f"  {'protocol':<18}{'ECE':>9}{'MCE':>9}{'Brier':>9}"
          f"{'mean conf':>11}{'accuracy':>10}{'gap':>9}")
    for name, c in calibrations.items():
        print(f"  {name:<18}{c['ece']:>9.4f}{c['mce']:>9.4f}{c['brier']:>9.4f}"
              f"{c['mean_confidence']:>11.4f}{c['accuracy']:>10.4f}"
              f"{c['overconfidence']:>+9.4f}")

    labels = sorted(set(gold_s))
    f1_stratified = f1_score(gold_s, predicted_s, average=None, labels=labels,
                             zero_division=0)
    f1_merchant = f1_score(gold_m, predicted_m, average=None, labels=labels,
                           zero_division=0)
    counts = [gold_s.count(label) for label in labels]

    print("\nper-category F1")
    print(f"  {'category':<18}{'stratified':>12}{'merchant':>12}{'drop':>9}")
    order = sorted(range(len(labels)), key=lambda i: f1_merchant[i])
    for i in order:
        print(f"  {labels[i]:<18}{f1_stratified[i]:>12.3f}{f1_merchant[i]:>12.3f}"
              f"{f1_stratified[i]-f1_merchant[i]:>9.3f}")

    print("\nwhat the model learned, as plain weights (stratified fit)")
    showcase = ["Mortgage", "Restaurants", "Income", "Shopping"]
    evidence = {}
    for category in showcase:
        tokens = top_tokens(vec_s, clf_s, category)
        evidence[category] = tokens
        print(f"  {category:<14}" +
              ", ".join(f"{t} {w:+.2f}" for t, w in tokens[:6]))

    # ---- figures ----------------------------------------------------------
    print("\nfigures")
    names = list(results)
    short = ["Stratified", "Merchant held out", "Strictly novel"]
    fig1 = grouped_bars(
        "Logistic regression: accuracy and macro-F1 by protocol", "", short,
        [("Accuracy", [results[n]["accuracy"] for n in names], ACCENT),
         ("Macro-F1", [results[n]["macro_f1"] for n in names], GOOD)],
        "model3_protocols.png", percent=True)

    fig2 = per_class_bars(
        "Logistic regression: F1 per category (merchant held out)",
        labels, list(f1_merchant), "model3_per_class_f1.png", counts=counts)

    matrix = confusion_matrix(gold_m, predicted_m, labels=labels)
    fig3 = confusion_heatmap(
        "Logistic regression: what gets confused with what (merchant held out)",
        labels, matrix, "model3_confusion.png")

    fig4 = reliability_diagram(
        "Logistic regression: is its confidence trustworthy?",
        calibrations["merchant"]["bins"], "model3_reliability.png",
        ece=calibrations["merchant"]["ece"],
        subtitle="merchant held out. Bars on the diagonal mean stated confidence "
                 "matches observed accuracy.")

    tokens = evidence["Restaurants"]
    fig5 = horizontal_bars(
        "Logistic regression explains itself: top evidence for 'Restaurants'",
        "weight pushing toward this category",
        [t for t, _w in tokens], [w for _t, w in tokens],
        "model3_evidence_restaurants.png", percent=False)

    # ---- confusions -------------------------------------------------------
    data = np.array(matrix, dtype=float)
    totals = data.sum(axis=1, keepdims=True)
    shares = np.divide(data, totals, out=np.zeros_like(data), where=totals != 0)
    confusions = []
    for i, true_label in enumerate(labels):
        for j, predicted_label in enumerate(labels):
            if i != j and shares[i, j] >= 0.08:
                confusions.append((true_label, predicted_label, shares[i, j],
                                   int(data[i, j])))
    confusions.sort(key=lambda c: -c[2])
    print("\nlargest confusions (merchant held out)")
    for t, p, s, n in confusions[:6]:
        print(f"  {t:<16} -> {p:<16}{s:>7.1%}  ({n})")

    # ---- document ---------------------------------------------------------
    print("\ndocument")
    strat, merch, novel_r = (results[n] for n in names)
    cal_m, cal_n = calibrations["merchant"], calibrations["strictly_novel"]
    worst = [labels[i] for i in order[:4]]
    best = [labels[i] for i in order[-4:]][::-1]

    write_doc(
        "model3_logistic_regression.docx",
        "Model 3 of 5 - TF-IDF with Logistic Regression",
        "Unit 2, model comparison. The first model whose confidence is measured, "
        "and the first that explains itself exactly.",
        [
        ("h1", "1. What this model is"),
        ("p", "Multinomial logistic regression learns one weight per feature per "
              "category and converts the resulting scores into probabilities. It "
              "uses exactly the same TF-IDF features as Model 2, so any "
              "difference between the two reflects the classifier alone."),
        ("p", "Two properties make it important to this project beyond its "
              "accuracy."),
        ("bullet", "**It is linear, so it explains itself exactly.** A feature's "
                   "contribution to a category score is its weight multiplied by "
                   "its value. Nothing needs to be approximated after the fact; "
                   "the explanation is the arithmetic the model already did."),
        ("bullet", "**Its probabilities are fitted, not inferred.** Logistic "
                   "regression maximises likelihood, which optimises the "
                   "probabilities themselves. The architecture sends "
                   "low-confidence predictions to a human, so a model whose "
                   "confidence is untrustworthy would push errors through that "
                   "gate unflagged."),
        ("p", "class_weight is set to balanced, because deduplication left the "
              "categories uneven - Restaurants has 3,110 rows against Rent's 874 "
              "- and macro-F1 penalises a model that favours the larger ones. "
              "Naive Bayes addresses the same imbalance through its class priors "
              "rather than a weight, which is a difference worth noting when "
              "comparing the two."),
        ("h1", "2. Results"),
        ("table", [["Protocol", "n", "Accuracy", "Macro-F1"]] +
                  [[label, f"{r['n']:,}", f"{r['accuracy']:.4f}",
                    f"{r['macro_f1']:.4f}"]
                   for label, r in zip(
                       ["Stratified (validation)",
                        "Merchant held out (validation)",
                        "of which strictly novel"], [strat, merch, novel_r])]),
        ("figure", (fig1, "Figure 1. Accuracy and macro-F1 across the three "
                          "evaluation protocols.")),
        ("figure", (fig2, "Figure 2. F1 for each of the 17 categories under the "
                          "merchant-held-out protocol, weakest at the bottom.")),
        ("p", f"The strongest categories are {', '.join(best)}; the weakest are "
              f"{', '.join(worst)}. The pattern matches Model 2: categories "
              "carrying their own vocabulary survive the merchant holdout, while "
              "those defined by merchant identity do not."),
        ("figure", (fig3, "Figure 3. Confusion matrix under the merchant-held-out "
                          "protocol, each row normalised to its own total.")),
        ("table", [["True category", "Predicted as", "Share", "Rows"]] +
                  [[t, p, f"{s:.1%}", f"{n:,}"] for t, p, s, n in confusions[:6]]),
        ("h1", "3. Calibration"),
        ("p", "Accuracy says how often a model is right. Calibration says "
              "whether its confidence can be believed, which is what determines "
              "whether the review gate works. A model claiming 90% confidence "
              "should be right about 90% of the time."),
        ("table", [["Protocol", "ECE", "MCE", "Brier", "Mean confidence",
                    "Accuracy", "Gap"]] +
                  [[name.replace("_", " "), f"{c['ece']:.4f}", f"{c['mce']:.4f}",
                    f"{c['brier']:.4f}", f"{c['mean_confidence']:.4f}",
                    f"{c['accuracy']:.4f}", f"{c['overconfidence']:+.4f}"]
                   for name, c in calibrations.items()]),
        ("figure", (fig4, "Figure 4. Reliability diagram, merchant held out. "
                          "Bars sitting on the dashed diagonal indicate stated "
                          "confidence matching observed accuracy. The lower panel "
                          "shows how many predictions fall in each confidence "
                          "band.")),
        ("p", f"On the merchant-held-out protocol the expected calibration error "
              f"is {cal_m['ece']:.4f}, with mean confidence {cal_m['mean_confidence']:.4f} "
              f"against accuracy {cal_m['accuracy']:.4f}, a gap of "
              f"{cal_m['overconfidence']:+.4f}. On strictly novel merchants the "
              f"gap is {cal_n['overconfidence']:+.4f} with mean confidence "
              f"{cal_n['mean_confidence']:.4f} against accuracy "
              f"{cal_n['accuracy']:.4f}."),
        ("h1", "4. The model explains itself"),
        ("p", "Because the model is linear, the evidence behind any prediction "
              "can be read directly from the weights. No surrogate model or "
              "sampling procedure is involved, so the explanation is exact rather "
              "than approximate."),
        ("figure", (fig5, "Figure 5. The word features that push hardest toward "
                          "the Restaurants category, taken straight from the "
                          "fitted weights.")),
        ("table", [["Category", "Strongest evidence learned"]] +
                  [[category,
                    ", ".join(f"{t} ({w:+.2f})" for t, w in tokens[:5])]
                   for category, tokens in evidence.items()]),
        ("p", "These weights are worth reading as a sanity check on the model. "
              "Evidence that is generic banking vocabulary will transfer to "
              "merchants never seen; evidence that is a brand name will not, and "
              "predicts exactly where the merchant-held-out score will fall."),
        ("h1", "5. Comparison so far"),
        ("table", [["Model", "Stratified macro-F1", "Merchant held out",
                    "Strictly novel"],
                   ["1. Rule baseline (17% coverage)", "0.2924", "0.2849", "0.3068"],
                   ["2. Naive Bayes", "0.9809", "0.5903", "0.4657"],
                   ["3. Logistic regression", f"{strat['macro_f1']:.4f}",
                    f"{merch['macro_f1']:.4f}", f"{novel_r['macro_f1']:.4f}"]]),
        ("h1", "6. Limitations"),
        ("bullet", "Training is slower than Naive Bayes by roughly an order of "
                   f"magnitude ({train_seconds:.2f} seconds here), which is only "
                   "worth paying for if the accuracy or the calibration justify it."),
        ("bullet", "The explanation is exact for this model, but reports word "
                   "features only. Character n-grams contribute to the decision "
                   "and are omitted from the display because they read as "
                   "fragments rather than evidence."),
        ("bullet", "Performance still depends heavily on merchant familiarity. "
                   "Being linear and interpretable does not supply information "
                   "the description does not contain."),
        ("h1", "7. For the report"),
        ("p", f"TF-IDF with logistic regression reaches a macro-F1 of "
              f"{strat['macro_f1']:.4f} on familiar merchants, "
              f"{merch['macro_f1']:.4f} when merchants are held out and "
              f"{novel_r['macro_f1']:.4f} on strictly novel merchants, with an "
              f"expected calibration error of {cal_m['ece']:.4f} on the "
              "merchant-held-out protocol. Being linear, it supplies exact "
              "token-level explanations at no additional inference cost, which "
              "is the property the application's explainability requirement "
              "depends on."),
        ])

    out = ROOT / "results" / "model3_logistic_regression.json"
    out.write_text(json.dumps({
        "model": MODEL, "machine_learning": True,
        "train_seconds": round(train_seconds, 3), "n_features": n_features,
        "results": results,
        "calibration": {k: {m: v for m, v in c.items() if m != "bins"}
                        for k, c in calibrations.items()},
        "per_class_f1": {"labels": labels,
                         "stratified": [float(v) for v in f1_stratified],
                         "merchant": [float(v) for v in f1_merchant]},
        "largest_confusions": [{"true": t, "predicted": p, "share": round(s, 4),
                                "rows": n} for t, p, s, n in confusions[:12]],
        "learned_evidence": {k: [[t, round(w, 4)] for t, w in v]
                             for k, v in evidence.items()},
    }, indent=2), encoding="utf-8")
    print(f"\n[PASS] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
