#!/usr/bin/env python3
"""Unit 2, Model 2 of 5: multinomial Naive Bayes over TF-IDF features.

Run:
    .venv/bin/python scripts/run_model2_naive_bayes.py
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

from evaluate_model import load, print_table, score, strictly_novel  # noqa: E402
from reporting import (ACCENT, BAD, GOOD, confusion_heatmap, grouped_bars,  # noqa: E402
                       per_class_bars, write_doc)
from models.naive_bayes import build_classifier  # noqa: E402
from models.tfidf_features import build_vectoriser  # noqa: E402

MODEL = "TF-IDF + Multinomial Naive Bayes"


def fit(train):
    """Fit the vectoriser and classifier on the training split only."""
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


def main() -> int:
    results = {}

    # ---- protocol 1: stratified ------------------------------------------
    s_train = load("stratified_train.csv")
    s_validation = load("stratified_validation.csv")
    vec_s, clf_s, train_seconds, n_features = fit(s_train)
    print(f"fitted on {len(s_train):,} rows in {train_seconds:.2f}s "
          f"({n_features:,} features)")

    predicted, _probabilities, classes = predict(vec_s, clf_s, s_validation)
    gold_s = [r["category"] for r in s_validation]
    results["stratified (validation)"] = score(gold_s, predicted)
    predicted_s = predicted

    # ---- protocol 2: merchant held out -----------------------------------
    m_train = load("merchant_train.csv")
    m_validation = load("merchant_validation.csv")
    vec_m, clf_m, _seconds, _features = fit(m_train)
    predicted_m, _probabilities, _classes = predict(vec_m, clf_m, m_validation)
    gold_m = [r["category"] for r in m_validation]
    results["merchant held out (validation)"] = score(gold_m, predicted_m)

    # ---- protocol 3: strictly novel subset of protocol 2 ------------------
    novel = strictly_novel(m_train, m_validation)
    predicted_n, _probabilities, _classes = predict(vec_m, clf_m, novel)
    gold_n = [r["category"] for r in novel]
    results["  of which strictly novel"] = score(gold_n, predicted_n)

    print_table(f"MODEL 2 of 5 - {MODEL.upper()}", results)

    # ---- per-class detail -------------------------------------------------
    labels = sorted(set(gold_s))
    f1_stratified = f1_score(gold_s, predicted_s, average=None,
                             labels=labels, zero_division=0)
    f1_merchant = f1_score(gold_m, predicted_m, average=None,
                           labels=labels, zero_division=0)
    counts = [gold_s.count(label) for label in labels]

    print("\nper-category F1")
    print(f"  {'category':<18}{'stratified':>12}{'merchant':>12}{'drop':>9}")
    order = sorted(range(len(labels)), key=lambda i: f1_merchant[i])
    for i in order:
        print(f"  {labels[i]:<18}{f1_stratified[i]:>12.3f}{f1_merchant[i]:>12.3f}"
              f"{f1_stratified[i]-f1_merchant[i]:>9.3f}")

    # ---- figures ----------------------------------------------------------
    print("\nfigures")
    names = list(results)
    short = ["Stratified", "Merchant held out", "Strictly novel"]
    fig1 = grouped_bars(
        "Naive Bayes: performance collapses when the merchant is unfamiliar",
        "", short,
        [("Accuracy", [results[n]["accuracy"] for n in names], ACCENT),
         ("Macro-F1", [results[n]["macro_f1"] for n in names], GOOD)],
        "model2_protocols.png", percent=True)

    fig2 = per_class_bars(
        "Naive Bayes: F1 per category (merchant held out)",
        labels, list(f1_merchant), "model2_per_class_f1.png", counts=counts)

    matrix = confusion_matrix(gold_m, predicted_m, labels=labels)
    fig3 = confusion_heatmap(
        "Naive Bayes: what gets confused with what (merchant held out)",
        labels, matrix, "model2_confusion.png")

    # ---- biggest confusions, for the write-up -----------------------------
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
    for true_label, predicted_label, share, n in confusions[:8]:
        print(f"  {true_label:<16} -> {predicted_label:<16}{share:>7.1%}  ({n})")

    # ---- document ---------------------------------------------------------
    print("\ndocument")
    strat, merch, novel_r = (results[n] for n in names)
    worst = [labels[i] for i in order[:4]]
    best = [labels[i] for i in order[-4:]][::-1]

    write_doc(
        "model2_naive_bayes.docx",
        "Model 2 of 5 - TF-IDF with Multinomial Naive Bayes",
        "Unit 2, model comparison. The first model here that learns from data.",
        [
        ("h1", "1. What this model is"),
        ("p", "Multinomial Naive Bayes counts how strongly each text feature is "
              "associated with each category and applies Bayes' rule, assuming "
              "the features are independent once the category is known. That "
              "assumption is plainly false for text, since MORTGAGE and PMT "
              "co-occur constantly, which is why the method is called naive. It "
              "survives as a standard baseline because the violated assumption "
              "distorts the probabilities far more than it distorts the ranking "
              "of the classes."),
        ("p", "It is included as the **cheap learned baseline**. If a model that "
              f"trains in {train_seconds:.2f} seconds performs close to logistic "
              "regression or a support vector machine, the additional complexity "
              "of those models must be justified by something other than raw "
              "accuracy."),
        ("h2", "1.2 Input representation"),
        ("p", "The text is converted to TF-IDF features: word unigrams and "
              "bigrams capture whole terms and short phrases such as OVERDRAFT "
              "FEE, while character n-grams of length three and four capture "
              "fragments inside words, which lets an unfamiliar merchant still "
              "be recognised from its shape. Both vectorisers are fitted on the "
              "training split alone, producing "
              f"{n_features:,} features. Fitting them on all of the data would "
              "leak vocabulary and document frequencies from validation into "
              "training and inflate every number that follows."),
        ("p", "Models 2, 3 and 4 share this representation exactly, so "
              "differences between them reflect the classifier rather than the "
              "features."),
        ("h1", "2. Results"),
        ("table", [["Protocol", "n", "Accuracy", "Macro-F1"]] +
                  [[label, f"{r['n']:,}", f"{r['accuracy']:.4f}",
                    f"{r['macro_f1']:.4f}"]
                   for label, r in zip(
                       ["Stratified (validation)",
                        "Merchant held out (validation)",
                        "of which strictly novel"],
                       [strat, merch, novel_r])]),
        ("figure", (fig1, "Figure 1. Accuracy and macro-F1 across the three "
                          "evaluation protocols.")),
        ("p", f"On the stratified protocol the model reaches a macro-F1 of "
              f"{strat['macro_f1']:.4f}. Holding merchants out entirely reduces "
              f"that to {merch['macro_f1']:.4f}, and on the strictly novel subset "
              f"it falls to {novel_r['macro_f1']:.4f} - a drop of "
              f"{strat['macro_f1'] - novel_r['macro_f1']:.3f} between the easiest "
              "and hardest conditions, with the model and the metric unchanged. "
              "The only thing that varies is whether the merchant has been seen "
              "before, which identifies merchant familiarity as the dominant "
              "factor in this model's apparent accuracy."),
        ("h2", "2.1 Which categories hold up, and which do not"),
        ("figure", (fig2, "Figure 2. F1 for each of the 17 categories under the "
                          "merchant-held-out protocol, weakest at the bottom.")),
        ("p", f"The strongest categories are {', '.join(best)}. These are "
              "defined by vocabulary that recurs across merchants, so the model "
              "can still recognise them when the merchant changes."),
        ("p", f"The weakest are {', '.join(worst)}. These are defined by **who "
              "the merchant is** rather than by any word in the description, so "
              "once the merchant is unfamiliar there is little signal left."),
        ("h2", "2.2 What is confused with what"),
        ("figure", (fig3, "Figure 3. Confusion matrix under the merchant-held-out "
                          "protocol. Each row is normalised to the true "
                          "category's row total, so values are the percentage of "
                          "that category routed to each prediction.")),
        ("table", [["True category", "Predicted as", "Share", "Rows"]] +
                  [[t, p, f"{s:.1%}", f"{n:,}"]
                   for t, p, s, n in confusions[:8]]),
        ("p", "These confusions are informative rather than random. Categories "
              "collide where the underlying transactions genuinely resemble one "
              "another in text, which is a limit of the representation rather "
              "than a defect in the classifier."),
        ("h1", "3. Comparison with Model 1"),
        ("p", "The rule baseline answered 17.3% of transactions at 98.2% "
              f"precision. Naive Bayes answers **every** transaction and reaches "
              f"{strat['accuracy']:.1%} accuracy on the stratified protocol. The "
              "learned model is therefore not merely better, it addresses a "
              "different problem: the rules are a precise instrument over a "
              "narrow slice, while the model is a general one over everything. "
              "That difference is what motivates combining them in a cascade "
              "rather than choosing between them."),
        ("h1", "4. Limitations"),
        ("bullet", "The independence assumption is violated by construction, so "
                   "the predicted probabilities should not be trusted as "
                   "confidence estimates without calibration. This matters for "
                   "the review gate and is measured properly for later models."),
        ("bullet", "Performance depends heavily on merchant familiarity, as "
                   "Figure 1 shows. The stratified figure alone would materially "
                   "overstate how the system behaves for a new user."),
        ("bullet", "Categories defined by merchant identity rather than "
                   "vocabulary remain weak however the model is tuned, because "
                   "the necessary information is absent from the text."),
        ("h1", "5. For the report"),
        ("p", "TF-IDF with multinomial Naive Bayes reaches a macro-F1 of "
              f"{strat['macro_f1']:.4f} where merchants recur between training "
              f"and evaluation, {merch['macro_f1']:.4f} when merchants are held "
              f"out, and {novel_r['macro_f1']:.4f} on strictly novel merchants. "
              f"Training takes {train_seconds:.2f} seconds, which establishes a "
              "demanding cost baseline for the more expensive models that "
              "follow."),
        ])

    out = ROOT / "results" / "model2_naive_bayes.json"
    out.write_text(json.dumps({
        "model": MODEL,
        "machine_learning": True,
        "train_seconds": round(train_seconds, 3),
        "n_features": n_features,
        "results": results,
        "per_class_f1": {
            "labels": labels,
            "stratified": [float(v) for v in f1_stratified],
            "merchant": [float(v) for v in f1_merchant],
        },
        "largest_confusions": [
            {"true": t, "predicted": p, "share": round(s, 4), "rows": n}
            for t, p, s, n in confusions[:12]],
    }, indent=2), encoding="utf-8")
    print(f"\n[PASS] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
