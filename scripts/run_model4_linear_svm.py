#!/usr/bin/env python3
"""Unit 2, Model 4 of 5: calibrated linear SVM over TF-IDF features.

Run:
    .venv/bin/python scripts/run_model4_linear_svm.py
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
from reporting import (ACCENT, BAD, GOOD, confusion_heatmap, grouped_bars,  # noqa: E402
                       per_class_bars, reliability_diagram, write_doc)
from models.linear_svm import build_classifier  # noqa: E402
from models.tfidf_features import build_vectoriser  # noqa: E402

MODEL = "TF-IDF + Linear SVM (calibrated)"


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


def main() -> int:
    results, calibrations = {}, {}

    s_train = load("stratified_train.csv")
    s_validation = load("stratified_validation.csv")
    vec_s, clf_s, train_seconds, n_features = fit(s_train)
    print(f"fitted on {len(s_train):,} rows in {train_seconds:.2f}s "
          f"({n_features:,} features; the SVM is fitted three times plus a "
          f"calibrator)")

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

    print_table(f"MODEL 4 of 5 - {MODEL.upper()}", results)

    print("\ncalibration (the sigmoid wrapper is meant to make this usable)")
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

    # ---- figures ----------------------------------------------------------
    print("\nfigures")
    names = list(results)
    short = ["Stratified", "Merchant held out", "Strictly novel"]
    fig1 = grouped_bars(
        "Linear SVM: accuracy and macro-F1 by protocol", "", short,
        [("Accuracy", [results[n]["accuracy"] for n in names], ACCENT),
         ("Macro-F1", [results[n]["macro_f1"] for n in names], GOOD)],
        "model4_protocols.png", percent=True)

    fig2 = per_class_bars(
        "Linear SVM: F1 per category (merchant held out)",
        labels, list(f1_merchant), "model4_per_class_f1.png", counts=counts)

    matrix = confusion_matrix(gold_m, predicted_m, labels=labels)
    fig3 = confusion_heatmap(
        "Linear SVM: what gets confused with what (merchant held out)",
        labels, matrix, "model4_confusion.png")

    fig4 = reliability_diagram(
        "Linear SVM: is the calibrated confidence trustworthy?",
        calibrations["merchant"]["bins"], "model4_reliability.png",
        ece=calibrations["merchant"]["ece"],
        subtitle="merchant held out. Bars above the diagonal mean the model is "
                 "under-confident; below means over-confident.")

    # calibration comparison against logistic regression
    previous = json.loads(
        (ROOT / "results" / "model3_logistic_regression.json").read_text())
    protocols = ["stratified", "merchant", "strictly_novel"]
    fig5 = grouped_bars(
        "Calibration error: does wrapping an SVM match a model fitted on "
        "probabilities?",
        "expected calibration error (lower is better)",
        ["Stratified", "Merchant held out", "Strictly novel"],
        [("Logistic regression",
          [previous["calibration"][p]["ece"] for p in protocols], GOOD),
         ("Linear SVM (calibrated)",
          [calibrations[p]["ece"] for p in protocols], BAD)],
        "model4_calibration_vs_logreg.png", percent=False)

    data = np.array(matrix, dtype=float)
    totals = data.sum(axis=1, keepdims=True)
    shares = np.divide(data, totals, out=np.zeros_like(data), where=totals != 0)
    confusions = sorted(
        [(labels[i], labels[j], shares[i, j], int(data[i, j]))
         for i in range(len(labels)) for j in range(len(labels))
         if i != j and shares[i, j] >= 0.08],
        key=lambda c: -c[2])
    print("\nlargest confusions (merchant held out)")
    for t, p, s, n in confusions[:6]:
        print(f"  {t:<16} -> {p:<16}{s:>7.1%}  ({n})")

    # ---- document ---------------------------------------------------------
    print("\ndocument")
    strat, merch, novel_r = (results[n] for n in names)
    lr = previous["results"]
    cal_m = calibrations["merchant"]
    lr_cal_m = previous["calibration"]["merchant"]
    worst = [labels[i] for i in order[:4]]
    best = [labels[i] for i in order[-4:]][::-1]
    better_f1 = merch["macro_f1"] > lr["merchant held out (validation)"]["macro_f1"]
    worse_cal = cal_m["ece"] > lr_cal_m["ece"]

    write_doc(
        "model4_linear_svm.docx",
        "Model 4 of 5 - TF-IDF with a Calibrated Linear SVM",
        "Unit 2, model comparison. The strongest classical linear model, and the "
        "one that has to be wrapped before its confidence can be used.",
        [
        ("h1", "1. What this model is"),
        ("p", "A linear support vector machine finds the separating boundary with "
              "the widest margin between classes. On sparse, high-dimensional "
              "text it is usually the strongest of the classical linear models, "
              "which makes it the natural test of how much accuracy this feature "
              "representation can yield at all. It uses the same TF-IDF features "
              "as Models 2 and 3."),
        ("h2", "1.1 Why it needs wrapping"),
        ("p", "**The SVM produces distances, not probabilities.** It reports how "
              "far a point lies from each boundary, and those numbers carry no "
              "probabilistic meaning: a distance of 2.0 does not indicate twice "
              "the likelihood. Because this project routes low-confidence "
              "predictions to human review, a raw distance is unusable as a "
              "confidence score."),
        ("p", "CalibratedClassifierCV therefore wraps the SVM and learns the "
              "mapping from distance to probability by fitting a sigmoid on "
              "held-out folds of the training data, a procedure known as Platt "
              "scaling. This is a second model fitted on top of the first. "
              "Whether it yields trustworthy confidence is an empirical question, "
              "and answering it is the main reason this model is interesting here."),
        ("p", f"The cost is time. With three folds the SVM is fitted three times "
              f"plus the calibrator, taking {train_seconds:.2f} seconds against "
              f"{previous['train_seconds']:.2f} for logistic regression and "
              "0.58 for Naive Bayes."),
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
              f"{', '.join(worst)}. The ordering is consistent with Models 2 and "
              "3, which indicates the limit lies in what the description contains "
              "rather than in any particular classifier."),
        ("figure", (fig3, "Figure 3. Confusion matrix under the merchant-held-out "
                          "protocol, each row normalised to its own total.")),
        ("table", [["True category", "Predicted as", "Share", "Rows"]] +
                  [[t, p, f"{s:.1%}", f"{n:,}"] for t, p, s, n in confusions[:6]]),
        ("h1", "3. Calibration: did the wrapper work?"),
        ("table", [["Protocol", "ECE", "MCE", "Brier", "Mean confidence",
                    "Accuracy", "Gap"]] +
                  [[name.replace("_", " "), f"{c['ece']:.4f}", f"{c['mce']:.4f}",
                    f"{c['brier']:.4f}", f"{c['mean_confidence']:.4f}",
                    f"{c['accuracy']:.4f}", f"{c['overconfidence']:+.4f}"]
                   for name, c in calibrations.items()]),
        ("figure", (fig4, "Figure 4. Reliability diagram, merchant held out.")),
        ("figure", (fig5, "Figure 5. Expected calibration error compared with "
                          "logistic regression across all three protocols. Lower "
                          "is better.")),
        ("p", f"On the merchant-held-out protocol the wrapped SVM records an "
              f"expected calibration error of {cal_m['ece']:.4f} against "
              f"{lr_cal_m['ece']:.4f} for logistic regression, with mean "
              f"confidence {cal_m['mean_confidence']:.4f} against accuracy "
              f"{cal_m['accuracy']:.4f}, a gap of {cal_m['overconfidence']:+.4f}."),
        ("p", ("Calibrating a model after the fact is therefore "
               + ("measurably worse than fitting one on probabilities from the "
                  "start, which is what logistic regression does. "
                  if worse_cal else
                  "competitive with fitting one on probabilities from the start. ")
               + "This matters because the deployed system uses the confidence "
                 "score to decide what a human should check, so the comparison "
                 "between these two models is not settled by macro-F1 alone."),),
        ("h1", "4. Comparison so far"),
        ("table", [["Model", "Stratified", "Merchant held out", "Strictly novel",
                    "Merchant ECE", "Train (s)"],
                   ["1. Rule baseline (17% coverage)", "0.2924", "0.2849",
                    "0.3068", "n/a", "0.00"],
                   ["2. Naive Bayes", "0.9809", "0.5903", "0.4657", "n/a", "0.58"],
                   ["3. Logistic regression",
                    f"{lr['stratified (validation)']['macro_f1']:.4f}",
                    f"{lr['merchant held out (validation)']['macro_f1']:.4f}",
                    f"{lr['  of which strictly novel']['macro_f1']:.4f}",
                    f"{lr_cal_m['ece']:.4f}", f"{previous['train_seconds']:.2f}"],
                   ["4. Linear SVM (calibrated)", f"{strat['macro_f1']:.4f}",
                    f"{merch['macro_f1']:.4f}", f"{novel_r['macro_f1']:.4f}",
                    f"{cal_m['ece']:.4f}", f"{train_seconds:.2f}"]]),
        ("p", ("The SVM " + ("edges ahead of logistic regression on macro-F1 with "
                             "merchants held out"
                             if better_f1 else
                             "does not improve on logistic regression with "
                             "merchants held out")
               + ", while costing substantially more to train. Model selection "
                 "for this project therefore cannot rest on macro-F1 alone: the "
                 "deployed model also has to supply confidence the review gate "
                 "can act on, and an explanation the interface can display."),),
        ("h1", "5. Limitations"),
        ("bullet", "The probabilities are produced by a sigmoid fitted after the "
                   "fact rather than by the model itself, so they inherit "
                   "whatever the calibration procedure gets wrong."),
        ("bullet", "Explanation is harder than for logistic regression. The "
                   "underlying SVM is linear and has weights, but the deployed "
                   "object is an ensemble of three fitted folds wrapped in a "
                   "calibrator, so there is no single weight vector to read."),
        ("bullet", f"Training takes {train_seconds:.2f} seconds, roughly "
                   f"{train_seconds / 0.58:.0f} times Naive Bayes, which has to "
                   "be justified by the results rather than assumed."),
        ("h1", "6. For the report"),
        ("p", f"A calibrated linear SVM over the same TF-IDF features reaches a "
              f"macro-F1 of {strat['macro_f1']:.4f} on familiar merchants, "
              f"{merch['macro_f1']:.4f} with merchants held out and "
              f"{novel_r['macro_f1']:.4f} on strictly novel merchants, with an "
              f"expected calibration error of {cal_m['ece']:.4f}. It confirms "
              "that the ceiling of this representation has been reached: three "
              "different linear classifiers over identical features agree to "
              "within a few points on every protocol, and all of them fall by "
              "roughly half when the merchant becomes unfamiliar."),
        ])

    out = ROOT / "results" / "model4_linear_svm.json"
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
    }, indent=2), encoding="utf-8")
    print(f"\n[PASS] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
