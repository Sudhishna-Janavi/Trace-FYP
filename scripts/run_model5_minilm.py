#!/usr/bin/env python3
"""Unit 2, Model 5 of 5: MiniLM sentence embeddings with a logistic-regression head.

Also answers a question deferred from Unit 1. Every model so far has been fed
the shared normalised text, which is uppercase with long digit runs replaced by
<NUM>. That representation suits TF-IDF but is unlike the natural sentences
MiniLM was trained on. This script therefore runs the encoder twice - once on
the shared normalised text, once on lightly cleaned text that keeps natural case
and punctuation - so the comparison is not quietly handicapping the one model it
could disadvantage.

Run:
    .venv/bin/python scripts/run_model5_minilm.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# importing this first sets the offline environment variables
from models.minilm_embeddings import (EMBEDDING_DIM, build_classifier,  # noqa: E402
                                      load_encoder)

import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from evaluate_model import (calibration, load, print_table, score,  # noqa: E402
                            strictly_novel)
from reporting import (ACCENT, BAD, GOOD, confusion_heatmap, grouped_bars,  # noqa: E402
                       per_class_bars, reliability_diagram, write_doc)

MODEL = "MiniLM embeddings + Logistic Regression"
from evaluate_model import embedding_cache  # noqa: E402
CACHE = embedding_cache()


def light_clean(raw: str) -> str:
    """Minimal cleaning that keeps text looking like natural language.

    The direction marker becomes an ordinary word so the model still receives
    it, but case, punctuation and digits are left alone.
    """
    return re.sub(r"^\[(debit|credit)\]\s*", r"\1 ", raw, flags=re.IGNORECASE).strip()


def encode(encoder, texts: list[str], key: str) -> np.ndarray:
    """Encode with an on-disk cache, so reruns do not repeat the work."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.npy"
    if path.exists():
        cached = np.load(path)
        if len(cached) == len(texts):
            return cached
    print(f"   encoding {len(texts):,} texts ({key}) ...", flush=True)
    started = time.perf_counter()
    vectors = encoder.encode(texts, batch_size=256, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
    print(f"      {time.perf_counter() - started:.1f}s", flush=True)
    np.save(path, vectors)
    return vectors


def run_variant(encoder, field, tag: str):
    """Fit and evaluate all three protocols for one text representation."""
    results, calibrations, extra = {}, {}, {}

    s_train = load("stratified_train.csv")
    s_validation = load("stratified_validation.csv")
    train_vectors = encode(encoder, [field(r) for r in s_train], f"{tag}_strat_train")
    validation_vectors = encode(encoder, [field(r) for r in s_validation],
                                f"{tag}_strat_validation")

    classifier = build_classifier()
    started = time.perf_counter()
    classifier.fit(train_vectors, [r["category"] for r in s_train])
    head_seconds = time.perf_counter() - started

    classes = [str(c) for c in classifier.classes_]
    probabilities = classifier.predict_proba(validation_vectors)
    predicted_s = [classes[i] for i in probabilities.argmax(1)]
    gold_s = [r["category"] for r in s_validation]
    results["stratified (validation)"] = score(gold_s, predicted_s)
    calibrations["stratified"] = calibration(gold_s, probabilities, classes)

    m_train = load("merchant_train.csv")
    m_validation = load("merchant_validation.csv")
    mt_vectors = encode(encoder, [field(r) for r in m_train], f"{tag}_merch_train")
    mv_vectors = encode(encoder, [field(r) for r in m_validation],
                        f"{tag}_merch_validation")

    classifier_m = build_classifier()
    classifier_m.fit(mt_vectors, [r["category"] for r in m_train])
    classes_m = [str(c) for c in classifier_m.classes_]
    prob_m = classifier_m.predict_proba(mv_vectors)
    predicted_m = [classes_m[i] for i in prob_m.argmax(1)]
    gold_m = [r["category"] for r in m_validation]
    results["merchant held out (validation)"] = score(gold_m, predicted_m)
    calibrations["merchant"] = calibration(gold_m, prob_m, classes_m)

    novel = strictly_novel(m_train, m_validation)
    novel_index = {r["transaction_id"]: i for i, r in enumerate(m_validation)}
    rows = [novel_index[r["transaction_id"]] for r in novel]
    prob_n = prob_m[rows]
    predicted_n = [classes_m[i] for i in prob_n.argmax(1)]
    gold_n = [r["category"] for r in novel]
    results["  of which strictly novel"] = score(gold_n, predicted_n)
    calibrations["strictly_novel"] = calibration(gold_n, prob_n, classes_m)

    extra.update({"head_seconds": head_seconds, "gold_s": gold_s,
                  "pred_s": predicted_s, "gold_m": gold_m, "pred_m": predicted_m})
    return results, calibrations, extra


def main() -> int:
    print(f"loading encoder ({EMBEDDING_DIM}-dimensional, frozen) ...", flush=True)
    started = time.perf_counter()
    encoder = load_encoder()
    print(f"   loaded in {time.perf_counter() - started:.1f}s", flush=True)

    print("\n--- variant A: the shared normalised text (as used by Models 1-4) ---")
    results, calibrations, extra = run_variant(
        encoder, lambda r: r["normalised_description"], "norm")
    print_table(f"MODEL 5 of 5 - {MODEL.upper()}", results)

    print("\ncalibration")
    print(f"  {'protocol':<18}{'ECE':>9}{'mean conf':>11}{'accuracy':>10}{'gap':>9}")
    for name, c in calibrations.items():
        print(f"  {name:<18}{c['ece']:>9.4f}{c['mean_confidence']:>11.4f}"
              f"{c['accuracy']:>10.4f}{c['overconfidence']:>+9.4f}")

    print("\n--- variant B: lightly cleaned text, natural case and punctuation ---")
    light_results, light_cal, _extra = run_variant(
        encoder, lambda r: light_clean(r["raw_description"]), "light")
    print(f"  {'protocol':<34}{'normalised':>12}{'light':>9}{'diff':>9}")
    for key in results:
        a = results[key]["macro_f1"]
        b = light_results[key]["macro_f1"]
        print(f"  {key.strip():<34}{a:>12.4f}{b:>9.4f}{b - a:>+9.4f}")

    gold_s, predicted_s = extra["gold_s"], extra["pred_s"]
    gold_m, predicted_m = extra["gold_m"], extra["pred_m"]
    labels = sorted(set(gold_s))
    f1_stratified = f1_score(gold_s, predicted_s, average=None, labels=labels,
                             zero_division=0)
    f1_merchant = f1_score(gold_m, predicted_m, average=None, labels=labels,
                           zero_division=0)
    counts = [gold_s.count(label) for label in labels]
    order = sorted(range(len(labels)), key=lambda i: f1_merchant[i])

    print("\nper-category F1 (normalised text)")
    print(f"  {'category':<18}{'stratified':>12}{'merchant':>12}{'drop':>9}")
    for i in order:
        print(f"  {labels[i]:<18}{f1_stratified[i]:>12.3f}{f1_merchant[i]:>12.3f}"
              f"{f1_stratified[i]-f1_merchant[i]:>9.3f}")

    # ---- figures ----------------------------------------------------------
    print("\nfigures")
    names = list(results)
    short = ["Stratified", "Merchant held out", "Strictly novel"]
    fig1 = grouped_bars(
        "MiniLM embeddings: accuracy and macro-F1 by protocol", "", short,
        [("Accuracy", [results[n]["accuracy"] for n in names], ACCENT),
         ("Macro-F1", [results[n]["macro_f1"] for n in names], GOOD)],
        "model5_protocols.png", percent=True)

    fig2 = per_class_bars(
        "MiniLM embeddings: F1 per category (merchant held out)",
        labels, list(f1_merchant), "model5_per_class_f1.png", counts=counts)

    matrix = confusion_matrix(gold_m, predicted_m, labels=labels)
    fig3 = confusion_heatmap(
        "MiniLM embeddings: what gets confused with what (merchant held out)",
        labels, matrix, "model5_confusion.png")

    fig4 = reliability_diagram(
        "MiniLM embeddings: is its confidence trustworthy?",
        calibrations["merchant"]["bins"], "model5_reliability.png",
        ece=calibrations["merchant"]["ece"], subtitle="merchant held out.")

    fig5 = grouped_bars(
        "Does MiniLM prefer natural text? Shared normalisation against light cleaning",
        "macro-F1", short,
        [("Shared normalised text", [results[n]["macro_f1"] for n in names], ACCENT),
         ("Lightly cleaned text", [light_results[n]["macro_f1"] for n in names], BAD)],
        "model5_text_variant.png", percent=False)

    # ---- comparison against the TF-IDF models -----------------------------
    others = {}
    for key, filename in (("Naive Bayes", "model2_naive_bayes"),
                          ("Logistic regression", "model3_logistic_regression"),
                          ("Linear SVM", "model4_linear_svm")):
        others[key] = json.loads((ROOT / "results" / f"{filename}.json").read_text())

    fig6 = grouped_bars(
        "Do learned semantics beat surface patterns on bank codes?",
        "macro-F1", short,
        [("TF-IDF + Logistic regression",
          [others["Logistic regression"]["results"][n]["macro_f1"] for n in names], GOOD),
         ("TF-IDF + Linear SVM",
          [others["Linear SVM"]["results"][n]["macro_f1"] for n in names], ACCENT),
         ("MiniLM + Logistic regression",
          [results[n]["macro_f1"] for n in names], BAD)],
        "model5_vs_tfidf.png", percent=False)

    data = np.array(matrix, dtype=float)
    totals = data.sum(axis=1, keepdims=True)
    shares = np.divide(data, totals, out=np.zeros_like(data), where=totals != 0)
    confusions = sorted(
        [(labels[i], labels[j], shares[i, j], int(data[i, j]))
         for i in range(len(labels)) for j in range(len(labels))
         if i != j and shares[i, j] >= 0.08], key=lambda c: -c[2])
    print("\nlargest confusions (merchant held out)")
    for t, p, s, n in confusions[:6]:
        print(f"  {t:<16} -> {p:<16}{s:>7.1%}  ({n})")

    out = ROOT / "results" / "model5_minilm.json"
    out.write_text(json.dumps({
        "model": MODEL, "machine_learning": True,
        "encoder": "all-MiniLM-L6-v2 (frozen, pre-trained)",
        "embedding_dim": EMBEDDING_DIM,
        "head_train_seconds": round(extra["head_seconds"], 3),
        "results": results,
        "results_light_text": light_results,
        "calibration": {k: {m: v for m, v in c.items() if m != "bins"}
                        for k, c in calibrations.items()},
        "calibration_light_text": {k: {m: v for m, v in c.items() if m != "bins"}
                                   for k, c in light_cal.items()},
        "per_class_f1": {"labels": labels,
                         "stratified": [float(v) for v in f1_stratified],
                         "merchant": [float(v) for v in f1_merchant]},
        "largest_confusions": [{"true": t, "predicted": p, "share": round(s, 4),
                                "rows": n} for t, p, s, n in confusions[:12]],
        "figures": [str(f.name) for f in (fig1, fig2, fig3, fig4, fig5, fig6)],
    }, indent=2), encoding="utf-8")
    print(f"\n[PASS] wrote {out}")
    print("[INFO] document is written by scripts/write_model5_doc.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
