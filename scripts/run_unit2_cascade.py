#!/usr/bin/env python3
"""Unit 2, step 2: evaluate the assembled cascade, not the models in isolation.

The deployed system is not a single model. It is three layers in order:

    1. deterministic rules      answer when banking vocabulary is recognisable
    2. merchant memory          answer when this shop has been seen before
    3. a learned model          answer everything else, and flag low confidence
                                for human review

Every layer has been measured alone. None of the model documents says what the
assembled system does, which is the thing actually being deployed. This script
answers that, and sweeps the confidence threshold so the review burden can be
chosen deliberately rather than guessed.

Each candidate model is placed in slot 3 in turn, so the choice of model can be
made on how the whole system behaves rather than on the model's own score.

Run:
    ~/.venvs/oct14/bin/python scripts/run_unit2_cascade.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_model import embedding_cache, load, strictly_novel  # noqa: E402
from reporting import ACCENT, BAD, GOOD, MUTED, WARN, save  # noqa: E402
from models.merchant_memory import MerchantMemory  # noqa: E402
from models.rule_baseline import predict as rule_predict  # noqa: E402
from models.tfidf_features import build_vectoriser  # noqa: E402

THRESHOLDS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def fit_model(kind, train_rows, embedding_file=None):
    """Return a function mapping rows to (probabilities, classes)."""
    labels = [r["category"] for r in train_rows]
    if kind == "minilm":
        from models.minilm_embeddings import build_classifier
        x = np.load(embedding_cache() / embedding_file)
        classifier = build_classifier()
        classifier.fit(x, labels)
        classes = [str(c) for c in classifier.classes_]

        def predict(_rows, eval_embedding):
            return classifier.predict_proba(
                np.load(embedding_cache() / eval_embedding)), classes
        return predict

    if kind == "logreg":
        from models.logistic_regression import build_classifier
    elif kind == "svm":
        from models.linear_svm import build_classifier
    elif kind == "nb":
        from models.naive_bayes import build_classifier
    else:
        raise ValueError(kind)

    vectoriser = build_vectoriser()
    features = vectoriser.fit_transform(
        [r["normalised_description"] for r in train_rows])
    classifier = build_classifier()
    classifier.fit(features, labels)
    classes = [str(c) for c in classifier.classes_]

    def predict(rows, _eval_embedding=None):
        return classifier.predict_proba(vectoriser.transform(
            [r["normalised_description"] for r in rows])), classes
    return predict


def run_cascade(rows, memory, probabilities, classes, threshold):
    """Route each row through rules, then memory, then the model."""
    out = []
    for row, probs in zip(rows, probabilities, strict=True):
        text = row["normalised_description"]
        rule, _evidence = rule_predict(text)
        if rule is not None:
            out.append({"prediction": rule, "layer": "rule", "confidence": 1.0,
                        "review": False})
            continue
        remembered = memory.predict(text)
        if remembered is not None:
            out.append({"prediction": remembered, "layer": "memory",
                        "confidence": 1.0, "review": False})
            continue
        best = int(np.argmax(probs))
        confidence = float(probs[best])
        out.append({"prediction": classes[best], "layer": "model",
                    "confidence": confidence, "review": confidence < threshold})
    return out


def summarise(gold, decisions):
    predictions = [d["prediction"] for d in decisions]
    correct = [g == p for g, p in zip(gold, predictions, strict=True)]
    accepted = [i for i, d in enumerate(decisions) if not d["review"]]
    layers = {}
    for layer in ("rule", "memory", "model"):
        rows = [i for i, d in enumerate(decisions) if d["layer"] == layer]
        layers[layer] = {
            "share": len(rows) / len(decisions) if decisions else 0.0,
            "accuracy": (sum(correct[i] for i in rows) / len(rows)) if rows else 0.0,
        }
    return {
        "n": len(gold),
        "accuracy": sum(correct) / len(correct),
        "macro_f1": float(f1_score(gold, predictions, average="macro",
                                   zero_division=0)),
        "review_rate": 1 - len(accepted) / len(decisions),
        "accepted_accuracy": (sum(correct[i] for i in accepted) / len(accepted))
                             if accepted else 0.0,
        "layers": layers,
    }


def main() -> int:
    protocols = {
        "stratified": ("stratified_train.csv", "stratified_validation.csv",
                       "norm_strat_train.npy", "norm_strat_validation.npy"),
        "merchant": ("merchant_train.csv", "merchant_validation.csv",
                     "norm_merch_train.npy", "norm_merch_validation.npy"),
    }
    candidates = ["logreg", "svm", "minilm", "nb"]
    pretty = {"logreg": "Logistic regression", "svm": "Linear SVM",
              "minilm": "MiniLM embeddings", "nb": "Naive Bayes"}
    results = {}

    for protocol, (train_file, eval_file, emb_train, emb_eval) in protocols.items():
        train_rows = load(train_file)
        eval_rows = load(eval_file)
        memory = MerchantMemory()
        memory.learn(train_rows)
        gold = [r["category"] for r in eval_rows]

        novel_rows = strictly_novel(train_rows, eval_rows) if protocol == "merchant" else []
        novel_index = [i for i, r in enumerate(eval_rows)
                       if r in novel_rows] if novel_rows else []

        print(f"\n=== protocol: {protocol} "
              f"({len(train_rows):,} train / {len(eval_rows):,} eval) ===")
        print(f"    memory holds {len(memory):,} merchants")

        for kind in candidates:
            started = time.perf_counter()
            predictor = fit_model(kind, train_rows, emb_train)
            probabilities, classes = predictor(eval_rows, emb_eval)
            elapsed = time.perf_counter() - started

            sweep = []
            for threshold in THRESHOLDS:
                decisions = run_cascade(eval_rows, memory, probabilities,
                                        classes, threshold)
                entry = summarise(gold, decisions)
                entry["threshold"] = threshold
                if novel_index:
                    novel_gold = [gold[i] for i in novel_index]
                    novel_dec = [decisions[i] for i in novel_index]
                    entry["strictly_novel"] = summarise(novel_gold, novel_dec)
                sweep.append(entry)

            results.setdefault(protocol, {})[kind] = {
                "fit_seconds": round(elapsed, 2), "sweep": sweep}

            base = sweep[0]
            print(f"  {pretty[kind]:<22} cascade macro-F1 {base['macro_f1']:.4f}  "
                  f"accuracy {base['accuracy']:.4f}   "
                  f"layers: rule {base['layers']['rule']['share']:.0%} / "
                  f"memory {base['layers']['memory']['share']:.0%} / "
                  f"model {base['layers']['model']['share']:.0%}")

    # ---------------------------------------------------------- reporting --
    print("\n" + "=" * 92)
    print("CASCADE AGAINST THE MODEL ALONE (validation)")
    print("=" * 92)
    alone = {
        "logreg": "model3_logistic_regression", "svm": "model4_linear_svm",
        "minilm": "model5_minilm", "nb": "model2_naive_bayes",
    }
    keys = {"stratified": "stratified (validation)",
            "merchant": "merchant held out (validation)"}
    print(f"{'model':<22}{'protocol':<14}{'alone':>10}{'in cascade':>13}{'diff':>9}")
    gains = {}
    for protocol in protocols:
        for kind in candidates:
            solo = json.loads(
                (ROOT / "results" / f"{alone[kind]}.json").read_text()
            )["results"][keys[protocol]]["macro_f1"]
            casc = results[protocol][kind]["sweep"][0]["macro_f1"]
            gains[(protocol, kind)] = casc - solo
            print(f"{pretty[kind]:<22}{protocol:<14}{solo:>10.4f}"
                  f"{casc:>13.4f}{casc - solo:>+9.4f}")

    print("\n" + "=" * 92)
    print("REVIEW BURDEN: how the confidence threshold trades accuracy for "
          "human effort")
    print("merchant-held-out protocol, which is what a new user meets")
    print("=" * 92)
    print(f"{'model':<22}{'threshold':>11}{'sent to review':>16}"
          f"{'auto-accepted acc':>20}")
    for kind in candidates:
        for entry in results["merchant"][kind]["sweep"]:
            if entry["threshold"] in (0.0, 0.5, 0.6, 0.7, 0.8):
                print(f"{pretty[kind]:<22}{entry['threshold']:>11.2f}"
                      f"{entry['review_rate']:>15.1%}"
                      f"{entry['accepted_accuracy']:>20.4f}")
        print()

    out = ROOT / "results" / "unit2_cascade.json"
    out.write_text(json.dumps({
        "description": "rules -> merchant memory -> model, evaluated end to end",
        "thresholds": THRESHOLDS,
        "results": results,
        "cascade_gain_over_model_alone": {f"{p}/{k}": round(v, 4)
                                          for (p, k), v in gains.items()},
    }, indent=2), encoding="utf-8")
    print(f"[PASS] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
