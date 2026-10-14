#!/usr/bin/env python3
"""Unit 2, final step: score the chosen configuration on the TEST sets, once.

Everything until now has been measured on validation, so that the comparison
between models did not contaminate the figure finally reported. This script
spends that saved budget exactly once.

Configuration under test
    cascade = deterministic rules -> merchant memory -> MiniLM embeddings with a
              logistic-regression head -> confidence gate

Following standard practice, once a configuration has been selected on
validation it is refitted on train + validation before the single test run, so
no labelled data is wasted. The test rows have never been seen by any fitting,
any threshold choice, or any model comparison.

Run:
    ~/.venvs/oct14/bin/python scripts/run_unit2_final_test.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models.minilm_embeddings import build_classifier, load_encoder  # noqa: E402
from models.cascade import Cascade  # noqa: E402
from evaluate_model import (calibration, embedding_cache, load,  # noqa: E402
                            strictly_novel)

THRESHOLD = 0.60


class MiniLMModel:
    """Adapter giving the cascade the probability interface it expects."""

    def __init__(self, encoder, classifier, classes):
        self.encoder = encoder
        self.classifier = classifier
        self._classes = classes

    @property
    def classes(self):
        return self._classes

    def predict_proba(self, texts):
        vectors = self.encoder.encode(list(texts), batch_size=256,
                                      convert_to_numpy=True,
                                      normalize_embeddings=True,
                                      show_progress_bar=False)
        return self.classifier.predict_proba(vectors)


def embed(encoder, texts, key):
    cache = embedding_cache() / f"{key}.npy"
    if cache.exists():
        stored = np.load(cache)
        if len(stored) == len(texts):
            return stored
    print(f"   encoding {len(texts):,} texts ({key}) ...", flush=True)
    vectors = encoder.encode(texts, batch_size=256, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
    np.save(cache, vectors)
    return vectors


def evaluate(decisions, gold, labels):
    predictions = [d.category for d in decisions]
    correct = [g == p for g, p in zip(gold, predictions, strict=True)]
    accepted = [i for i, d in enumerate(decisions) if not d.needs_review]
    layers = {}
    for layer in ("rule", "memory", "model"):
        rows = [i for i, d in enumerate(decisions) if d.layer == layer]
        layers[layer] = {
            "share": len(rows) / len(decisions),
            "n": len(rows),
            "accuracy": (sum(correct[i] for i in rows) / len(rows)) if rows else 0.0,
        }
    return {
        "n": len(gold),
        "accuracy": sum(correct) / len(correct),
        "macro_f1": float(f1_score(gold, predictions, average="macro",
                                   labels=labels, zero_division=0)),
        "micro_f1": float(f1_score(gold, predictions, average="micro",
                                   labels=labels, zero_division=0)),
        "review_rate": 1 - len(accepted) / len(decisions),
        "accepted_accuracy": (sum(correct[i] for i in accepted) / len(accepted))
                             if accepted else 0.0,
        "layers": layers,
        "per_class": classification_report(gold, predictions, labels=labels,
                                           output_dict=True, zero_division=0),
    }


def main() -> int:
    print("loading the frozen encoder ...", flush=True)
    encoder = load_encoder()
    report = {}

    for protocol, prefix, emb in (("in_distribution", "stratified", "strat"),
                                  ("unseen_merchant", "merchant", "merch")):
        print(f"\n=== {protocol} ===", flush=True)
        # refit on train + validation now that the configuration is fixed
        fit_rows = load(f"{prefix}_train.csv") + load(f"{prefix}_validation.csv")
        test_rows = load(f"{prefix}_test.csv")
        labels = sorted({r["category"] for r in fit_rows})

        train_vectors = np.vstack([
            embed(encoder, [r["normalised_description"]
                            for r in load(f"{prefix}_train.csv")],
                  f"norm_{emb}_train"),
            embed(encoder, [r["normalised_description"]
                            for r in load(f"{prefix}_validation.csv")],
                  f"norm_{emb}_validation"),
        ])
        test_vectors = embed(
            encoder, [r["normalised_description"] for r in test_rows],
            f"norm_{emb}_test")

        classifier = build_classifier()
        started = time.perf_counter()
        classifier.fit(train_vectors, [r["category"] for r in fit_rows])
        fit_seconds = time.perf_counter() - started
        classes = [str(c) for c in classifier.classes_]

        # the cascade, with the model slot pre-scored so the encoder is not
        # called twice for the same rows
        cascade = Cascade(MiniLMModel(encoder, classifier, classes),
                          threshold=THRESHOLD)
        cascade.learn_memory(fit_rows)

        class Precomputed:
            def __init__(self, vectors, rows):
                self.lookup = {r["normalised_description"]: i
                               for i, r in enumerate(rows)}
                self.vectors = vectors

            @property
            def classes(self):
                return classes

            def predict_proba(self, texts):
                rows = [self.lookup[t] for t in texts]
                return classifier.predict_proba(self.vectors[rows])

        cascade.model = Precomputed(test_vectors, test_rows)

        started = time.perf_counter()
        decisions = cascade.predict([r["normalised_description"]
                                     for r in test_rows])
        predict_seconds = time.perf_counter() - started
        gold = [r["category"] for r in test_rows]
        result = evaluate(decisions, gold, labels)
        result |= {"fit_seconds": round(fit_seconds, 2),
                   "predict_seconds": round(predict_seconds, 3)}

        # the model layer's own calibration, on the rows that reached it
        model_rows = [i for i, d in enumerate(decisions) if d.layer == "model"]
        if model_rows:
            probs = classifier.predict_proba(test_vectors[model_rows])
            result["model_layer_calibration"] = {
                k: v for k, v in
                calibration([gold[i] for i in model_rows], probs, classes).items()
                if k != "bins"}

        report[protocol] = result
        print(f"   macro-F1 {result['macro_f1']:.4f}   "
              f"accuracy {result['accuracy']:.4f}   "
              f"review {result['review_rate']:.1%}   "
              f"accepted accuracy {result['accepted_accuracy']:.4f}")
        print(f"   layers: rule {result['layers']['rule']['share']:.0%} / "
              f"memory {result['layers']['memory']['share']:.0%} / "
              f"model {result['layers']['model']['share']:.0%}")

        if protocol == "unseen_merchant":
            novel = strictly_novel(fit_rows, test_rows)
            index = {r["transaction_id"]: i for i, r in enumerate(test_rows)}
            rows = [index[r["transaction_id"]] for r in novel]
            novel_result = evaluate([decisions[i] for i in rows],
                                    [gold[i] for i in rows], labels)
            report["strictly_novel"] = novel_result
            print(f"   strictly novel subset: n={novel_result['n']:,}  "
                  f"macro-F1 {novel_result['macro_f1']:.4f}  "
                  f"accuracy {novel_result['accuracy']:.4f}")

    print("\n" + "=" * 86)
    print("FINAL TEST-SET RESULT - cascade with MiniLM, threshold "
          f"{THRESHOLD}")
    print("These sets were untouched until now and are not reused.")
    print("=" * 86)
    print(f"{'protocol':<22}{'n':>8}{'accuracy':>11}{'macro-F1':>11}"
          f"{'review':>10}{'accepted acc':>14}")
    for name in ("in_distribution", "unseen_merchant", "strictly_novel"):
        r = report[name]
        print(f"{name:<22}{r['n']:>8,}{r['accuracy']:>11.4f}"
              f"{r['macro_f1']:>11.4f}{r['review_rate']:>10.1%}"
              f"{r['accepted_accuracy']:>14.4f}")
    print("=" * 86)

    out = ROOT / "results" / "unit2_final_test.json"
    out.write_text(json.dumps({
        "configuration": {
            "cascade": "rules -> merchant memory -> MiniLM + logistic head "
                       "-> confidence gate",
            "model": "all-MiniLM-L6-v2 (frozen) + logistic regression",
            "threshold": THRESHOLD,
            "refit_on": "train + validation",
        },
        "note": "test sets used exactly once; never used for selection or tuning",
        "results": report,
    }, indent=2), encoding="utf-8")
    print(f"[PASS] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
