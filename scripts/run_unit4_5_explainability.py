#!/usr/bin/env python3
"""Unit 4.5: measure and validate the explanation layer.

Three questions, in order of importance to the application.

1. How much does an explanation cost, per cascade layer? The answer decides
   whether the interface can show one instantly on click or needs a loading
   state, so it has to be measured before the interface is designed.

2. Is it stable? LIME fits a surrogate to sampled perturbations. If the same
   transaction yields different reasons on a refresh, a user would rightly
   distrust the feature.

3. Can the post-hoc methods be trusted at all? For the deployed embedding model
   there is no ground truth to compare against, so LIME and SHAP are first run
   on a linear model, where the exact contribution of every token is
   computable, and their agreement with it is measured. Passing that test is
   what earns the right to use them where truth is unavailable.

Run:
    ~/.venvs/oct14/bin/python scripts/run_unit4_5_explainability.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backend.explainability import (explain_lime, explain_shap,  # noqa: E402
                                    unique_tokens)
from evaluate_model import embedding_cache, load  # noqa: E402
from models.cascade import Cascade  # noqa: E402
from models.merchant_memory import MerchantMemory  # noqa: E402

SAMPLE = 60


class EmbeddingModel:
    """The deployed model: frozen encoder plus a logistic head."""

    def __init__(self, encoder, classifier, classes):
        self.encoder, self.classifier, self._classes = encoder, classifier, classes

    @property
    def classes(self):
        return self._classes

    def predict_proba(self, texts):
        vectors = self.encoder.encode(list(texts), batch_size=256,
                                      convert_to_numpy=True,
                                      normalize_embeddings=True,
                                      show_progress_bar=False)
        return self.classifier.predict_proba(vectors)


class LinearModel:
    """A TF-IDF linear model, used only as a testbed with known ground truth."""

    def __init__(self, vectoriser, classifier, classes):
        self.vectoriser, self.classifier, self._classes = (
            vectoriser, classifier, classes)
        names = vectoriser.get_feature_names_out()
        self.is_word = np.array(
            [str(n).startswith("tfidfvectorizer-1__") for n in names])
        self.names = np.array([str(n).split("__", 1)[-1] for n in names])

    @property
    def classes(self):
        return self._classes

    def predict_proba(self, texts):
        return self.classifier.predict_proba(self.vectoriser.transform(list(texts)))

    def exact_tokens(self, normalised, category):
        """The true contribution of each token: weight times value."""
        vector = self.vectoriser.transform([normalised])
        weights = self.classifier.coef_[self._classes.index(category)]
        out: dict[str, float] = {}
        for i in vector.nonzero()[1]:
            if not self.is_word[i]:
                continue
            name = str(self.names[i])
            if " " in name:
                continue
            out[name] = out.get(name, 0.0) + float(weights[i] * vector[0, i])
        return out


def _key(token: str) -> str:
    """Canonical form for comparing attributions across sources.

    The vectoriser lowercases and strips punctuation, so its feature names are
    'wendy' and 'num', while the normalised text yields 'WENDY' and '<NUM>'.
    Comparing them raw makes every token look like a mismatch and produces a
    top-3 overlap of exactly zero, which reads as total disagreement rather
    than as the lookup failure it is.
    """
    import re as _re
    return _re.sub(r"[^a-z0-9]+", "", token.lower())


def agreement(a: dict, b: dict):
    a = {_key(k): v for k, v in a.items() if _key(k)}
    b = {_key(k): v for k, v in b.items() if _key(k)}
    keys = sorted(set(a) | set(b))
    if len(keys) < 2:
        return None
    va = np.array([a.get(k, 0.0) for k in keys])
    vb = np.array([b.get(k, 0.0) for k in keys])
    if np.allclose(va, va[0]) or np.allclose(vb, vb[0]):
        return None
    rho = spearmanr(va, vb).statistic
    sign = float(np.mean(np.sign(va) == np.sign(vb)))
    top = lambda d: set(sorted(d, key=lambda k: -abs(d[k]))[:3])
    ta, tb = top(a), top(b)
    jaccard = len(ta & tb) / len(ta | tb) if (ta | tb) else 1.0
    return {"spearman": float(rho) if np.isfinite(rho) else 0.0,
            "sign_agreement": sign, "jaccard_top3": jaccard}


def main() -> int:
    report = {}

    # ================================================================== 1 ===
    print("=" * 78)
    print("PART 1  Validating LIME and SHAP where the truth is computable")
    print("=" * 78)
    from models.logistic_regression import build_classifier as linear_head
    from models.tfidf_features import build_vectoriser

    train = load("stratified_train.csv")
    validation = load("stratified_validation.csv")
    vectoriser = build_vectoriser()
    features = vectoriser.fit_transform(
        [r["normalised_description"] for r in train])
    classifier = linear_head()
    classifier.fit(features, [r["category"] for r in train])
    linear = LinearModel(vectoriser, classifier,
                         [str(c) for c in classifier.classes_])

    import random
    rng = random.Random(1131)
    sample = [r for r in validation
              if 2 <= len(unique_tokens(r["normalised_description"])) <= 12]
    rng.shuffle(sample)
    sample = sample[:SAMPLE]

    scores = {"lime": [], "shap": []}
    residuals, fidelities = [], []
    for row in sample:
        text = row["normalised_description"]
        probabilities = linear.predict_proba([text])[0]
        category = linear.classes[int(np.argmax(probabilities))]
        truth = linear.exact_tokens(text, category)
        for name, fn in (("lime", explain_lime), ("shap", explain_shap)):
            result = fn(linear, text, category)
            got = agreement(truth, dict(result.tokens))
            if got:
                scores[name].append(got)
            if name == "shap":
                residuals.append(result.detail.get("additivity_residual", 0.0))
            else:
                fidelities.append(result.detail.get("surrogate_r2", 0.0))

    print(f"\n{len(sample)} transactions, compared against exact contributions")
    print(f"  {'method':<8}{'Spearman':>11}{'sign agree':>13}{'top-3 Jaccard':>16}")
    for name in ("lime", "shap"):
        rows = scores[name]
        mean = lambda k: statistics.fmean(r[k] for r in rows)
        print(f"  {name:<8}{mean('spearman'):>11.4f}{mean('sign_agreement'):>13.4f}"
              f"{mean('jaccard_top3'):>16.4f}")
        report.setdefault("validation_on_linear_model", {})[name] = {
            k: round(mean(k), 4) for k in
            ("spearman", "sign_agreement", "jaccard_top3")}
    print(f"\n  SHAP additivity residual   {statistics.fmean(residuals):.2e}  "
          "(exact Shapley satisfies this to machine precision)")
    print(f"  LIME surrogate fidelity    {statistics.fmean(fidelities):.4f}  "
          "(how well its own surrogate fits)")
    report["validation_on_linear_model"]["shap_additivity_residual"] = \
        statistics.fmean(residuals)
    report["validation_on_linear_model"]["lime_surrogate_r2"] = \
        statistics.fmean(fidelities)
    report["validation_on_linear_model"]["n"] = len(sample)

    # ================================================================== 2 ===
    print("\n" + "=" * 78)
    print("PART 2  Cost and stability on the deployed model")
    print("=" * 78)
    from models.minilm_embeddings import build_classifier, load_encoder
    encoder = load_encoder()
    train_vectors = np.load(embedding_cache() / "norm_strat_train.npy")
    head = build_classifier()
    head.fit(train_vectors, [r["category"] for r in train])
    deployed = EmbeddingModel(encoder, head, [str(c) for c in head.classes_])

    memory = MerchantMemory()
    memory.learn(train)
    cascade = Cascade(deployed, threshold=0.60, memory=memory)

    demo = load("../demo/demo_transactions.csv") if False else None
    with (ROOT / "dataset" / "demo" / "demo_transactions.csv").open(
            encoding="utf-8", newline="") as handle:
        import csv as _csv
        demo = list(_csv.DictReader(handle))
    rng.shuffle(demo)
    subset = demo[:SAMPLE]
    decisions = cascade.predict([r["normalised_description"] for r in subset])

    layers = {}
    for decision, row in zip(decisions, subset, strict=True):
        layers.setdefault(decision.layer, []).append(row)
    print("\nwhich layer decided, in the demonstration data")
    for layer in ("rule", "memory", "model"):
        rows = layers.get(layer, [])
        print(f"  {layer:<10}{len(rows):>4} of {len(subset)}  "
              f"({len(rows)/len(subset):.0%})")

    timings = {}
    for name, fn in (("shap", explain_shap), ("lime", explain_lime)):
        model_rows = layers.get("model", [])[:20]
        durations, calls = [], []
        for row in model_rows:
            text = row["normalised_description"]
            probabilities = deployed.predict_proba([text])[0]
            category = deployed.classes[int(np.argmax(probabilities))]
            started = time.perf_counter()
            result = fn(deployed, text, category)
            durations.append((time.perf_counter() - started) * 1000)
            calls.append(result.model_calls)
        timings[name] = {"median_ms": statistics.median(durations),
                         "mean_ms": statistics.fmean(durations),
                         "max_ms": max(durations),
                         "mean_model_calls": statistics.fmean(calls),
                         "n": len(durations)}
        print(f"\n  {name.upper()} on the model layer: median "
              f"{timings[name]['median_ms']:.0f} ms, max "
              f"{timings[name]['max_ms']:.0f} ms, "
              f"{timings[name]['mean_model_calls']:.0f} encoder passes")
    report["cost"] = timings
    report["layer_shares"] = {k: len(v) / len(subset) for k, v in layers.items()}

    # stability: the same transaction explained repeatedly
    print("\nstability: explaining the same transaction five times")
    stability = {}
    for name, fn in (("shap", explain_shap), ("lime", explain_lime)):
        jaccards = []
        for row in layers.get("model", [])[:8]:
            text = row["normalised_description"]
            probabilities = deployed.predict_proba([text])[0]
            category = deployed.classes[int(np.argmax(probabilities))]
            runs = [dict(fn(deployed, text, category).tokens) for _ in range(5)]
            top = [set(sorted(r, key=lambda k: -abs(r[k]))[:3]) for r in runs]
            pairs = [len(a & b) / len(a | b) for i, a in enumerate(top)
                     for b in top[i + 1:] if (a | b)]
            if pairs:
                jaccards.append(statistics.fmean(pairs))
        stability[name] = statistics.fmean(jaccards) if jaccards else 1.0
        print(f"  {name:<6} top-3 agreement across runs: {stability[name]:.4f}")
    report["stability_top3"] = stability

    # ================================================================== 3 ===
    print("\n" + "=" * 78)
    print("PART 3  What a user actually sees, per layer")
    print("=" * 78)
    from backend.explainability import explain
    for layer in ("rule", "memory", "model"):
        rows = layers.get(layer, [])
        if not rows:
            continue
        row = rows[0]
        decision = cascade.predict([row["normalised_description"]])[0]
        started = time.perf_counter()
        result = explain(deployed, decision, row["normalised_description"],
                         memory=memory, method="shap")
        elapsed = (time.perf_counter() - started) * 1000
        print(f"\n  [{layer}]  {row['raw_description'][:58]}")
        print(f"     -> {result.category}  ({result.method}, "
              f"{'exact' if result.exact else 'approximate'}, {elapsed:.0f} ms)")
        print(f"     {result.headline[:150]}")

    out = ROOT / "results" / "unit4_5_explainability.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[PASS] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
