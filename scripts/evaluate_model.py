#!/usr/bin/env python3
"""Shared evaluation harness for Unit 2.

Every model is scored here, on the same data, with the same metrics, so the
comparison is like for like.

Protocols
    stratified     merchant may appear in both train and test (returning user)
    merchant       merchant held out entirely (new user)
    strictly-novel the subset of the merchant protocol whose rows contain no
                   merchant word seen anywhere in training: the true cold start

Metrics
    accuracy     fraction correct. Misleading on its own, because deduplication
                 left the classes unbalanced (Restaurants 3,110 vs Rent 874).
    macro-F1     F1 per category, then averaged, so every category counts
                 equally regardless of size. This is the headline number.
    coverage     fraction of rows the model committed to. Only meaningful for
                 models that can abstain, such as the rule layer.
    precision-when-fired
                 of the rows it did answer, how many were right.

Models are compared on VALIDATION. Test sets stay untouched until one winner
has been chosen, otherwise the final figure is tuned-on and not honest.
"""
from __future__ import annotations

import collections
import csv
import re
from pathlib import Path

from sklearn.metrics import f1_score

DATA = Path(__file__).resolve().parents[1] / "dataset" / "processed"

NOISE = {
    "DEBIT", "CREDIT", "POS", "DBT", "ECOM", "GIRO", "REF", "USA", "US", "EFT",
    "PYMT", "PPD", "ID", "PURCHASE", "PAYMENT", "CARD", "ACH", "WEB", "RECUR",
    "XFER", "DES", "INDN", "CO",
}
PROCESSOR = {
    "PAYPAL", "PP", "PYPL", "TST", "CLV", "SQ", "INST", "PREAPPROVED", "EXPRESS",
    "CHECKOUT", "WITHDRAWAL", "FROM", "PREAUTHORIZED", "DEPOSIT", "BILL", "USER",
    "MKTPL", "TO",
}
SPREAD_LIMIT = 10   # a token under <=10 leading words is merchant-like


def load(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _tokens(text: str) -> list[str]:
    out = []
    for word in text.split():
        if word in NOISE or word in PROCESSOR or word == "<NUM>" or word.isdigit():
            continue
        if len(word) <= 2:
            continue
        if len(word) >= 6 and re.search(r"\d", word) and re.fullmatch(r"[A-Z0-9]+", word):
            continue
        out.append(word)
    return out


def strictly_novel(train: list[dict], evaluation: list[dict]) -> list[dict]:
    """Rows whose merchant words never occur in training.

    Conservative by design: a row sharing only an ordinary word such as CAR or
    WASH is excluded from this subset even though that is legitimate signal, so
    the figure is a lower bound rather than an exact cold-start measurement.
    """
    spread: dict[str, set[str]] = collections.defaultdict(set)
    for row in train + evaluation:
        tokens = _tokens(row["normalised_description"])
        lead = tokens[0] if tokens else "UNKNOWN"
        for word in tokens:
            spread[word].add(lead)

    seen = set()
    for row in train:
        seen.update(w for w in _tokens(row["normalised_description"])
                    if len(spread[w]) <= SPREAD_LIMIT)

    novel = []
    for row in evaluation:
        words = [w for w in _tokens(row["normalised_description"])
                 if len(spread[w]) <= SPREAD_LIMIT]
        if not any(w in seen for w in words):
            novel.append(row)
    return novel


def score(gold: list[str], predicted: list[str | None], abstain_label: str = "<none>"):
    """Score predictions, treating None as an abstention."""
    fired = [(g, p) for g, p in zip(gold, predicted, strict=True) if p is not None]
    filled = [p if p is not None else abstain_label for p in predicted]
    correct = sum(1 for g, p in zip(gold, filled, strict=True) if g == p)
    return {
        "n": len(gold),
        "accuracy": correct / len(gold) if gold else 0.0,
        "macro_f1": float(f1_score(gold, filled, average="macro", zero_division=0)),
        "coverage": len(fired) / len(gold) if gold else 0.0,
        "precision_when_fired": (
            sum(1 for g, p in fired if g == p) / len(fired) if fired else 0.0),
    }


def print_table(title: str, rows: dict[str, dict]) -> None:
    print("=" * 74)
    print(title)
    print("=" * 74)
    print(f"{'protocol':<34}{'n':>7}{'accuracy':>11}{'macro-F1':>11}{'coverage':>11}")
    for name, values in rows.items():
        print(f"{name:<34}{values['n']:>7,}{values['accuracy']:>11.4f}"
              f"{values['macro_f1']:>11.4f}{values['coverage']:>11.1%}")
    print()
    for name, values in rows.items():
        if values["coverage"] < 0.999:
            print(f"  {name}: when it answered, it was right "
                  f"{values['precision_when_fired']:.1%} of the time")
    print("=" * 74)


def calibration(gold: list[str], probabilities, classes: list[str], bins: int = 10):
    """How well a model's confidence matches how often it is actually right.

    A model that says 90% should be right about 90% of the time. Where it is not,
    the confidence score cannot be used to decide what to send for human review,
    which is what this project's architecture depends on.

    ECE  expected calibration error: the average gap between stated confidence
         and observed accuracy, weighted by how many predictions fall in each
         confidence bin. Lower is better; 0 is perfect.
    MCE  the worst single bin's gap.
    Brier
         mean squared error of the full probability vector. Rewards being both
         right and appropriately confident.
    """
    import numpy as np

    probabilities = np.asarray(probabilities)
    confidence = probabilities.max(axis=1)
    predicted = [classes[i] for i in probabilities.argmax(axis=1)]
    correct = np.array([p == g for p, g in zip(predicted, gold, strict=True)],
                       dtype=float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece, mce, buckets = 0.0, 0.0, []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lower) & (confidence <= upper)
        if not mask.any():
            buckets.append({"lower": float(lower), "upper": float(upper),
                            "n": 0, "confidence": 0.0, "accuracy": 0.0})
            continue
        bin_confidence = float(confidence[mask].mean())
        bin_accuracy = float(correct[mask].mean())
        weight = float(mask.sum()) / len(confidence)
        gap = abs(bin_confidence - bin_accuracy)
        ece += weight * gap
        mce = max(mce, gap)
        buckets.append({"lower": float(lower), "upper": float(upper),
                        "n": int(mask.sum()), "confidence": bin_confidence,
                        "accuracy": bin_accuracy})

    index = {name: i for i, name in enumerate(classes)}
    onehot = np.zeros_like(probabilities)
    for row, label in enumerate(gold):
        if label in index:
            onehot[row, index[label]] = 1.0
    brier = float(((probabilities - onehot) ** 2).sum(axis=1).mean())

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier": brier,
        "mean_confidence": float(confidence.mean()),
        "accuracy": float(correct.mean()),
        "overconfidence": float(confidence.mean() - correct.mean()),
        "bins": buckets,
    }


# Embeddings are large and regenerable, and this project lives in a cloud-synced
# folder, so the cache is kept outside it. Override with OCT14_EMBEDDING_CACHE.
def embedding_cache() -> Path:
    import os
    path = Path(os.environ.get(
        "OCT14_EMBEDDING_CACHE", Path.home() / ".cache" / "oct14-embeddings"))
    path.mkdir(parents=True, exist_ok=True)
    return path
