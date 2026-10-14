#!/usr/bin/env python3
"""Freeze the deployed classifier so the app does not refit on every start.

The configuration is the one Unit 2 selected and measured: MiniLM sentence
embeddings with a logistic-regression head, fitted on the stratified train and
validation splits. The test split is excluded, so the transactions the
application displays were never seen during fitting and the demonstration is a
genuine demonstration of generalisation.

Only the head is pickled. The encoder is loaded from its local snapshot at
runtime, because a sentence-transformer does not pickle usefully.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_model import embedding_cache, load  # noqa: E402
from models.minilm_embeddings import build_classifier  # noqa: E402


def main() -> int:
    out = ROOT / "backend" / "artifacts"
    out.mkdir(parents=True, exist_ok=True)

    rows = load("stratified_train.csv") + load("stratified_validation.csv")
    vectors = np.vstack([
        np.load(embedding_cache() / "norm_strat_train.npy"),
        np.load(embedding_cache() / "norm_strat_validation.npy"),
    ])
    labels = [r["category"] for r in rows]
    print(f"fitting on {len(rows):,} rows x {vectors.shape[1]} dimensions ...")

    head = build_classifier()
    head.fit(vectors, labels)
    classes = [str(c) for c in head.classes_]

    with (out / "head.pkl").open("wb") as handle:
        pickle.dump({"classifier": head, "classes": classes}, handle)

    # the merchant memory is part of the deployed cascade and is learned from
    # exactly the same rows, so it is frozen here too
    from models.merchant_memory import MerchantMemory
    memory = MerchantMemory()
    memory.learn(rows)
    with (out / "memory.pkl").open("wb") as handle:
        pickle.dump(memory, handle)

    final = json.loads((ROOT / "results" / "unit2_final_test.json").read_text())
    card = {
        "model": "all-MiniLM-L6-v2 (frozen) + logistic regression",
        "cascade": "user corrections -> rules -> merchant memory -> model",
        "threshold": 0.60,
        "fitted_on": "stratified train + validation",
        "fitted_rows": len(rows),
        "classes": classes,
        "merchants_memorised": len(memory),
        "performance": {
            name: {k: result[k] for k in
                   ("n", "accuracy", "macro_f1", "review_rate",
                    "accepted_accuracy")}
            for name, result in final["results"].items()
        },
        "note": "the application's demonstration data is drawn from the test "
                "split, which was excluded from fitting",
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2),
                                         encoding="utf-8")
    print(f"[PASS] head.pkl, memory.pkl, model_card.json -> {out}")
    print(f"   {len(classes)} classes, {len(memory):,} merchants memorised")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
