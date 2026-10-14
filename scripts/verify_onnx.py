#!/usr/bin/env python3
"""Check that the ONNX encoder reproduces the PyTorch one it replaces.

Swapping inference engines silently changes the system every reported figure
describes. Unless the embeddings and, more importantly, the predictions agree,
the accuracy quoted in the report would no longer be the accuracy of the
deployed application. This script is the evidence that they do.

Three comparisons, in increasing order of what actually matters:

    1. embeddings      cosine similarity and maximum absolute difference
    2. probabilities   the classifier head's output on both representations
    3. predictions     the labels a user would see, and the cascade's decision
                       to accept or hold each one for review

Floating-point arithmetic reorders differently across engines, so exact
equality is not expected. Disagreement about any prediction is.

Run:
    ~/.venvs/oct14/bin/python scripts/verify_onnx.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SAMPLE = 1500
REVIEW_THRESHOLD = 0.60


def main() -> int:
    from evaluate_model import load

    rows = load("stratified_test.csv")
    rng = np.random.default_rng(1131)
    if len(rows) > SAMPLE:
        rows = [rows[i] for i in rng.choice(len(rows), SAMPLE, replace=False)]
    texts = [r["normalised_description"] for r in rows]
    print(f"comparing on {len(texts):,} held-out transactions\n")

    from models.minilm_embeddings import load_encoder as load_torch
    from models.onnx_encoder import OnnxEncoder

    print("encoding with PyTorch ...")
    torch_vectors = load_torch().encode(
        texts, batch_size=256, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False)
    print("encoding with ONNX Runtime ...")
    onnx_vectors = OnnxEncoder().encode(texts, batch_size=256)

    # ---------------------------------------------------------- embeddings
    cosine = (torch_vectors * onnx_vectors).sum(axis=1)
    delta = np.abs(torch_vectors - onnx_vectors)
    print("\n1. embeddings")
    print(f"   cosine similarity   min {cosine.min():.8f}  "
          f"mean {cosine.mean():.8f}")
    print(f"   absolute difference max {delta.max():.3e}  "
          f"mean {delta.mean():.3e}")

    # -------------------------------------------------------- probabilities
    with (ROOT / "backend" / "artifacts" / "head.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    head, classes = bundle["classifier"], bundle["classes"]
    torch_probabilities = head.predict_proba(torch_vectors)
    onnx_probabilities = head.predict_proba(onnx_vectors)
    probability_delta = np.abs(torch_probabilities - onnx_probabilities)
    print("\n2. class probabilities")
    print(f"   absolute difference max {probability_delta.max():.3e}  "
          f"mean {probability_delta.mean():.3e}")

    # ----------------------------------------------------------- decisions
    torch_labels = [classes[i] for i in torch_probabilities.argmax(axis=1)]
    onnx_labels = [classes[i] for i in onnx_probabilities.argmax(axis=1)]
    disagreements = [i for i, (a, b) in enumerate(zip(torch_labels, onnx_labels))
                     if a != b]

    torch_gate = torch_probabilities.max(axis=1) >= REVIEW_THRESHOLD
    onnx_gate = onnx_probabilities.max(axis=1) >= REVIEW_THRESHOLD
    gate_disagreements = int((torch_gate != onnx_gate).sum())

    print("\n3. what a user would see")
    print(f"   predicted category differs   {len(disagreements)} "
          f"of {len(texts):,}")
    print(f"   accept-or-review differs     {gate_disagreements} "
          f"of {len(texts):,}")
    for index in disagreements[:5]:
        print(f"      {texts[index][:52]!r}")
        print(f"         torch {torch_labels[index]} -> onnx {onnx_labels[index]}")

    # accuracy against the truth, which is the figure the report quotes
    truth = [r["category"] for r in rows]
    torch_accuracy = np.mean([p == t for p, t in zip(torch_labels, truth)])
    onnx_accuracy = np.mean([p == t for p, t in zip(onnx_labels, truth)])
    print(f"\n   accuracy  torch {torch_accuracy:.6f}   onnx {onnx_accuracy:.6f}"
          f"   difference {abs(torch_accuracy - onnx_accuracy):.6f}")

    ok = (len(disagreements) == 0 and gate_disagreements == 0
          and cosine.min() > 0.9999)
    print("\n[PASS] ONNX reproduces PyTorch; every reported figure still holds"
          if ok else
          "\n[FAIL] the engines disagree; the deployed system would not be the "
          "system the report measures")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
