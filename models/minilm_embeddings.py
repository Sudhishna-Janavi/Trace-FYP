"""Model 5 of 5: MiniLM sentence embeddings with a logistic-regression head.

The only model here that represents *meaning* rather than counting surface
patterns. all-MiniLM-L6-v2 is a small transformer, pre-trained by others on
roughly a billion sentence pairs, which maps any text to 384 numbers positioned
so that texts with similar meaning land near one another.

What is and is not trained here
    The encoder is **downloaded and frozen**. No neural network is trained by
    this project; the encoder is borrowed exactly as published. Only a logistic
    regression head is fitted, on top of the 384 numbers it produces. The same
    head as Model 3, so the comparison between Models 3 and 5 isolates the
    representation: TF-IDF counting against learned semantics.

The hypothesis being tested
    Bank descriptions are codes rather than prose - TST*GOLDEN DRAGON, PPD ID,
    PYPL*365 MARKET. A semantic encoder should in principle recognise that an
    unfamiliar BISTRO relates to a familiar RESTAURANT, which is precisely the
    weakness every previous model showed when merchants were held out. Whether
    that advantage materialises on text this far from natural language is the
    question this model answers.

Offline loading
    Importing sentence_transformers without the offline environment variables
    set causes it to attempt network calls that block for roughly 155 seconds
    before timing out. The variables are set here, before the import, and the
    encoder is loaded from a resolved local snapshot path.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# must be set before sentence_transformers is imported anywhere
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def _snapshot_path() -> str:
    """Resolve the cached snapshot directory, so nothing is fetched at runtime.

    HF_HOME is consulted first because a deployed image bakes the weights in at
    build time and points this variable at them; the project-relative paths are
    what a local checkout uses.
    """
    pattern = (f"models--sentence-transformers--{MODEL_ID}/snapshots/*")
    bases = [Path(os.environ["HF_HOME"]) / "hub",
             Path(os.environ["HF_HOME"]),
             ROOT / ".cache" / "huggingface",
             ROOT.parent / "codebase" / ".cache" / "huggingface"]
    for base in bases:
        found = sorted(base.glob(pattern))
        if found:
            return str(found[-1])
    raise RuntimeError(
        f"{MODEL_ID} not found in a local cache. Looked in: "
        + ", ".join(str(b) for b in bases)
        + ". Download it once with network access (scripts/fetch_model.py), "
          "or copy an existing cache into one of those directories.")


def load_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_snapshot_path())


def build_classifier():
    """The same head as Model 3, so only the representation differs."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
