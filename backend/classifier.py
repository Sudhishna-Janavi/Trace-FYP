"""The deployed classifier, loaded once per process.

Loading the encoder takes a few seconds and the head is a pickle, so both are
held in a module-level singleton rather than rebuilt per request.

The cascade served here adds a layer the evaluation could not have: a user's
own corrections sit in front of everything else. A category a person has
explicitly fixed must never be silently overridden by a rule or a model, and
the correction is theirs alone, so it is looked up per user.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
REVIEW_THRESHOLD = 0.60


@dataclass(frozen=True)
class Prediction:
    category: str
    confidence: float
    decided_by: str          # correction | rule | memory | model
    evidence: str
    needs_review: bool


class DeployedModel:
    """Encoder plus head, with the batching interface the cascade expects."""

    def __init__(self):
        from models.minilm_embeddings import load_encoder
        with (ARTIFACTS / "head.pkl").open("rb") as handle:
            bundle = pickle.load(handle)
        self.encoder = load_encoder()
        self.head = bundle["classifier"]
        self._classes = bundle["classes"]

    @property
    def classes(self) -> list[str]:
        return self._classes

    def predict_proba(self, texts):
        vectors = self.encoder.encode(list(texts), batch_size=256,
                                      convert_to_numpy=True,
                                      normalize_embeddings=True,
                                      show_progress_bar=False)
        return self.head.predict_proba(vectors)


@lru_cache(maxsize=1)
def get_model() -> DeployedModel:
    return DeployedModel()


@lru_cache(maxsize=1)
def get_memory():
    with (ARTIFACTS / "memory.pkl").open("rb") as handle:
        return pickle.load(handle)


@lru_cache(maxsize=1)
def get_model_card() -> dict:
    import json
    return json.loads((ARTIFACTS / "model_card.json").read_text())


def classify(texts: list[str], corrections: dict[str, str]) -> list[Prediction]:
    """Run the full cascade over a batch.

    `corrections` maps a merchant fingerprint to the category this user has
    chosen for it. The model is called only for the rows that reach it, which
    for a returning user is a minority of them.
    """
    from models.merchant_memory import fingerprint
    from models.rule_baseline import predict as rule_predict

    memory = get_memory()
    out: list[Prediction | None] = [None] * len(texts)
    pending: list[int] = []

    for index, text in enumerate(texts):
        key = fingerprint(text)
        if key and key in corrections:
            out[index] = Prediction(
                corrections[key], 1.0, "correction",
                f"you previously set “{key}” to this category", False)
            continue
        category, phrase = rule_predict(text)
        if category is not None:
            out[index] = Prediction(category, 1.0, "rule",
                                    f"matched “{phrase}”", False)
            continue
        remembered = memory.predict(text)
        if remembered is not None:
            out[index] = Prediction(remembered, 1.0, "memory",
                                    f"merchant “{key}” seen before",
                                    False)
            continue
        pending.append(index)

    if pending:
        model = get_model()
        probabilities = model.predict_proba([texts[i] for i in pending])
        classes = model.classes
        for position, index in enumerate(pending):
            row = probabilities[position]
            best = int(row.argmax())
            confidence = float(row[best])
            out[index] = Prediction(
                classes[best], confidence, "model",
                "no correction, rule or known merchant matched",
                confidence < REVIEW_THRESHOLD)

    return [p for p in out if p is not None]
