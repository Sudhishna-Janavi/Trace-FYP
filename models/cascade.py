"""The deployed categorisation cascade.

One implementation, used by both the evaluation scripts and the web
application, so that what is measured is literally what is shipped.

Order of decision
    1. deterministic rules   answer when recognisable banking vocabulary is
                             present. Merchant-independent, so unaffected by
                             whether the shop is familiar.
    2. merchant memory       answer when this merchant has been seen before.
                             Entirely merchant-dependent, so it covers most
                             transactions for a returning user and almost none
                             for a new one.
    3. learned model         answer everything else.
    4. confidence gate       a model answer below the threshold is marked for
                             human review rather than shown as settled.

Layers 1 and 2 are deliberate opposites. The rules hold roughly steady at ~17%
coverage whether merchants are familiar or not; the memory covers 82% for a
returning user and 4% for a new one. Neither is better: they fail in different
places, which is why both sit in front of the model.

A user-correction memory belongs between layers 1 and 2 in the full design. It
is not implemented here because it requires real users, and it is added in the
application rather than in this evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from models.merchant_memory import MerchantMemory
from models.rule_baseline import predict as rule_predict


@dataclass(frozen=True)
class Decision:
    category: str
    layer: str                 # "rule" | "memory" | "model"
    confidence: float
    needs_review: bool
    evidence: str              # matched phrase, remembered merchant, or source
    alternatives: tuple[tuple[str, float], ...] = ()


class ProbabilityModel(Protocol):
    """Anything that can score a batch of normalised descriptions."""

    def predict_proba(self, texts: Sequence[str]): ...

    @property
    def classes(self) -> list[str]: ...


class Cascade:
    """Rules, then merchant memory, then a model, then a confidence gate."""

    def __init__(self, model: ProbabilityModel, threshold: float = 0.60,
                 memory: MerchantMemory | None = None) -> None:
        self.model = model
        self.threshold = threshold
        self.memory = memory or MerchantMemory()

    def learn_memory(self, rows) -> "Cascade":
        """Populate the merchant memory from labelled training rows."""
        self.memory.learn(rows)
        return self

    def predict(self, normalised_descriptions: Sequence[str]) -> list[Decision]:
        """Route a batch. The model is called only for rows that reach layer 3.

        Batching matters: an embedding model is far cheaper per row when called
        once for many rows than once per row, and in a returning user's data
        most rows never reach it at all.
        """
        decisions: list[Decision | None] = [None] * len(normalised_descriptions)
        pending: list[int] = []

        for index, text in enumerate(normalised_descriptions):
            category, evidence = rule_predict(text)
            if category is not None:
                decisions[index] = Decision(category, "rule", 1.0, False,
                                            f"matched rule phrase: {evidence}")
                continue
            remembered = self.memory.predict(text)
            if remembered is not None:
                decisions[index] = Decision(remembered, "memory", 1.0, False,
                                            "merchant seen during training")
                continue
            pending.append(index)

        if pending:
            probabilities = self.model.predict_proba(
                [normalised_descriptions[i] for i in pending])
            classes = self.model.classes
            for position, index in enumerate(pending):
                row = probabilities[position]
                order = sorted(range(len(classes)), key=lambda c: -row[c])
                best = order[0]
                confidence = float(row[best])
                decisions[index] = Decision(
                    classes[best], "model", confidence,
                    confidence < self.threshold,
                    "no rule or known merchant matched; the model decided",
                    tuple((classes[c], float(row[c])) for c in order[1:4]))

        return [d for d in decisions if d is not None]

    def layer_shares(self, decisions: Sequence[Decision]) -> dict[str, float]:
        total = len(decisions) or 1
        return {layer: sum(1 for d in decisions if d.layer == layer) / total
                for layer in ("rule", "memory", "model")}
