"""Layer-aware explanations for the categorisation cascade.

Each layer of the cascade decides for a different reason, so each explains
itself differently. Treating them uniformly would either waste effort or
misrepresent what happened.

    rule     matched a phrase. The matched phrase is the explanation: exact,
             free, and needs no model call at all.
    memory   this merchant was seen during training. The history is the
             explanation: exact, free.
    model    predicted from the text. The model is an embedding model, whose
             384 dimensions carry no word-level meaning, so a token-level
             explanation has to be reconstructed after the fact with LIME or
             SHAP. Approximate, and the only layer that costs anything.

For a returning user roughly four transactions in five are decided by the first
two layers, so most explanations are exact and instant. The cascade therefore
reduces the cost of explainability as well as improving accuracy.

Both post-hoc methods operate on the same coalition space, so their outputs are
comparable. A coalition is a binary vector over the unique tokens of a
description; a token that is on keeps all of its occurrences, and the model is
queried on the text rebuilt from each coalition.

    LIME   Ribeiro et al. (2016): a weighted ridge surrogate fitted around the
           full-text prediction.
    SHAP   Lundberg and Lee (2017): Shapley values. Every coalition is
           enumerated when the description is short enough, which it almost
           always is here, so the values are exact Shapley values rather than
           sampled estimates.

Determinism: the random seed is derived from a hash of the text, so the same
transaction always produces the same explanation. Without this a user could
refresh a page and see different reasons for the same decision.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from math import factorial

import numpy as np
from sklearn.linear_model import Ridge

MAX_EXACT_TOKENS = 12
LIME_KERNEL_WIDTH = 0.25
LIME_RIDGE_ALPHA = 1.0
SAMPLED_COALITIONS = 512


@dataclass
class Explanation:
    layer: str                       # rule | memory | model
    category: str
    confidence: float
    method: str                      # matched_phrase | merchant_history | lime | shap
    exact: bool
    headline: str                    # one sentence a user can read
    tokens: list[tuple[str, float]] = field(default_factory=list)
    detail: dict = field(default_factory=dict)
    model_calls: int = 0


# --------------------------------------------------------------- free, exact

def explain_rule(category: str, matched_phrase: str) -> Explanation:
    return Explanation(
        layer="rule", category=category, confidence=1.0,
        method="matched_phrase", exact=True,
        headline=f"The description contains “{matched_phrase}”, which "
                 f"always indicates {category}.",
        tokens=[(matched_phrase, 1.0)],
        detail={"matched_phrase": matched_phrase},
    )


def explain_memory(category: str, fingerprint: str, history: dict) -> Explanation:
    total = sum(history.values())
    same = history.get(category, 0)
    others = {k: v for k, v in history.items() if k != category}
    headline = (f"You have {total} previous transaction"
                f"{'s' if total != 1 else ''} at “{fingerprint}”, "
                f"{same} of which {'were' if same != 1 else 'was'} {category}.")
    if others:
        headline += (" Other categories seen for this merchant: "
                     + ", ".join(f"{k} ({v})" for k, v in others.items()) + ".")
    return Explanation(
        layer="memory", category=category, confidence=1.0,
        method="merchant_history", exact=True, headline=headline,
        tokens=[(fingerprint, 1.0)],
        detail={"merchant": fingerprint, "history": history,
                "occurrences": total},
    )


# ------------------------------------------------------- post-hoc, the model

def _seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def unique_tokens(normalised: str) -> list[str]:
    seen: dict[str, None] = {}
    for token in normalised.split():
        seen.setdefault(token, None)
    return list(seen)


def _text_from(tokens: list[str], mask: np.ndarray) -> str:
    return " ".join(t for t, keep in zip(tokens, mask, strict=True) if keep)


def _all_coalitions(n: int) -> np.ndarray:
    return np.array([[(i >> b) & 1 for b in range(n)] for i in range(2 ** n)],
                    dtype=float)


def _sampled_coalitions(n: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    masks = rng.binomial(1, 0.5, size=(count, n)).astype(float)
    masks[0, :] = 0.0
    masks[1, :] = 1.0
    return masks


def _coalition_values(model, tokens, coalitions, target):
    texts = [_text_from(tokens, mask) for mask in coalitions]
    return model.predict_proba(texts)[:, target]


def explain_lime(model, normalised: str, category: str) -> Explanation:
    """A weighted ridge surrogate fitted around the full-text prediction."""
    tokens = unique_tokens(normalised)
    classes = model.classes
    target = classes.index(category)
    probabilities = model.predict_proba([normalised])[0]
    confidence = float(probabilities[target])
    if len(tokens) < 2:
        return Explanation("model", category, confidence, "lime", False,
                           "Too few terms to attribute.", model_calls=1)

    exact = len(tokens) <= MAX_EXACT_TOKENS
    coalitions = (_all_coalitions(len(tokens)) if exact
                  else _sampled_coalitions(len(tokens), SAMPLED_COALITIONS,
                                           _seed(normalised)))
    values = _coalition_values(model, tokens, coalitions, target)

    full = np.ones(len(tokens))
    denominator = np.linalg.norm(coalitions, axis=1) * np.linalg.norm(full)
    similarity = np.divide(coalitions @ full, denominator,
                           out=np.zeros(len(coalitions)), where=denominator != 0)
    weights = np.exp(-(((1.0 - similarity) ** 2) / (LIME_KERNEL_WIDTH ** 2)))
    surrogate = Ridge(alpha=LIME_RIDGE_ALPHA)
    surrogate.fit(coalitions, values, sample_weight=weights)
    fidelity = float(surrogate.score(coalitions, values, sample_weight=weights))

    pairs = sorted(zip(tokens, (float(c) for c in surrogate.coef_)),
                   key=lambda kv: -abs(kv[1]))
    top = [t for t, w in pairs if w > 0][:3]
    return Explanation(
        "model", category, confidence, "lime", False,
        _headline(top, category),
        tokens=pairs,
        detail={"surrogate_r2": round(fidelity, 4), "exact_enumeration": exact},
        model_calls=len(coalitions) + 1,
    )


def _exact_shapley(coalitions, values, n) -> np.ndarray:
    by_mask = {}
    for mask, value in zip(coalitions, values, strict=True):
        by_mask[int("".join(str(int(b)) for b in mask[::-1]), 2)] = float(value)
    phi = np.zeros(n)
    for j in range(n):
        bit = 1 << j
        for key, value in by_mask.items():
            if key & bit:
                continue
            size = bin(key).count("1")
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            phi[j] += weight * (by_mask[key | bit] - value)
    return phi


def explain_shap(model, normalised: str, category: str) -> Explanation:
    """Shapley values, enumerated exactly when the description is short."""
    tokens = unique_tokens(normalised)
    classes = model.classes
    target = classes.index(category)
    probabilities = model.predict_proba([normalised])[0]
    confidence = float(probabilities[target])
    if len(tokens) < 2:
        return Explanation("model", category, confidence, "shap", False,
                           "Too few terms to attribute.", model_calls=1)

    exact = len(tokens) <= MAX_EXACT_TOKENS
    if exact:
        coalitions = _all_coalitions(len(tokens))
        values = _coalition_values(model, tokens, coalitions, target)
        phi = _exact_shapley(coalitions, values, len(tokens))
        baseline, full_value = float(values[0]), float(values[-1])
        calls = len(coalitions) + 1
    else:
        coalitions = _sampled_coalitions(len(tokens), SAMPLED_COALITIONS,
                                         _seed(normalised))
        values = _coalition_values(model, tokens, coalitions, target)
        # a least-squares Shapley estimate over the sampled coalitions
        phi = np.linalg.lstsq(coalitions, values - values[0], rcond=None)[0]
        baseline, full_value = float(values[0]), confidence
        calls = len(coalitions) + 1

    # efficiency axiom: phi_0 + sum(phi) should equal f(full)
    residual = abs(baseline + float(phi.sum()) - full_value)
    pairs = sorted(zip(tokens, (float(v) for v in phi)),
                   key=lambda kv: -abs(kv[1]))
    top = [t for t, w in pairs if w > 0][:3]
    return Explanation(
        "model", category, confidence, "shap", False,
        _headline(top, category),
        tokens=pairs,
        detail={"additivity_residual": float(residual),
                "exact_enumeration": exact, "baseline": baseline},
        model_calls=calls,
    )


def _headline(top: list[str], category: str) -> str:
    if not top:
        return (f"No single term drove this towards {category}; the decision "
                "rests on the description as a whole.")
    readable = ", ".join(f"“{t}”" for t in top)
    return (f"The terms {readable} pushed this towards {category}. "
            "This explanation is reconstructed after the fact and is an "
            "approximation of the model's reasoning.")


def explain(model, decision, normalised: str, memory=None,
            method: str = "shap") -> Explanation:
    """Explain whichever layer actually made the decision."""
    if decision.layer == "rule":
        phrase = decision.evidence.replace("matched rule phrase: ", "")
        return explain_rule(decision.category, phrase)
    if decision.layer == "memory":
        from models.merchant_memory import fingerprint
        key = fingerprint(normalised)
        history = dict(memory._seen.get(key, {})) if memory else {}
        return explain_memory(decision.category, key, history)
    if method == "lime":
        return explain_lime(model, normalised, decision.category)
    return explain_shap(model, normalised, decision.category)
