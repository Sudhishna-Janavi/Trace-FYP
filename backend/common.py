"""What the HTML pages and the JSON API both need.

Trace has two front ends. The web application renders Jinja templates on the
server; the mobile application asks for JSON and draws its own screens. They
are two presentations of one system, and everything below is the part that
belongs to neither of them in particular: the labels a layer is given, the
wording that accompanies an explanation, and the dispatch that picks the
explanation a deciding layer can actually support.

Keeping it here rather than in `app.py` means the two front ends cannot drift
apart. A category explained one way on the web is explained the same way on
the phone, because it is the same function answering.
"""
from __future__ import annotations

import time

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend import classifier
from backend.database import Account, Transaction, User

MIN_PASSWORD = 8

LAYER_LABELS = {
    "correction": "Your correction", "rule": "Rule",
    "memory": "Merchant memory", "model": "Model",
}

# The import summary bars each layer's share of the transactions it just
# classified. Colour stays semantic, as everywhere else: the accent for the
# three layers that answer exactly, amber for the model, the only one that can
# be wrong. Giving the exact three one colour is the point, not an oversight.
LAYERS = (
    ("correction", "Your corrections", "checked before anything else",
     "var(--accent)"),
    ("rule", "Rules", "unambiguous banking phrases", "var(--accent)"),
    ("memory", "Merchant memory", "merchants seen before", "var(--accent)"),
    ("model", "The model", "everything else", "var(--review)"),
)

# The sidebar on the web, the tab bar on the phone. One order, one source.
NAV = [
    ("/", "Dashboard", "dashboard"),
    ("/transactions", "Transactions", "transactions"),
    ("/review", "Review", "review"),
    ("/insights", "Insights", "insights"),
    ("/accounts", "Accounts", "accounts"),
    ("/how-it-works", "How it works", "how"),
]

PROTOCOL_LABELS = {
    "in_distribution": "Merchant seen before",
    "unseen_merchant": "Merchant held out",
    "strictly_novel": "Cold start",
}
PROTOCOL_MEANS = {
    "in_distribution": "The merchant appeared in training under a different "
                       "transaction. This is the easiest case and the one most "
                       "papers report.",
    "unseen_merchant": "Every transaction for a merchant was held out "
                       "together, so the merchant itself is new.",
    "strictly_novel": "Neither the merchant nor anything resembling its "
                      "wording was in training. This is a new user's first "
                      "upload.",
}

EXPLANATION_NOTES = {
    "user_correction": "You set this category yourself, so nothing was "
                       "predicted here. Trace remembers it for this merchant.",
    "matched_phrase": "A deterministic rule decided this. The reason above is "
                      "the whole of the logic, not a summary of it.",
    "merchant_history": "Your own history decided this, by looking the "
                        "merchant up. No model was involved.",
    "shap": "Shapley values, enumerated exactly over every combination of "
            "terms in this description. The weights are the model's, but the "
            "reasoning shown is reconstructed afterwards.",
    "lime": "LIME fits a simple linear model to the predictions around this "
            "description. The weights come from that stand-in, not from the "
            "model itself.",
}

# Measured on the deployed cascade. The model row is re-measured against ONNX
# Runtime rather than quoting the PyTorch figure from Unit 4.5 (31 ms median),
# because the engine the application actually runs is the one a reader should
# be told about. See results/unit4_5.log and scripts/verify_onnx.py.
EXPLANATION_COST = [
    {"layer": "Rule", "method": "The phrase that matched",
     "exact": True, "cost": "under 1 ms"},
    {"layer": "Merchant memory", "method": "Your own history at the merchant",
     "exact": True, "cost": "under 1 ms"},
    {"layer": "Model", "method": "Shapley values over the terms",
     "exact": False, "cost": "51 ms median, 113 ms worst"},
]

# The demonstration feeds are the only source that assigns these ids. When a
# column recording an account's provenance exists this becomes a lookup rather
# than an inference from an identifier.
DEMO_ACCOUNT_IDS = ("acc-chase", "acc-capitalone", "acc-applecard")


def is_demo(session: Session, user: User) -> bool:
    """Whether this user's data is the generated demonstration set.

    It matters for the benchmark comparison: the demonstration amounts were
    calibrated to the same survey the comparison uses, so agreement there is
    arithmetic rather than evidence.
    """
    return bool(session.scalar(
        select(Account).where(Account.user_id == user.id,
                              Account.external_id.in_(DEMO_ACCOUNT_IDS))))


def explanation_for(row: Transaction, method: str):
    """Dispatch to the explanation the deciding layer can actually support."""
    from backend import explainability as xai

    if row.decided_by == "correction" or row.corrected_category:
        # Six transactions in the demonstration set — 0.8% — have no merchant
        # fingerprint at all, because every token in the description is bank
        # noise or a reference code. Naming the merchant in that case produces
        # an empty pair of quotation marks, which reads as a fault rather than
        # as the limitation it is, so the sentence says what actually happened.
        if row.merchant_fingerprint:
            headline = (f"You told Trace that “{row.merchant_fingerprint}” is "
                        f"{row.category}, so that is what it uses.")
            tokens = [(row.merchant_fingerprint, 1.0)]
        else:
            headline = (f"You set this to {row.category} yourself, so that is "
                        f"what it uses. This description has no recognisable "
                        f"merchant, so the correction applies to this "
                        f"transaction alone.")
            tokens = []
        return xai.Explanation(
            layer="correction", category=row.category, confidence=1.0,
            method="user_correction", exact=True,
            headline=headline, tokens=tokens)

    if row.decided_by == "rule":
        from models.rule_baseline import predict as rule_predict
        _, phrase = rule_predict(row.normalised_description)
        return xai.explain_rule(row.predicted_category, phrase or "")

    if row.decided_by == "memory":
        memory = classifier.get_memory()
        history = dict(memory._seen.get(row.merchant_fingerprint, {}))
        return xai.explain_memory(row.predicted_category,
                                  row.merchant_fingerprint, history)

    model = classifier.get_model()
    if method == "lime":
        return xai.explain_lime(model, row.normalised_description,
                                row.predicted_category)
    return xai.explain_shap(model, row.normalised_description,
                            row.predicted_category)


def explanation_payload(row: Transaction, method: str = "shap") -> dict:
    """Why one transaction carries the category it does.

    The explanation that is honest depends on which cascade layer decided.
    Three of the four layers know their own reasons exactly and cost nothing
    to explain; only the model needs a post-hoc approximation, and only there
    is the encoder called at all.

    Loading the encoder is a one-off cost of the first request, not a cost of
    this explanation, so it happens before the clock starts.
    """
    if row.decided_by == "model" and not row.corrected_category:
        classifier.get_model()
    started = time.perf_counter()
    result = explanation_for(row, method)
    return {
        "id": row.id,
        "description": row.raw_description,
        "amount": row.amount,
        "date": row.booking_date.strftime("%d %b %Y"),
        "bank": row.account.bank,
        "category": result.category,
        "confidence": result.confidence,
        "layer": result.layer,
        "layer_label": LAYER_LABELS.get(result.layer, result.layer),
        "method": result.method,
        "exact": result.exact,
        "headline": result.headline,
        "tokens": [[token, round(weight, 4)] for token, weight in result.tokens],
        "note": EXPLANATION_NOTES[result.method],
        "took_ms": round((time.perf_counter() - started) * 1000),
    }
