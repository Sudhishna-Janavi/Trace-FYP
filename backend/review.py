"""The user-correction layer: confirming and correcting what the cascade decided.

Why a correction is stored against a merchant rather than a transaction
    Correcting one row and leaving forty identical rows untouched would make
    the feature feel broken. A user who says "RALPH'S is Groceries, not
    Insurance" is making a statement about the merchant, so the correction is
    keyed by merchant fingerprint, applied to every matching transaction the
    user has not already reviewed individually, and consulted first the next
    time that merchant appears.

Why an already-reviewed row is left alone
    A row the user has personally confirmed or corrected is their decision.
    A later correction for the same merchant must not quietly overwrite it,
    so retroactive application skips anything already reviewed.

This is the first layer of the cascade, and the only one that is free, exact
and always right by definition: it is not a prediction at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.database import Correction, Transaction, User


def _now() -> datetime:
    """A naive UTC timestamp, matching how the column is declared.

    `datetime.utcnow()` is deprecated and returns a naive value anyway, so the
    timezone is made explicit and then dropped rather than assumed.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class ReviewResult:
    transaction_id: int
    category: str
    corrected: bool
    also_updated: int          # other rows at this merchant that changed
    remaining: int             # how many still need review


def pending_count(session: Session, user: User) -> int:
    return session.scalar(
        select(func.count()).select_from(Transaction)
        .where(Transaction.user_id == user.id,
               Transaction.needs_review.is_(True),
               Transaction.reviewed_at.is_(None))) or 0


def confirm(session: Session, user: User, row: Transaction) -> ReviewResult:
    """Accept the predicted category. No correction is recorded.

    Confirming is not evidence about the merchant in the way a correction is:
    the user agreed with one prediction, which does not license overriding the
    model elsewhere. Only this row is marked reviewed.
    """
    row.reviewed_at = _now()
    row.needs_review = False
    session.commit()
    return ReviewResult(row.id, row.category, False, 0,
                        pending_count(session, user))


def correct(session: Session, user: User, row: Transaction,
            category: str) -> ReviewResult:
    """Record the user's category for this merchant and apply it."""
    fingerprint = row.merchant_fingerprint

    row.corrected_category = category
    row.reviewed_at = _now()
    row.needs_review = False

    also = 0
    if fingerprint:
        existing = session.scalar(
            select(Correction).where(Correction.user_id == user.id,
                                     Correction.merchant_fingerprint == fingerprint))
        if existing is None:
            existing = Correction(user_id=user.id,
                                  merchant_fingerprint=fingerprint,
                                  category=category)
            session.add(existing)
        else:
            existing.category = category

        # every other transaction at this merchant the user has not personally
        # ruled on: one correction should not have to be made forty times
        siblings = session.scalars(
            select(Transaction).where(
                Transaction.user_id == user.id,
                Transaction.merchant_fingerprint == fingerprint,
                Transaction.id != row.id,
                Transaction.reviewed_at.is_(None))).all()
        for sibling in siblings:
            if sibling.category == category:
                continue
            sibling.corrected_category = category
            sibling.needs_review = False
            also += 1
        session.flush()
        # how many transactions this correction currently governs, not how many
        # times it has been edited: a user who changes their mind twice has
        # still only taught Trace about one merchant
        existing.times_applied = session.scalar(
            select(func.count()).select_from(Transaction)
            .where(Transaction.user_id == user.id,
                   Transaction.merchant_fingerprint == fingerprint,
                   Transaction.corrected_category == category)) or 0

    session.commit()
    return ReviewResult(row.id, category, True, also,
                        pending_count(session, user))
