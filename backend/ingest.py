"""Turn bank export files into categorised transactions for one user.

The pipeline is the whole system in one function:

    export files -> adapters -> canonical rows -> cascade -> database

Nothing here knows which bank a row came from after the adapter stage, which is
the point of having adapters at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.adapters import ADAPTERS, ingest as parse_feeds
from backend.classifier import classify
from backend.database import Account, Correction, Transaction, User

ROOT = Path(__file__).resolve().parents[1]
DEMO_FEEDS = ROOT / "dataset" / "demo" / "feeds"

ACCOUNT_NAMES = {
    "acc-chase": ("Chase", "Current Account",
                  "CSV, one signed Amount column, MM/DD/YYYY dates"),
    "acc-capitalone": ("Capital One", "Savings Account",
                       "CSV, separate Debit and Credit columns, ISO dates"),
    "acc-applecard": ("Apple Card", "Credit Card",
                      "12 monthly CSVs, purchases positive, payments negative"),
}


@dataclass
class IngestResult:
    transactions: int
    accounts: int
    by_layer: dict[str, int]
    needs_review: int
    files: int


def user_corrections(session: Session, user: User) -> dict[str, str]:
    rows = session.scalars(
        select(Correction).where(Correction.user_id == user.id)).all()
    return {row.merchant_fingerprint: row.category for row in rows}


def ingest_for_user(session: Session, user: User,
                    feed_directory: Path = DEMO_FEEDS,
                    replace: bool = True) -> IngestResult:
    """Parse every feed, classify, and store against this user."""
    from models.merchant_memory import fingerprint

    if replace:
        # re-connecting replaces rather than duplicates; corrections survive,
        # because they are the user's own decisions rather than imported data
        session.execute(delete(Transaction).where(Transaction.user_id == user.id))
        session.execute(delete(Account).where(Account.user_id == user.id))
        session.flush()

    canonical = parse_feeds(feed_directory)
    if not canonical:
        return IngestResult(0, 0, {}, 0, 0)

    accounts: dict[str, Account] = {}
    for row in canonical:
        if row.account_id in accounts:
            continue
        bank, name, fmt = ACCOUNT_NAMES.get(
            row.account_id, (row.bank, row.bank, "unknown format"))
        account = Account(user_id=user.id, external_id=row.account_id,
                          bank=bank, name=name, source_format=fmt)
        session.add(account)
        accounts[row.account_id] = account
    session.flush()

    corrections = user_corrections(session, user)
    # the cascade is trained and written against normalised text, so normalise
    # once here and reuse it for classification, the fingerprint and storage
    normalised = [_normalise(r.raw_description) for r in canonical]
    predictions = classify(normalised, corrections)

    by_layer: dict[str, int] = {}
    review = 0
    for row, text, prediction in zip(canonical, normalised, predictions,
                                     strict=True):
        by_layer[prediction.decided_by] = by_layer.get(prediction.decided_by, 0) + 1
        review += int(prediction.needs_review)
        session.add(Transaction(
            user_id=user.id, account_id=accounts[row.account_id].id,
            raw_description=row.raw_description,
            normalised_description=text,
            merchant_fingerprint=fingerprint(text),
            amount=row.amount, booking_date=row.booking_date,
            predicted_category=prediction.category,
            confidence=prediction.confidence,
            decided_by=prediction.decided_by, evidence=prediction.evidence,
            needs_review=prediction.needs_review))
    session.commit()

    files = len(list(feed_directory.glob("*.csv")))
    return IngestResult(len(canonical), len(accounts), by_layer, review, files)


def _normalise(raw: str) -> str:
    """The same normalisation the model was trained with."""
    import re
    text = raw.upper()
    text = re.sub(r"^\[(DEBIT|CREDIT)\]\s*", lambda m: m.group(1) + " ", text)
    text = re.sub(r"\b\d{4,}\b", "<NUM>", text)
    text = re.sub(r"[^A-Z0-9<> ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
