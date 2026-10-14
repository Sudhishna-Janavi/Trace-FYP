"""Source adapters: heterogeneous bank exports into one canonical schema.

Every linked institution exports a different shape. The adapters absorb those
differences so that nothing downstream has to know which bank a row came from,
which is what makes aggregating several accounts possible at all. This is the
Adapter pattern: one contract, several implementations.

The differences they reconcile are documented real-world conventions, not
invented ones:

    Chase         a single signed Amount column, month-first dates
    Capital One   separate Debit and Credit columns, one blank on every row,
                  plus a transaction date and a posted date that differ
    Apple Card    one file per statement month, and purchases written POSITIVE
                  with payments NEGATIVE, the reverse of the other two

That last one matters more than it looks. An adapter that ignores it produces a
dashboard where every purchase appears as income, which is precisely the class
of error an adapter layer exists to prevent.

Canonical sign convention: negative is money out, positive is money in.

Every feed also carries the bank's own Category column, as real exports do, and
every adapter discards it. Using it would mean the bank had already performed
the categorisation this project exists to perform.
"""
from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


@dataclass(frozen=True)
class CanonicalTransaction:
    external_id: str
    raw_description: str
    amount: float            # signed: negative out, positive in
    booking_date: date
    direction: str           # "debit" | "credit"
    account_id: str
    bank: str

    @staticmethod
    def of(external_id, description, amount, booking_date, account_id, bank):
        return CanonicalTransaction(
            external_id=external_id,
            raw_description=description,
            amount=round(amount, 2),
            booking_date=booking_date,
            direction="credit" if amount > 0 else "debit",
            account_id=account_id,
            bank=bank,
        )


class SourceAdapter(ABC):
    """The contract every linked institution must satisfy."""

    account_id: str
    bank_name: str
    #: fields the export carries that the canonical schema deliberately drops
    discards: tuple[str, ...] = ()

    @abstractmethod
    def parse(self, source: Path) -> list[CanonicalTransaction]:
        """Read one export and return canonical rows."""


class ChaseAdapter(SourceAdapter):
    """One signed Amount column; dates are month-first."""

    account_id = "acc-chase"
    bank_name = "Chase"
    discards = ("Category", "Type", "Memo")

    def parse(self, source: Path) -> list[CanonicalTransaction]:
        out = []
        with source.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                out.append(CanonicalTransaction.of(
                    row["Memo"], row["Description"], float(row["Amount"]),
                    datetime.strptime(row["Transaction Date"], "%m/%d/%Y").date(),
                    self.account_id, self.bank_name))
        return out


class CapitalOneAdapter(SourceAdapter):
    """Separate Debit and Credit columns; exactly one carries a value."""

    account_id = "acc-capitalone"
    bank_name = "Capital One"
    discards = ("Card No.", "Category", "Posted Date")

    def parse(self, source: Path) -> list[CanonicalTransaction]:
        out = []
        with source.open(encoding="utf-8", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                debit = (row.get("Debit") or "").strip()
                credit = (row.get("Credit") or "").strip()
                if bool(debit) == bool(credit):
                    raise ValueError(
                        f"{source.name} row {index + 2}: exactly one of Debit "
                        f"and Credit must carry a value, got {debit!r}/{credit!r}")
                amount = -float(debit) if debit else float(credit)
                # the transaction date is used, not the posted date: the user
                # made the purchase on the day they remember, not the day the
                # bank settled it
                out.append(CanonicalTransaction.of(
                    f"{self.account_id}-{index}", row["Description"], amount,
                    date.fromisoformat(row["Transaction Date"]),
                    self.account_id, self.bank_name))
        return out


class AppleCardAdapter(SourceAdapter):
    """Twelve monthly files, and the sign convention is inverted."""

    account_id = "acc-applecard"
    bank_name = "Apple Card"
    discards = ("Merchant", "Category", "Type", "Clearing Date")

    def parse(self, source: Path) -> list[CanonicalTransaction]:
        """`source` is the directory holding the monthly exports."""
        files = sorted(source.glob("applecard_*.csv"))
        if not files:
            raise FileNotFoundError(f"no Apple Card monthly exports in {source}")
        out = []
        for path in files:
            with path.open(encoding="utf-8", newline="") as handle:
                for index, row in enumerate(csv.DictReader(handle)):
                    # Apple Card writes purchases positive and payments
                    # negative. Negating restores the canonical convention.
                    amount = -float(row["Amount (USD)"])
                    out.append(CanonicalTransaction.of(
                        f"{path.stem}-{index}", row["Description"], amount,
                        datetime.strptime(row["Transaction Date"],
                                          "%m/%d/%Y").date(),
                        self.account_id, self.bank_name))
        return out


ADAPTERS: dict[str, SourceAdapter] = {
    "chase_activity.csv": ChaseAdapter(),
    "capitalone_transactions.csv": CapitalOneAdapter(),
    "applecard": AppleCardAdapter(),      # a directory, not a single file
}


def ingest(feed_directory: Path) -> list[CanonicalTransaction]:
    """Read every feed in a directory and return one canonical list."""
    rows: list[CanonicalTransaction] = []
    for name, adapter in ADAPTERS.items():
        source = feed_directory if name == "applecard" else feed_directory / name
        if name != "applecard" and not source.exists():
            continue
        rows += adapter.parse(source)
    return sorted(rows, key=lambda r: r.booking_date)
