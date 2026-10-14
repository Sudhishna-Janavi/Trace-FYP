"""Model 1b of 6: a merchant-memory baseline. Lookup, not learning.

The deliberate opposite of Model 1. The rule baseline contains no merchant names
and relies entirely on generic banking vocabulary. This one contains nothing but
merchant names: it records which category each merchant was given during
training and looks the answer up again later.

Together the two bracket the problem. Rules are merchant-independent and
therefore unaffected by whether a shop is familiar. Memory is purely
merchant-dependent and therefore useless the moment a shop is not. The learned
models sit between those extremes, and the gap between these two baselines is
the space those models have to occupy.

Why the fingerprint is computed from the description alone
    The prepared CSVs already carry a `merchant_key`, but that key was derived
    using statistics over the whole corpus, which a deployed system would not
    have when a single new transaction arrives. Using it here would quietly
    hand this baseline information it could not possess in practice, so the
    fingerprint below is computed from one description in isolation.

Matching is exact. A fuzzy variant that tolerates small spelling differences
would raise coverage and is how the deployed cascade would be built, but exact
matching keeps this baseline simple to interpret: it answers if and only if it
has seen the merchant before.
"""
from __future__ import annotations

import collections
import re

NOISE = {
    "DEBIT", "CREDIT", "POS", "DBT", "ECOM", "GIRO", "REF", "USA", "US", "EFT",
    "PYMT", "PPD", "ID", "PURCHASE", "PAYMENT", "CARD", "ACH", "WEB", "RECUR",
    "XFER", "DES", "INDN", "CO",
}
PROCESSOR = {
    "PAYPAL", "PP", "PYPL", "TST", "CLV", "SQ", "INST", "PREAPPROVED", "EXPRESS",
    "CHECKOUT", "WITHDRAWAL", "FROM", "PREAUTHORIZED", "DEPOSIT", "BILL", "USER",
    "MKTPL", "TO",
}
FINGERPRINT_TOKENS = 2


def fingerprint(normalised: str) -> str:
    """A merchant key derivable from a single description, with no corpus stats.

    Bank noise, processor prefixes, numbers and opaque reference codes are
    dropped; the leading remaining tokens name the merchant.
    """
    useful = []
    for word in normalised.split():
        if word in NOISE or word in PROCESSOR or word == "<NUM>" or word.isdigit():
            continue
        if len(word) <= 2:
            continue
        if len(word) >= 6 and re.search(r"\d", word) and re.fullmatch(r"[A-Z0-9]+", word):
            continue
        useful.append(word)
        if len(useful) == FINGERPRINT_TOKENS:
            break
    return " ".join(useful)


class MerchantMemory:
    def __init__(self) -> None:
        self._seen: dict[str, collections.Counter] = {}

    def learn(self, rows) -> None:
        for row in rows:
            key = fingerprint(row["normalised_description"])
            if not key:
                continue
            self._seen.setdefault(key, collections.Counter())[row["category"]] += 1

    def predict(self, normalised: str) -> str | None:
        """The remembered category, or None when the merchant is unknown.

        A merchant recorded under two categories equally often is ambiguous, and
        the memory abstains rather than picking arbitrarily.
        """
        counts = self._seen.get(fingerprint(normalised))
        if not counts:
            return None
        top = counts.most_common(2)
        if len(top) > 1 and top[0][1] == top[1][1]:
            return None
        return top[0][0]

    def __len__(self) -> int:
        return len(self._seen)
