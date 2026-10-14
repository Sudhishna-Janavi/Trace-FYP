#!/usr/bin/env python3
"""Unit 4 verification: do the adapters reconstruct the original transactions?

Each feed was written in a different format, so parsing it back and comparing
against the source is the only way to know the adapters are correct rather than
merely plausible. The Apple Card sign inversion in particular would otherwise
fail silently, turning every purchase into income.

Run:
    ~/.venvs/oct14/bin/python scripts/verify_feeds.py
"""
from __future__ import annotations

import collections
import csv
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.adapters import ADAPTERS, ingest  # noqa: E402

DEMO = ROOT / "dataset" / "demo"
FEEDS = DEMO / "feeds"


def main() -> int:
    with (DEMO / "demo_transactions.csv").open(encoding="utf-8", newline="") as h:
        original = list(csv.DictReader(h))
    parsed = ingest(FEEDS)

    failures = []

    # ---- 1. nothing lost, nothing invented -------------------------------
    print(f"{'check':<52}{'result':>22}")
    ok = len(parsed) == len(original)
    print(f"  {'row count preserved':<50}"
          f"{f'{len(parsed):,} vs {len(original):,}':>22}  {'OK' if ok else 'FAIL'}")
    failures += [] if ok else ["row count"]

    # ---- 2. the money survives every format ------------------------------
    # amounts are matched as a multiset, since adapters assign their own ids
    original_amounts = collections.Counter(
        round(float(r["amount"]), 2) for r in original)
    parsed_amounts = collections.Counter(round(r.amount, 2) for r in parsed)
    ok = original_amounts == parsed_amounts
    missing = (original_amounts - parsed_amounts).most_common(3)
    print(f"  {'every amount round-trips exactly':<50}{'':>22}  "
          f"{'OK' if ok else 'FAIL ' + str(missing)}")
    failures += [] if ok else ["amounts"]

    # ---- 3. the Apple Card sign inversion was undone ---------------------
    # if it were missed, Apple Card purchases would appear as income
    apple = [r for r in parsed if r.account_id == "acc-applecard"]
    apple_credits = sum(1 for r in apple if r.amount > 0)
    source_apple = [r for r in original if r["account_id"] == "acc-applecard"]
    source_credits = sum(1 for r in source_apple if float(r["amount"]) > 0)
    ok = apple_credits == source_credits
    print(f"  {'Apple Card signs restored':<50}"
          f"{f'{apple_credits} credits vs {source_credits}':>22}  "
          f"{'OK' if ok else 'FAIL'}")
    failures += [] if ok else ["apple signs"]

    # ---- 4. dates survive three different formats ------------------------
    original_dates = collections.Counter(r["booking_date"] for r in original)
    parsed_dates = collections.Counter(r.booking_date.isoformat() for r in parsed)
    ok = original_dates == parsed_dates
    print(f"  {'every date round-trips across 3 formats':<50}{'':>22}  "
          f"{'OK' if ok else 'FAIL'}")
    failures += [] if ok else ["dates"]

    # ---- 5. descriptions are untouched -----------------------------------
    original_text = collections.Counter(r["raw_description"] for r in original)
    parsed_text = collections.Counter(r.raw_description for r in parsed)
    ok = original_text == parsed_text
    print(f"  {'descriptions unchanged':<50}{'':>22}  {'OK' if ok else 'FAIL'}")
    failures += [] if ok else ["descriptions"]

    # ---- 6. no feed leaks the answer -------------------------------------
    leaked = []
    for path in sorted(FEEDS.glob("*.csv")):
        header = path.open(encoding="utf-8").readline()
        for banned in ("category_code", "true_category", "label", "ground_truth"):
            if banned.lower() in header.lower():
                leaked.append(f"{path.name}:{banned}")
    print(f"  {'no feed carries the true label':<50}{'':>22}  "
          f"{'OK' if not leaked else 'FAIL ' + str(leaked)}")
    failures += [] if not leaked else ["label leak"]

    # ---- 7. the bank's own category is present but discarded -------------
    header = (FEEDS / "chase_activity.csv").open(encoding="utf-8").readline()
    carries = "Category" in header
    discarded = "Category" in ADAPTERS["chase_activity.csv"].discards
    ok = carries and discarded
    print(f"  {'bank Category carried by feed, dropped by adapter':<50}"
          f"{'':>22}  {'OK' if ok else 'FAIL'}")
    failures += [] if ok else ["bank category"]

    # ---- per-account summary ---------------------------------------------
    print(f"\n{'account':<16}{'rows':>7}{'money out':>14}{'money in':>13}"
          f"{'files':>7}")
    for account in ("acc-chase", "acc-capitalone", "acc-applecard"):
        rows = [r for r in parsed if r.account_id == account]
        out = sum(-r.amount for r in rows if r.amount < 0)
        income = sum(r.amount for r in rows if r.amount > 0)
        files = (len(list(FEEDS.glob("applecard_*.csv")))
                 if account == "acc-applecard" else 1)
        print(f"  {rows[0].bank:<14}{len(rows):>7}{out:>14,.2f}{income:>13,.2f}"
              f"{files:>7}")

    span = sorted(r.booking_date for r in parsed)
    print(f"\n  period {span[0]} to {span[-1]}")
    print(f"  discarded by adapters: " + ", ".join(
        f"{a.bank_name} {list(a.discards)}" for a in ADAPTERS.values()))

    if failures:
        print(f"\n[FAIL] {len(failures)} check(s) failed: {failures}")
        return 1
    print(f"\n[PASS] all 7 checks passed; {len(parsed):,} transactions "
          "reconstructed from 14 files in 3 formats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
