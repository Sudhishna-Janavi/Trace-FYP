#!/usr/bin/env python3
"""Unit 4: split the demo transactions into three heterogeneous bank feeds.

The project's claim is multi-account aggregation, which needs more than one
source and more than one format. Each feed here is modelled on a publicly
documented export layout, so the conventions the adapters must reconcile are
real differences rather than invented ones.

  Chase          one signed Amount column, US month-first dates
  Capital One    separate Debit and Credit columns, one blank on every row,
                 and two dates: a transaction date and a posted date, which
                 genuinely differ because a Friday purchase posts on Monday
  Apple Card     twelve monthly files rather than one, and purchases written
                 as POSITIVE with payments NEGATIVE, the reverse of the others

Exact headers vary by account type and could not be fully verified, so these
are representative rather than exact replicas. That limitation is stated in the
design record rather than glossed over.

Every feed also carries fields the canonical schema discards, including the
bank's own Category column. Real exports carry them, and discarding the bank's
category is essential: using it would mean the bank had already done the
categorisation this project exists to perform.

Run:
    ~/.venvs/oct14/bin/python scripts/build_feeds.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A bank's own guess, carried by real exports and deliberately discarded.
BANK_CATEGORY = {
    "Restaurants": "Dining", "Groceries": "Groceries", "Transportation": "Gas/Automotive",
    "Healthcare": "Health Care", "Entertainment": "Entertainment", "Shopping": "Merchandise",
    "Personal Care": "Other Services", "Education": "Other Services", "Travel": "Travel",
    "Utilities": "Utilities", "Rent": "Other", "Mortgage": "Other", "Insurance": "Insurance",
    "Subscription": "Internet", "Fees": "Fee/Interest Charge", "Income": "Payment/Credit",
    "Transfer": "Payment/Credit",
}


def posted_date(booking: date, rng: random.Random) -> date:
    """A card transaction posts a day or two after it is made."""
    return booking + timedelta(days=rng.choice([0, 1, 1, 2, 3]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=Path,
                        default=ROOT / "dataset" / "demo" / "demo_transactions.csv")
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "dataset" / "demo" / "feeds")
    parser.add_argument("--seed", type=int, default=1131)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    with args.demo.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_account = collections.defaultdict(list)
    for row in rows:
        by_account[row["account_id"]].append(row)

    written = {}

    # ---- Chase: one signed Amount, month-first dates -----------------------
    chase = sorted(by_account["acc-chase"], key=lambda r: r["booking_date"])
    path = args.outdir / "chase_activity.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Transaction Date", "Post Date", "Description",
                         "Category", "Type", "Amount", "Memo"])
        for row in chase:
            booking = date.fromisoformat(row["booking_date"])
            amount = float(row["amount"])
            writer.writerow([
                booking.strftime("%m/%d/%Y"),
                posted_date(booking, rng).strftime("%m/%d/%Y"),
                row["raw_description"],
                BANK_CATEGORY.get(row["category"], "Other"),
                "Sale" if amount < 0 else "Payment",
                f"{amount:.2f}",
                row["transaction_id"],
            ])
    written[path.name] = len(chase)

    # ---- Capital One: split Debit and Credit columns ----------------------
    capital = sorted(by_account["acc-capitalone"], key=lambda r: r["booking_date"])
    path = args.outdir / "capitalone_transactions.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Transaction Date", "Posted Date", "Card No.",
                         "Description", "Category", "Debit", "Credit"])
        for row in capital:
            booking = date.fromisoformat(row["booking_date"])
            amount = float(row["amount"])
            writer.writerow([
                booking.isoformat(),
                posted_date(booking, rng).isoformat(),
                "1874",
                row["raw_description"],
                BANK_CATEGORY.get(row["category"], "Other"),
                f"{abs(amount):.2f}" if amount < 0 else "",
                f"{amount:.2f}" if amount > 0 else "",
            ])
    written[path.name] = len(capital)

    # ---- Apple Card: one file per month, and inverted signs ---------------
    apple = by_account["acc-applecard"]
    monthly = collections.defaultdict(list)
    for row in apple:
        monthly[row["booking_date"][:7]].append(row)
    apple_total = 0
    for month in sorted(monthly):
        path = args.outdir / f"applecard_{month.replace('-', '_')}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Transaction Date", "Clearing Date", "Description",
                             "Merchant", "Category", "Type", "Amount (USD)"])
            for row in sorted(monthly[month], key=lambda r: r["booking_date"]):
                booking = date.fromisoformat(row["booking_date"])
                amount = float(row["amount"])
                writer.writerow([
                    booking.strftime("%m/%d/%Y"),
                    posted_date(booking, rng).strftime("%m/%d/%Y"),
                    row["raw_description"],
                    row["merchant_key"],
                    BANK_CATEGORY.get(row["category"], "Other"),
                    "Purchase" if amount < 0 else "Payment",
                    # Apple Card writes purchases positive and payments
                    # negative, the reverse of the other two feeds
                    f"{-amount:.2f}",
                ])
        apple_total += len(monthly[month])
        written[path.name] = len(monthly[month])

    # ---- ground truth, kept away from every feed --------------------------
    truth = args.outdir.parent / "ground_truth.csv"
    with truth.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "transaction_id", "category", "merchant_key", "account_id",
            "is_recurring"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    manifest = {
        "note": "formats are modelled on publicly documented export layouts; "
                "exact headers vary by account type and are not exact replicas",
        "canonical_sign_convention": "negative is money out, positive is money in",
        "feeds": {
            "chase_activity.csv": {
                "bank": "Chase", "account_id": "acc-chase", "rows": len(chase),
                "date_format": "MM/DD/YYYY", "amount": "one signed column",
                "discarded_by_adapter": ["Category", "Type", "Memo"],
                "source": "https://beancount.io/tools/csv-to-beancount/chase",
            },
            "capitalone_transactions.csv": {
                "bank": "Capital One", "account_id": "acc-capitalone",
                "rows": len(capital), "date_format": "YYYY-MM-DD",
                "amount": "separate Debit and Credit columns",
                "discarded_by_adapter": ["Card No.", "Category"],
                "source": "https://tiller.com/export-capital-one-transactions/",
            },
            "applecard_YYYY_MM.csv": {
                "bank": "Apple Card", "account_id": "acc-applecard",
                "rows": apple_total, "files": len(monthly),
                "date_format": "MM/DD/YYYY",
                "amount": "purchases POSITIVE, payments NEGATIVE: inverted "
                          "relative to the other feeds",
                "discarded_by_adapter": ["Merchant", "Category", "Type"],
                "source": "https://community.tiller.com/t/apple-card-statement-"
                          "export-import/3025",
            },
        },
        "bank_category_is_discarded": "every feed carries the bank's own "
            "Category column, as real exports do. The adapters discard it: "
            "using it would mean the bank had already performed the "
            "categorisation this project exists to perform.",
    }
    (args.outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"{'file':<34}{'rows':>7}")
    for name, count in sorted(written.items()):
        print(f"  {name:<32}{count:>7}")
    print(f"\n  Chase          {len(chase):>5} rows, 1 file,  signed Amount")
    print(f"  Capital One    {len(capital):>5} rows, 1 file,  Debit/Credit columns")
    print(f"  Apple Card     {apple_total:>5} rows, {len(monthly)} files, "
          f"INVERTED signs")
    print(f"  total          {len(rows):>5} rows")
    print(f"\n[PASS] wrote {len(written)} feed files + manifest.json to {args.outdir}")
    print(f"[PASS] wrote {truth} (never read by any adapter)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
