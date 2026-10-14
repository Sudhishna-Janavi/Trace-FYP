#!/usr/bin/env python3
"""Unit 3: attach money, dates and accounts to held-out transactions.

The benchmark supplies a description and a category and nothing else. A finance
dashboard with no money in it can only count rows, so amounts and dates are
generated here.

The central guarantee
    Amounts, dates and account identifiers are **never model features**. Every
    accuracy, F1, calibration and explainability figure in Unit 2 was computed
    from real third-party text and real third-party labels alone, and adding
    money changes none of them. They exist only to drive the analytics layer.

Method
    1. Each category is mapped to a line of the U.S. Bureau of Labor Statistics
       Consumer Expenditure Survey 2024, giving an annual household spend.
    2. Dividing by an assumed number of transactions per year gives a mean per
       transaction. The annual figures are official; the transaction counts are
       this project's assumptions, and both are recorded.
    3. Amounts are drawn from a log-normal distribution with that mean, because
       money is strictly positive and right-skewed. A normal distribution would
       produce negative prices.
    4. The number of transactions drawn per category is proportional to those
       same transactions-per-year figures, so the mix resembles a household
       rather than the class-balanced dataset.
    5. Recurring categories emit a full twelve-month series at a fixed day of
       month with near-identical amounts, so cadence detection has something
       real to find.
    6. Accounts are assigned by the statement format actually present in the
       description, which is real heterogeneity rather than invented.

Source rows come only from the held-out test split, so the deployed model has
never been trained on any row the demo displays.

Run:
    ~/.venvs/oct14/bin/python scripts/build_demo_data.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import re
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_model import load  # noqa: E402

# ---------------------------------------------------------------- BLS 2024 --
# "annual" is from the BLS Consumer Expenditure Survey 2024 release, Table A,
# unless marked assumed. "per_year" is this project's assumption throughout.
# "share" adjusts a population average to a per-participating-household figure:
# BLS averages include households that spend nothing in a category, so rented
# dwellings at $5,660 is the average across renters and owners alike, not what
# a renter pays.
BLS = {
    "Restaurants":    {"annual": 3945,   "per_year": 150, "source": "bls",
                       "line": "Food away from home"},
    "Groceries":      {"annual": 6224,   "per_year": 100, "source": "bls",
                       "line": "Food at home"},
    "Transportation": {"annual": 13318,  "per_year": 120, "source": "bls",
                       "line": "Transportation"},
    "Healthcare":     {"annual": 6197,   "per_year": 24,  "source": "bls",
                       "line": "Healthcare"},
    "Entertainment":  {"annual": 3609,   "per_year": 90,  "source": "bls",
                       "line": "Entertainment"},
    "Personal Care":  {"annual": 978,    "per_year": 24,  "source": "bls",
                       "line": "Personal care products and services"},
    "Education":      {"annual": 1569,   "per_year": 6,   "source": "bls",
                       "line": "Education"},
    "Shopping":       {"annual": 2001,   "per_year": 80,  "source": "bls",
                       "line": "Apparel and services"},
    "Mortgage":       {"annual": 9310,   "per_year": 12,  "source": "bls",
                       "line": "Owned dwellings", "share": 0.65},
    "Rent":           {"annual": 5660,   "per_year": 12,  "source": "bls",
                       "line": "Rented dwellings", "share": 0.35},
    "Travel":         {"annual": 1347,   "per_year": 4,   "source": "bls",
                       "line": "Other lodging"},
    "Insurance":      {"annual": 2568,   "per_year": 12,  "source": "bls",
                       "line": "Vehicle insurance 1,993 + life and other "
                               "personal insurance 575"},
    # Scaled to take-home pay; see AFTER_TAX_RATIO below.
    "Income":         {"annual": 104207, "per_year": 24,  "source": "bls",
                       "line": "Average income before taxes",
                       "after_tax": True},
    # Not broken out in Table A, which shows selected subcategories only.
    "Utilities":      {"annual": 4736,   "per_year": 36,  "source": "assumed",
                       "line": "Utilities, fuels and public services; not "
                               "itemised in Table A"},
    "Subscription":   {"annual": 1200,   "per_year": 36,  "source": "assumed",
                       "line": "no BLS line; streaming and software subscriptions"},
    "Fees":           {"annual": 240,    "per_year": 6,   "source": "assumed",
                       "line": "no BLS line; bank charges"},
    "Transfer":       {"annual": 12000,  "per_year": 24,  "source": "assumed",
                       "line": "not an expenditure; movement between the user's "
                               "own accounts"},
}

# What reaches a bank account is pay after tax, after Social Security and after
# payroll deductions, so the BLS "income before taxes" line cannot be deposited
# as it stands: doing so showed a household saving half of everything it earned.
#
# BLS published "income after taxes" alongside it until the 2023 release and
# then discontinued it, stating it can no longer produce federal tax, state tax
# or after-tax income estimates for the CE tables, LABSTAT or the public use
# microdata. The 2024 release therefore has no after-tax line to cite.
#
# The 2024 before-tax figure is scaled by the last ratio BLS itself published,
# from 2023: 87,869 after tax against 101,805 before tax. Using their own
# published relationship keeps the adjustment traceable to the source rather
# than to an assumed tax rate of this project's invention.
AFTER_TAX_RATIO = 87869 / 101805      # 0.8631, BLS CE 2023

# How many distinct merchants each recurring category should have. A household
# has one landlord, not three, so this cannot be derived from the row quota.
RECURRING_MERCHANTS = {
    "Rent": 1, "Mortgage": 1, "Utilities": 3, "Insurance": 2, "Subscription": 4,
    # Salary is the most regular payment a household has. Treating Income as a
    # one-off category scattered it to random dates and left January with no
    # salary at all, and two payments one day apart.
    "Income": 1,
}
# Categories paid twice a month rather than once.
FORTNIGHTLY = {"Income"}
RECURRING = set(RECURRING_MERCHANTS)
# Spread for one-off spending. These categories are deliberately NOT reconciled
# to their BLS annual figure: a household's restaurant and grocery spending
# genuinely varies from the national average, and forcing each to land on it
# exactly would be less realistic, not more. With roughly a hundred independent
# draws the realised totals settle within a tenth of target on their own, and
# the categories drawn only a handful of times are left visibly noisy because
# that is what a small number of real purchases looks like.
SIGMA = 0.55          # log-normal spread; wider means more large outliers
# A household's rent or insurance premium is a fixed commitment near the
# typical figure, not a lottery, so recurring bases are drawn far more tightly.
# Their level no longer depends on this spread at all — the draws are rescaled
# to the BLS annual figure afterwards — so it now controls only how different
# two subscriptions look from each other.
SIGMA_RECURRING = 0.15
MONTHS = 12

# Statement formats genuinely present in the source data, verified by counting.
ACCOUNTS = [
    ("acc-chase", "Chase", "Current Account", "ach",
     re.compile(r"\b(PPD|WEB) ID\b")),
    ("acc-capitalone", "Capital One", "Savings Account", "prose",
     re.compile(r"(?i)\b(WITHDRAWAL FROM|PREAUTHORIZED DEPOSIT FROM)\b")),
    ("acc-applecard", "Apple Card", "Credit Card", "address",
     re.compile(r"<NUM>\s+\d{0,4}[A-Z]{2}\s+USA\b|\b[A-Z]{2}\s+USA$")),
]


def annual_target(category: str) -> float:
    """What this category should total over a year, for one household."""
    cfg = BLS[category]
    annual = cfg["annual"] / cfg.get("share", 1.0)
    if cfg.get("after_tax"):
        annual *= AFTER_TAX_RATIO
    return annual


def mean_per_transaction(category: str) -> float:
    return annual_target(category) / BLS[category]["per_year"]


def draw(rng: random.Random, mean: float, sigma: float = SIGMA) -> float:
    """Log-normal with the given arithmetic mean."""
    mu = math.log(mean) - (sigma ** 2) / 2
    return max(round(rng.lognormvariate(mu, sigma), 2), 0.50)


def pick_account(normalised: str, rng: random.Random) -> tuple[str, str, str]:
    for account_id, bank, name, _fmt, pattern in ACCOUNTS:
        if pattern.search(normalised):
            return account_id, bank, name
    account_id, bank, name, _fmt, _p = rng.choice(ACCOUNTS)
    return account_id, bank, name


def month_start(start: date, offset: int) -> date:
    year = start.year + (start.month - 1 + offset) // 12
    month = (start.month - 1 + offset) % 12 + 1
    return date(year, month, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    # One BLS household-year is about 700 transactions. Asking for many more
    # inflates the discretionary categories while housing stays fixed at twelve
    # payments, so the category proportions drift away from the BLS shares the
    # whole calibration exists to reproduce.
    parser.add_argument("--rows", type=int, default=0,
                        help="0 means one household-year, which keeps the "
                             "category proportions faithful to BLS")
    parser.add_argument("--start", default="2025-10-01")
    parser.add_argument("--seed", type=int, default=1131)
    # A household rents or owns; it does not do both. Emitting Rent and
    # Mortgage together produced a housing share of 44% against the BLS 33%.
    parser.add_argument("--tenure", choices=("rent", "own"), default="rent")
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "dataset" / "demo")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    year, month, day = map(int, args.start.split("-"))
    start = date(year, month, day)

    excluded = "Mortgage" if args.tenure == "rent" else "Rent"
    print(f"household tenure: {args.tenure} "
          f"({excluded} therefore produces no transactions)\n")

    pool = load("stratified_test.csv")
    by_category: dict[str, list[dict]] = collections.defaultdict(list)
    for row in pool:
        by_category[row["category"]].append(row)
    for rows in by_category.values():
        rng.shuffle(rows)

    # ---- how many transactions per category: proportional to a household ----
    household_year = sum(c["per_year"] for c in BLS.values())
    target = args.rows or household_year
    scale = target / household_year
    quota = {name: max(1, round(cfg["per_year"] * scale))
             for name, cfg in BLS.items()}
    quota[excluded] = 0
    # a recurring category's row count is merchants x months, not a quota share
    for name, merchants in RECURRING_MERCHANTS.items():
        per_month = 2 if name in FORTNIGHTLY else 1
        quota[name] = 0 if name == excluded else merchants * MONTHS * per_month

    print(f"a BLS household transacts about {household_year} times a year; "
          f"scaling by {scale:.2f} for {target} rows\n")
    print(f"  {'category':<16}{'BLS annual':>12}{'/yr':>6}{'mean/txn':>11}"
          f"{'rows':>7}  source")
    for name, cfg in sorted(BLS.items(), key=lambda kv: -quota[kv[0]]):
        print(f"  {name:<16}{cfg['annual']:>12,}{cfg['per_year']:>6}"
              f"{mean_per_transaction(name):>11,.2f}{quota[name]:>7}  "
              f"{cfg['source']}")

    emitted: list[dict] = []
    used: set[str] = set()

    # ---- recurring categories: a full twelve-month series per merchant ----
    for category in sorted(RECURRING):
        if category == excluded:
            continue
        merchants = RECURRING_MERCHANTS[category]
        mean = mean_per_transaction(category)
        # distinct merchant names, so two services do not share a fingerprint
        # and read as one erratic subscription
        candidates = by_category[category]
        if category == "Income":
            # the recurring income merchant is an employer, so prefer rows that
            # read as payroll rather than, say, a refund
            payroll = [r for r in candidates
                       if re.search(r"\b(PAYROLL|DIR DEP|DIRECT DEP|SALARY)\b",
                                    r["normalised_description"])]
            candidates = payroll or candidates
        chosen, seen_keys = [], set()
        for row in candidates:
            key = row["merchant_key"]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            chosen.append(row)
            if len(chosen) == merchants:
                break
        # Two separate quantities decided this category's annual total and
        # nothing made them agree. How MANY payments are emitted comes from
        # `merchants x MONTHS`; how BIG each one is comes from dividing the BLS
        # annual figure by `per_year`. Where those disagreed the household was
        # billed the wrong amount for the year: Insurance names two merchants
        # against a `per_year` of twelve, so it emitted 24 payments sized for
        # 12 and charged 2.27x the BLS figure.
        #
        # A single draw also decided twelve months of a fixed commitment, so an
        # unlucky tail set the whole year wrong. With one merchant there is no
        # second draw to average against: one roll put salary at the clamp
        # ceiling and inflated annual income by 25%.
        #
        # Both are fixed by drawing every merchant's base first and then
        # rescaling them together, so the payments this category actually emits
        # sum to its BLS annual figure by construction. Variety between
        # merchants survives; only the overall level is pinned.
        per_month = 2 if category in FORTNIGHTLY else 1
        payments_each = MONTHS * per_month
        bases = [draw(rng, mean, SIGMA_RECURRING) for _ in chosen]
        reconcile = annual_target(category) / (sum(bases) * payments_each)
        bases = [base * reconcile for base in bases]

        for row, base in zip(chosen, bases):
            used.add(row["transaction_id"])
            fixed_day = rng.randint(1, 28)
            account = pick_account(row["normalised_description"], rng)
            # a fortnightly category is paid on two fixed days each month
            days = ([fixed_day, min(fixed_day + 14, 28)]
                    if per_month == 2 else [fixed_day])
            occurrences = [(offset, d) for offset in range(MONTHS) for d in days]
            for sequence, (offset, pay_day) in enumerate(occurrences):
                when = month_start(start, offset).replace(day=pay_day)
                # a recurring commitment repeats a near-identical amount
                amount = round(base * rng.uniform(0.97, 1.03), 2)
                emitted.append({
                    **row,
                    "transaction_id": f"{row['transaction_id']}-m{sequence:02d}",
                    "amount": f"{amount if row['direction'] == 'credit' else -amount:.2f}",
                    "booking_date": when.isoformat(),
                    "account_id": account[0], "bank": account[1],
                    "account_name": account[2], "is_recurring": "1",
                })

    # ---- everything else: one-off transactions across the year ----
    for category, wanted in quota.items():
        if category == excluded or category in RECURRING:
            continue
        already = sum(1 for e in emitted if e["category"] == category)
        mean = mean_per_transaction(category)
        available = [r for r in by_category[category]
                     if r["transaction_id"] not in used]
        for row in available[: max(0, wanted - already)]:
            used.add(row["transaction_id"])
            amount = draw(rng, mean)
            signed = amount if row["direction"] == "credit" else -amount
            when = start + timedelta(days=rng.randint(0, 364))
            account = pick_account(row["normalised_description"], rng)
            emitted.append({
                **row, "amount": f"{signed:.2f}",
                "booking_date": when.isoformat(),
                "account_id": account[0], "bank": account[1],
                "account_name": account[2], "is_recurring": "0",
            })

    emitted.sort(key=lambda r: r["booking_date"])

    fields = ["transaction_id", "raw_description", "normalised_description",
              "merchant_key", "category", "direction", "amount",
              "booking_date", "account_id", "bank", "account_name",
              "is_recurring"]
    out = args.outdir / "demo_transactions.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(emitted)

    # ------------------------------------------- reconcile against BLS ----
    # Recurring commitments are pinned to their BLS annual figure, so any drift
    # here is a bug rather than sampling noise. Only the per-payment jitter of
    # plus or minus 3% should show, which over twelve or more payments leaves
    # well under one percent.
    print(f"\n{'recurring category':<22}{'BLS/yr':>11}{'emitted':>11}{'drift':>8}")
    worst = 0.0
    for category in sorted(RECURRING):
        if category == excluded:
            continue
        got = sum(abs(float(r["amount"])) for r in emitted
                  if r["category"] == category)
        target = annual_target(category)
        drift = got / target - 1
        worst = max(worst, abs(drift))
        print(f"  {category:<20}{target:>11,.0f}{got:>11,.0f}{drift:>+7.1%}")
    assert worst < 0.02, f"recurring category drifted {worst:.1%} from BLS"
    print(f"  worst drift {worst:.2%} — within the per-payment jitter\n")

    # ---------------------------------------------------------- verify ----
    from statistics import median
    amounts = collections.defaultdict(list)
    for row in emitted:
        amounts[row["category"]].append(abs(float(row["amount"])))
    spend = sum(abs(float(r["amount"])) for r in emitted
                if r["category"] not in ("Income", "Transfer"))
    income = sum(float(r["amount"]) for r in emitted if r["category"] == "Income")
    months = sorted({r["booking_date"][:7] for r in emitted})
    accounts = collections.Counter(r["bank"] for r in emitted)

    print(f"\n{'category':<16}{'rows':>7}{'median':>12}{'total':>14}{'share':>8}")
    for category, values in sorted(amounts.items(),
                                   key=lambda kv: -sum(kv[1])):
        total = sum(values)
        share = (total / spend * 100
                 if category not in ("Income", "Transfer") else 0)
        print(f"  {category:<14}{len(values):>7}{median(values):>12,.2f}"
              f"{total:>14,.0f}{share:>7.1f}%")

    print(f"\n  transactions      {len(emitted):,}")
    print(f"  distinct sources  {len(used):,}  (recurring rows repeat monthly)")
    print(f"  period            {months[0]} to {months[-1]}  ({len(months)} months)")
    print(f"  total spend       {spend:,.0f}")
    print(f"  total income      {income:,.0f}")
    print(f"  net               {income - spend:+,.0f}")
    print(f"  accounts          {dict(accounts)}")

    model = {
        "source": "U.S. Bureau of Labor Statistics, Consumer Expenditure Survey "
                  "2024 (released 19 December 2025)",
        "url": "https://www.bls.gov/news.release/cesan.nr0.htm",
        "guarantee": "amounts, dates and accounts are never model features; "
                     "every Unit 2 metric comes from text and labels alone",
        "drawn_from": "stratified_test.csv, held out from all model fitting",
        "distribution": f"log-normal, sigma {SIGMA}",
        "period": {"from": months[0], "to": months[-1], "months": len(months)},
        "household_share_adjustment":
            "BLS category averages include households spending nothing in that "
            "category. Rent and Mortgage are divided by the share of households "
            "that rent (0.35) or own (0.65) so the figure reflects a "
            "participating household rather than the population mean.",
        "assumptions": "every transactions-per-year figure is this project's "
                       "assumption, not BLS",
        "categories": {
            name: {**cfg,
                   "mean_per_transaction": round(mean_per_transaction(name), 2),
                   "rows_emitted": len(amounts.get(name, [])),
                   "observed_median": round(median(amounts[name]), 2)
                                      if amounts.get(name) else None}
            for name, cfg in BLS.items()},
        "accounts": [{"id": a[0], "bank": a[1], "name": a[2], "format": a[3]}
                     for a in ACCOUNTS],
        "recurring_categories": sorted(RECURRING),
        "recurring_merchants_per_category": RECURRING_MERCHANTS,
        "fortnightly_categories": sorted(FORTNIGHTLY),
        "tenure": args.tenure,
        "tenure_note": "a household rents or owns, not both; emitting Rent and "
                       "Mortgage together gave a housing share of 44% against "
                       f"the BLS figure of 33%. {excluded} is therefore empty.",
    }
    (args.outdir / "amount_model.json").write_text(
        json.dumps(model, indent=2), encoding="utf-8")

    print(f"\n[PASS] wrote {out}")
    print(f"[PASS] wrote {args.outdir / 'amount_model.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
