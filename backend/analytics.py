"""Money-based analytics over one user's transactions.

Spending is the sum of signed amounts, not a count of rows. Counting rows would
make a hundred coffees look larger than a year of rent.

Transfers between a user's own accounts are excluded throughout: moving money
from current to savings is not spending, and counting it would inflate every
total. Income is reported separately rather than netted into categories.

Every query is scoped by user_id. Scoping lives here rather than in the routes,
so a missing filter is a visible omission in one place.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
from datetime import date
from statistics import median

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database import Account, Transaction, User

# How each of the 17 categories affects a spending total.
INCOME = {"Income"}
NEUTRAL = {"Transfer"}


def is_spend(category: str) -> bool:
    return category not in INCOME and category not in NEUTRAL


@dataclass
class Totals:
    spending: float
    income: float
    net: float
    transactions: int
    needs_review: int
    accounts: int
    first: date | None
    last: date | None


def load_transactions(session: Session, user: User) -> list[Transaction]:
    return list(session.scalars(
        select(Transaction).where(Transaction.user_id == user.id)
        .order_by(Transaction.booking_date.desc())))


def totals(rows: list[Transaction], accounts: int) -> Totals:
    spending = sum(-r.amount for r in rows if is_spend(r.category) and r.amount < 0)
    income = sum(r.amount for r in rows if r.category in INCOME and r.amount > 0)
    dates = sorted(r.booking_date for r in rows) if rows else []
    return Totals(
        spending=spending, income=income, net=income - spending,
        transactions=len(rows),
        needs_review=sum(1 for r in rows if r.needs_review and not r.reviewed_at),
        accounts=accounts,
        first=dates[0] if dates else None, last=dates[-1] if dates else None,
    )


def monthly(rows: list[Transaction]) -> list[dict]:
    """Income and spending per calendar month, oldest first."""
    buckets: dict[str, dict] = collections.defaultdict(
        lambda: {"spending": 0.0, "income": 0.0})
    for row in rows:
        key = row.booking_date.strftime("%Y-%m")
        if row.category in INCOME and row.amount > 0:
            buckets[key]["income"] += row.amount
        elif is_spend(row.category) and row.amount < 0:
            buckets[key]["spending"] += -row.amount
    return [{"month": key, "label": _month_label(key), **values}
            for key, values in sorted(buckets.items())]


def _month_label(key: str) -> str:
    year, month = key.split("-")
    return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
            "Oct", "Nov", "Dec"][int(month) - 1]


def by_category(rows: list[Transaction]) -> list[dict]:
    totals_by: dict[str, float] = collections.defaultdict(float)
    counts: dict[str, int] = collections.defaultdict(int)
    for row in rows:
        if is_spend(row.category) and row.amount < 0:
            totals_by[row.category] += -row.amount
            counts[row.category] += 1
    overall = sum(totals_by.values()) or 1.0
    return sorted(
        [{"category": name, "amount": value, "count": counts[name],
          "share": value / overall * 100}
         for name, value in totals_by.items()],
        key=lambda item: -item["amount"])


def by_account(session: Session, user: User, rows: list[Transaction]) -> list[dict]:
    accounts = {a.id: a for a in session.scalars(
        select(Account).where(Account.user_id == user.id))}
    spend: dict[int, float] = collections.defaultdict(float)
    counts: dict[int, int] = collections.defaultdict(int)
    for row in rows:
        if is_spend(row.category) and row.amount < 0:
            spend[row.account_id] += -row.amount
            counts[row.account_id] += 1
    overall = sum(spend.values()) or 1.0
    return sorted(
        [{"bank": accounts[account_id].bank, "name": accounts[account_id].name,
          "format": accounts[account_id].source_format,
          "amount": value, "count": counts[account_id],
          "share": value / overall * 100}
         for account_id, value in spend.items() if account_id in accounts],
        key=lambda item: -item["amount"])


def recurring(rows: list[Transaction], limit: int = 8) -> list[dict]:
    """Merchants charging a similar amount at roughly monthly intervals.

    Three occurrences on distinct dates give two or more intervals, so a cadence
    can be observed rather than guessed from a single gap.
    """
    groups: dict[str, list[Transaction]] = collections.defaultdict(list)
    for row in rows:
        if is_spend(row.category) and row.amount < 0 and row.merchant_fingerprint:
            groups[row.merchant_fingerprint].append(row)

    found = []
    for fingerprint, items in groups.items():
        dates = sorted({r.booking_date for r in items})
        if len(dates) < 3:
            continue
        gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
        average = sum(gaps) / len(gaps)
        if not 20 <= average <= 45:
            continue
        amounts = [-r.amount for r in items]
        middle = median(amounts)
        # a subscription repeats a similar amount
        if middle <= 0 or any(abs(a - middle) > 0.25 * middle for a in amounts):
            continue
        found.append({
            "merchant": fingerprint, "category": items[-1].category,
            "description": items[-1].raw_description,
            "occurrences": len(items), "average": sum(amounts) / len(amounts),
            "interval": average, "annual": sum(amounts) / len(amounts) * 12,
        })
    return sorted(found, key=lambda item: -item["annual"])[:limit]


SHELTER = ("MORTGAGE", "MTG", "RENT", "LEASE", "ESCROW", "PRINCIPAL")
INSURANCE = ("INSURANCE", "PREMIUM", "POLICY")


def _bucket(row: Transaction) -> str:
    """Compare like with like.

    A category spanning very different payment sizes would flag its larger
    members every time. Rent and insurance are separated from general bills for
    that reason.
    """
    if row.category in ("Rent", "Mortgage"):
        return "housing"
    if row.category == "Insurance":
        return "insurance"
    return row.category


def unusual(rows: list[Transaction], recurring_merchants: set[str],
            limit: int = 8) -> list[dict]:
    """Amounts far above the usual for their category.

    Recurring merchants are excluded: a payment that repeats monthly at a
    stable amount is by definition not unusual, and including them would flag
    every rent payment.
    """
    buckets: dict[str, list[Transaction]] = collections.defaultdict(list)
    for row in rows:
        if not is_spend(row.category) or row.amount >= 0:
            continue
        if row.merchant_fingerprint in recurring_merchants:
            continue
        buckets[_bucket(row)].append(row)

    found = []
    for name, items in buckets.items():
        if len(items) < 8:
            continue
        amounts = [-r.amount for r in items]
        middle = median(amounts)
        deviation = median(abs(a - middle) for a in amounts)
        # a tightly clustered bucket would otherwise produce enormous scores
        deviation = max(deviation, 0.10 * middle)
        for row, amount in zip(items, amounts):
            score = 0.6745 * (amount - middle) / deviation if deviation else 0.0
            if score < 3.5:
                continue
            found.append({
                "id": row.id, "description": row.raw_description,
                "category": row.category, "date": row.booking_date,
                "amount": amount, "baseline": middle, "score": score,
            })
    return sorted(found, key=lambda item: -item["score"])[:limit]


METHODOLOGY = [
    "Spending is the sum of signed amounts, not a count of transactions.",
    "Transfers between your own accounts are movements rather than spending, "
    "and are excluded from every total.",
    "Predictions the system was not confident about are held for your review "
    "and are shown separately.",
    "A recurring payment repeats at a merchant on at least three distinct "
    "dates, roughly monthly, with every amount within 25% of the median.",
    "An unusual amount is measured against the median and median absolute "
    "deviation for its category, excluding recurring merchants.",
    "Amounts and dates are generated values calibrated to the US Bureau of "
    "Labor Statistics Consumer Expenditure Survey 2024. They are never used to "
    "categorise anything.",
]


# ------------------------------------------------- change, trend, benchmark --

def category_months(rows: list[Transaction]) -> dict[str, dict[str, float]]:
    """Spending per category per calendar month, as {month: {category: total}}."""
    out: dict[str, dict[str, float]] = collections.defaultdict(
        lambda: collections.defaultdict(float))
    for row in rows:
        if is_spend(row.category) and row.amount < 0:
            out[row.booking_date.strftime("%Y-%m")][row.category] += -row.amount
    return {month: dict(values) for month, values in out.items()}


def complete_months(rows: list[Transaction], today: date | None = None) -> list[str]:
    """Months with data, excluding one that has not finished yet.

    Comparing a half-finished month against a complete one invents a fall in
    spending that never happened. A month is only complete once the calendar
    has moved past it.
    """
    today = today or date.today()
    current = today.strftime("%Y-%m")
    return sorted(m for m in category_months(rows) if m < current)


def changes(rows: list[Transaction], today: date | None = None) -> dict:
    """What moved in the most recent complete month.

    Two baselines, because they answer different questions. Against the
    previous month is what a user notices; against their own twelve-month
    average is what actually indicates a change in habit, since a single
    expensive month makes a month-on-month comparison meaningless.
    """
    per_month = category_months(rows)
    months = complete_months(rows, today)
    if len(months) < 2:
        return {"available": False, "month": None, "previous": None, "rows": []}

    latest, previous = months[-1], months[-2]
    names = {name for month in months for name in per_month[month]}
    out = []
    for name in names:
        now = per_month[latest].get(name, 0.0)
        before = per_month[previous].get(name, 0.0)
        average = sum(per_month[m].get(name, 0.0) for m in months) / len(months)
        out.append({
            "category": name, "now": now, "previous": before, "average": average,
            "vs_previous": now - before, "vs_average": now - average,
        })
    out.sort(key=lambda item: -abs(item["vs_average"]))
    return {"available": True, "month": latest, "previous": previous,
            "months": len(months), "rows": out}


def _month_name(key: str) -> str:
    year, month = key.split("-")
    return f"{MONTH_NAMES[int(month) - 1]} {year}"


MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


@dataclass
class Signals:
    """Descriptive facts about a user's own spending.

    Deliberately not a score. Any single number claiming to rate financial
    health would need a threshold, and every threshold available here would be
    invented rather than derived. These are statements the data supports.
    """
    months: int
    lowest: float
    highest: float
    average: float
    trend_per_month: float       # dollars per month, from a least-squares fit
    overspent_months: int        # months where spending exceeded income
    overspent_list: list[str]


def signals(rows: list[Transaction], today: date | None = None) -> Signals | None:
    per_month = category_months(rows)
    months = complete_months(rows, today)
    if len(months) < 3:
        return None

    spend = [sum(per_month[m].values()) for m in months]
    income: dict[str, float] = collections.defaultdict(float)
    for row in rows:
        if row.category in INCOME and row.amount > 0:
            income[row.booking_date.strftime("%Y-%m")] += row.amount

    # least squares on (index, spend): the slope is dollars per month
    n = len(spend)
    mean_x = (n - 1) / 2
    mean_y = sum(spend) / n
    variance = sum((i - mean_x) ** 2 for i in range(n))
    slope = (sum((i - mean_x) * (y - mean_y) for i, y in enumerate(spend))
             / variance) if variance else 0.0

    over = [m for m, total in zip(months, spend) if total > income.get(m, 0.0)]
    return Signals(
        months=n, lowest=min(spend), highest=max(spend), average=mean_y,
        trend_per_month=slope, overspent_months=len(over),
        overspent_list=[_month_name(m) for m in over],
    )


# Share of a household's spending each category takes, from the same BLS 2024
# lines the demonstration amounts were calibrated against. Held here rather
# than imported from the generator, because a deployed app has no generator.
BLS_SHARES = {
    "Rent": 25.2, "Transportation": 20.8, "Healthcare": 9.7, "Groceries": 9.7,
    "Utilities": 7.4, "Restaurants": 6.2, "Entertainment": 5.6, "Insurance": 4.0,
    "Shopping": 3.1, "Education": 2.4, "Travel": 2.1, "Subscription": 1.9,
    "Personal Care": 1.5, "Fees": 0.4,
}


def benchmark(rows: list[Transaction]) -> list[dict]:
    """Each category's share of spending against the BLS household average.

    A comparison, not a judgement: the survey describes what households do,
    not what they ought to do.
    """
    totals: dict[str, float] = collections.defaultdict(float)
    for row in rows:
        if is_spend(row.category) and row.amount < 0:
            totals[row.category] += -row.amount
    overall = sum(totals.values()) or 1.0
    out = []
    for name, value in totals.items():
        if name not in BLS_SHARES:
            continue
        share = value / overall * 100
        out.append({"category": name, "amount": value, "share": share,
                    "benchmark": BLS_SHARES[name],
                    "gap": share - BLS_SHARES[name]})
    return sorted(out, key=lambda item: -item["amount"])


def layer_counts(rows: list[Transaction]) -> list[dict]:
    """How many of this user's transactions each cascade layer decided."""
    counts = collections.Counter(
        "correction" if row.corrected_category else row.decided_by
        for row in rows)
    order = ["correction", "rule", "memory", "model"]
    total = sum(counts.values()) or 1
    return [{"layer": name, "count": counts.get(name, 0),
             "share": counts.get(name, 0) / total * 100}
            for name in order]
