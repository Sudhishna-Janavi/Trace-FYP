"""Model 1 of 5: a hand-written rule baseline. No machine learning.

This is the control group. It exists so the project can state what plain
keyword matching achieves, and therefore what the machine-learned models have
to beat before they are worth their complexity.

Design rule: the patterns use **generic banking vocabulary only** - MORTGAGE,
RENT, PAYROLL, OVERDRAFT FEE - and never merchant brand names. A rule saying
"NETFLIX means Subscription" would score well on merchants already seen and
teach us nothing about new ones; worse, it would quietly become a lookup table
rather than a baseline. Generic vocabulary is what a real deployed rule layer
uses, and it transfers to merchants the system has never encountered.

A rule abstains when nothing matches. Abstention is the point: a rule layer is
meant to be highly precise on a narrow slice and hand everything else to a
model, so coverage and precision-when-fired matter more than raw accuracy.
"""
from __future__ import annotations

import re

# (category, pattern, required direction or None)
# Order matters: the first match wins, so specific patterns precede general ones.
RULES: tuple[tuple[str, str, str | None], ...] = (
    # --- money in. Direction is required: "TRANSFER FROM SAVINGS" as a credit
    #     is a transfer, not income, so the word alone is not enough.
    ("Income", r"\b(PAYROLL|DIR DEP|DIRECT DEP|SALARY|PENSION|TREAS \d*|"
               r"EARNINGS|PAYOUT|STIPEND|ANNUITY)\b", "CREDIT"),
    # Money arriving from a named person is income, not a movement between the
    # user's own accounts. Diagnosed on train: the Transfer rule was claiming
    # 343 of these because it matched the word ZELLE regardless of direction.
    ("Income", r"\b(ZELLE|VENMO|CASH APP)\b.*\bFROM\b", "CREDIT"),

    # --- money between the user's own accounts
    ("Transfer", r"\b(ZELLE|VENMO|WIRE TRANSFER|ONLINE TRANSFER|INTERNET TRANSFER|"
                 r"DDA TO DDA|ACH TRANSFER|MONEYLINK|TRANSFER (TO|FROM)|"
                 r"AUTOMATIC PAYMENT|CARD PAYMENT|ATM DEPOSIT)\b", None),

    # --- housing, split the way the taxonomy splits it
    ("Mortgage", r"\b(MORTGAGE|MTG PMT|PRINCIPAL PMT|ESCROW)\b", None),
    ("Rent", r"\b(RENT PAYMENT|RENT PMT|LEASE PAYMENT|APARTMENTS|PROPERTY RENT|"
             r"RESIDENTIAL RENT)\b", None),

    # --- recurring household obligations
    # bare PREMIUM removed: on train it matched TELEGRAM PREMIUM, ZOHO PREMIUM
    # and X CORP PREMIUM, which are subscriptions, 76 times
    ("Insurance", r"\b(INSURANCE|INS PREMIUM|INS BILLING|POLICY PMT)\b", None),
    ("Utilities", r"\b(ELECTRIC|WATER UTIL|MUNICIPAL WATER|UTILITY|UTIL PMT|"
                  r"GAS COMPANY|REFUSE|TRASH|SEWER|POWER COMPANY)\b", None),

    # --- bank's own charges
    ("Fees", r"\b(OVERDRAFT|SERVICE FEE|SERVICE CHARGE|ATM FEE|ATM SURCHARGE|"
             r"LATE FEE|MAINTENANCE FEE|PAPER STATEMENT|RETURNED ITEM|"
             r"FOREIGN TRANSACTION FEE|CARD REPLACEMENT FEE)\b", None),

    # --- Subscription is deliberately left to the model.
    # The word SUBSCRIPTION does not determine the category: on train it appears
    # in TICKETMASTER SUBSCRIPTION (Entertainment), SUBSCRIPTION PAYMENT DELL
    # (Shopping) and SUBSCRIPTION PAYMENT BRILLIANT ORG (Education). Deciding it
    # requires knowing the merchant, which is exactly what a rule layer cannot
    # do and a learned model can. Rules should abstain rather than guess.

    # --- a few other unambiguous generic phrasings
    ("Healthcare", r"\b(PHARMACY|DENTAL|MEDICAL|CLINIC|HEALTHCARE|TELEHEALTH|"
                   r"HOSPITAL|OPTICAL|VISION CENTER)\b", None),
    ("Education", r"\b(TUITION|UNIVERSITY PMT|COLLEGE PMT|SCHOOL FEES|"
                  r"STUDENT LOAN|COURSE FEE)\b", None),
    # PARKING is guarded: AIRPORT PARKING is labelled Travel, not Transportation
    ("Transportation", r"\b(FUEL|GAS STATION|TOLL|TRANSIT|DMV|CAR WASH|"
                       r"CHARGING STATION)\b|(?<!AIRPORT )\bPARKING\b", None),
    ("Groceries", r"\b(GROCERY|SUPERMARKET|FARMERS MARKET)\b", None),
    ("Restaurants", r"\b(RESTAURANT|PIZZA|COFFEE|CAFE|GRILL|DINER|BAKERY|"
                    r"STEAKHOUSE|SUSHI)\b", None),
)

COMPILED = tuple((category, re.compile(pattern), direction)
                 for category, pattern, direction in RULES)


def predict(normalised: str) -> tuple[str | None, str | None]:
    """Return (category, matched_evidence), or (None, None) when no rule fires.

    `normalised` is the shared normalised_description, which already begins with
    DEBIT or CREDIT, so direction is read from the text itself.
    """
    is_credit = normalised.startswith("CREDIT")
    for category, pattern, required in COMPILED:
        if required == "CREDIT" and not is_credit:
            continue
        if required == "DEBIT" and is_credit:
            continue
        found = pattern.search(normalised)
        if found:
            return category, found.group(0)
    return None, None
