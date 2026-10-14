#!/usr/bin/env python3
"""Unit 1: turn the raw benchmark into leakage-safe train / validation / test sets.

This script touches no models. It only prepares data, so the data can be
inspected and trusted before anything learns from it.

Pipeline
    1. read the raw CSV
    2. normalise each description with one shared rule set
    3. deduplicate on (normalised text, category)
    4. build Split A - stratified random 70/15/15
    5. build Split B - merchant-grouped 70/15/15
    6. write the six CSVs plus a verification report

Why two splits
    Split A measures performance on a merchant the model has already seen,
    which is the common case for a returning user. Split B holds entire
    merchants out, which is what happens when a new user arrives with shops the
    system has never encountered. Reporting only A overstates the system;
    reporting only B understates everyday use. Both are therefore produced.

Usage
    python scripts/prepare_dataset.py \
        --input dataset/US/transactions-synthetic.csv \
        --outdir dataset/processed
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
from pathlib import Path

# ---------------------------------------------------------------- vocabulary

# Bank plumbing: words that appear across every merchant and name none of them.
NOISE = {
    "DEBIT", "CREDIT", "POS", "DBT", "ECOM", "GIRO", "REF", "USA", "US", "EFT",
    "PYMT", "PPD", "ID", "PURCHASE", "PAYMENT", "CARD", "ACH", "WEB", "RECUR",
    "XFER", "DES", "INDN", "CO",
}

# Payment processors and statement-format markers. "PAYPAL" in "PAYPAL*SAFEWAY"
# describes how the payment was routed, not who was paid.
PROCESSOR = {
    "PAYPAL", "PP", "PYPL", "TST", "CLV", "SQ", "INST", "PREAPPROVED", "EXPRESS",
    "CHECKOUT", "WITHDRAWAL", "FROM", "PREAUTHORIZED", "DEPOSIT", "BILL", "USER",
    "MKTPL", "TO",
}

# Categories that are money coming in, money moving between the user's own
# accounts, or money going out. Used later by the analytics layer; recorded here
# so the taxonomy lives in one place.
SPENDING_EFFECT = {
    "Income": "income",
    "Transfer": "transfer",
}  # everything else is an expense

# ------------------------------------------------------------- normalisation


def normalise(text: str) -> str:
    """The single shared cleaning rule applied to every description.

    Every model receives exactly this output, so the comparison measures models
    rather than preprocessing choices.

        [debit] PUBLIX 7860 UNIVERSITY LN OAKLAND 94601-4574CA USA
        DEBIT PUBLIX <NUM> UNIVERSITY LN OAKLAND <NUM> 4574CA USA
    """
    text = text.upper()
    # the direction marker is kept as an ordinary leading token: it is the
    # dataset's primary signal separating Income from Transfer, and a real
    # system always knows the direction from the sign of the amount
    text = re.sub(r"^\[(DEBIT|CREDIT)\]\s*", lambda m: m.group(1) + " ", text)
    # long digit runs are store, order and reference numbers. Collapsing them
    # stops the model memorising "CHEVRON 4580" instead of learning "CHEVRON".
    # Short numbers survive, so "365 MARKET" keeps its 365.
    text = re.sub(r"\b\d{4,}\b", "<NUM>", text)
    text = re.sub(r"[^A-Z0-9<> ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def direction_of(raw: str) -> str:
    return "credit" if raw.lower().startswith("[credit]") else "debit"


# --------------------------------------------------------- merchant grouping


def candidate_tokens(normalised: str) -> list[str]:
    """Tokens that could plausibly name a merchant."""
    out = []
    for word in normalised.split():
        if word in NOISE or word in PROCESSOR or word == "<NUM>" or word.isdigit():
            continue
        if len(word) <= 2:
            continue
        # opaque order and reference codes mix letters with digits
        # (AER4TZS75Q, DTBLH00A1U). A long run of pure letters is a real brand
        # name (STARBUCKS, WALGREENS) and must be kept.
        if len(word) >= 6 and re.search(r"\d", word) and re.fullmatch(r"[A-Z0-9]+", word):
            continue
        out.append(word)
    return out


def merchant_keys(rows: list[dict]) -> list[str]:
    """Assign each row the merchant it actually names.

    Taking the first candidate word is not enough. One brand produces many first
    words across statement formats (AMAZON, AMZN, AMAZON MKTPL), and a
    description often leads with a generic word belonging to hundreds of
    merchants (STARBUCKS <NUM> MARKET CT -> MARKET).

    A token's *spread* is how many distinct leading words it co-occurs with.
    Generic words spread widely (MARKET across ~312); merchant names barely at
    all (STARBUCKS across ~2). Each row is therefore keyed by its
    narrowest-spread token, which is the word that most specifically identifies
    the merchant. Ties keep the earliest token, since statement formats lead
    with the merchant.
    """
    leading: list[str] = []
    for row in rows:
        tokens = candidate_tokens(row["normalised_description"])
        leading.append(tokens[0] if tokens else "UNKNOWN")

    spread: dict[str, set[str]] = collections.defaultdict(set)
    for row, lead in zip(rows, leading, strict=True):
        for word in candidate_tokens(row["normalised_description"]):
            spread[word].add(lead)

    keys = []
    for row in rows:
        tokens = candidate_tokens(row["normalised_description"])
        if not tokens:
            keys.append("UNKNOWN")
            continue
        keys.append(min(tokens, key=lambda w: (len(spread[w]), tokens.index(w))))
    return keys


# -------------------------------------------------------------------- splits


def stratified_split(rows: list[dict], rng: random.Random):
    """Random 70/15/15 that preserves each category's proportion."""
    by_category: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    train, validation, test = [], [], []
    for items in by_category.values():
        rng.shuffle(items)
        n = len(items)
        a, b = int(n * 0.70), int(n * 0.85)
        train += items[:a]
        validation += items[a:b]
        test += items[b:]
    for part in (train, validation, test):
        rng.shuffle(part)
    return train, validation, test


def merchant_split(rows: list[dict], rng: random.Random):
    """70/15/15 where every row of a merchant lands in the same partition.

    Packing merchants by size alone is not enough. Categories differ in how
    their merchants are shaped: Rent is a handful of large landlords, while
    Restaurants is thousands of small independents. Size-only packing sent every
    Rent merchant to training, leaving that category with zero validation and
    zero test rows, which then scored F1 = 0 and silently corrupted macro-F1.

    Merchants are therefore packed **within each category**. Each merchant is
    assigned to its dominant category, and that category's merchants are
    distributed largest-first toward a 70/15/15 split of that category's rows.
    Merchant disjointness is preserved because a merchant still goes wholly to
    one partition; what changes is that every category now appears in all three.
    """
    members: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        members[row["merchant_key"]].append(row)

    # a merchant is assigned to the category most of its rows carry
    by_category: dict[str, list[tuple[str, list[dict]]]] = collections.defaultdict(list)
    for key, items in members.items():
        dominant = collections.Counter(r["category"] for r in items).most_common(1)[0][0]
        by_category[dominant].append((key, items))

    targets = {"train": 0.70, "validation": 0.15, "test": 0.15}
    parts: dict[str, list[dict]] = collections.defaultdict(list)

    for _category, merchants in by_category.items():
        total = sum(len(items) for _key, items in merchants)
        filled = {name: 0 for name in targets}
        for _key, items in sorted(merchants, key=lambda kv: -len(kv[1])):
            name = min(targets, key=lambda n: filled[n] - targets[n] * total)
            parts[name] += items
            filled[name] += len(items)

    for name in targets:
        rng.shuffle(parts[name])
    return parts["train"], parts["validation"], parts["test"]


# ---------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1131)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---- 1. read ----------------------------------------------------------
    with args.input.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    print(f"[1/6] read {len(raw_rows):,} rows from {args.input}")

    # ---- 2. normalise -----------------------------------------------------
    prepared = []
    for index, row in enumerate(raw_rows):
        normalised = normalise(row["description"])
        if not normalised:
            continue
        prepared.append({
            "transaction_id": f"tx-{index:06d}",
            "raw_description": row["description"],
            "normalised_description": normalised,
            "merchant_key": "",          # filled in below
            "category": row["category"],
            "direction": direction_of(row["description"]),
        })
    print(f"[2/6] normalised {len(prepared):,} rows")

    # ---- 3. deduplicate ---------------------------------------------------
    # Done before splitting. Afterwards would be useless: the duplicates would
    # already have crossed the train/test boundary.
    seen: set[tuple[str, str]] = set()
    rows: list[dict] = []
    for row in prepared:
        key = (row["normalised_description"], row["category"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    removed = len(prepared) - len(rows)
    print(f"[3/6] deduplicated {len(prepared):,} -> {len(rows):,} "
          f"({removed:,} duplicate (text, category) pairs removed)")

    # a description carrying two different categories is genuine ambiguity and
    # is kept, so the model is free to be uncertain there
    texts = collections.Counter(r["normalised_description"] for r in rows)
    ambiguous = sum(1 for _t, n in texts.items() if n > 1)

    # ---- 4. merchant keys -------------------------------------------------
    for row, key in zip(rows, merchant_keys(rows), strict=True):
        row["merchant_key"] = key
    groups = collections.Counter(r["merchant_key"] for r in rows)
    print(f"[4/6] identified {len(groups):,} merchants "
          f"(largest: {groups.most_common(1)[0][0]} with "
          f"{groups.most_common(1)[0][1]:,} rows)")

    rng = random.Random(args.seed)
    strat = stratified_split(rows, rng)
    merch = merchant_split(rows, rng)
    print(f"[5/6] built both splits")

    # ---- 5. write ---------------------------------------------------------
    fields = ["transaction_id", "raw_description", "normalised_description",
              "merchant_key", "category", "direction"]

    def write(name: str, data: list[dict]) -> int:
        with (args.outdir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
        return len(data)

    counts = {
        "stratified": {part: write(f"stratified_{part}.csv", data)
                       for part, data in zip(("train", "validation", "test"), strat)},
        "merchant": {part: write(f"merchant_{part}.csv", data)
                     for part, data in zip(("train", "validation", "test"), merch)},
    }

    # ---- 6. verify --------------------------------------------------------
    s_train, _s_val, s_test = strat
    m_train, _m_val, m_test = merch

    strat_text_overlap = len(
        {r["normalised_description"] for r in s_train}
        & {r["normalised_description"] for r in s_test})

    merchants_train = {r["merchant_key"] for r in m_train} - {"UNKNOWN"}
    merchants_test = {r["merchant_key"] for r in m_test} - {"UNKNOWN"}
    merchant_overlap = merchants_train & merchants_test

    # independent spot check: well-known brands must sit on one side only
    probes = ("STARBUCKS", "CHEVRON", "SAFEWAY", "AMAZON", "NETFLIX",
              "WALGREENS", "TARGET", "UBER", "LYFT", "COSTCO")
    spanning = []
    for probe in probes:
        in_train = any(probe in r["normalised_description"] for r in m_train)
        in_test = any(probe in r["normalised_description"] for r in m_test)
        if in_train and in_test:
            spanning.append(probe)

    assert not merchant_overlap, f"merchant leakage: {sorted(merchant_overlap)[:10]}"

    # every category must survive into every partition, or macro-F1 silently
    # counts an unscoreable category as a zero
    coverage = {}
    starved = []
    for category in sorted({r["category"] for r in rows}):
        counts_by_part = [sum(1 for r in part if r["category"] == category)
                          for part in (m_train, _m_val, m_test)]
        coverage[category] = counts_by_part
        if min(counts_by_part[1:]) < 30:
            starved.append(category)
    assert not starved, f"categories starved in the merchant split: {starved}"

    meta = {
        "source_file": str(args.input),
        "seed": args.seed,
        "raw_rows": len(raw_rows),
        "rows_after_deduplication": len(rows),
        "duplicates_removed": removed,
        "texts_with_more_than_one_category": ambiguous,
        "categories": len({r["category"] for r in rows}),
        "class_distribution": dict(
            collections.Counter(r["category"] for r in rows).most_common()),
        "direction_distribution": dict(
            collections.Counter(r["direction"] for r in rows)),
        "merchants": len(groups),
        "largest_merchant_rows": groups.most_common(1)[0][1],
        "splits": counts,
        "verification": {
            "stratified_exact_text_shared_train_test": strat_text_overlap,
            "merchant_groups_shared_train_test": len(merchant_overlap),
            "probe_brands_spanning_train_and_test": spanning,
            "merchant_split_rows_per_category": coverage,
        },
        "normalisation_rules": [
            "uppercase",
            "[debit]/[credit] becomes a leading DEBIT/CREDIT token",
            "runs of four or more digits become <NUM>",
            "punctuation becomes a space",
            "repeated whitespace collapses",
        ],
        "notes": [
            "normalised_description is the only model input; category is the label.",
            "merchant_key exists to build the merchant split and is never a feature.",
            "Deduplication happens before splitting, otherwise duplicates would "
            "already span the train/test boundary.",
        ],
    }
    (args.outdir / "dataset_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    # ---- report -----------------------------------------------------------
    print(f"[6/6] wrote 6 CSVs + dataset_meta.json to {args.outdir}\n")
    print("=" * 66)
    print("VERIFICATION REPORT")
    print("=" * 66)
    print(f"  raw rows                     {len(raw_rows):>8,}")
    print(f"  after deduplication          {len(rows):>8,}   ({removed:,} removed)")
    print(f"  categories                   {meta['categories']:>8}")
    print(f"  merchants identified         {len(groups):>8,}")
    print(f"  texts with 2+ categories     {ambiguous:>8}   (kept: genuine ambiguity)")
    print()
    print("  Split A - stratified random (merchant may appear on both sides)")
    for part, n in counts["stratified"].items():
        print(f"      {part:<12}{n:>8,}")
    print("  Split B - merchant-grouped (a merchant lands on one side only)")
    for part, n in counts["merchant"].items():
        print(f"      {part:<12}{n:>8,}")
    print()
    print("  leakage checks")
    print(f"      identical text in both stratified train and test   "
          f"{strat_text_overlap:>5}   (expected: 0)")
    print(f"      merchants in both merchant train and test          "
          f"{len(merchant_overlap):>5}   (expected: 0)")
    print(f"      probe brands spanning train and test               "
          f"{len(spanning):>5}   {spanning if spanning else ''}")
    smallest = min(coverage.items(), key=lambda kv: min(kv[1][1:]))
    print(f"      smallest category in merchant valid/test           "
          f"{min(smallest[1][1:]):>5}   ({smallest[0]})")
    print()
    print("  class balance after deduplication")
    for name, n in meta["class_distribution"].items():
        print(f"      {name:<18}{n:>7,}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
