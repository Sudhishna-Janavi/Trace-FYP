#!/usr/bin/env python3
"""Unit 2, Model 1 of 5: evaluate the hand-written rule baseline.

Run:
    python scripts/run_model1_rules.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_model import load, print_table, score, strictly_novel  # noqa: E402
from reporting import ACCENT, BAD, GOOD, grouped_bars, horizontal_bars, write_doc  # noqa: E402
from models.rule_baseline import RULES, predict  # noqa: E402


def main() -> int:
    results = {}

    # ---- protocol 1: stratified validation ----
    rows = load("stratified_validation.csv")
    gold = [r["category"] for r in rows]
    predicted = [predict(r["normalised_description"])[0] for r in rows]
    results["stratified (validation)"] = score(gold, predicted)

    # ---- protocol 2: merchant-held-out validation ----
    merchant_train = load("merchant_train.csv")
    merchant_validation = load("merchant_validation.csv")
    gold = [r["category"] for r in merchant_validation]
    predicted = [predict(r["normalised_description"])[0] for r in merchant_validation]
    results["merchant held out (validation)"] = score(gold, predicted)

    # ---- protocol 3: the strictly novel subset of protocol 2 ----
    novel = strictly_novel(merchant_train, merchant_validation)
    gold = [r["category"] for r in novel]
    predicted = [predict(r["normalised_description"])[0] for r in novel]
    results["  of which strictly novel"] = score(gold, predicted)

    print_table("MODEL 1 of 5 - RULE BASELINE (no machine learning)", results)

    # ---- which rules actually fire, and how well ----
    rows = load("stratified_validation.csv")
    per_rule: dict[str, list[bool]] = {}
    for row in rows:
        category, _evidence = predict(row["normalised_description"])
        if category is None:
            continue
        per_rule.setdefault(category, []).append(category == row["category"])

    print("\nper-rule behaviour on stratified validation")
    print(f"  {'rule fires as':<18}{'times':>8}{'correct':>10}")
    for category, hits in sorted(per_rule.items(), key=lambda kv: -len(kv[1])):
        print(f"  {category:<18}{len(hits):>8,}{sum(hits)/len(hits):>10.1%}")
    silent = [c for c, _p, _d in RULES if c not in per_rule]
    if silent:
        print(f"\n  rules that never fired: {silent}")

    # ---------------------------------------------------------- figures ----
    print("\nfigures")
    names = list(results)
    short = ["Stratified", "Merchant held out", "Strictly novel"]
    fig1 = grouped_bars(
        "Rule baseline: it answers rarely, but is almost never wrong",
        "", short,
        [("Coverage (how often it answers)", [results[n]["coverage"] for n in names], ACCENT),
         ("Precision when it answers", [results[n]["precision_when_fired"] for n in names], GOOD),
         ("Accuracy over all rows", [results[n]["accuracy"] for n in names], BAD)],
        "model1_coverage_precision.png", percent=True)

    ordered = sorted(per_rule.items(), key=lambda kv: -len(kv[1]))
    fig2 = horizontal_bars(
        "Rule baseline: precision of each rule (stratified validation)",
        "precision when the rule fires",
        [c for c, _v in ordered],
        [sum(v) / len(v) for _c, v in ordered],
        "model1_per_rule_precision.png",
        counts=[len(v) for _c, v in ordered], percent=True, threshold=0.95)

    # --------------------------------------------------------- findings ----
    print("\ndocument")
    strat, merch, novel_r = (results[n] for n in names)
    write_doc(
        "model1_rule_baseline.docx",
        "Model 1 of 5 - Rule Baseline",
        "Unit 2, model comparison. No machine learning: hand-written patterns only.",
        [
        ("h1", "1. What this model is"),
        ("p", "A list of hand-written IF-THEN patterns, checked in order, first "
              "match wins. If no pattern matches, the model abstains rather than "
              "guessing. It performs no learning of any kind, so it is the "
              "control group: it establishes what plain keyword matching "
              "achieves, and therefore what a learned model must beat to justify "
              "its complexity."),
        ("p", "**The patterns use generic banking vocabulary only** - MORTGAGE, "
              "RENT, PAYROLL, OVERDRAFT FEE - and contain no merchant names. A "
              "rule asserting that NETFLIX means Subscription would score well on "
              "merchants already seen while revealing nothing about new ones, and "
              "would turn the baseline into a lookup table built from the answers."),
        ("h2", "1.1 Worked example"),
        ("code",
         "Input:  DEBIT FLAGSTAR BANK MORTGAGE <NUM> PPD ID <NUM>\n"
         "  Income?    needs CREDIT, this is DEBIT   -> skip\n"
         "  Transfer?  ZELLE|WIRE|VENMO ...          -> no match\n"
         "  Mortgage?  MORTGAGE|MTG PMT ...          -> MATCH on 'MORTGAGE'\n"
         "Answer: Mortgage                            (correct)\n\n"
         "Input:  DEBIT TST GOLDEN DRAGON OAKLAND\n"
         "  ... all twelve rules ...                 -> no match\n"
         "Answer: abstains                            (true answer: Restaurants)"),
        ("p", "The second case is the normal case. A human reads 'Golden Dragon' "
              "and infers a restaurant; the rules cannot, because the description "
              "contains a shop name and no banking vocabulary."),
        ("h1", "2. Results"),
        ("table", [["Protocol", "n", "Accuracy", "Macro-F1", "Coverage",
                    "Precision when it answered"]] +
                  [[label, f"{r['n']:,}", f"{r['accuracy']:.4f}",
                    f"{r['macro_f1']:.4f}", f"{r['coverage']:.1%}",
                    f"{r['precision_when_fired']:.1%}"]
                   for label, r in zip(
                       ["Stratified (validation)",
                        "Merchant held out (validation)",
                        "of which strictly novel"],
                       [strat, merch, novel_r])]),
        ("figure", (fig1, "Figure 1. Coverage, precision and accuracy for the "
                          "rule baseline across the three evaluation protocols.")),
        ("h2", "2.1 Reading the accuracy column"),
        ("p", f"Accuracy of {strat['accuracy']:.4f} looks like failure and is not. "
              "The rules abstain on more than eighty percent of transactions, and "
              "every abstention is counted as a miss. Accuracy here is simply "
              "coverage multiplied by precision: "
              f"{strat['coverage']:.1%} of rows answered, "
              f"{strat['precision_when_fired']:.1%} of those correct, giving "
              f"{strat['accuracy']:.4f}. The meaningful quantity for a layer that "
              "is allowed to abstain is precision on the rows it commits to."),
        ("h2", "2.2 The finding that matters"),
        ("p", "**Precision does not fall as the task becomes harder.** It is "
              f"{strat['precision_when_fired']:.1%} on merchants the system has "
              f"seen, {merch['precision_when_fired']:.1%} when merchants are held "
              f"out entirely, and {novel_r['precision_when_fired']:.1%} on the "
              "strictly novel subset where no merchant word occurs anywhere in "
              "training. Because the patterns describe banking vocabulary rather "
              "than particular shops, they are indifferent to whether the merchant "
              "is familiar."),
        ("p", "This is the argument for retaining a deterministic rule layer in "
              "the deployed cascade. The learned models are expected to be "
              "weakest precisely where this layer is strongest, so the two are "
              "complementary rather than competing."),
        ("figure", (fig2, "Figure 2. Precision of each individual rule on the "
                          "stratified validation set, with the number of times "
                          "each fired. The dashed line marks 95%.")),
        ("h1", "3. Faults found and corrected"),
        ("p", "Four faults were diagnosed on the **training** split rather than "
              "validation, so the set used to compare models remains untouched."),
        ("table", [["Fault", "Evidence on train", "Correction"],
                   ["Transfer claimed incoming Zelle",
                    "CREDIT ZELLE PAYMENT FROM MARGARET ROBINSON is Income, 343 times",
                    "money arriving from a named person is treated as income"],
                   ["Insurance matched software tiers",
                    "TELEGRAM PREMIUM, ZOHO PREMIUM, X CORP PREMIUM, 76 times",
                    "the bare word PREMIUM was removed"],
                   ["Transportation claimed airport parking",
                    "AIRPORT PARKING INC is Travel, 30 times",
                    "PARKING guarded against a preceding AIRPORT"],
                   ["Subscription was unresolvable",
                    "the word appears in TICKETMASTER SUBSCRIPTION (Entertainment), "
                    "SUBSCRIPTION PAYMENT DELL (Shopping) and SUBSCRIPTION PAYMENT "
                    "BRILLIANT ORG (Education)",
                    "the rule was removed entirely"]]),
        ("p", "The fourth is a finding rather than a repair. Deciding Subscription "
              "requires knowing the merchant, which a rule layer cannot do and a "
              "learned model can. Removing the rule lowered coverage from 19.6% to "
              f"{strat['coverage']:.1%} while raising precision from 85.1% to "
              f"{strat['precision_when_fired']:.1%}: the layer answers less often "
              "and is substantially more trustworthy when it does."),
        ("h1", "4. Limitations"),
        ("bullet", "Coverage is low by construction. More than eighty percent of "
                   "transactions carry no banking vocabulary and are left entirely "
                   "to the models that follow."),
        ("bullet", "The rules were written by hand against this taxonomy. They do "
                   "not transfer to a different label set without being rewritten, "
                   "which is a maintenance cost a learned model does not have."),
        ("bullet", "Categories dominated by merchant identity rather than "
                   "vocabulary - Shopping, Entertainment, Personal Care, Travel - "
                   "receive no rule at all and are invisible to this layer."),
        ("h1", "5. For the report"),
        ("p", f"The rule baseline answers {strat['coverage']:.1%} of transactions "
              f"at {strat['precision_when_fired']:.1%} precision, and its precision "
              "is unaffected by whether the merchant has been seen before, "
              f"reaching {novel_r['precision_when_fired']:.1%} on strictly novel "
              "merchants. The remaining four fifths of transactions carry no "
              "deterministic signal and define the problem the machine-learned "
              "models exist to solve."),
        ])

    out = ROOT / "results" / "model1_rules.json"
    out.write_text(json.dumps({
        "model": "rule baseline",
        "machine_learning": False,
        "results": results,
        "per_rule_fires": {k: {"fired": len(v), "precision": sum(v) / len(v)}
                           for k, v in per_rule.items()},
    }, indent=2), encoding="utf-8")
    print(f"\n[PASS] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
