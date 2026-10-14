#!/usr/bin/env python3
"""Unit 2, Model 1b of 6: a merchant-memory baseline.

Run:
    .venv/bin/python scripts/run_model1b_merchant_memory.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_model import load, print_table, score, strictly_novel  # noqa: E402
from reporting import ACCENT, BAD, GOOD, grouped_bars, write_doc  # noqa: E402
from models.merchant_memory import MerchantMemory  # noqa: E402
from models.rule_baseline import predict as rule_predict  # noqa: E402


def main() -> int:
    results = {}

    # ---- protocol 1: stratified -------------------------------------------
    s_train = load("stratified_train.csv")
    s_validation = load("stratified_validation.csv")
    memory = MerchantMemory()
    memory.learn(s_train)
    print(f"memorised {len(memory):,} merchants from {len(s_train):,} training rows")

    gold_s = [r["category"] for r in s_validation]
    predicted_s = [memory.predict(r["normalised_description"]) for r in s_validation]
    results["stratified (validation)"] = score(gold_s, predicted_s)

    # ---- protocol 2: merchant held out ------------------------------------
    m_train = load("merchant_train.csv")
    m_validation = load("merchant_validation.csv")
    memory_m = MerchantMemory()
    memory_m.learn(m_train)
    gold_m = [r["category"] for r in m_validation]
    predicted_m = [memory_m.predict(r["normalised_description"])
                   for r in m_validation]
    results["merchant held out (validation)"] = score(gold_m, predicted_m)

    # ---- protocol 3: strictly novel ---------------------------------------
    novel = strictly_novel(m_train, m_validation)
    gold_n = [r["category"] for r in novel]
    predicted_n = [memory_m.predict(r["normalised_description"]) for r in novel]
    results["  of which strictly novel"] = score(gold_n, predicted_n)

    print_table("MODEL 1b of 6 - MERCHANT MEMORY (lookup, no learning)", results)

    # ---- how the two deterministic layers complement each other -----------
    rule_fires = [rule_predict(r["normalised_description"])[0] is not None
                  for r in s_validation]
    memory_fires = [p is not None for p in predicted_s]
    both = sum(1 for a, b in zip(rule_fires, memory_fires) if a and b)
    rule_only = sum(1 for a, b in zip(rule_fires, memory_fires) if a and not b)
    memory_only = sum(1 for a, b in zip(rule_fires, memory_fires) if b and not a)
    neither = sum(1 for a, b in zip(rule_fires, memory_fires) if not a and not b)
    total = len(s_validation)

    print("\nhow the two deterministic layers divide the work "
          "(stratified validation)")
    for label, value in (("rule only", rule_only), ("memory only", memory_only),
                         ("both agree to answer", both),
                         ("neither: must go to a model", neither)):
        print(f"  {label:<30}{value:>6,}  ({value/total:5.1%})")

    # accuracy of the combined deterministic cascade, where either fires
    covered = [(g, rp if rp else mp)
               for g, rp, mp in zip(gold_s,
                                    [rule_predict(r["normalised_description"])[0]
                                     for r in s_validation],
                                    predicted_s)
               if rp or mp]
    cascade_precision = (sum(1 for g, p in covered if g == p) / len(covered)
                         if covered else 0.0)
    print(f"  rules + memory together answer {len(covered)/total:.1%} of rows "
          f"at {cascade_precision:.1%} precision")

    # ---- figures ----------------------------------------------------------
    print("\nfigures")
    names = list(results)
    short = ["Stratified", "Merchant held out", "Strictly novel"]
    fig1 = grouped_bars(
        "Merchant memory: perfect recall of known shops, blind to new ones",
        "", short,
        [("Coverage (how often it answers)",
          [results[n]["coverage"] for n in names], ACCENT),
         ("Precision when it answers",
          [results[n]["precision_when_fired"] for n in names], GOOD),
         ("Accuracy over all rows",
          [results[n]["accuracy"] for n in names], BAD)],
        "model1b_coverage_precision.png", percent=True)

    rules = json.loads((ROOT / "results" / "model1_rules.json").read_text())
    fig2 = grouped_bars(
        "Two deterministic layers, opposite strengths",
        "coverage", short,
        [("Rules (merchant-independent)",
          [rules["results"][n]["coverage"] for n in names], GOOD),
         ("Merchant memory (merchant-dependent)",
          [results[n]["coverage"] for n in names], BAD)],
        "model1b_vs_rules.png", percent=True)

    # ---- document ---------------------------------------------------------
    print("\ndocument")
    strat, merch, novel_r = (results[n] for n in names)
    write_doc(
        "model1b_merchant_memory.docx",
        "Model 1b of 6 - Merchant Memory",
        "Unit 2, model comparison. Lookup, not learning: the deliberate opposite "
        "of the rule baseline.",
        [
        ("h1", "1. What this model is"),
        ("p", "A lookup table. During training it records which category each "
              "merchant received; at prediction time it returns that category "
              "again. If the merchant was never seen it abstains. No learning, "
              "no generalisation, no features."),
        ("p", "It is the deliberate opposite of Model 1. The rule baseline "
              "contains **no merchant names** and depends entirely on generic "
              "banking vocabulary. This one contains **nothing but** merchant "
              "names. Between them they bracket the problem, and the gap is the "
              "space the learned models have to occupy."),
        ("h2", "1.1 A deliberate restriction"),
        ("p", "The prepared CSVs already carry a `merchant_key`, but that key was "
              "derived from statistics over the whole corpus, which a deployed "
              "system would not possess when a single new transaction arrives. "
              "Using it would hand this baseline information it could not have in "
              "practice, so the fingerprint is instead computed from one "
              "description in isolation: drop bank noise, processor prefixes, "
              "numbers and reference codes, then keep the leading two tokens."),
        ("p", "Matching is exact. A fuzzy variant tolerating small spelling "
              "differences would raise coverage and is how the deployed cascade "
              "would be built, but exact matching keeps the baseline "
              "interpretable: it answers if and only if it has seen the merchant."),
        ("h1", "2. Results"),
        ("table", [["Protocol", "n", "Accuracy", "Macro-F1", "Coverage",
                    "Precision when it answered"]] +
                  [[label, f"{r['n']:,}", f"{r['accuracy']:.4f}",
                    f"{r['macro_f1']:.4f}", f"{r['coverage']:.1%}",
                    f"{r['precision_when_fired']:.1%}"]
                   for label, r in zip(
                       ["Stratified (validation)",
                        "Merchant held out (validation)",
                        "of which strictly novel"], [strat, merch, novel_r])]),
        ("figure", (fig1, "Figure 1. Coverage, precision and accuracy across the "
                          "three protocols.")),
        ("p", f"The behaviour is exactly as designed and the numbers say so "
              f"plainly. On the stratified protocol the memory answers "
              f"{strat['coverage']:.1%} of transactions at "
              f"{strat['precision_when_fired']:.1%} precision. With merchants held "
              f"out, coverage falls to {merch['coverage']:.1%}, and on strictly "
              f"novel merchants to {novel_r['coverage']:.1%}."),
        ("p", "This is not a failure of the method; it is the definition of it. "
              "A lookup table cannot answer a question it has never been asked. "
              "The value of running it is that it quantifies how much of the "
              "stratified protocol's apparent difficulty is simply merchant "
              "recall."),
        ("h1", "3. The two deterministic layers are complementary"),
        ("figure", (fig2, "Figure 2. Coverage of the rule baseline against the "
                          "merchant memory across the three protocols. The rules "
                          "hold roughly steady; the memory collapses.")),
        ("table", [["Layer", "Stratified coverage", "Merchant held out",
                    "Strictly novel"],
                   ["Rules (merchant-independent)",
                    f"{rules['results'][names[0]]['coverage']:.1%}",
                    f"{rules['results'][names[1]]['coverage']:.1%}",
                    f"{rules['results'][names[2]]['coverage']:.1%}"],
                   ["Merchant memory (merchant-dependent)",
                    f"{strat['coverage']:.1%}", f"{merch['coverage']:.1%}",
                    f"{novel_r['coverage']:.1%}"]]),
        ("p", "The rules barely move between protocols because they describe "
              "banking vocabulary rather than particular shops. The memory "
              "collapses between protocols because it describes nothing else. "
              "Neither is better; they fail in different places, which is the "
              "argument for arranging them in a cascade rather than choosing "
              "between them."),
        ("table", [["On the stratified validation set", "Rows", "Share"],
                   ["A rule answers, memory does not", f"{rule_only:,}",
                    f"{rule_only/total:.1%}"],
                   ["Memory answers, no rule does", f"{memory_only:,}",
                    f"{memory_only/total:.1%}"],
                   ["Both answer", f"{both:,}", f"{both/total:.1%}"],
                   ["Neither answers: must go to a model", f"{neither:,}",
                    f"{neither/total:.1%}"]]),
        ("p", f"Taken together the two deterministic layers answer "
              f"{len(covered)/total:.1%} of transactions at "
              f"{cascade_precision:.1%} precision, leaving {neither/total:.1%} "
              "that carry neither recognisable vocabulary nor a known merchant. "
              "That remainder is the genuine machine-learning problem, and it is "
              "what the comparison between Models 2 to 5 is really about."),
        ("h1", "4. Limitations"),
        ("bullet", "It cannot generalise at all. Coverage on unfamiliar merchants "
                   "is near zero by construction, and no tuning changes that."),
        ("bullet", "Memory grows with every merchant ever seen, and stale entries "
                   "persist after a merchant changes its trading name or line of "
                   "business."),
        ("bullet", "Exact matching misses trivial spelling variants that a fuzzy "
                   "matcher would catch, so the coverage reported here is a lower "
                   "bound for what a deployed memory layer would achieve."),
        ("h1", "5. For the report"),
        ("p", f"A merchant-memory lookup answers {strat['coverage']:.1%} of "
              f"transactions at {strat['precision_when_fired']:.1%} precision when "
              f"merchants recur between training and evaluation, and "
              f"{novel_r['coverage']:.1%} when they do not. Contrasting it with "
              "the rule baseline, whose coverage is almost unchanged across the "
              "same protocols, demonstrates empirically why the system combines "
              "deterministic layers with a learned model rather than relying on "
              "any one of them."),
        ])

    out = ROOT / "results" / "model1b_merchant_memory.json"
    out.write_text(json.dumps({
        "model": "Merchant memory (lookup)", "machine_learning": False,
        "merchants_memorised": len(memory),
        "results": results,
        "cascade_on_stratified": {
            "rule_only": rule_only, "memory_only": memory_only, "both": both,
            "neither": neither, "total": total,
            "combined_coverage": len(covered) / total,
            "combined_precision": cascade_precision,
        },
    }, indent=2), encoding="utf-8")
    print(f"\n[PASS] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
