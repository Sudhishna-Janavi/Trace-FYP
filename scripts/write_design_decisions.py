#!/usr/bin/env python3
"""Write the running record of design decisions and their justifications.

This is the document to draw on when writing the report: every choice made in
this project, why it was made, what evidence supported it, and what was given
up. Regenerate it after any decision changes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reporting import write_doc  # noqa: E402


def load_demo():
    path = ROOT / "dataset" / "demo" / "amount_model.json"
    return json.loads(path.read_text()) if path.exists() else None


def load(name):
    path = ROOT / "results" / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:
    cmp_ = load("unit2_comparison")
    casc = load("unit2_cascade")
    final = load("unit2_final_test")
    demo = load_demo()
    meta = json.loads(
        (ROOT / "dataset" / "processed" / "dataset_meta.json").read_text())

    blocks = [
        ("p", "A running record of every design decision taken in this project, "
              "the evidence behind it, and what was traded away. Each entry is "
              "written to be lifted directly into the report."),

        ("h1", "1. Data"),
        ("h2", "1.1 Which dataset"),
        ("table", [["Decision", "A public third-party synthetic benchmark of US "
                                "bank-statement formats (68,000 rows, MIT licence)"],
                   ["Alternatives", "Generate the data ourselves; use a smaller "
                                    "Indian narration dataset; request a real "
                                    "corpus from published authors"],
                   ["Why", "No public dataset of real labelled bank-transaction "
                           "descriptions exists, because a transaction narrative "
                           "is personal data. Published work either uses "
                           "proprietary in-house data (BTG Pactual, 2023), "
                           "develops weak supervision to avoid needing labels "
                           "(Toran et al., 2023), or releases a corpus only on "
                           "request (Garcia-Mendez et al., 2024)."],
                   ["Trade-off", "The data is synthetic and must be described as "
                                 "such. What makes it defensible is that it is "
                                 "third-party, publicly downloadable, and "
                                 "reproducible because its generator is open "
                                 "source."]]),

        ("h2", "1.2 Seventeen categories rather than eleven"),
        ("table", [["Decision", "Keep the benchmark's 17 categories"],
                   ["Alternatives", "Map down to the 11 used in the earlier report"],
                   ["Why", "No mapping to justify; finer categories are more "
                           "useful to the application, since Subscription "
                           "separate from Entertainment improves recurring-payment "
                           "detection and Rent separate from Mortgage improves the "
                           "bills breakdown. Classification granularity and "
                           "display granularity are independent, so categories "
                           "can still be grouped in the interface."],
                   ["Trade-off", "Deduplication left the classes uneven "
                                 f"(Restaurants {meta['class_distribution']['Restaurants']:,} "
                                 f"rows against Rent {meta['class_distribution']['Rent']:,}), "
                                 "so macro-F1 rather than accuracy must be the "
                                 "headline metric, and class weights are used "
                                 "when fitting."]]),

        ("h2", "1.3 Deduplication before splitting"),
        ("table", [["Decision",
                    f"Deduplicate on (normalised text, category): "
                    f"{meta['raw_rows']:,} rows reduce to "
                    f"{meta['rows_after_deduplication']:,}"],
                   ["Why", "The benchmark is template-generated and repeats "
                           "itself heavily: DEBIT MORTGAGE PAYMENT appears 607 "
                           "times. Without deduplication a large share of test "
                           "text appears verbatim in training, and the headline "
                           "figure measures memorisation."],
                   ["Order", "Deduplication happens before splitting. Afterwards "
                             "would be pointless, because duplicates would "
                             "already have crossed the train/test boundary."],
                   ["Kept deliberately",
                    f"{meta['texts_with_more_than_one_category']} descriptions "
                    "carry two different categories. These are genuine "
                    "ambiguity and are retained so a model may be uncertain."]]),

        ("h2", "1.4 Keeping the direction marker in the text"),
        ("table", [["Decision", "[debit] and [credit] become a leading DEBIT or "
                                "CREDIT token rather than being stripped"],
                   ["Why", "The dataset's documentation names the sign as the "
                           "primary signal separating Income from Transfer: "
                           "a Zelle payment received is income, the same payment "
                           "sent is a transfer. Using it is realistic rather "
                           "than generous, because a real system always knows "
                           "the direction from the sign of the amount."],
                   ["Evidence", "The rule baseline's Transfer rule initially "
                                "claimed 343 incoming Zelle payments that were "
                                "Income; adding direction fixed it."]]),

        ("h2", "1.5 One shared normalisation for every model"),
        ("table", [["Decision", "Uppercase; direction marker as a token; runs of "
                                "four or more digits become <NUM>; punctuation "
                                "to space"],
                   ["Why", "Every model receives identical text, so the "
                           "comparison measures models rather than preprocessing "
                           "choices."],
                   ["Tested, not assumed", "The normalisation suits TF-IDF but "
                                           "is unlike the prose MiniLM was "
                                           "trained on, so MiniLM was also run "
                                           "on lightly cleaned text keeping "
                                           "natural case and punctuation. It was "
                                           "marginally better on the easy "
                                           "protocol and worse on both harder "
                                           "ones, so the shared representation "
                                           "is not handicapping it."]]),

        ("h1", "2. Evaluation protocol"),
        ("h2", "2.1 Three protocols rather than one split"),
        ("table", [["In-distribution", "Deduplicated stratified 70/15/15. A "
                                       "merchant may appear on both sides. "
                                       "Models a returning user."],
                   ["Unseen merchant", "Every row of a merchant lands in one "
                                       "partition only. Models a new user."],
                   ["Strictly novel", "The subset of the unseen-merchant test "
                                      "rows containing no merchant word that "
                                      "occurs anywhere in training. The genuine "
                                      "cold-start case."],
                   ["Why", "Quoting only the first overstates the system; "
                           "quoting only the third understates everyday use. "
                           "Reporting all three brackets the truth honestly."]]),

        ("h2", "2.2 Why the third protocol is needed at all"),
        ("p", "A perfectly merchant-disjoint split is not constructible on this "
              "corpus. Brands co-occur inside single descriptions, so merging "
              "every co-occurring pair transitively collapses most of the data "
              "into one group that cannot then be split 70/15/15. Rather than "
              "report a figure that quietly benefits from the residual overlap, "
              "the subset with no shared merchant vocabulary is reported "
              "separately. It is a conservative lower bound by construction, "
              "since a row sharing only an ordinary word such as CAR is excluded "
              "from it."),

        ("h2", "2.3 Identifying merchants"),
        ("table", [["Decision", "Key each row by its narrowest-spread token"],
                   ["Why not the first word", "One brand produces many first "
                                              "words across statement formats, "
                                              "and descriptions often lead with "
                                              "a generic word belonging to "
                                              "hundreds of merchants: "
                                              "STARBUCKS <NUM> MARKET CT would "
                                              "be keyed MARKET."],
                   ["How", "A token's spread is how many distinct leading words "
                           "it co-occurs with. MARKET spreads across ~312 "
                           "merchants, STARBUCKS across ~2, so the "
                           "narrowest-spread token most specifically identifies "
                           "the merchant."],
                   ["Result", f"{meta['merchants']:,} merchants identified"]]),

        ("h2", "2.4 Models compared on validation, test reserved"),
        ("p", "All model comparison, threshold selection and diagnosis happens "
              "on the validation sets. Comparing models on test would make the "
              "final figure one that had been selected for, and therefore "
              "optimistic. The test sets are used exactly once, by the chosen "
              "configuration, after it has been refitted on train plus "
              "validation. Faults found in the rule baseline were likewise "
              "diagnosed on the training split, not on validation."),

        ("h1", "3. Models"),
        ("h2", "3.1 Why these six"),
        ("table", [["Model", "Question it answers"],
                   ["Rule baseline", "Do we need machine learning at all? "
                                     "Establishes the floor, using generic "
                                     "banking vocabulary only and no merchant "
                                     "names."],
                   ["Merchant memory", "How much of the apparent difficulty is "
                                       "simply recalling known shops? The "
                                       "deliberate opposite of the rules."],
                   ["Naive Bayes", "How far does pure word counting get us, at "
                                   "almost no cost?"],
                   ["Logistic regression", "Can one model give accuracy, "
                                           "trustworthy confidence and exact "
                                           "explanations together?"],
                   ["Linear SVM", "What is the accuracy ceiling of this feature "
                                  "representation?"],
                   ["MiniLM embeddings", "Does representing meaning beat "
                                         "counting surface patterns on text that "
                                         "is code rather than prose?"]]),

        ("h2", "3.2 Shared features for models 2 to 4"),
        ("p", "Naive Bayes, logistic regression and the linear SVM use exactly "
              "the same TF-IDF representation (word 1-2 grams plus character 3-4 "
              "grams), fitted on the training split alone. Differences between "
              "them therefore reflect the classifier rather than accidental "
              "differences in text representation. Fitting the vectoriser on all "
              "the data would leak vocabulary and document frequencies from "
              "validation into training."),
    ]

    if cmp_:
        models = cmp_["models"]
        blocks += [
            ("h2", "3.3 What the comparison showed"),
            ("table", [["Model", "In-distribution", "Unseen merchant",
                        "Strictly novel", "Explanation"]] +
                      [[name, f"{m['macro_f1']['Stratified']:.4f}",
                        f"{m['macro_f1']['Merchant held out']:.4f}",
                        f"{m['macro_f1']['Strictly novel']:.4f}",
                        m["explanation"]]
                       for name, m in models.items()]),
            ("p", "No model wins on every axis. The linear SVM is the most "
                  "accurate on familiar merchants but its confidence collapses "
                  "under distribution shift. MiniLM is the most accurate on "
                  "unfamiliar merchants and the best calibrated, but cannot "
                  "explain itself token by token. Logistic regression is "
                  "mid-table on accuracy while being well calibrated and exactly "
                  "interpretable."),
        ]

    blocks += [
        ("h1", "4. Architecture"),
        ("h2", "4.1 A cascade rather than a single model"),
        ("p", "The deployed system routes each transaction through deterministic "
              "rules, then merchant memory, then the learned model, with a "
              "confidence gate marking low-confidence model answers for human "
              "review."),
        ("p", "The justification is empirical. The rules hold roughly steady at "
              "17% coverage whether merchants are familiar or not, because they "
              "describe banking vocabulary. The merchant memory covers 82% of a "
              "returning user's transactions at 99% precision and 4% of a new "
              "user's. They are opposites, and neither is better: they fail in "
              "different places, which is why both sit in front of the model."),
    ]

    if casc:
        gains = casc["cascade_gain_over_model_alone"]
        blocks += [
            ("table", [["Model in the final slot", "Change when placed in the "
                                                   "cascade, returning user",
                        "Change, new user"]] +
                      [[name, f"{gains[f'stratified/{k}']:+.4f}",
                        f"{gains[f'merchant/{k}']:+.4f}"]
                       for k, name in (("logreg", "Logistic regression"),
                                       ("svm", "Linear SVM"),
                                       ("minilm", "MiniLM"),
                                       ("nb", "Naive Bayes"))]),
            ("p", "Assembling the cascade improves every model for a new user. "
                  "MiniLM gains most for a returning user, because the "
                  "deterministic layers cover precisely its weakness. The two "
                  "strongest linear models lose a little there, because the "
                  "memory occasionally overrides a correct model prediction."),
        ]

    blocks += [
        ("h2", "4.2 How each layer actually works"),
        ("h3", "Layer 1: deterministic rules"),
        ("p", "A list of hand-written IF-THEN patterns checked in order, first "
              "match wins. It performs no learning. The patterns use generic "
              "banking vocabulary only and contain no merchant names, because a "
              "rule asserting that NETFLIX means Subscription would score well "
              "on merchants already seen while revealing nothing about new ones."),
        ("code",
         "Input:  DEBIT FLAGSTAR BANK MORTGAGE <NUM> PPD ID <NUM>\n"
         "  Income?    needs CREDIT, this is DEBIT   -> skip\n"
         "  Transfer?  ZELLE|WIRE|VENMO ...          -> no match\n"
         "  Mortgage?  MORTGAGE|MTG PMT ...          -> MATCH on 'MORTGAGE'\n"
         "Answer: Mortgage\n\n"
         "Input:  DEBIT TST GOLDEN DRAGON OAKLAND\n"
         "  ... all twelve rules ...                 -> no match\n"
         "Answer: abstains, passes to the next layer"),
        ("h3", "Layer 2: merchant memory"),
        ("p", "**No mathematics and no learning: this layer is name matching.** "
              "It records which category each merchant name received during "
              "training and recalls it later, like a contacts list for shops."),
        ("p", "A description is first stripped to the shop's name. Bank plumbing "
              "(DEBIT, PPD, ID), payment processors (PYPL, SQ, PAYPAL), long "
              "numbers and opaque reference codes are discarded, and the first "
              "two remaining words are kept."),
        ("code",
         "DEBIT FLAGSTAR BANK MORTGAGE <NUM> PPD ID <NUM>  ->  'FLAGSTAR BANK'\n"
         "DEBIT PYPL BISHOPS CUT                           ->  'BISHOPS CUT'\n"
         "DEBIT SQ PIZZA PLACE MINNEAPOLIS                 ->  'PIZZA PLACE'"),
        ("p", "Training consists only of counting which category each stripped "
              "name was given. There are no weights and no equations."),
        ("code",
         "'SAFEWAY'      ->  Groceries        seen 11 times\n"
         "'CHEVRON'      ->  Transportation   seen 7 times\n"
         "'NETFLIX COM'  ->  Subscription     seen 1 time\n\n"
         "6,156 merchants recorded from the training split."),
        ("p", "At prediction time the same stripping is applied and the name is "
              "looked up."),
        ("code",
         "HIT   new row: DEBIT SAFEWAY 888\n"
         "      stripped: 'SAFEWAY'     -> found -> Groceries   (correct)\n\n"
         "MISS  new row: DEBIT RAPIDAPI <NUM> MAIN BLVD SEATTLE WA USA\n"
         "      stripped: 'RAPIDAPI MAIN' -> not found -> abstains"),
        ("p", "The training row said SAFEWAY 1197 and the new row says SAFEWAY "
              "888, yet both reduce to SAFEWAY once the store number is removed. "
              "That is precisely why the layer works at all."),
        ("p", "It abstains in two situations rather than guessing: when the "
              "merchant is unknown, and when a merchant was recorded under two "
              "categories equally often, which is genuine ambiguity."),
        ("p", "The fingerprint is computed from a single description rather than "
              "from the merchant_key column in the prepared files, because that "
              "column was derived using statistics over the whole corpus - "
              "information a deployed system would not have when one new "
              "transaction arrives."),
        ("h3", "Layer 3: the learned model, and the confidence gate"),
        ("p", "Everything the first two layers decline reaches the model, which "
              "always answers. An answer below the confidence threshold is "
              "marked for human review rather than presented as settled."),
        ("h3", "Why the order is rules, then memory, then model"),
        ("table", [["Layer", "Knows", "Returning user", "New user"],
                   ["Rules", "banking vocabulary", "17%", "17%"],
                   ["Merchant memory", "shop names", "82%", "4%"],
                   ["Model", "learned patterns", "the remainder", "the remainder"]]),
        ("p", "Memory is excellent for shops it has met and useless for shops it "
              "has not: it can recall but never reason. The rules are "
              "indifferent to whether a shop is familiar, because they describe "
              "vocabulary rather than merchants. They are therefore placed in "
              "front of the memory, and the model, which can generalise to "
              "merchants never seen, behind both."),
        ("h2", "4.3 The confidence threshold"),
        ("p", "The threshold decides how much human review the system asks for. "
              "It was chosen from the measured trade-off rather than inherited: "
              "below roughly 30% review all candidate models behave alike; at "
              "45% review the better-calibrated models reach 90% accuracy on "
              "what they accept automatically. A threshold of 0.60 was selected "
              "as the operating point."),
        ("h2", "4.4 A user-correction layer is specified but not yet built"),
        ("p", "The full design places a user-correction memory between the rules "
              "and the merchant memory, so that a category a user has fixed is "
              "never silently overridden. It requires real users and is "
              "therefore implemented in the application rather than in this "
              "evaluation."),

        ("h1", "5. Model selection for deployment"),
        ("h2", "5.1 The decision"),
        ("p", "**MiniLM sentence embeddings with a logistic-regression head** "
              "occupies the final slot of the cascade."),
        ("h2", "5.2 Why, and what it costs"),
        ("table", [["In favour", "Highest macro-F1 on unseen and strictly novel "
                                 "merchants, which is what a new user meets; "
                                 "best calibrated of all six models (ECE 0.0360)"],
                   ["Against", "It cannot produce token-level explanations "
                               "directly: its 384 dimensions are abstract and "
                               "carry no word-level meaning"],
                   ["Mitigation", "Explanations are produced post hoc with LIME "
                                  "and SHAP, which are established methods and "
                                  "already part of this project's scope"],
                   ["Rejected alternative", "Logistic regression, which gives "
                                            "exact explanations at no cost but "
                                            "is materially weaker on unfamiliar "
                                            "merchants"]]),
        ("h2", "5.3 Three caveats that must be stated, not hidden"),
        ("num", "**The explanation is approximate.** For a linear model the "
                "explanation is the arithmetic the model performed. For an "
                "embedding model it is a surrogate's estimate of what the model "
                "did."),
        ("num", "**LIME is stochastic by nature**, being fitted to randomly "
                "sampled perturbations. This was expected to make the same "
                "transaction yield different explanations across runs. Seeding "
                "the sampler from a hash of the description removes it "
                "entirely: measured top-3 stability is 1.0000 for both LIME and "
                "SHAP. The caveat is therefore resolved rather than merely "
                "acknowledged, and the resolution is itself a design decision."),
        ("num", "**It costs**, though far less than feared. Measured at a "
                "median of 33 milliseconds and 66 encoder passes for SHAP, "
                "against zero for the rule and memory layers. Fast enough to "
                "serve on demand with no loading state."),
        ("h2", "5.4 Validating the explanation method before relying on it"),
        ("p", "Because no exact attribution exists for an embedding model, LIME "
              "and SHAP cannot be checked against ground truth there. They are "
              "therefore validated on a linear model first, where exact "
              "contributions are computable, and only then applied to the "
              "deployed model. The linear model is used solely as a testbed for "
              "the explanation method and never produces a user-facing "
              "prediction."),
    ]

    if final:
        r = final["results"]
        blocks += [
            ("h1", "6. Final result"),
            ("p", "The configuration above, refitted on train plus validation and "
                  "evaluated once on the held-out test sets."),
            ("table", [["Protocol", "n", "Accuracy", "Macro-F1", "Sent to review",
                        "Accuracy of what is accepted"]] +
                      [[name.replace("_", " "), f"{r[name]['n']:,}",
                        f"{r[name]['accuracy']:.4f}", f"{r[name]['macro_f1']:.4f}",
                        f"{r[name]['review_rate']:.1%}",
                        f"{r[name]['accepted_accuracy']:.4f}"]
                       for name in ("in_distribution", "unseen_merchant",
                                    "strictly_novel") if name in r]),
        ]

    blocks += [
        ("h1", "7. Amounts, dates and accounts for the application"),
        ("h2", "7.1 Why they have to be generated"),
        ("p", "The benchmark supplies a description and a category and nothing "
              "else. A finance dashboard with no money in it can only count "
              "rows, so amounts, dates and account identifiers are generated."),
        ("p", "**The guarantee that makes this acceptable: they are never model "
              "features.** Every accuracy, F1, calibration and explainability "
              "figure reported in this project was computed from real "
              "third-party text and real third-party labels alone. Adding money "
              "changes none of them; it drives only the analytics layer."),
        ("h2", "7.2 Calibration to official statistics"),
        ("p", "Each category is mapped to a line of the U.S. Bureau of Labor "
              "Statistics Consumer Expenditure Survey 2024, released 19 December "
              "2025. The annual household figure divided by an assumed number of "
              "transactions per year gives a mean per transaction, and amounts "
              "are drawn from a log-normal distribution with that mean. Money is "
              "strictly positive and right-skewed, so a normal distribution "
              "would produce negative prices."),
        ("p", "**Fourteen of the seventeen categories take an official BLS "
              "figure.** Only Utilities, Subscription, Fees and Transfer are "
              "assumptions, and every transactions-per-year figure is this "
              "project's assumption rather than BLS. All of this is recorded in "
              "amount_model.json so the distinction can be stated in the report "
              "rather than glossed over."),
        ("h2", "7.3 Four corrections made during construction"),
        ("p", "Each was found by inspecting the generated data rather than "
              "assuming it was right, and each is a design decision in its own "
              "right."),
        ("table", [["Problem observed", "Cause", "Decision"],
                   ["Housing reached 44% of spending against the BLS figure near "
                    "a third",
                    "The household was emitting both rent and a mortgage",
                    "A household rents or owns, not both. Tenure is now an "
                    "explicit parameter and the other category is empty."],
                   ["Rent fell to 6.5% of spending",
                    "One landlord, and a single log-normal draw landed at $692 "
                    "against a $1,348 mean",
                    "A recurring commitment is a fixed obligation near the "
                    "typical figure, not a lottery. Recurring bases use a much "
                    "tighter spread and are held within a quarter of the mean."],
                   ["Category proportions drifted from the BLS shares",
                    "Scaling to 2,000 rows multiplies discretionary categories "
                    "while housing stays fixed at twelve payments a year",
                    "The default output is one true household-year, about 770 "
                    "transactions, which keeps the proportions faithful."],
                   ["January 2026 contained no salary at all, and two payments "
                    "fell one day apart",
                    "Income was treated as a one-off category and scattered to "
                    "random dates",
                    "Salary is the most regular payment a household has. Income "
                    "is now a fortnightly recurring commitment on two fixed days "
                    "each month."]]),
        ("h2", "7.4 Adjusting a population average to a household figure"),
        ("p", "BLS category averages include households that spend nothing in "
              "that category, so rented dwellings at $5,660 a year is the average "
              "across renters and owners alike rather than what a renter pays. "
              "Used directly it produced $472 monthly rent. Rent and mortgage "
              "are therefore divided by the share of households that rent (0.35) "
              "or own (0.65), giving about $1,348 and $1,194 a month. The "
              "adjustment is declared rather than hidden."),
        ("h2", "7.5 Which rows the demonstration uses"),
        ("p", "Transactions are drawn only from the held-out test split, so the "
              "deployed model has never been trained on any row the application "
              "displays. The stratified split is used rather than the "
              "merchant-held-out split, because with merchants held out the "
              "memory layer fires on 4% of rows and would be invisible in the "
              "demonstration, while the dashboard would sit at 65% accuracy. The "
              "stratified protocol exercises all three cascade layers and is "
              "also the realistic case for a working product, whose users mostly "
              "transact at merchants the system already knows."),
        ("p", "**The interface must nevertheless display all three protocol "
              "figures**, not only the one the demonstration reflects. Showing "
              "96% alone would imply a general accuracy the evaluation does not "
              "support."),
        ("h2", "7.6 Accounts"),
        ("p", "Transactions are assigned to one of three accounts by the "
              "statement format actually present in the description: an ACH "
              "reference marks a Chase row, a 'Withdrawal from' phrasing marks "
              "Capital One, and a trailing address with ZIP and state marks an "
              "Apple Card row. That heterogeneity is real and already present in "
              "the source data. The separate matter of each bank exporting a "
              "different file format is simulated, and the distinction between "
              "the two is stated plainly."),
    ]

    if demo:
        period = demo["period"]
        blocks += [
            ("h2", "7.7 The resulting dataset"),
            ("table", [["Property", "Value"],
                       ["Period", f"{period['from']} to {period['to']} "
                                  f"({period['months']} complete months)"],
                       ["Tenure", demo.get("tenure", "")],
                       ["Distribution", demo["distribution"]],
                       ["Source rows", demo["drawn_from"]],
                       ["BLS release", demo["source"]]]),
            ("table", [["Category", "BLS annual", "Per year", "Mean per "
                        "transaction", "Source"]] +
                      [[name, f"{cfg['annual']:,}", str(cfg["per_year"]),
                        f"{cfg['mean_per_transaction']:,.2f}", cfg["source"]]
                       for name, cfg in sorted(
                           demo["categories"].items(),
                           key=lambda kv: -kv[1]["annual"])]),
        ]

    blocks += [
        ("h1", "8. Multi-account aggregation"),
        ("h2", "8.1 Why three feeds exist"),
        ("p", "The project's claim is multi-account aggregation, which requires "
              "more than one source and more than one format. The demonstration "
              "transactions are therefore split into three bank feeds, each "
              "written in a different export format, and reassembled by one "
              "adapter per bank."),
        ("h2", "8.2 What is real and what is simulated"),
        ("p", "This distinction should be stated plainly in the report rather "
              "than left for an examiner to find."),
        ("table", [["Element", "Status"],
                   ["Different description formats per bank (ACH references, "
                    "'Withdrawal from' phrasing, trailing addresses)",
                    "Real: already present in the source data and verified by "
                    "counting"],
                   ["The problem of reconciling bank exports",
                    "Real: it is what Plaid, Yodlee and SGFinDex exist to solve"],
                   ["Apple Card's inverted sign convention",
                    "Real: independently documented behaviour"],
                   ["Separate Debit and Credit columns, differing date semantics",
                    "Real: documented Capital One export layout"],
                   ["The specific file layouts used here",
                    "Simulated: modelled on documented layouts, but exact "
                    "headers vary by account type and could not be fully "
                    "verified"]]),
        ("p", "**The limitation to acknowledge:** the adapters are exercised "
              "against representative fixtures, not against live bank exports, "
              "so this demonstrates the design and the code rather than proving "
              "robustness against real institutional data. Connecting one feed "
              "to the Open Bank Project sandbox would make the claim genuinely "
              "external, and is recorded as future work."),
        ("h2", "8.3 The conventions the adapters must reconcile"),
        ("table", [["Feed", "Format", "The problem it poses"],
                   ["Chase", "one signed Amount column, MM/DD/YYYY dates",
                    "month-first date parsing"],
                   ["Capital One", "separate Debit and Credit columns, ISO "
                                   "dates, transaction and posted dates",
                    "exactly one column carries a value per row, and the "
                    "adapter must choose which date represents the purchase"],
                   ["Apple Card", "twelve monthly files, MM/DD/YYYY, purchases "
                                  "POSITIVE and payments NEGATIVE",
                    "the sign convention is inverted, and the adapter must "
                    "discover and concatenate twelve files"]]),
        ("p", "The Apple Card inversion matters more than it appears. An "
              "adapter that ignores it produces a dashboard in which every "
              "purchase is displayed as income, and nothing else in the system "
              "would detect that. It is precisely the class of failure an "
              "adapter layer exists to prevent, and it is a documented "
              "real-world convention rather than an invented obstacle."),
        ("h2", "8.4 The bank's own category is discarded deliberately"),
        ("p", "Every feed carries a Category column, as real bank exports do. "
              "**Every adapter discards it.** Using it would mean the bank had "
              "already performed the categorisation this project exists to "
              "perform, which would make the entire evaluation circular. The "
              "field is carried precisely so that discarding it is visible in "
              "the code rather than implied."),
        ("p", "Each adapter also drops other fields a real export carries and "
              "the canonical schema does not need, so the adapters genuinely "
              "select fields rather than merely renaming them: Chase drops Type "
              "and Memo, Capital One drops the card number and the posted date, "
              "Apple Card drops the merchant, type and clearing date."),
        ("h2", "8.5 Verification"),
        ("p", "Correctness here cannot be assumed, because a sign error or a "
              "misparsed date fails silently. The adapters are therefore checked "
              "by round-trip: every feed is parsed back and compared against the "
              "transactions it was written from. Seven checks pass, covering row "
              "count, exact amounts, restored Apple Card signs, dates across "
              "three formats, unchanged descriptions, the absence of any true "
              "label in any feed, and the presence-but-discarding of the bank's "
              "category."),

        ("h1", "9. Decisions still open"),
        ("bullet", "Whether the merchant memory should use fuzzy matching. It "
                   "currently matches exactly, so its coverage is a lower bound."),
        ("bullet", "Whether to fine-tune the encoder rather than freeze it. It is "
                   "the only remaining change that would alter the representation "
                   "itself, which is the only thing that has moved results so far."),
        ("bullet", "Whether a neural network head improves on a linear one over "
                   "the same embeddings. Written and awaiting a run."),
    ]

    blocks += [
        ("h1", "10. Sources"),
        ("h2", "10.1 Data"),
        ("bullet", "DoDataThings. *US Bank Transaction Categories v2*. Hugging "
                   "Face Datasets, MIT licence. "
                   "https://huggingface.co/datasets/DoDataThings/"
                   "us-bank-transaction-categories-v2"),
        ("bullet", "coderanand. *Indian Banking Transaction Text Dataset*. "
                   "Kaggle. https://www.kaggle.com/datasets/coderanand/"
                   "indian-banking-transaction-text-dataset"),
        ("bullet", "U.S. Bureau of Labor Statistics. *Consumer "
                   "Expenditures - 2024*, USDL-25-1586, released 19 December "
                   "2025. https://www.bls.gov/news.release/cesan.nr0.htm"),
        ("h2", "10.2 Prior work on transaction categorisation"),
        ("bullet", "Garcia-Mendez, S., Fernandez-Gavilanes, M., "
                   "Juncal-Martinez, J., Gonzalez-Castano, F. J. and Barba "
                   "Seara, O. (2024). *Identifying Banking Transaction "
                   "Descriptions via Support Vector Machine Short-Text "
                   "Classification Based on a Specialized Labelled Corpus*. "
                   "arXiv:2404.08664. Real customer transactions, released to "
                   "researchers on request."),
        ("bullet", "Toran, L., Van Der Walt, C., Sammarone, A. and Keller, A. "
                   "(2023). *Scalable and Weakly Supervised Bank Transaction "
                   "Classification*. arXiv:2305.18430. Evidence that labelled "
                   "transaction data is expensive and difficult to obtain."),
        ("bullet", "*Hierarchical Classification of Financial Transactions "
                   "Through Context-Fusion of Transformer-based Embeddings and "
                   "Taxonomy-aware Attention Layer* (2023). arXiv:2312.07730. "
                   "Uses 151,867 real card transactions held in-house."),
        ("h2", "10.3 Methods"),
        ("bullet", "Ribeiro, M. T., Singh, S. and Guestrin, C. (2016). *Why "
                   "Should I Trust You? Explaining the Predictions of Any "
                   "Classifier*. KDD 2016. The LIME method."),
        ("bullet", "Lundberg, S. M. and Lee, S.-I. (2017). *A Unified Approach "
                   "to Interpreting Model Predictions*. NeurIPS 2017. The SHAP "
                   "method."),
        ("bullet", "Reimers, N. and Gurevych, I. (2019). *Sentence-BERT: "
                   "Sentence Embeddings using Siamese BERT-Networks*. EMNLP "
                   "2019. The sentence-transformers family, of which "
                   "all-MiniLM-L6-v2 is a member."),
        ("bullet", "Pedregosa, F. et al. (2011). *Scikit-learn: Machine "
                   "Learning in Python*. JMLR 12, 2825-2830."),
        ("bullet", "Platt, J. (1999). *Probabilistic Outputs for Support Vector "
                   "Machines*. The sigmoid calibration applied to the SVM."),
        ("h2", "10.4 Bank export formats"),
        ("bullet", "Tiller. *How to Export Your Capital One Transactions to a "
                   "Spreadsheet*. Documents the Transaction Date, Posted Date, "
                   "Card No., Description, Category, Debit, Credit layout. "
                   "https://tiller.com/export-capital-one-transactions/"),
        ("bullet", "beancount.io. *CSV to Beancount: Capital One* and *Chase*. "
                   "Independent corroboration of both layouts. "
                   "https://beancount.io/tools/csv-to-beancount/capital-one"),
        ("bullet", "TidBITS (2020). *Apple Card (Finally) Gains CSV Statement "
                   "Export*. Documents the per-statement-month export. "
                   "https://tidbits.com/2020/01/23/"
                   "apple-card-finally-gains-csv-statement-export/"),
        ("bullet", "Tiller Community. *Apple Card Statement Export/Import*. "
                   "Reports that purchases export as positive and payments as "
                   "negative. https://community.tiller.com/t/"
                   "apple-card-statement-export-import/3025"),
        ("h2", "10.5 Note on citation"),
        ("p", "The bank-format sources are community documentation rather than "
              "official publications, because none of the three banks publishes "
              "a formal export specification. They are cited as the basis for "
              "the representative formats used here, and the report should "
              "describe them as such."),
    ]

    write_doc("design_decisions.docx", "Design Decisions and Justifications",
              "A running record. Regenerated whenever a decision changes.",
              blocks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
