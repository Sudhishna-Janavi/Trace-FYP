#!/usr/bin/env python3
"""Unit 5: the application. Architecture, interface decisions, deployment."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import matplotlib.pyplot as plt  # noqa: E402
from reporting import ACCENT, GOOD, MUTED, WARN, save, write_doc  # noqa: E402

# observed on the demonstration account, 772 transactions
LAYERS = [("your corrections", 0), ("rules", 91), ("merchant memory", 508),
          ("the model", 173)]
FLAGGED = 46


def figure_layers() -> Path:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    names = [l[0] for l in LAYERS]
    counts = [l[1] for l in LAYERS]
    colours = [GOOD, GOOD, GOOD, ACCENT]
    bars = ax.barh(names[::-1], counts[::-1], color=colours[::-1], height=0.62)
    total = sum(counts)
    for bar, value in zip(bars, counts[::-1]):
        ax.text(value + 8, bar.get_y() + bar.get_height() / 2,
                f"{value}  ({value / total:.0%})", va="center", fontsize=9.5,
                color=MUTED)
    ax.set_xlim(0, max(counts) * 1.28)
    ax.set_title("Which layer decided, on 772 transactions", fontsize=11,
                 fontweight="bold", loc="left", pad=10)

    exact = sum(c for (n, c) in LAYERS if n != "the model")
    ax2.bar(["exact and free", "needs approximating"], [exact, total - exact],
            color=[GOOD, WARN], width=0.55)
    for index, value in enumerate([exact, total - exact]):
        ax2.text(index, value + 10, f"{value / total:.0%}", ha="center",
                 fontsize=10.5, color=MUTED)
    ax2.set_ylim(0, total * 0.95)
    ax2.set_title("What an explanation costs", fontsize=11, fontweight="bold",
                  loc="left", pad=10)

    for a in (ax, ax2):
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.grid(axis="x" if a is ax else "y", color="#e1e5e2", linewidth=0.8)
        a.set_axisbelow(True)
        a.tick_params(colors=MUTED, labelsize=9.5)
    fig.suptitle("The cascade in production: the model is the exception, not "
                 "the rule", fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return save(fig, "unit5_cascade_in_production.png")


def main() -> int:
    card = json.loads((ROOT / "backend" / "artifacts" / "model_card.json").read_text())
    p = card["performance"]
    fig1 = figure_layers()

    write_doc(
        "unit5_application.docx",
        "Unit 5: the application",
        "A deployed multi-account dashboard in which every category can be "
        "questioned, corrected, and traced to the layer that decided it.",
        [
        ("h1", "1. What the application has to prove"),
        ("p", "Units 2 and 4.5 established that a cascade beats the model "
              "alone and that most explanations can be exact. Neither claim "
              "means anything until a user can act on it. The application "
              "exists to make three things visible: that transactions from "
              "incompatible bank exports aggregate into one ledger, that "
              "every category carries a reason the user can read, and that a "
              "category the user disagrees with can be corrected and stays "
              "corrected."),

        ("h1", "2. Architecture"),
        ("p", "FastAPI with server-rendered Jinja templates and no build step. "
              "The absence of a frontend framework is deliberate: the whole "
              "application is one process, which is what allows it to run on "
              "a free instance, and every page renders without scripting."),
        ("table", [["layer", "module", "responsibility"],
                   ["adapters", "backend/adapters.py",
                    "three bank export formats into one shape"],
                   ["ingestion", "backend/ingest.py",
                    "feeds through the cascade into storage"],
                   ["classification", "backend/classifier.py",
                    "the four-layer cascade and confidence gate"],
                   ["explanation", "backend/explainability.py",
                    "per-layer, exact where possible"],
                   ["corrections", "backend/review.py",
                    "the user-correction memory"],
                   ["analytics", "backend/analytics.py",
                    "money, recurrence, anomalies, benchmarks"],
                   ["charts", "backend/charts.py",
                    "inline SVG, server-side"]]),
        ("p", "Charts are drawn as SVG on the server rather than by a "
              "charting library. This keeps deployment to a single process "
              "with no asset pipeline, renders with scripting unavailable, "
              "and — the reason that matters in practice — lets every mark "
              "reference the same CSS custom properties as the page, so "
              "charts follow the light and dark themes and stay sharp at any "
              "zoom when captured for a document."),

        ("h1", "3. The cascade in production"),
        ("figure", (fig1, "Figure 1. Which layer decided each of the "
                          "demonstration account's 772 transactions, and what "
                          "that implies for the cost of explaining them. "
                          "78% were decided by a layer that knows its own "
                          "reason exactly.")),
        ("p", "The ordering is not an optimisation detail. Three of the four "
              "layers are exact, free, and able to state their own reason, so "
              "the cascade reduces what explanation costs as well as "
              "improving accuracy on unfamiliar merchants. Reaching the model "
              "at all is the exception."),

        ("h1", "4. The confidence gate and the review queue"),
        ("p", f"The model declines below {card['threshold']:.0%} confidence "
              "rather than filing a transaction somewhere plausible and "
              "leaving the user to discover it later. On the demonstration "
              f"account {FLAGGED} of 772 transactions ({FLAGGED / 772:.1%}) "
              "are held for review, against the "
              f"{p['in_distribution']['review_rate']:.1%} measured offline "
              "under the matching protocol. The running application "
              "reproduces the measurement rather than quoting it."),
        ("p", "The queue is sorted by ascending confidence, so the model's "
              "least certain decisions are seen first. A representative case: "
              "“[debit] P.F. CHANG'S #896 COLUMBUS” was filed as Groceries at "
              "24% confidence, and the Shapley explanation shows why — the "
              "terms carrying the most weight were “896” (a store number), "
              "“DEBIT” and “COLUMBUS”, while the brand token “CHANG” carried "
              "a negative weight. The model had never seen the merchant and "
              "had nothing to go on but noise. That single view justifies the "
              "gate, the queue and the correction memory at once."),

        ("h1", "5. Corrections are about merchants, not transactions"),
        ("p", "Correcting one row and leaving forty identical rows untouched "
              "would make the feature feel broken. A correction is therefore "
              "keyed by merchant fingerprint, applied to every matching "
              "transaction the user has not personally ruled on, and "
              "consulted before any rule or model the next time that merchant "
              "appears. One correction updated 23 sibling rows in testing."),
        ("p", "Confirming is deliberately not symmetric with correcting. "
              "Agreeing with one prediction is a statement about that "
              "prediction, not about the merchant, and does not license "
              "overriding the model elsewhere — so only that row is marked. A "
              "row the user has personally ruled on is never overwritten by a "
              "later merchant-level correction."),

        ("h1", "6. Honesty in the interface"),
        ("p", "Three places where the application tells the user something "
              "that is against its own interest, because the alternative "
              "would be misleading:"),
        ("num", "The “How it works” page reports all three evaluation "
                f"protocols — macro F1 {p['in_distribution']['macro_f1']:.3f} "
                f"when the merchant is familiar, "
                f"{p['unseen_merchant']['macro_f1']:.3f} when it is held out, "
                f"{p['strictly_novel']['macro_f1']:.3f} at a cold start. "
                "Reporting only the first would be true and misleading at "
                "once."),
        ("num", "Explanations from the model layer are badged Approximate and "
                "say so in words; the other three layers are badged Exact. "
                "A single uniform presentation would imply the model's "
                "reconstruction is as sound as a matched rule phrase."),
        ("num", "The Insights page states that its comparison against the "
                "BLS household average is circular for the demonstration "
                "data, because those amounts were generated from the same "
                "survey lines. Close agreement there is arithmetic, not "
                "evidence."),
        ("p", "The spending signals are descriptive and carry no score. Any "
              "financial-health rating would need a threshold, and every "
              "threshold available here would be invented; the application "
              "says so on the page and gives no financial advice."),

        ("h1", "7. Multi-account aggregation"),
        ("p", "Three export formats are reconciled: a single signed amount "
              "column with US dates, separate debit and credit columns with "
              "ISO dates, and twelve monthly files in which purchases are "
              "positive and payments negative. Every adapter discards the "
              "bank's own category column deliberately. Three institutions "
              "disagree about what a category is and none agrees with the "
              "others, so a category that holds across all of a person's "
              "accounts has to be derived from the description — which is the "
              "reason the classifier exists."),

        ("h1", "8. Deployment"),
        ("p", "The application is deployed and reachable at "
              "https://trace-fyp.onrender.com, as a Docker image on a free "
              "512 MB instance with a managed Postgres database. Peak "
              "resident memory is 314 MB, which required replacing the "
              "inference engine; that change and its verification are "
              "documented in the Unit 3 corrections paper."),
        ("bullet", "Session cookies are signed, HTTP-only, and marked secure "
                   "when deployed. The application refuses to start in a "
                   "deployment environment without its own signing key, "
                   "because the development fallback is published in the "
                   "repository and a quiet fallback would make that failure "
                   "invisible."),
        ("bullet", "Every user-visible row carries a user identifier and "
                   "every query is scoped by it in one module, so a missing "
                   "filter is a visible omission in one place rather than a "
                   "silent leak across many."),
        ("bullet", "The encoder is baked into the image at build time and the "
                   "application is offline at runtime, so a missing model "
                   "fails the build rather than the first user request."),

        ("h1", "9. Limitations"),
        ("p", "Measured on the 772-transaction demonstration set, both "
              "concerning the merchant fingerprint:"),
        ("bullet", "Six transactions (0.8%) produce an empty fingerprint, "
                   "because every token is bank noise, a processor prefix or "
                   "a reference code — for example “[debit] BP*86NSEW6Y3PW”. "
                   "Merchant memory cannot learn these and a correction "
                   "cannot be stored against them. The correction path guards "
                   "the empty key, so this degrades rather than breaks."),
        ("bullet", "Seventy-four brands split across more than one "
                   "fingerprint, because the key is the first two meaningful "
                   "tokens and the second is often a city: SAFEWAY, SAFEWAY "
                   "BOSTON and SAFEWAY COLUMBUS are three separate merchants "
                   "to the system. Memory and corrections therefore fragment "
                   "across branches of one chain. Exact matching was chosen so "
                   "that the baseline answers if and only if it has seen the "
                   "merchant before; a fuzzy or prefix match would raise "
                   "coverage at the cost of that property."),
        ("bullet", "Statement upload is not implemented. The application "
                   "reads exported files through its adapters, but the "
                   "onboarding path offers only the demonstration set. "
                   "Reading balances directly from a bank requires that "
                   "bank's authorisation under an open-banking scheme, which "
                   "is an institutional agreement rather than a technical "
                   "obstacle."),

        ("h1", "10. For the report"),
        ("bullet", "The cascade's value in production is twofold: better "
                   "accuracy on unfamiliar merchants, and a lower cost of "
                   "explanation, because 78% of decisions come from a layer "
                   "that knows its own reason exactly."),
        ("bullet", "A deployed review queue turns a confidence threshold from "
                   "a reported statistic into a visible workflow."),
        ("bullet", "The live application independently reproduces the offline "
                   "review rate, which is a stronger claim than quoting it."),
        ("bullet", "Interface honesty is a design decision with consequences, "
                   "not a disclaimer: badging approximate explanations and "
                   "declaring the benchmark circular both cost the system "
                   "apparent authority and are both correct."),
        ])
    print("[PASS] docs/unit5_application.docx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
