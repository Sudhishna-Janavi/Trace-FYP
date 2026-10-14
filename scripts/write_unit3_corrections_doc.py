#!/usr/bin/env python3
"""Unit 3 corrections and the ONNX inference change.

Two pieces of work that happened after the original Unit 3 document was
written: three faults found in the demonstration-data generator by reading its
output against the survey it claims to follow, and the replacement of the
inference engine, which had to be proved to leave every reported figure intact.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import matplotlib.pyplot as plt  # noqa: E402
from reporting import ACCENT, BAD, GOOD, MUTED, save, write_doc  # noqa: E402

# measured before and after the fixes; see results/ and the generator's own
# reconciliation table, which it now prints and asserts on every run
BEFORE = {"Income": 1.25, "Insurance": 2.27, "Subscription": 1.46,
          "Utilities": 1.17, "Rent": 0.94}
AFTER = {"Income": 1.000, "Insurance": 0.995, "Subscription": 0.998,
         "Utilities": 1.001, "Rent": 1.001}

MEMORY = [("application imported", 67, 66), ("merchant memory", 72, 71),
          ("encoder loaded", 401, 107), ("peak under load", 401, 314)]


def figure_reconciliation() -> Path:
    names = list(BEFORE)
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    width, positions = 0.38, range(len(names))
    ax.bar([p - width / 2 for p in positions], [BEFORE[n] for n in names],
           width, label="before", color=BAD)
    ax.bar([p + width / 2 for p in positions], [AFTER[n] for n in names],
           width, label="after", color=GOOD)
    ax.axhline(1.0, color=MUTED, linewidth=1.1, linestyle="--")
    ax.text(len(names) - 0.4, 1.03, "the BLS figure", fontsize=9, color=MUTED)
    for index, name in enumerate(names):
        ax.text(index - width / 2, BEFORE[name] + 0.04, f"{BEFORE[name]:.2f}",
                ha="center", fontsize=9, color=MUTED)
    ax.set_xticks(list(positions)); ax.set_xticklabels(names)
    ax.set_ylabel("emitted total / BLS annual figure")
    ax.set_ylim(0, 2.5)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Recurring categories against the survey they are calibrated to",
                 fontsize=12, fontweight="bold", loc="left", pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e1e5e2", linewidth=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    fig.tight_layout()
    return save(fig, "unit3_bls_reconciliation.png")


def figure_memory() -> Path:
    labels = [m[0] for m in MEMORY]
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    width, positions = 0.38, range(len(labels))
    ax.bar([p - width / 2 for p in positions], [m[1] for m in MEMORY], width,
           label="PyTorch", color=BAD)
    ax.bar([p + width / 2 for p in positions], [m[2] for m in MEMORY], width,
           label="ONNX Runtime", color=ACCENT)
    ax.axhline(512, color=MUTED, linewidth=1.1, linestyle="--")
    ax.text(0.05, 522, "512 MB — the free instance limit", fontsize=9, color=MUTED)
    for index, entry in enumerate(MEMORY):
        ax.text(index - width / 2, entry[1] + 9, f"{entry[1]}", ha="center",
                fontsize=9, color=MUTED)
        ax.text(index + width / 2, entry[2] + 9, f"{entry[2]}", ha="center",
                fontsize=9, color=MUTED)
    ax.set_xticks(list(positions)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("resident memory (MB)"); ax.set_ylim(0, 610)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("The weights are the same 86 MB; the framework was the cost",
                 fontsize=12, fontweight="bold", loc="left", pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e1e5e2", linewidth=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    fig.tight_layout()
    return save(fig, "unit3_onnx_memory.png")


def main() -> int:
    fig1, fig2 = figure_reconciliation(), figure_memory()

    write_doc(
        "unit3_corrections_and_onnx.docx",
        "Unit 3 corrections, and changing the inference engine",
        "Three faults found by reading the generator's output against the "
        "survey it claims to follow, and an engine replacement that had to be "
        "proved harmless before it was adopted.",
        [
        ("h1", "1. Why these corrections exist"),
        ("p", "The demonstration amounts are generated, and generated data is "
              "only as good as the checking applied to it. Reading the "
              "dashboard against the survey the generator claims to follow "
              "exposed three faults that no unit test would have caught, "
              "because each produced output that was individually plausible."),
        ("p", "The symptom was a household apparently saving half of "
              "everything it earned: income $130,229 against spending "
              "$65,496."),

        ("h1", "2. Income was gross, not take-home"),
        ("p", "The generator mapped income to the BLS line “average income "
              "before taxes” ($104,207 in 2024) and deposited it into a "
              "current account. No bank statement shows gross pay: what "
              "arrives is net of tax, social security and payroll deductions."),
        ("p", "The obvious fix — use the matching after-tax line — is not "
              "available. **BLS discontinued “income after taxes” with the "
              "2024 release**, stating it can no longer produce federal tax, "
              "state tax or after-tax income estimates for the CE tables, "
              "LABSTAT or the public use microdata."),
        ("p", "The 2024 before-tax figure is therefore scaled by the last "
              "ratio BLS itself published, from 2023: $87,869 after tax "
              "against $101,805 before, a factor of 0.8631. Using the "
              "agency's own published relationship keeps the adjustment "
              "traceable to the source rather than to an assumed tax rate of "
              "this project's invention. Take-home becomes $89,942."),

        ("h1", "3. Recurring categories did not reconcile to their own target"),
        ("p", "Two quantities decided a recurring category's annual total and "
              "nothing made them agree. How many payments were emitted came "
              "from merchants × months; how large each was came from dividing "
              "the BLS annual figure by an assumed payments-per-year. Where "
              "those disagreed, the household was billed the wrong amount for "
              "the year."),
        ("p", "Insurance names two merchants against a payments-per-year of "
              "twelve, so it emitted 24 payments sized for 12 and charged "
              "2.27× the BLS figure — which is why insurance appeared as 9.4% "
              "of spending against a survey figure near 3.9%."),
        ("figure", (fig1, "Figure 1. Each recurring category's emitted annual "
                          "total as a multiple of the BLS figure it is "
                          "calibrated to, before and after. The generator now "
                          "prints this table and asserts that drift stays "
                          "under 2% on every run.")),

        ("h1", "4. A single draw decided a whole year"),
        ("p", "Each recurring merchant's payment amount was drawn once and "
              "repeated for twelve months. Income has one merchant — one "
              "employer — so a single random draw set the entire year's "
              "salary, and this seed landed on the clamp ceiling at +25%. "
              "A clamp bounds such an error without removing it; with one "
              "merchant there is no second draw to average against."),
        ("p", "All three faults are fixed by one change: every merchant's base "
              "is drawn first, then the set is rescaled so the payments the "
              "category actually emits sum to its BLS annual figure by "
              "construction. Variety between merchants survives; only the "
              "overall level is pinned."),

        ("h1", "5. What was deliberately not corrected"),
        ("p", "One-off categories are left sampling freely. A household's "
              "restaurant and grocery spending genuinely varies from the "
              "national average, and forcing each to land on it exactly would "
              "be less realistic rather than more. With roughly a hundred "
              "independent draws those totals settle within a tenth of target "
              "on their own; categories drawn a handful of times are left "
              "visibly noisy, because that is what a small number of real "
              "purchases looks like."),
        ("table", [["", "before", "after"],
                   ["income", "$130,229", "$89,922"],
                   ["spending", "$65,496", "$61,704"],
                   ["net", "+$64,733", "+$28,217"],
                   ["insurance share", "9.4%", "4.7%"]]),
        ("p", "A surplus remains, and it is honest rather than residual error: "
              "the seventeen categories the classifier recognises do not cover "
              "everything a household spends on — no apparel, pensions, cash "
              "contributions or miscellaneous — so modelled spending sits below "
              "the BLS average annual expenditure of $78,535. That is a limit "
              "of the label set, not of the generator."),

        ("h1", "6. Changing the inference engine"),
        ("p", "Deploying the application to an instance that does not require "
              "a payment card means fitting inside 512 MB. With PyTorch "
              "loaded the application measured 401 MB resident, leaving too "
              "little headroom to be safe."),
        ("p", "The encoder was moved to an ONNX graph under ONNX Runtime. The "
              "weights are the same 86 MB either way; PyTorch was the cost. "
              "Inference here is a fixed sequence of matrix multiplications "
              "with no training, no autograd and no dynamic graph, so the "
              "framework supporting those things was paying rent it did not "
              "earn."),
        ("figure", (fig2, "Figure 2. Resident memory at each stage, under both "
                          "engines. Removing PyTorch also strips roughly a "
                          "gigabyte of wheels from the deployment image.")),

        ("h1", "7. Why the change had to be verified, not assumed"),
        ("p", "Swapping inference engines silently changes the system that "
              "every reported figure describes. Unless the predictions agree, "
              "the accuracy quoted in this report would no longer be the "
              "accuracy of the deployed application. `scripts/verify_onnx.py` "
              "is the evidence that they do."),
        ("table", [["comparison", "result"],
                   ["embedding cosine similarity", "min 0.99999988"],
                   ["embedding absolute difference", "max 2.5e-07"],
                   ["class probability difference", "max 2.7e-06"],
                   ["predicted category differs", "0 of 1,500"],
                   ["accept-or-review differs", "0 of 1,500"],
                   ["accuracy", "identical to six decimal places"]]),
        ("p", "Re-importing the full demonstration set through the live "
              "cascade changed the category or review flag on 0 of 772 "
              "transactions. Floating-point arithmetic reorders differently "
              "across engines, so exact equality of embeddings is not "
              "expected; disagreement about any prediction would have been "
              "disqualifying."),
        ("p", "One figure was re-measured rather than carried over. Shapley "
              "explanations take a median of 51 ms under ONNX against 31 ms "
              "under PyTorch. The application's own interface quotes the ONNX "
              "figure, because that is the engine that runs."),

        ("h1", "8. For the report"),
        ("bullet", "Generated data needs reconciliation against its own "
                   "source, not only unit tests. All three faults produced "
                   "individually plausible output."),
        ("bullet", "A discontinued statistic is a citable methodological "
                   "fact. Scaling by the agency's last published ratio is "
                   "defensible; assuming a tax rate would not be."),
        ("bullet", "An engine substitution is a change to the system under "
                   "measurement and must be verified at the level of "
                   "predictions, not loss or embeddings alone."),
        ("bullet", "Where a measured figure changes, the interface should "
                   "quote the new one."),
        ])
    print("[PASS] docs/unit3_corrections_and_onnx.docx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
