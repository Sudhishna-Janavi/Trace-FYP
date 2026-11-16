# Handoff — Integrated Finance Dashboard (CCDS25-1131)


---

## What this project is

A finance dashboard that reads bank transaction descriptions, classifies each into
one of 17 spending categories, **explains why**, and aggregates three accounts into
one view. The explanation is the point: the FYP's claim is *traceable* categorisation.

Classification runs as a **cascade**, not a single model:

```
1. deterministic rules      banking vocabulary; merchant-independent
2. merchant memory          shop names seen before; merchant-dependent
3. MiniLM + logistic head   everything else
4. confidence gate          below 0.60 goes to human review
```

A **user-correction memory** belongs between layers 1 and 2. It is specified but
**not built** — it needs real users, so it ships with the app (Unit 5).

---
## The headline numbers

Final test-set result, used **once**, cascade with MiniLM, threshold 0.60:

| Protocol | n | Accuracy | Macro-F1 | Sent to review | Accuracy of what is accepted |
|---|---|---|---|---|---|
| In-distribution (familiar merchants) | 4,786 | 0.9636 | 0.9681 | 5.2% | 0.9896 |
| Unseen merchant | 4,780 | 0.6544 | 0.6808 | 33.8% | 0.8291 |
| Strictly novel (cold start) | 2,245 | 0.5555 | 0.5636 | 42.8% | 0.7689 |

**The app must display all three**, not only the first. Showing 96% alone implies a
general accuracy the evaluation does not support.

Model comparison (validation macro-F1):

| Model | In-dist | Unseen | Strictly novel | Merchant ECE |
|---|---|---|---|---|
| Rule baseline (17% coverage) | 0.2924 | 0.2849 | 0.3068 | n/a |
| Merchant memory (82%/4% coverage) | 0.8500 | 0.0752 | 0.0156 | n/a |
| Naive Bayes | 0.9809 | 0.5903 | 0.4657 | n/a |
| Logistic regression | 0.9976 | 0.5980 | 0.4636 | 0.0406 |
| Linear SVM | 0.9984 | 0.6393 | 0.5146 | 0.1452 |
| **MiniLM (deployed)** | 0.9167 | **0.6479** | **0.5936** | **0.0360** |

**Why MiniLM was chosen despite losing on in-distribution accuracy:** it is the
best on unfamiliar merchants, which is what a new user meets, and the best
calibrated. It cannot explain itself directly, so explanations come from SHAP.
That trade-off is deliberate and quantified: about 0.10 macro-F1 given up.

Explainability (Unit 4.5):

- SHAP agrees with exact contributions at Spearman **0.7971**, 76.8% top-3 overlap
- LIME is worse on every measure (0.7383, 66.3%) — retained for comparison only
- Cost: SHAP **33 ms** median, 66 encoder passes. LIME 17 ms.
- Stability **1.0000** for both, because seeds derive from a hash of the text
- **68% of explanations are exact and free** (rule + memory layers)

---

## Directory

```
oct14/
├── HANDOFF.md                      this file
├── README.md                       project overview, Units 1–2
├── backend/
│   ├── adapters.py                 3 bank adapters -> canonical schema
│   └── explainability.py           layer-aware explanations (rule/memory/SHAP)
├── models/
│   ├── cascade.py                  THE DEPLOYED CASCADE. app imports this
│   ├── rule_baseline.py            layer 1
│   ├── merchant_memory.py          layer 2
│   ├── minilm_embeddings.py        layer 3 (deployed model)
│   ├── logistic_regression.py      comparison + XAI validation testbed
│   ├── linear_svm.py, naive_bayes.py, tfidf_features.py, mlp_head.py
├── scripts/
│   ├── prepare_dataset.py          Unit 1
│   ├── evaluate_model.py           shared harness: load(), score(), calibration()
│   ├── reporting.py                shared figures + .docx writer
│   ├── run_model1..5, run_model1b  per-model evaluations
│   ├── run_unit2_comparison.py     cross-model comparison
│   ├── run_unit2_cascade.py        cascade + threshold sweep
│   ├── run_unit2_final_test.py     THE single test run
│   ├── build_demo_data.py          Unit 3: BLS amounts, dates, accounts
│   ├── build_feeds.py              Unit 4: split into 3 bank formats
│   ├── verify_feeds.py             round-trip checks (7 assertions)
│   ├── run_unit4_5_explainability.py
│   └── write_*_doc.py              document generators
├── dataset/
│   ├── US/transactions-synthetic.csv       raw, 68,000 rows (not published)
│   ├── Indian/                             second source, unused so far (not published)
│   ├── processed/                          6 split CSVs + meta (not published)
│   └── demo/
│       ├── demo_transactions.csv           772 rows, the app's data
│       ├── amount_model.json               BLS figures + provenance
│       ├── ground_truth.csv                labels, never read by adapters
│       └── feeds/                          14 files in 3 formats + manifest
├── results/                        experiment output (not published)
└── docs/                           project documents (not published)
```

**Embeddings cache lives OUTSIDE the project** at `~/.cache/oct14-embeddings`
(see Gotchas). Resolve it with `evaluate_model.embedding_cache()`.

---

## Data flow

```
transactions-synthetic.csv  (68,000 raw rows)
   │  prepare_dataset.py: normalise, deduplicate to 31,863, two splits
   ▼
stratified_{train,validation,test}.csv      merchant_{...}.csv
   │                                          (merchant-held-out protocol)
   │  build_demo_data.py: BLS amounts + dates + accounts, from TEST only
   ▼
demo_transactions.csv  (772 rows, Oct 2025 – Sep 2026)
   │  build_feeds.py: split by statement format
   ▼
feeds/chase_activity.csv, capitalone_transactions.csv, applecard_*.csv (x12)
   │  adapters.py: parse back to canonical
   ▼
CanonicalTransaction  →  cascade.predict()  →  explainability.explain()
```

**Critical invariant:** amounts, dates and account ids are **never model features**.
Every metric comes from text and labels alone.

---

## Commands

```bash
# run everything with this interpreter
~/.venvs/oct14/bin/python scripts/<script>.py

# rebuild the data pipeline end to end
python scripts/prepare_dataset.py --input dataset/US/transactions-synthetic.csv \
       --outdir dataset/processed
python scripts/build_demo_data.py
python scripts/build_feeds.py
python scripts/verify_feeds.py

# the app, once built (serves pages AND api; no separate frontend server)
~/.venvs/oct14/bin/uvicorn backend.app:app --reload --port 8000
```

---

## Open decisions

- Fuzzy merchant matching (currently exact, so coverage is a lower bound)
- Fine-tuning the encoder rather than freezing it
- Model 6 (neural head) is written but never run
- A personal RAG chatbot, floated as a stretch extension only
- Connecting one feed to the Open Bank Project sandbox, which would make the
  multi-account claim genuinely external rather than self-referential

