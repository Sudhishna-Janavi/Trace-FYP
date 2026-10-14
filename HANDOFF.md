# Handoff — Integrated Finance Dashboard (CCDS25-1131)

Everything a fresh session needs to continue this project. Read this first.

**Working directory:** `/Users/sudhishna/Desktop/FYP_Final_ver2/oct14`
**Python:** `~/.venvs/oct14/bin/python` — **not** a venv inside the project (see Gotchas)
**Deadline:** 16 October 2026. Units 1–4.5 are done; Unit 5 (the web app) remains.

---

## 1. What this project is

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

## 2. Working agreements with the user

These matter as much as the code.

- **Never start building until the user types "go".** Propose, then stop. They
  said so explicitly after work began on implied approval.
- They work **unit by unit** and verify each with their own eyes before the next.
- They want **design decisions and their justifications recorded** as the work
  happens, for the report.
- Be honest about what is real versus simulated. Several findings in this project
  came from admitting a measurement was wrong rather than defending it.

---

## 3. Where things stand

| Unit | What | State |
|---|---|---|
| 1 | Data preparation: normalise, deduplicate, two leakage-safe splits | done |
| 2 | Six models compared, cascade evaluated, final test run | done |
| 3 | BLS-calibrated amounts, dates, accounts | done |
| 4 | Three heterogeneous bank feeds + adapters | done |
| 4.5 | Layer-aware explanations, validated | done |
| **5** | **The web app** | **not started** |
| — | Model 6 (neural net head) written, never run | optional |
| — | Report rewrite | not started |

---

## 4. The headline numbers

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

## 5. Directory

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
│   ├── US/transactions-synthetic.csv       raw, 68,000 rows
│   ├── Indian/                             second source, unused so far
│   ├── processed/                          6 split CSVs + dataset_meta.json
│   └── demo/
│       ├── demo_transactions.csv           772 rows, the app's data
│       ├── amount_model.json               BLS figures + provenance
│       ├── ground_truth.csv                labels, never read by adapters
│       └── feeds/                          14 files in 3 formats + manifest
├── results/                        JSON per unit + figures/ (30+ PNGs)
└── docs/                           9 .docx deliverables + mockup/
```

**Embeddings cache lives OUTSIDE the project** at `~/.cache/oct14-embeddings`
(see Gotchas). Resolve it with `evaluate_model.embedding_cache()`.

---

## 6. Data flow

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

## 7. Unit 5 — what to build

### Stack (decided)
FastAPI + Jinja2 templates + vanilla CSS/JS. **No React, no build step.**
SQLAlchemy with `DATABASE_URL`: SQLite locally, PostgreSQL deployed.
Auth: session cookies + bcrypt via `passlib`.

### Schema
```
users         id, email, password_hash, created_at, theme
accounts      id, user_id, bank, name, source_format
transactions  id, user_id, account_id, raw_description, normalised_description,
              merchant_fingerprint, amount, booking_date, predicted_category,
              confidence, decided_by_layer, needs_review, reviewed_at,
              corrected_category
corrections   id, user_id, merchant_fingerprint, category, created_at
              ^ this table IS the user-correction memory layer
```
**Every transaction is scoped by `user_id`.** Users never see each other's data.

### Pages
| Page | Purpose |
|---|---|
| Landing | what it does, sign in / sign up |
| Sign up / Sign in | email + password, bcrypt |
| Onboarding | "Connect bank data" (loads the 3 feeds) or "Upload statements" |
| Dashboard | KPIs, 12-month chart, category breakdown, recurring |
| Transactions | filterable ledger, mono descriptions, tabular amounts |
| Explanation panel | **layer-aware**: exact for rule/memory, SHAP for model |
| Review queue | low-confidence items; confirm or correct -> corrections table |
| Insights | recurring, unusual amounts, per-account |
| Accounts | the 3 banks and their actual export formats |
| How it works | cascade diagram + **all three protocol scores** |

### Design system
Mockup at `docs/mockup/dashboard.html` — **follow it.** Inter for UI,
JetBrains Mono for bank descriptions (they are machine output, so mono is
honest). One accent, hairline borders, no gradients, no emoji. Light and dark
via CSS custom properties on `:root` and `[data-theme="dark"]`.

### Build order
1. schema + auth  2. ingestion  3. API  4. design tokens + base template
5. dashboard + transactions + explanation panel  6. review queue + corrections
7. insights, accounts, how-it-works  8. deploy (Render or Railway + Postgres)

Estimated 7–8 hours, plus 1–2 for deployment.

---

## 8. Gotchas learned the hard way

**Do not put a virtualenv inside this project.** `~/Desktop` is iCloud-synced.
A 733 MB venv there made `fileproviderd` saturate the disk and every Python
import hung — including imports that had worked seconds earlier. The venv lives
at `~/.venvs/oct14` and the embeddings cache at `~/.cache/oct14-embeddings`.

**sentence_transformers hangs for ~155 s without offline flags.** They are set
inside `models/minilm_embeddings.py` before the import. Do not import
sentence_transformers before that module.

**The MiniLM snapshot is NOT in this project.** `_snapshot_path()` falls back to
`../codebase/.cache/huggingface/`. Copy it in before submitting.

**Three bugs of the same shape have occurred.** Each was a comparison that
silently measured the wrong thing while its assertions passed:
1. the merchant split asserted zero fingerprint overlap while Starbucks sat on
   both sides
2. the merchant split starved Rent of all validation rows, so F1 = 0 meant
   "nothing to score", not "the model failed"
3. token attributions compared uppercase text tokens against lowercase
   vectoriser feature names, giving a top-3 overlap of exactly 0.0000
**A metric that is exactly 0 or exactly 1 across many samples is a bug, not a
finding.** Check the units match before believing a comparison.

---

## 9. Commands

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

## 10. Open decisions

- Fuzzy merchant matching (currently exact, so coverage is a lower bound)
- Fine-tuning the encoder rather than freezing it
- Model 6 (neural head) is written but never run
- A personal RAG chatbot, floated as a stretch extension only
- Connecting one feed to the Open Bank Project sandbox, which would make the
  multi-account claim genuinely external rather than self-referential


## Unit 5 build status (14 Oct 2026)

| Phase | Scope | State |
|---|---|---|
| 1 | Shell, accounts, sessions, themes | done |
| 2 | Onboarding, adapters, ingestion | done |
| 3 | Navigation, placeholders, model card | done |
| 4 | Dashboard, ledger, explanation panel | done |
| 5 | Review queue and corrections | done |
| 6 | Insights, Accounts, How it works | done |
| 7 | Polish, empty states, edge cases | done |
| 8 | Deploy (set `TRACE_SECRET_KEY`) | to do |

### Demo data corrections made in Phase 5

Three faults were found in `scripts/build_demo_data.py` and fixed:

1. **Income was gross, not take-home.** The BLS line used is "average income
   before taxes", which cannot be deposited into a current account. BLS
   discontinued its "income after taxes" line with the 2024 release, so the
   2024 before-tax figure is scaled by the last ratio BLS published itself
   (2023: 87,869 / 101,805 = 0.8631). `AFTER_TAX_RATIO` in the generator.
2. **Recurring categories did not reconcile to their BLS annual figure.** The
   number of payments came from `merchants x MONTHS` while the size of each
   came from `annual / per_year`, and nothing made the two agree. Insurance
   emitted 2.27x its BLS total, Subscription 1.46x. Bases are now drawn and
   then rescaled so each recurring category sums to its BLS figure; the
   generator asserts the drift is under 2%.
3. **A single draw set a whole year.** With one merchant (rent, salary) one
   random draw governed twelve months, and the clamp pinned income at its
   +25% ceiling. The rescale in (2) removes this.

One-off categories are deliberately left unreconciled: a household's restaurant
spending genuinely varies from the national average.

Resulting dashboard: income $89,922, spending $61,704, net +$28,217.

The remaining surplus is honest and worth stating in the report as a
limitation: the 15 categories the classifier knows do not cover everything a
household spends on (no apparel, pensions, cash contributions or
miscellaneous), so modelled spending sits below the BLS $78,535 average.

### Phase 6 notes

`backend/charts.py` is new: every chart is inline SVG built on the server, so
there is no build step, nothing is fetched from a CDN, charts render without
scripting, and the marks reference the same CSS custom properties as the page
and therefore follow the light and dark themes. The 33 matplotlib PNGs in
`results/figures/` are deliberately **not** reused in the app: they are the
right artefacts for the written report but are white-background rasters of
research results, which would look broken in dark theme.

Two honesty caveats are printed on the Insights page itself, conditional on the
data being the generated demonstration set (`_is_demo` in `app.py`):

- The **BLS benchmark comparison is circular** for demo data, because the demo
  amounts were generated from those same survey lines. Agreement is arithmetic,
  not evidence; the small differences that remain are the classifier placing a
  transaction in the wrong category. It becomes meaningful on real uploads.
- **Month-to-month movement is sampling noise** in demo data, because one-off
  purchases fall on random dates rather than reflecting a change in habit.

The spending signals are deliberately descriptive and carry no score. Any
financial-health rating would need a threshold, and every threshold available
here would be invented. Trace does not give financial advice.

A partial month is never compared against a complete one
(`analytics.complete_months`), which would otherwise invent a fall in spending
for any user looking mid-month.

### Phase 7 notes

**A crash on every brand-new account.** `/insights` returned 500 for any user
with fewer than two complete months, because the `monthname` filter was called
inside the page's script block, outside the `{% if change.available %}` guard
protecting the rest of that section. A marker creating a fresh account would
have hit it on their first visit. The donut, benchmark and signal sections are
now guarded too, and the empty case offers a link to add data.

**Deploy blockers, all fixed before Phase 8.**

- `TRACE_SECRET_KEY` now makes the application **refuse to start** when a
  deployment environment is detected and the key is still the development
  default. The default is published in this repository, and anyone holding it
  can mint a session cookie for any account; a quiet fallback would have made
  that failure invisible.
- Session cookies gain `secure=True` when deployed, so they never travel over
  plain HTTP.
- `@app.on_event("startup")` replaced with a lifespan handler, and
  `datetime.utcnow()` with an explicit UTC timestamp. Both are deprecated.

**The ledger silently truncated.** It capped at 250 of 772 rows with no way to
reach the rest. Now paginated at 100 per page, with filters carried in the
links and out-of-range pages clamped rather than erroring.

**Accessibility.** A skip link over the six nav items, visible `:focus-visible`
outlines, clickable table rows reachable and operable by keyboard with labels
naming what they do, the explanation panel marked as a dialog that takes focus
and hands it back on close, and `prefers-reduced-motion` honoured.

**Themed 404 and 500 pages**, with API paths answering JSON and page paths
answering a page. The 500 says the data was not changed and leaks no traceback.

### Two limitations left documented rather than fixed

Both concern `fingerprint()` in `models/merchant_memory.py`, measured on the
772-row demonstration set:

1. **6 transactions (0.8%) produce an empty fingerprint** — every token was
   bank noise, a processor prefix or a reference code, as in
   `[debit] BP*86NSEW6Y3PW`. Merchant memory can never learn these and a
   correction cannot be stored against them. The correction path guards against
   the empty key, so this degrades rather than breaks.
2. **74 brands split across more than one fingerprint**, because the key is the
   first two meaningful tokens and the second is often a city: `SAFEWAY`,
   `SAFEWAY BOSTON` and `SAFEWAY COLUMBUS` are three separate merchants to the
   system. Memory and corrections therefore fragment across branches of the
   same chain. A fuzzy or prefix match would raise coverage; exact matching was
   chosen so the baseline answers if and only if it has seen the merchant
   before, which is noted in that module's docstring.

### Running the app

```bash
cd /Users/sudhishna/Desktop/FYP_Final_ver2/oct14
~/.venvs/oct14/bin/python -m uvicorn backend.app:app --reload --port 8000
```

Test account: `sj@ntu.edu.sg`. 772 transactions, 3 accounts, 46 flagged for review.
