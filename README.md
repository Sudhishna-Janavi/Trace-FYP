# Trace

An explainable multi-account personal finance dashboard. Every transaction it
categorises can tell you why.

**Live application — <https://trace-fyp.onrender.com/>**

NTU College of Computing and Data Science, Final Year Project CCDS25-1131.

---

## What it does

Bank statements exported from three institutions, in three mutually
incompatible formats, are normalised into one ledger and categorised into 17
categories. Commercial products do this too. What they do not do is explain
themselves: a user who disagrees with a category can only override it, learning
nothing about the system and teaching it nothing durable.

Trace makes every decision accountable. Select any transaction and it states
which of four layers decided it, on what evidence, and whether that reason is
exact or reconstructed after the fact.

## The cascade

Categorisation is not one model. It is four layers, cheapest first, each
answering only what the one before it could not:

| Layer | Mechanism | Explanation |
|---|---|---|
| 1. Your corrections | Categories you set yourself, keyed by merchant | Exact |
| 2. Rules | Unambiguous banking phrases — `PAYROLL`, `RENT PMT` | Exact |
| 3. Merchant memory | Merchants seen before, looked up not predicted | Exact |
| 4. The model | MiniLM sentence encoder + logistic head | Approximate |

Below 60% confidence the model declines rather than guesses, and the
transaction goes to a review queue.

The ordering is an explainability decision as much as a performance one. Three
of the four layers state their reasoning in full, so post-hoc approximation —
LIME or Shapley values — is needed only for the minority of transactions that
reach the model. On the test set that is 16.3% of them.

## Quickstart

**Python 3.12 or 3.13.** Not 3.14 — `psycopg-binary` publishes no wheel for it
and the install fails. `runtime.txt` pins 3.12.7, which is what the deployed
image uses. The steps below were run end to end against a clean clone on
macOS with 3.13.15.

```bash
git clone https://github.com/Sudhishna-Janavi/Trace-FYP.git
cd Trace-FYP
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

The sentence encoder is not in this repository (86 MB of third-party weights).
Fetch it, and keep `HF_HOME` pointing at the same place when you run the app —
the application refuses to reach the network at runtime:

```bash
export HF_HOME="$PWD/.cache/huggingface"
.venv/bin/python scripts/fetch_model.py
```

Create the tables:

```bash
.venv/bin/python scripts/init_db.py
```

Run it. `TRACE_DEMO_PASSWORD` is what seeds the demonstration account — without
it the application starts with no data and no user:

```bash
export TRACE_DEMO_PASSWORD="choose-something"
.venv/bin/python -m uvicorn --app-dir . backend.app:app --port 8000
```

First start prints what it seeded:

```
[seed] demonstration account created: 772 transactions across 3 accounts, 46 for review
```

Sign in at <http://127.0.0.1:8000/> as `demo@trace.app` with the password you
set. `TRACE_DEMO_EMAIL` and `TRACE_DEMO_NAME` override the defaults.

## Project layout

```
backend/        the web application
  app.py          routes, error handlers, startup
  classifier.py   the four-layer cascade
  review.py       confirmation and correction
  explainability.py  LIME and exact Shapley enumeration
  adapters.py     one canonical schema from three bank formats
  analytics.py    totals, recurring payments, outliers, benchmarks
  charts.py       inline SVG, rendered server-side
  database.py     SQLAlchemy models; SQLite locally, Postgres deployed
  templates/      Jinja2
models/         the classifiers compared in the report
  rule_baseline.py  merchant_memory.py  naive_bayes.py
  logistic_regression.py  linear_svm.py  minilm_embeddings.py
  onnx_encoder.py   the encoder, reimplemented for ONNX Runtime
scripts/        data preparation, evaluation, verification
results/        every number quoted below, as JSON
dataset/demo/   the statement feeds the deployed app seeds itself from
docs/           design decisions, model documentation, research log
```

## Results

Test sets were used exactly once, after all selection and tuning. Cascade with
MiniLM at threshold 0.60. Source: `results/unit2_final_test.json`.

| Protocol | n | Accuracy | Macro-F1 | Sent to review | Accuracy on what it accepted |
|---|---|---|---|---|---|
| Stratified | 4,786 | 0.9636 | 0.9681 | 5.2% | 0.9896 |
| Merchant held out | 4,780 | 0.6544 | 0.6808 | 33.8% | 0.8291 |
| Strictly novel | 2,245 | 0.5555 | 0.5636 | 42.8% | 0.7689 |

Three protocols, not one. Stratified measures a returning user whose merchants
are familiar; merchant-held-out measures a new user whose merchants are not;
strictly novel measures text sharing no brand token with training. A single
split would have reported only the first number.

Model comparison, macro-F1 on validation — `results/unit2_comparison.json`:

| Model | Stratified | Merchant held out | Strictly novel | Merchant ECE |
|---|---|---|---|---|
| Rule baseline *(17% coverage)* | 0.2924 | 0.2849 | 0.3068 | — |
| Merchant memory *(82% / 4% coverage)* | 0.8500 | 0.0752 | 0.0156 | — |
| TF-IDF + Naive Bayes | 0.9809 | 0.5903 | 0.4657 | — |
| TF-IDF + Logistic regression | 0.9976 | 0.5980 | 0.4636 | 0.0406 |
| TF-IDF + Linear SVM | **0.9984** | 0.6393 | 0.5146 | 0.1452 |
| MiniLM + Logistic regression | 0.9167 | **0.6479** | **0.5936** | **0.0360** |

No model wins outright. The SVM is the most accurate on familiar merchants and
the wrong choice here: its confidence collapses under shift (ECE 0.145), and a
confidence gate is only as good as the confidence feeding it. MiniLM is last on
familiar merchants, first on unfamiliar ones, and the best calibrated.

## Reproducing the evaluation

`results/` contains the output of every experiment, so the numbers above can be
checked against this repository as it stands.

**Re-running the experiments needs data this repository does not contain.** The
corpus is a third-party public benchmark of synthetic US bank descriptions, and
redistributing it here is neither necessary nor ours to do. To regenerate:

```bash
.venv/bin/pip install -r requirements-training.txt   # adds torch, matplotlib, pandas
.venv/bin/python scripts/prepare_dataset.py \
    --input dataset/US/transactions-synthetic.csv --outdir dataset/processed
.venv/bin/python scripts/run_model1_rules.py         # and the other run_*.py
```

`scripts/prepare_dataset.py` documents the splitting and leakage-control
procedure. `scripts/verify_onnx.py` checks the ONNX encoder against the PyTorch
pipeline it replaced.

## Deployment

Docker on Render, with Postgres on Neon. `DEPLOY.md` has the full procedure.

PyTorch is deliberately absent from `requirements.txt`. The encoder runs as an
ONNX graph instead, which cut peak memory from 401 MB to 314 MB and made a
512 MB instance viable. `scripts/verify_onnx.py` confirmed the substitution
changed none of 1,500 held-out predictions.

| Variable | Required | Purpose |
|---|---|---|
| `TRACE_SECRET_KEY` | in production | Signs session cookies. Startup fails without it |
| `DATABASE_URL` | no | Postgres if set, SQLite otherwise |
| `TRACE_DEMO_PASSWORD` | no | Seeds the demonstration account |
| `HF_HOME` | no | Where the encoder weights live |

## Security

Passwords are bcrypt-hashed; sessions are signed cookies, `Secure` and
`HttpOnly` when deployed. The application refuses to start in production with
the development secret key. Every query is scoped to the signed-in user.

`trace.db` is gitignored and has never been committed — it holds real password
hashes.

## Limitations

Stated in full in the report; the ones that affect anyone reading this code:

- **Statement upload is not implemented.** The adapter path exists and is
  exercised by the demonstration feeds, but the interface has no upload route.
- **The corpus is synthetic.** It is a third-party public benchmark, defended
  on independence and reproducibility rather than realism.
- **Merchant fingerprints fragment** on some descriptions, capping what the
  memory layer can reach.
- **No user study was conducted.** Interface decisions are argued from stated
  principles, not validated with users.

## Further reading

- `docs/RESEARCH_LOG.md` — the unit-by-unit development record, including a
  leakage error found and corrected mid-project
- `docs/design_decisions.docx` — decisions and their justifications
- `DEPLOY.md` — deployment procedure
