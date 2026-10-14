# Deploying Trace

## What the measurements say

The application runs the encoder as an **ONNX graph under onnxruntime, not
PyTorch**. Measured locally, with the encoder loaded and inference run:

| Stage | PyTorch | ONNX Runtime |
|---|---:|---:|
| Application imported | 67 MB | 66 MB |
| Merchant memory loaded | 72 MB | 71 MB |
| Encoder loaded | 401 MB | 107 MB |
| Peak, under explanation workload | 401 MB | **314 MB** |

The weights are the same 86 MB either way. PyTorch was the cost: inference here
is a fixed sequence of matrix multiplications with no training, no autograd and
no dynamic graph, so its runtime was paying rent it did not earn. Removing it
also strips roughly a gigabyte of wheels from the build.

**This was verified before it was adopted.** `scripts/verify_onnx.py` compares
both engines on 1,500 held-out transactions:

- embeddings: minimum cosine similarity 0.99999988, maximum absolute
  difference 2.5e-07 — float32 rounding, nothing more
- class probabilities: maximum absolute difference 2.7e-06
- **predicted category differs on 0 of 1,500; accept-or-review differs on 0**
- accuracy identical to six decimal places

Re-importing the full demonstration set through the live cascade changed the
category or review flag on **0 of 772** transactions. Every figure the report
quotes still describes the system that is deployed.

Explanation latency was re-measured rather than carried over: Shapley values
take a median of 51 ms under ONNX against 31 ms under PyTorch. The "How it
works" page quotes the ONNX figure, since that is what runs.

314 MB leaves roughly 200 MB of headroom on a 512 MB instance, which is enough
to be safe. The build context is 5 MB and the model artefacts 304 KB.

## Before anything else

Generate a secret key. The application **refuses to start** in a deployment
environment without one, because the development fallback is published in this
repository and would let anyone forge a session cookie:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `TRACE_SECRET_KEY` | **yes** | Signs session cookies. Startup fails without it. |
| `TRACE_ENV` | yes | Set to `production`. Enables the key check and `secure` cookies. |
| `DATABASE_URL` | no | Postgres if set, SQLite file otherwise. `postgres://` is rewritten automatically. |
| `TRACE_DEMO_PASSWORD` | recommended | Seeds a demonstration account on first boot, so the URL opens on a working dashboard rather than an empty form. |
| `TRACE_DEMO_EMAIL` | no | Defaults to `demo@trace.app`. |
| `PORT` | no | Supplied by the host. |

Without `TRACE_DEMO_PASSWORD` nothing is seeded and the link opens on a sign-in
form, which is not useful to someone you sent it to.

## Storage

SQLite works and needs nothing, but most hosts give a container an ephemeral
filesystem: **every redeploy would wipe every account and correction.** Attach
Postgres and set `DATABASE_URL` if the data must survive a deploy. For a
demonstration that is reseeded on each boot anyway, SQLite is acceptable.

## Deploying with Docker

The `Dockerfile` runs anywhere that takes a container and bakes the encoder in
at build time, so a missing model fails the build rather than the first
request.

```bash
docker build -t trace .
docker run -p 8000:7860 \
  -e TRACE_ENV=production \
  -e TRACE_SECRET_KEY=... \
  -e TRACE_DEMO_PASSWORD=... \
  trace
```

## Notes

- **One worker.** Each uvicorn worker loads its own copy of the encoder, so a
  second worker doubles the memory. `Procfile` and `Dockerfile` both pin it.
- **First request after a cold start is slow.** The encoder loads on a
  background thread at boot, which takes a few seconds; a host that sleeps the
  instance pays that again on wake. Open the URL yourself shortly before anyone
  else does.
- **Schema changes do not migrate.** `create_all()` creates missing tables, not
  missing columns. Adding a field to a model means altering the live table by
  hand or recreating the database.
