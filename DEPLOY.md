# Deploying Trace

## What the measurements say

Memory, measured locally with the encoder loaded and inference run:

| Stage | Resident |
|---|---:|
| Application imported | 67 MB |
| Merchant memory loaded | 72 MB |
| **Encoder loaded** | **401 MB** |
| After inference | 401 MB |

Limiting torch to one thread did not help (408 MB), so the encoder's ~330 MB is
structural rather than an allocator artefact.

**This rules out a 512 MB instance as a safe target.** Render's free and
Starter tiers are both 512 MB, which leaves roughly 110 MB for uvicorn, the
Postgres driver and concurrent requests. It might hold; it might be killed by
the OOM reaper on the first click your marker makes, which is the worst
possible moment to find out. Pick a host with at least 1 GB.

The build context is 5 MB and the model artefacts are 304 KB; the encoder is
~90 MB and is baked into the image at build time by `scripts/fetch_model.py`.

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
