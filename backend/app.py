"""Trace — the application.

Phase 1: accounts and sessions. Later phases add ingestion, the dashboard,
explanations and the review queue.
"""
from __future__ import annotations

import tempfile
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi import Response
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from backend.auth import (COOKIE_NAME, COOKIE_SECURE, current_user,
                          find_by_email,
                          hash_password, issue_session, require_user,
                          verify_password)
from backend.common import (EXPLANATION_COST, LAYER_LABELS, LAYERS,
                            MIN_PASSWORD, NAV, PROTOCOL_LABELS,
                            PROTOCOL_MEANS, explanation_payload,
                            is_demo)
from backend.database import (Account, Correction, Transaction, User,
                              create_all, get_session)
from backend.ingest import ingest_for_user
from backend.uploads import clear as clear_uploads, stage as stage_uploads
from backend import analytics, api, charts, classifier, review

ROOT = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(_: FastAPI):
    create_all()
    try:
        from backend.seed import seed_demo_account
        print(f"[seed] {seed_demo_account()}", flush=True)
    except Exception as error:          # a failed seed must not stop the app
        print(f"[seed] failed: {error}", flush=True)
    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="Trace", docs_url=None, redoc_url=None,
              lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")

# The mobile application is a second presentation of this system: same
# analytics, same cascade, same explanations, returned as JSON instead of
# HTML. It is mounted here rather than run as a separate service so that
# there is one deployment, one database and one set of answers.
app.include_router(api.router)


@app.middleware("http")
async def api_cors(request: Request, call_next):
    """Allow a browser-based client to call the API, and nothing else.

    A native application is not subject to the same-origin policy, so the
    phone needs none of this. A client running in a browser does, and
    without it every request from one is refused before it is sent.

    The headers are added only under /api/v1, never to the pages, and
    credentials are deliberately not allowed: this API authenticates with a
    bearer token the caller must already hold, not with an ambient cookie, so
    a page on another origin gains nothing by being able to call it. It
    cannot read the token, and without the token every answer is a 401.
    """
    if not request.url.path.startswith("/api/v1"):
        return await call_next(request)

    if request.method == "OPTIONS":
        response = Response(status_code=204)
    else:
        response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
    response.headers["Access-Control-Max-Age"] = "600"
    return response

# Money is shown to the reader, so formatting belongs in one place rather than
# repeated across templates.
templates.env.filters["money"] = lambda v: (
    ("\u2212" if v < 0 else "") + "$" + f"{abs(v):,.2f}")
templates.env.filters["money0"] = lambda v: (
    ("\u2212" if v < 0 else "") + "$" + f"{abs(v):,.0f}")

# A corrected row must be filtered under the category the user sees, not the
# one the model first guessed.
EFFECTIVE_CATEGORY = func.coalesce(Transaction.corrected_category,
                                   Transaction.predicted_category)

templates.env.filters["monthname"] = lambda key: analytics._month_name(key)


def page_context(request, user: User, session: Session, active: str) -> dict:
    """Everything the shell needs, on every signed-in page."""
    review_count = session.scalar(
        select(func.count()).select_from(Transaction)
        .where(Transaction.user_id == user.id,
               Transaction.needs_review.is_(True),
               Transaction.reviewed_at.is_(None))) or 0
    return {"user": user, "theme": user.theme, "nav_items": NAV,
            "active": active, "review_count": review_count,
            "correctable": sorted(classifier.get_model_card()["classes"])}


def placeholder(request, user, session, active, title, blurb, detail, phase):
    return templates.TemplateResponse(request, "placeholder.html", {
        **page_context(request, user, session, active),
        "title": title, "blurb": blurb, "detail": detail, "phase": phase,
    })


def _warm() -> None:
    """Load the encoder and run one forward pass before anyone asks for it.

    Done on a background thread so the server still answers immediately. The
    first sentence through a freshly loaded encoder costs several hundred
    milliseconds that have nothing to do with the explanation being computed,
    and paying that here keeps the timings the panel reports honest.
    """
    try:
        import models.rule_baseline  # noqa: F401
        classifier.get_memory()
        classifier.get_model().predict_proba(["WARMUP"])
    except Exception:  # a cold model is a slow first click, not a failure
        pass


def _see_other(path: str) -> RedirectResponse:
    """303 so that a POST redirects to a GET rather than re-posting."""
    return RedirectResponse(path, status_code=303)


# ------------------------------------------------------------------ pages ---

@app.get("/", response_class=HTMLResponse)
def index(request: Request, user: User | None = Depends(current_user),
          session: Session = Depends(get_session)):
    if user is None:
        return _see_other("/signin")
    return _home(request, user, session)


def _home(request: Request, user: User, session: Session):
    accounts = session.scalar(
        select(func.count()).select_from(Account).where(Account.user_id == user.id))
    transactions = session.scalar(
        select(func.count()).select_from(Transaction)
        .where(Transaction.user_id == user.id))
    # a user with no data has nothing to show, so send them to onboarding
    if not transactions:
        return _see_other("/connect")
    rows = analytics.load_transactions(session, user)
    totals = analytics.totals(rows, accounts or 0)
    months = analytics.monthly(rows)
    recurring = analytics.recurring(rows)
    hour = datetime.now().hour
    greeting = ("Good morning" if hour < 12
                else "Good afternoon" if hour < 18 else "Good evening")
    return templates.TemplateResponse(request, "dashboard.html", {
        **page_context(request, user, session, "dashboard"),
        "t": totals, "months": months, "chart": charts.cashflow(months),
        "categories": analytics.by_category(rows), "recurring": recurring,
        "recent": rows[:8], "greeting": greeting,
    })


@app.get("/connect", response_class=HTMLResponse)
def connect_form(request: Request, user: User = Depends(require_user)):
    return templates.TemplateResponse(request, "onboarding.html",
                                      {"user": user, "theme": user.theme})


@app.post("/connect", response_class=HTMLResponse)
def connect(request: Request, user: User = Depends(require_user),
            session: Session = Depends(get_session)):
    result = ingest_for_user(session, user)
    if not result.transactions:
        return templates.TemplateResponse(request, "onboarding.html", {
            "user": user, "theme": user.theme,
            "error": "No statement files were found to import.",
        }, status_code=500)
    return templates.TemplateResponse(request, "connected.html", {
        "user": user, "theme": user.theme, "result": result,
        "layers": [l for l in LAYERS if result.by_layer.get(l[0])],
    })


@app.post("/connect/upload", response_class=HTMLResponse)
async def connect_upload(request: Request,
                         statements: list[UploadFile] = File(default=[]),
                         user: User = Depends(require_user),
                         session: Session = Depends(get_session)):
    """Import statement files the user chose themselves.

    The files are recognised by their columns rather than their names, staged
    under the names the adapters expect, and then handed to exactly the same
    ingestion path the demonstration button uses. Staging happens in a
    temporary directory that is removed whatever the outcome: an upload is
    one user's data and has no business persisting on the server's disk.
    """
    payload = [(f.filename or "file.csv", await f.read()) for f in statements]
    if not payload:
        return templates.TemplateResponse(request, "onboarding.html", {
            "user": user, "theme": user.theme,
            "error": "No files were chosen.",
        }, status_code=400)

    staging = Path(tempfile.mkdtemp(prefix="trace-upload-"))
    try:
        staged = stage_uploads(payload, staging)
        if not staged.files:
            detail = "; ".join(f"{name} — {why}" for name, why in staged.rejected[:3])
            return templates.TemplateResponse(request, "onboarding.html", {
                "user": user, "theme": user.theme,
                "error": ("None of those files looked like a supported bank "
                          "export. " + detail),
            }, status_code=400)
        try:
            result = ingest_for_user(session, user, staging)
        except Exception as error:
            # A malformed row should say which file and why, not present the
            # generic failure page in the middle of someone's import.
            session.rollback()
            return templates.TemplateResponse(request, "onboarding.html", {
                "user": user, "theme": user.theme,
                "error": f"Those files could not be read: {error}",
            }, status_code=400)
    finally:
        clear_uploads(staging)

    if not result.transactions:
        return templates.TemplateResponse(request, "onboarding.html", {
            "user": user, "theme": user.theme,
            "error": "Those files held no transactions.",
        }, status_code=400)

    return templates.TemplateResponse(request, "connected.html", {
        "user": user, "theme": user.theme, "result": result,
        "layers": [l for l in LAYERS if result.by_layer.get(l[0])],
        "uploaded": staged,
    })


@app.get("/signup", response_class=HTMLResponse)
def signup_form(request: Request, user: User | None = Depends(current_user)):
    if user is not None:
        return _see_other("/")
    return templates.TemplateResponse(request, "auth.html", {
        "mode": "signup", "heading": "Create your account",
        "blurb": "Trace categorises your transactions and shows you why.",
        "action": "/signup", "cta": "Create account", "form": {},
        "switch_text": "Already have an account?", "switch_href": "/signin",
        "switch_cta": "Sign in",
    })


@app.post("/signup", response_class=HTMLResponse)
def signup(request: Request, display_name: str = Form(""),
           email: str = Form(""), password: str = Form(""),
           session: Session = Depends(get_session)):
    email = email.lower().strip()
    display_name = display_name.strip()
    error = None
    if not display_name:
        error = "Please enter your name."
    elif "@" not in email or "." not in email.split("@")[-1]:
        error = "Please enter a valid email address."
    elif len(password) < MIN_PASSWORD:
        error = f"Choose a password of at least {MIN_PASSWORD} characters."
    elif find_by_email(session, email):
        error = "An account already exists with that email."

    if error:
        return templates.TemplateResponse(request, "auth.html", {
            "mode": "signup", "heading": "Create your account",
            "blurb": "Trace categorises your transactions and shows you why.",
            "action": "/signup", "cta": "Create account", "error": error,
            "form": {"display_name": display_name, "email": email},
            "switch_text": "Already have an account?", "switch_href": "/signin",
            "switch_cta": "Sign in",
        }, status_code=400)

    user = User(email=email, display_name=display_name,
                password_hash=hash_password(password))
    session.add(user)
    session.commit()

    response = _see_other("/")
    response.set_cookie(COOKIE_NAME, issue_session(user.id), httponly=True,
                        samesite="lax", secure=COOKIE_SECURE,
                        max_age=60 * 60 * 24 * 14)
    return response


@app.get("/signin", response_class=HTMLResponse)
def signin_form(request: Request, user: User | None = Depends(current_user)):
    if user is not None:
        return _see_other("/")
    return templates.TemplateResponse(request, "auth.html", {
        "mode": "signin", "heading": "Sign in",
        "blurb": "Welcome back.", "action": "/signin", "cta": "Sign in",
        "form": {}, "switch_text": "New here?", "switch_href": "/signup",
        "switch_cta": "Create an account",
    })


@app.post("/signin", response_class=HTMLResponse)
def signin(request: Request, email: str = Form(""), password: str = Form(""),
           session: Session = Depends(get_session)):
    user = find_by_email(session, email)
    # the same message either way, so the form cannot be used to discover
    # which email addresses have accounts
    if user is None or not verify_password(password, user.password_hash):
        return templates.TemplateResponse(request, "auth.html", {
            "mode": "signin", "heading": "Sign in", "blurb": "Welcome back.",
            "action": "/signin", "cta": "Sign in",
            "error": "That email and password do not match an account.",
            "form": {"email": email}, "switch_text": "New here?",
            "switch_href": "/signup", "switch_cta": "Create an account",
        }, status_code=401)

    response = _see_other("/")
    response.set_cookie(COOKIE_NAME, issue_session(user.id), httponly=True,
                        samesite="lax", secure=COOKIE_SECURE,
                        max_age=60 * 60 * 24 * 14)
    return response


@app.post("/signout")
def signout():
    response = _see_other("/signin")
    response.delete_cookie(COOKIE_NAME)
    return response


# ------------------------------------------------------------ app pages ---

@app.get("/transactions", response_class=HTMLResponse)
def transactions_page(request: Request, q: str = "", category: str = "",
                      account: str = "", state: str = "", page: int = 1,
                      user: User = Depends(require_user),
                      session: Session = Depends(get_session)):
    PER_PAGE = 100
    page = max(page, 1)
    stmt = select(Transaction).where(Transaction.user_id == user.id)
    if q:
        stmt = stmt.where(Transaction.raw_description.ilike(f"%{q}%"))
    if category:
        stmt = stmt.where(EFFECTIVE_CATEGORY == category)
    if account:
        stmt = stmt.where(Transaction.account_id == int(account))
    if state == "review":
        stmt = stmt.where(Transaction.needs_review.is_(True),
                          Transaction.reviewed_at.is_(None))
    elif state == "confirmed":
        stmt = stmt.where(Transaction.needs_review.is_(False))

    matching = session.scalar(
        select(func.count()).select_from(stmt.subquery())) or 0
    pages = max(1, -(-matching // PER_PAGE))
    page = min(page, pages)
    rows = list(session.scalars(
        stmt.options(selectinload(Transaction.account))
        .order_by(Transaction.booking_date.desc())
        .offset((page - 1) * PER_PAGE).limit(PER_PAGE)))
    total = session.scalar(
        select(func.count()).select_from(Transaction)
        .where(Transaction.user_id == user.id)) or 0
    accounts = list(session.scalars(
        select(Account).where(Account.user_id == user.id).order_by(Account.bank)))
    all_categories = sorted(c for (c,) in session.execute(
        select(EFFECTIVE_CATEGORY)
        .where(Transaction.user_id == user.id).distinct()))

    return templates.TemplateResponse(request, "transactions.html", {
        **page_context(request, user, session, "transactions"),
        "rows": rows, "matching": matching, "total": total,
        "accounts": accounts, "all_categories": all_categories,
        "q": q, "category": category, "account": account, "state": state,
        "filtered": bool(q or category or account or state),
        "layer_labels": LAYER_LABELS,
        "page": page, "pages": pages, "per_page": PER_PAGE,
        "first_row": (page - 1) * PER_PAGE + 1 if matching else 0,
        "last_row": min(page * PER_PAGE, matching),
        "query_string": urlencode(
            {k: v for k, v in (("q", q), ("category", category),
                               ("account", account), ("state", state)) if v}),
    })


@app.get("/review", response_class=HTMLResponse)
def review_page(request: Request, user: User = Depends(require_user),
                session: Session = Depends(get_session)):
    """Everything the cascade was not confident enough to decide alone.

    The gate at 0.60 exists so that the model declines rather than guesses.
    This is where those declined rows are handed to the person who actually
    knows the answer.
    """
    rows = list(session.scalars(
        select(Transaction)
        .options(selectinload(Transaction.account))
        .where(Transaction.user_id == user.id,
               Transaction.needs_review.is_(True),
               Transaction.reviewed_at.is_(None))
        .order_by(Transaction.confidence.asc())))
    reviewed = session.scalar(
        select(func.count()).select_from(Transaction)
        .where(Transaction.user_id == user.id,
               Transaction.reviewed_at.is_not(None))) or 0
    corrections = list(session.scalars(
        select(Correction).where(Correction.user_id == user.id)
        .order_by(Correction.created_at.desc())))
    return templates.TemplateResponse(request, "review.html", {
        **page_context(request, user, session, "review"),
        "rows": rows, "reviewed": reviewed, "corrections": corrections,
        "categories": sorted(classifier.get_model_card()["classes"]),
        "threshold": classifier.REVIEW_THRESHOLD,
        "layer_labels": LAYER_LABELS,
    })


@app.post("/api/transactions/{transaction_id}/confirm")
def confirm_transaction(transaction_id: int,
                        user: User = Depends(require_user),
                        session: Session = Depends(get_session)):
    row = _owned(session, user, transaction_id)
    if row is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(asdict(review.confirm(session, user, row)))


@app.post("/api/transactions/{transaction_id}/category")
def correct_transaction(transaction_id: int, category: str = Form(...),
                        user: User = Depends(require_user),
                        session: Session = Depends(get_session)):
    row = _owned(session, user, transaction_id)
    if row is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    if category not in classifier.get_model_card()["classes"]:
        return JSONResponse({"error": "unknown category"}, status_code=400)
    return JSONResponse(asdict(review.correct(session, user, row, category)))


def _owned(session: Session, user: User, transaction_id: int):
    """The transaction, but only if it belongs to this user."""
    row = session.get(Transaction, transaction_id)
    return row if row is not None and row.user_id == user.id else None


@app.get("/insights", response_class=HTMLResponse)
def insights_page(request: Request, user: User = Depends(require_user),
                  session: Session = Depends(get_session)):
    rows = analytics.load_transactions(session, user)
    categories = analytics.by_category(rows)
    wheel, segments = charts.donut(categories)
    recurring = analytics.recurring(rows)
    change = analytics.changes(rows)
    return templates.TemplateResponse(request, "insights.html", {
        **page_context(request, user, session, "insights"),
        "categories": categories, "wheel": wheel, "segments": segments,
        "recurring": recurring,
        "unusual": analytics.unusual(rows, {r["merchant"] for r in recurring}),
        "change": change,
        "change_chart": charts.diverging(change["rows"], "vs_average"),
        "change_chart_prev": charts.diverging(change["rows"], "vs_previous"),
        "signals": analytics.signals(rows),
        "benchmark": analytics.benchmark(rows),
        "methodology": analytics.METHODOLOGY,
        "is_demo": is_demo(session, user),
    })


@app.get("/accounts", response_class=HTMLResponse)
def accounts_page(request: Request, user: User = Depends(require_user),
                  session: Session = Depends(get_session)):
    rows = analytics.load_transactions(session, user)
    accounts = list(session.scalars(
        select(Account).where(Account.user_id == user.id).order_by(Account.bank)))
    spend = {a["bank"]: a for a in analytics.by_account(session, user, rows)}
    detail = []
    for account in accounts:
        mine = [r for r in rows if r.account_id == account.id]
        dates = sorted(r.booking_date for r in mine)
        detail.append({
            "account": account, "rows": len(mine),
            "first": dates[0] if dates else None,
            "last": dates[-1] if dates else None,
            "spend": spend.get(account.bank, {}).get("amount", 0.0),
            "share": spend.get(account.bank, {}).get("share", 0.0),
        })
    return templates.TemplateResponse(request, "accounts.html", {
        **page_context(request, user, session, "accounts"),
        "detail": detail, "total": len(rows),
    })


@app.get("/how-it-works", response_class=HTMLResponse)
def how_page(request: Request, user: User = Depends(require_user),
             session: Session = Depends(get_session)):
    """The cascade, and how accurate it actually is.

    Reporting only the in-distribution figure would be true and misleading at
    once, so all three protocols are shown together.
    """
    card = classifier.get_model_card()
    rows = analytics.load_transactions(session, user)
    flagged = sum(1 for r in rows if r.needs_review and not r.reviewed_at)
    return templates.TemplateResponse(request, "how.html", {
        **page_context(request, user, session, "how"),
        "card": card, "performance": card["performance"],
        "labels": PROTOCOL_LABELS, "means": PROTOCOL_MEANS,
        "chart": charts.protocols(card["performance"], PROTOCOL_LABELS),
        "layers": analytics.layer_counts(rows),
        "total": len(rows), "flagged": flagged,
        "observed_rate": flagged / len(rows) * 100 if rows else 0.0,
        "explain_cost": EXPLANATION_COST,
    })


@app.get("/api/transactions/{transaction_id}/explanation")
def explanation(transaction_id: int, method: str = "shap",
                user: User = Depends(require_user),
                session: Session = Depends(get_session)):
    """Why one transaction carries the category it does.

    The explanation that is honest depends on which cascade layer decided.
    Three of the four layers know their own reasons exactly and cost nothing
    to explain; only the model needs a post-hoc approximation, and only there
    is the encoder called at all.
    """
    row = _owned(session, user, transaction_id)
    if row is None:
        # the same answer either way, so a probe cannot tell a missing row
        # from one belonging to somebody else
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(explanation_payload(row, method))


@app.post("/theme")
def set_theme(request: Request, theme: str = Form("system"),
              user: User = Depends(require_user),
              session: Session = Depends(get_session)):
    if theme in ("light", "dark", "system"):
        user.theme = theme
        session.commit()
    return {"theme": user.theme}


@app.exception_handler(404)
def not_found(request: Request, exc):
    """A missing page should still look like the application.

    An API path gets JSON, because something is parsing that response; a page
    gets a page.
    """
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": "not found"}, status_code=404)
    return templates.TemplateResponse(request, "error.html", {
        "theme": "system", "title": "Page not found",
        "message": "That address does not lead anywhere. It may have been "
                   "mistyped, or the page may have moved.",
        "back": "/", "back_label": "Back to your dashboard",
    }, status_code=404)


@app.exception_handler(500)
def server_error(request: Request, exc):
    """Apologise without leaking a traceback to whoever asked."""
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": "server error"}, status_code=500)
    return templates.TemplateResponse(request, "error.html", {
        "theme": "system", "title": "Something went wrong",
        "message": "Trace could not finish that request. Your data has not "
                   "been changed. Trying again often works.",
        "back": "/", "back_label": "Back to your dashboard",
    }, status_code=500)


# ----------------------------------------------- installing on a phone ---
# The service worker and the manifest are served from the root rather than
# from /static, because a worker's scope cannot rise above the path it was
# served from: at /static/sw.js it could only control /static.

@app.get("/sw.js", include_in_schema=False)
def service_worker():
    return FileResponse(ROOT / "static" / "sw.js", media_type="text/javascript",
                        headers={"Cache-Control": "no-cache",
                                 "Service-Worker-Allowed": "/"})


@app.get("/manifest.webmanifest", include_in_schema=False)
def manifest():
    return FileResponse(ROOT / "static" / "manifest.webmanifest",
                        media_type="application/manifest+json")


@app.get("/offline", response_class=HTMLResponse, include_in_schema=False)
def offline(request: Request):
    """Precached at install, so there is something to show with no network."""
    return templates.TemplateResponse(request, "error.html", {
        "theme": "system", "title": "You are offline",
        "message": "Trace needs a connection to read your transactions. "
                   "This page was stored on your device so that something "
                   "sensible appears until one returns.",
        "back": "/", "back_label": "Try again",
    })


@app.get("/healthz")
def healthz(session: Session = Depends(get_session)):
    users = session.scalar(select(func.count()).select_from(User)) or 0
    return {"status": "ok", "users": users}
