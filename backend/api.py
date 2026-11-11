"""The JSON API the mobile application talks to.

Why this exists

    The web application renders HTML on the server: a browser asks for a page
    and receives one. A native application cannot use that. It draws its own
    screens from native components and needs the data behind those screens,
    not a document describing them.

    So this module is a second presentation of the same system. It calls the
    same analytics, the same cascade and the same explanation code the pages
    call — `backend.common` holds everything both of them need — and differs
    only in what it hands back. Nothing here reimplements a decision.

Why the authentication is a header rather than a cookie

    A browser carries cookies automatically; a native application has no
    cookie jar and no same-site protection to rely on. It is given the same
    signed token the cookie already held, and sends it back as a bearer
    token. The signer, the key and the payload are unchanged — only the
    transport differs — so a token is worth exactly what a cookie was worth
    and no more.

Everything is scoped to the authenticated user, as in the pages: `_owned`
answers for a transaction only when it belongs to the caller, and returns the
same "not found" for a row that is missing and for one that is somebody
else's, so the API cannot be used to discover which ids exist.
"""
from __future__ import annotations

import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from fastapi import (APIRouter, Depends, File, Header, HTTPException, Query,
                     UploadFile)
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from backend import analytics, charts, classifier, common, review
from backend.auth import (find_by_email, hash_password, issue_session,
                          read_session, verify_password)
from backend.common import (EXPLANATION_COST, LAYER_LABELS, LAYERS,
                            MIN_PASSWORD, PROTOCOL_LABELS, PROTOCOL_MEANS)
from backend.database import (Account, Correction, Transaction, User,
                              get_session)
from backend.ingest import ingest_for_user
from backend.uploads import clear as clear_uploads, stage as stage_uploads

router = APIRouter(prefix="/api/v1", tags=["mobile"])

PER_PAGE = 100

# A corrected row must be filtered under the category the user sees, not the
# one the model first guessed.
EFFECTIVE_CATEGORY = func.coalesce(Transaction.corrected_category,
                                   Transaction.predicted_category)


# ----------------------------------------------------------- the caller ---

def api_user(authorization: str | None = Header(default=None),
             session: Session = Depends(get_session)) -> User:
    """The signed-in user, from the bearer token, or 401.

    The pages redirect an unauthenticated visitor to the sign-in form. An
    application wants an answer it can act on, so this says 401 and lets the
    client decide to show its own sign-in screen.
    """
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    user_id = read_session(token)
    user = session.get(User, user_id) if user_id is not None else None
    if user is None:
        raise HTTPException(status_code=401, detail="not signed in")
    return user


def _owned(session: Session, user: User, transaction_id: int) -> Transaction | None:
    row = session.get(Transaction, transaction_id)
    return row if row is not None and row.user_id == user.id else None


def _require(session: Session, user: User, transaction_id: int) -> Transaction:
    row = _owned(session, user, transaction_id)
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    return row


# --------------------------------------------------------- serialisation ---

def _user(user: User) -> dict:
    return {"id": user.id, "display_name": user.display_name,
            "email": user.email, "theme": user.theme}


def _account(account: Account) -> dict:
    return {"id": account.id, "bank": account.bank, "name": account.name,
            "source_format": account.source_format,
            "currency": account.currency, "external_id": account.external_id}


def _transaction(row: Transaction, bank: str | None = None) -> dict:
    """One ledger row, with everything a screen might show about it.

    `decided_by` and the correction are both reported rather than collapsed
    into one field: the application shows which layer answered, and a row the
    user corrected must not claim the model decided it.
    """
    layer = "correction" if row.corrected_category else row.decided_by
    return {
        "id": row.id,
        "description": row.raw_description,
        "merchant": row.merchant_fingerprint,
        "amount": row.amount,
        "date": row.booking_date,
        "account_id": row.account_id,
        "bank": bank if bank is not None else row.account.bank,
        "category": row.category,
        "predicted_category": row.predicted_category,
        "corrected_category": row.corrected_category,
        "confidence": row.confidence,
        "layer": layer,
        "layer_label": LAYER_LABELS.get(layer, layer),
        "needs_review": bool(row.needs_review and not row.reviewed_at),
        "reviewed": row.reviewed_at is not None,
    }


def _review_count(session: Session, user: User) -> int:
    return session.scalar(
        select(func.count()).select_from(Transaction)
        .where(Transaction.user_id == user.id,
               Transaction.needs_review.is_(True),
               Transaction.reviewed_at.is_(None))) or 0


def _ingest_payload(result, staged=None) -> dict:
    """The import summary, including which layer answered how much of it."""
    payload = {
        "transactions": result.transactions,
        "accounts": result.accounts,
        "files": result.files,
        "needs_review": result.needs_review,
        "by_layer": result.by_layer,
        "layers": [{"key": key, "label": label, "note": note,
                    "count": result.by_layer.get(key, 0)}
                   for key, label, note, _colour in LAYERS
                   if result.by_layer.get(key)],
    }
    if staged is not None:
        payload["uploaded"] = {
            "files": staged.files, "banks": staged.banks,
            "banks_phrase": staged.banks_phrase,
            "rejected": [{"name": name, "why": why}
                         for name, why in staged.rejected],
        }
    return payload


# ---------------------------------------------------------------- bodies ---

class SignUpBody(BaseModel):
    display_name: str = ""
    email: str = ""
    password: str = ""


class SignInBody(BaseModel):
    email: str = ""
    password: str = ""


class CategoryBody(BaseModel):
    category: str


class ThemeBody(BaseModel):
    theme: str


# ------------------------------------------------------------------ auth ---

@router.post("/auth/signup")
def signup(body: SignUpBody, session: Session = Depends(get_session)):
    """Create an account. The same rules the web form applies."""
    email = body.email.lower().strip()
    display_name = body.display_name.strip()

    if not display_name:
        raise HTTPException(400, "Please enter your name.")
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(400, "Please enter a valid email address.")
    if len(body.password) < MIN_PASSWORD:
        raise HTTPException(
            400, f"Choose a password of at least {MIN_PASSWORD} characters.")
    if find_by_email(session, email):
        raise HTTPException(400, "An account already exists with that email.")

    user = User(email=email, display_name=display_name,
                password_hash=hash_password(body.password))
    session.add(user)
    session.commit()
    return {"token": issue_session(user.id), "user": _user(user),
            "has_data": False}


@router.post("/auth/signin")
def signin(body: SignInBody, session: Session = Depends(get_session)):
    user = find_by_email(session, body.email)
    # the same message either way, so the endpoint cannot be used to discover
    # which email addresses have accounts
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            401, "That email and password do not match an account.")
    rows = session.scalar(
        select(func.count()).select_from(Transaction)
        .where(Transaction.user_id == user.id)) or 0
    return {"token": issue_session(user.id), "user": _user(user),
            "has_data": bool(rows)}


@router.get("/me")
def me(user: User = Depends(api_user), session: Session = Depends(get_session)):
    """Who the token belongs to, and the few things every screen needs."""
    rows = session.scalar(
        select(func.count()).select_from(Transaction)
        .where(Transaction.user_id == user.id)) or 0
    return {
        "user": _user(user),
        "has_data": bool(rows),
        "transactions": rows,
        "review_count": _review_count(session, user),
        "categories": sorted(classifier.get_model_card()["classes"]),
        "threshold": classifier.REVIEW_THRESHOLD,
    }


@router.post("/theme")
def set_theme(body: ThemeBody, user: User = Depends(api_user),
              session: Session = Depends(get_session)):
    if body.theme in ("light", "dark", "system"):
        user.theme = body.theme
        session.commit()
    return {"theme": user.theme}


# --------------------------------------------------------------- connect ---

@router.post("/connect/demo")
def connect_demo(user: User = Depends(api_user),
                 session: Session = Depends(get_session)):
    """Load the bundled three-bank demonstration set."""
    result = ingest_for_user(session, user)
    if not result.transactions:
        raise HTTPException(500, "No statement files were found to import.")
    return _ingest_payload(result)


@router.post("/connect/upload")
async def connect_upload(statements: list[UploadFile] = File(default=[]),
                         user: User = Depends(api_user),
                         session: Session = Depends(get_session)):
    """Import statement files chosen on the device.

    Identical to the web path, deliberately: the files are recognised by their
    columns rather than their names, staged under the names the adapters
    expect, and handed to the same ingestion. Staging happens in a temporary
    directory removed whatever the outcome — an upload is one person's data
    and has no business persisting on the server's disk.
    """
    payload = [(f.filename or "file.csv", await f.read()) for f in statements]
    if not payload:
        raise HTTPException(400, "No files were chosen.")

    staging = Path(tempfile.mkdtemp(prefix="trace-upload-"))
    try:
        staged = stage_uploads(payload, staging)
        if not staged.files:
            detail = "; ".join(f"{name} — {why}"
                               for name, why in staged.rejected[:3])
            raise HTTPException(
                400, "None of those files looked like a supported bank "
                     "export. " + detail)
        try:
            result = ingest_for_user(session, user, staging)
        except HTTPException:
            raise
        except Exception as error:
            session.rollback()
            raise HTTPException(400, f"Those files could not be read: {error}")
    finally:
        clear_uploads(staging)

    if not result.transactions:
        raise HTTPException(400, "Those files held no transactions.")
    return _ingest_payload(result, staged)


# ------------------------------------------------------------- dashboard ---

@router.get("/dashboard")
def dashboard(user: User = Depends(api_user),
              session: Session = Depends(get_session)):
    accounts = session.scalar(
        select(func.count()).select_from(Account)
        .where(Account.user_id == user.id)) or 0
    rows = analytics.load_transactions(session, user)
    if not rows:
        return {"has_data": False}

    months = analytics.monthly(rows)
    hour = datetime.now().hour
    greeting = ("Good morning" if hour < 12
                else "Good afternoon" if hour < 18 else "Good evening")
    return {
        "has_data": True,
        "greeting": greeting,
        "totals": asdict(analytics.totals(rows, accounts)),
        "months": months,
        # the same SVG the web page draws, rendered on the server. The phone
        # displays it rather than rebuilding the chart from a second library,
        # so the two front ends cannot disagree about what the data looks like
        "chart": charts.cashflow(months),
        "categories": analytics.by_category(rows),
        "recurring": analytics.recurring(rows),
        "recent": [_transaction(r) for r in rows[:8]],
        "review_count": _review_count(session, user),
    }


# ---------------------------------------------------------- transactions ---

@router.get("/transactions")
def transactions(q: str = "", category: str = "", account: str = "",
                 state: str = "", page: int = Query(1, ge=1),
                 user: User = Depends(api_user),
                 session: Session = Depends(get_session)):
    stmt = select(Transaction).where(Transaction.user_id == user.id)
    if q:
        stmt = stmt.where(Transaction.raw_description.ilike(f"%{q}%"))
    if category:
        stmt = stmt.where(EFFECTIVE_CATEGORY == category)
    if account:
        try:
            stmt = stmt.where(Transaction.account_id == int(account))
        except ValueError:
            raise HTTPException(400, "account must be a number")
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
        select(Account).where(Account.user_id == user.id)
        .order_by(Account.bank)))

    return {
        "rows": [_transaction(r) for r in rows],
        "matching": matching, "total": total,
        "page": page, "pages": pages, "per_page": PER_PAGE,
        "accounts": [_account(a) for a in accounts],
        "all_categories": sorted(c for (c,) in session.execute(
            select(EFFECTIVE_CATEGORY)
            .where(Transaction.user_id == user.id).distinct())),
        "filters": {"q": q, "category": category, "account": account,
                    "state": state},
    }


@router.get("/transactions/{transaction_id}")
def transaction(transaction_id: int, user: User = Depends(api_user),
                session: Session = Depends(get_session)):
    return _transaction(_require(session, user, transaction_id))


@router.get("/transactions/{transaction_id}/explanation")
def explanation(transaction_id: int, method: str = "shap",
                user: User = Depends(api_user),
                session: Session = Depends(get_session)):
    """Why this transaction carries the category it does.

    The same function the web panel calls, so a phone and a browser looking at
    one transaction are told the same thing by the same code.
    """
    row = _require(session, user, transaction_id)
    return common.explanation_payload(row, method)


@router.post("/transactions/{transaction_id}/confirm")
def confirm(transaction_id: int, user: User = Depends(api_user),
            session: Session = Depends(get_session)):
    row = _require(session, user, transaction_id)
    return asdict(review.confirm(session, user, row))


@router.post("/transactions/{transaction_id}/category")
def correct(transaction_id: int, body: CategoryBody,
            user: User = Depends(api_user),
            session: Session = Depends(get_session)):
    row = _require(session, user, transaction_id)
    if body.category not in classifier.get_model_card()["classes"]:
        raise HTTPException(400, "unknown category")
    return asdict(review.correct(session, user, row, body.category))


# ------------------------------------------------------------- the queue ---

@router.get("/review")
def review_queue(user: User = Depends(api_user),
                 session: Session = Depends(get_session)):
    """Everything the cascade was not confident enough to decide alone."""
    rows = list(session.scalars(
        select(Transaction).options(selectinload(Transaction.account))
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
    return {
        "rows": [_transaction(r) for r in rows],
        "reviewed": reviewed,
        "threshold": classifier.REVIEW_THRESHOLD,
        "categories": sorted(classifier.get_model_card()["classes"]),
        "corrections": [{"merchant": c.merchant_fingerprint,
                         "category": c.category,
                         "times_applied": c.times_applied,
                         "created_at": c.created_at}
                        for c in corrections],
    }


# -------------------------------------------------------------- insights ---

@router.get("/insights")
def insights(user: User = Depends(api_user),
             session: Session = Depends(get_session)):
    rows = analytics.load_transactions(session, user)
    if not rows:
        return {"has_data": False}
    categories = analytics.by_category(rows)
    wheel, segments = charts.donut(categories)
    recurring = analytics.recurring(rows)
    change = analytics.changes(rows)
    signals = analytics.signals(rows)
    return {
        "has_data": True,
        "categories": categories,
        "wheel": wheel, "segments": segments,
        "recurring": recurring,
        "unusual": analytics.unusual(rows, {r["merchant"] for r in recurring}),
        "change": change,
        "change_chart": charts.diverging(change["rows"], "vs_average"),
        "change_chart_prev": charts.diverging(change["rows"], "vs_previous"),
        "signals": asdict(signals) if signals else None,
        "benchmark": analytics.benchmark(rows),
        "methodology": analytics.METHODOLOGY,
        "is_demo": common.is_demo(session, user),
    }


# -------------------------------------------------------------- accounts ---

@router.get("/accounts")
def accounts(user: User = Depends(api_user),
             session: Session = Depends(get_session)):
    rows = analytics.load_transactions(session, user)
    linked = list(session.scalars(
        select(Account).where(Account.user_id == user.id)
        .order_by(Account.bank)))
    spend = {a["bank"]: a for a in analytics.by_account(session, user, rows)}
    detail = []
    for account in linked:
        mine = [r for r in rows if r.account_id == account.id]
        dates = sorted(r.booking_date for r in mine)
        detail.append({
            "account": _account(account),
            "rows": len(mine),
            "first": dates[0] if dates else None,
            "last": dates[-1] if dates else None,
            "spend": spend.get(account.bank, {}).get("amount", 0.0),
            "share": spend.get(account.bank, {}).get("share", 0.0),
        })
    return {"detail": detail, "total": len(rows)}


# ---------------------------------------------------------- how it works ---

@router.get("/how-it-works")
def how_it_works(user: User = Depends(api_user),
                 session: Session = Depends(get_session)):
    """The cascade, and how accurate it actually is.

    Reporting only the in-distribution figure would be true and misleading at
    once, so all three protocols are returned together.
    """
    card = classifier.get_model_card()
    rows = analytics.load_transactions(session, user)
    flagged = sum(1 for r in rows if r.needs_review and not r.reviewed_at)
    return {
        "card": card,
        "performance": card["performance"],
        "labels": PROTOCOL_LABELS,
        "means": PROTOCOL_MEANS,
        "chart": charts.protocols(card["performance"], PROTOCOL_LABELS),
        "layers": analytics.layer_counts(rows),
        "total": len(rows),
        "flagged": flagged,
        "observed_rate": flagged / len(rows) * 100 if rows else 0.0,
        "explain_cost": EXPLANATION_COST,
        "threshold": classifier.REVIEW_THRESHOLD,
    }
