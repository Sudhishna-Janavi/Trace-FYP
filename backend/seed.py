"""Create the demonstration account on first boot.

A deployed URL that opens on an empty sign-in form is useless to anyone sent
the link: they would have to register, then find data to upload. Seeding a
single account means the link opens on a working dashboard.

Seeding is idempotent and never touches an account that already exists, so a
redeploy leaves any corrections the visitor has made intact.
"""
from __future__ import annotations

import os

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.auth import hash_password
from backend.database import Transaction, User, SessionLocal
from backend.ingest import ingest_for_user

DEMO_EMAIL = os.getenv("TRACE_DEMO_EMAIL", "demo@trace.app")
DEMO_PASSWORD = os.getenv("TRACE_DEMO_PASSWORD", "")
DEMO_NAME = os.getenv("TRACE_DEMO_NAME", "Alex Demo")


def seed_demo_account() -> str:
    """Ensure the demonstration account exists. Returns what it did."""
    if not DEMO_PASSWORD:
        return ("skipped: TRACE_DEMO_PASSWORD is unset, so no demonstration "
                "account was created")

    with SessionLocal() as session:          # type: Session
        user = session.scalar(select(User).where(User.email == DEMO_EMAIL))
        if user is None:
            user = User(email=DEMO_EMAIL, display_name=DEMO_NAME,
                        password_hash=hash_password(DEMO_PASSWORD), theme="system")
            session.add(user)
            session.commit()
            created = True
        else:
            created = False

        rows = session.scalar(
            select(func.count()).select_from(Transaction)
            .where(Transaction.user_id == user.id)) or 0
        if rows:
            return f"demonstration account present with {rows} transactions"

        result = ingest_for_user(session, user)
        session.commit()
        return (f"demonstration account {'created' if created else 'refilled'}: "
                f"{result.transactions} transactions across "
                f"{result.accounts} accounts, {result.needs_review} for review")
