"""Database schema for Trace.

One engine, two deployments: SQLite locally because it is a file and needs
nothing installed, PostgreSQL in production because most hosting platforms have
an ephemeral filesystem and a SQLite file would not survive a restart.
SQLAlchemy hides the difference, so DATABASE_URL is the only thing that changes.

Every row a user can see carries a user_id. Scoping is enforced in the queries
rather than left to the interface, so a missing filter is a visible omission in
the data layer rather than an invisible leak between accounts.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import (Boolean, Date, DateTime, Float, ForeignKey, Integer,
                        String, Text, UniqueConstraint, create_engine, func)
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column,
                            relationship, sessionmaker)

ROOT = Path(__file__).resolve().parents[1]


def _normalise(url: str) -> str:
    """Make a host's connection string usable by SQLAlchemy 2.

    Render, Railway and Heroku all hand out `postgres://`, a scheme SQLAlchemy
    removed support for, and none of them name a driver. Rewriting here means
    the environment variable can be pasted in exactly as the host gives it.
    """
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


DATABASE_URL = _normalise(
    os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'trace.db'}"))


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(255))
    theme: Mapped[str] = mapped_column(String(8), default="system")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    accounts: Mapped[list["Account"]] = relationship(
        back_populates="user", cascade="all, delete-orphan")
    transactions: Mapped[list["Transaction"]] = relationship(
        back_populates="user", cascade="all, delete-orphan")
    corrections: Mapped[list["Correction"]] = relationship(
        back_populates="user", cascade="all, delete-orphan")


class Account(Base):
    """One linked institution. Created when a user connects or uploads data."""

    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    external_id: Mapped[str] = mapped_column(String(32))   # acc-chase, ...
    bank: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(64))
    # the export shape this account's adapter had to reconcile, shown in the UI
    source_format: Mapped[str] = mapped_column(String(160))
    currency: Mapped[str] = mapped_column(String(3), default="USD")

    user: Mapped[User] = relationship(back_populates="accounts")
    transactions: Mapped[list["Transaction"]] = relationship(
        back_populates="account", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("user_id", "external_id"),)


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)

    raw_description: Mapped[str] = mapped_column(Text)
    normalised_description: Mapped[str] = mapped_column(Text)
    merchant_fingerprint: Mapped[str] = mapped_column(String(96), index=True)
    amount: Mapped[float] = mapped_column(Float)     # negative out, positive in
    booking_date: Mapped[date] = mapped_column(Date, index=True)

    predicted_category: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    # which cascade layer decided: rule | memory | correction | model.
    # Stored because the explanation shown depends on it.
    decided_by: Mapped[str] = mapped_column(String(16), index=True)
    evidence: Mapped[str] = mapped_column(Text, default="")
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    # set when a user confirms or changes the category
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    corrected_category: Mapped[str | None] = mapped_column(String(32), nullable=True)

    user: Mapped[User] = relationship(back_populates="transactions")
    account: Mapped[Account] = relationship(back_populates="transactions")

    @property
    def category(self) -> str:
        """What the user should be shown: their correction wins."""
        return self.corrected_category or self.predicted_category


class Correction(Base):
    """The user-correction memory layer.

    A category a user has fixed is never silently overridden, so this sits in
    front of the merchant memory in the cascade. It is per user: one person
    deciding that a merchant is Groceries says nothing about anyone else.
    """

    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    merchant_fingerprint: Mapped[str] = mapped_column(String(96), index=True)
    category: Mapped[str] = mapped_column(String(32))
    times_applied: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    user: Mapped[User] = relationship(back_populates="corrections")

    __table_args__ = (UniqueConstraint("user_id", "merchant_fingerprint"),)


connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
# A free Postgres instance closes idle connections, and a pooled connection
# that was closed server-side fails the next request rather than reconnecting.
pool_args = {} if DATABASE_URL.startswith("sqlite") else {
    "pool_pre_ping": True, "pool_size": 5, "max_overflow": 2,
    "pool_recycle": 280,
}
engine = create_engine(DATABASE_URL, future=True, connect_args=connect_args,
                       **pool_args)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


def get_session():
    with SessionLocal() as session:
        yield session


def create_all() -> None:
    Base.metadata.create_all(engine)
