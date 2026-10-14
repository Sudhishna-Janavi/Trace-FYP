"""Authentication: password hashing and signed session cookies.

Passwords are hashed with bcrypt and never stored or logged in plain text.
The session cookie holds only the user id, signed so it cannot be forged, and
is HttpOnly so page scripts cannot read it.
"""
from __future__ import annotations

import os

from fastapi import Depends, HTTPException, Request, status
import base64
import hashlib

import bcrypt
from itsdangerous import BadSignature, URLSafeSerializer
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database import User, get_session

# A development default keeps local runs working. Deployment must set its own:
# anyone holding the key can mint a session cookie for any account, and a
# default published in the source is held by everyone. Rather than trust that
# the variable was remembered, the application refuses to start without it.
DEV_SECRET = "dev-only-not-for-deployment"
SECRET_KEY = os.getenv("TRACE_SECRET_KEY", DEV_SECRET)
# Set by the host on Render, Railway, Fly and Heroku alike; locally it is unset.
IS_DEPLOYED = bool(os.getenv("TRACE_ENV") == "production"
                   or os.getenv("RENDER") or os.getenv("RAILWAY_ENVIRONMENT")
                   or os.getenv("FLY_APP_NAME") or os.getenv("DYNO"))

if IS_DEPLOYED and SECRET_KEY == DEV_SECRET:
    raise RuntimeError(
        "TRACE_SECRET_KEY is not set. Refusing to start with the development "
        "key, which is published in this repository and would let anyone sign "
        "a session cookie for any account. Generate one with: "
        "python -c 'import secrets; print(secrets.token_urlsafe(48))'")

COOKIE_NAME = "trace_session"
# A session cookie must not travel over plain HTTP once the site has TLS.
COOKIE_SECURE = IS_DEPLOYED

_signer = URLSafeSerializer(SECRET_KEY, salt="trace-session")


def _prepare(password: str) -> bytes:
    """Reduce a password of any length to something bcrypt accepts.

    bcrypt refuses inputs longer than 72 bytes rather than truncating them, so
    a long passphrase would otherwise raise instead of being hashed. Taking a
    SHA-256 digest first gives a fixed 32 bytes, base64-encoded to 44
    characters, which is always within the limit and preserves the entropy of
    the original. This is the standard remedy and is why passlib is not used:
    its backend probe passes an over-length password and fails on contact with
    any current bcrypt release.
    """
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prepare(password), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_prepare(password), hashed.encode("ascii"))
    except (ValueError, TypeError):
        return False


def issue_session(user_id: int) -> str:
    return _signer.dumps({"uid": user_id})


def read_session(token: str | None) -> int | None:
    if not token:
        return None
    try:
        return int(_signer.loads(token)["uid"])
    except (BadSignature, KeyError, ValueError, TypeError):
        return None


def current_user(request: Request,
                 session: Session = Depends(get_session)) -> User | None:
    """The signed-in user, or None. Use for pages that work either way."""
    user_id = read_session(request.cookies.get(COOKIE_NAME))
    if user_id is None:
        return None
    return session.get(User, user_id)


def require_user(request: Request,
                 session: Session = Depends(get_session)) -> User:
    """The signed-in user, or a redirect to sign in."""
    user = current_user(request, session)
    if user is None:
        raise HTTPException(status_code=status.HTTP_303_SEE_OTHER,
                            headers={"Location": "/signin"})
    return user


def find_by_email(session: Session, email: str) -> User | None:
    return session.scalar(select(User).where(User.email == email.lower().strip()))
