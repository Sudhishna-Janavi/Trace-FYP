#!/usr/bin/env python3
"""Create the database tables. Safe to re-run; existing tables are left alone."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.database import DATABASE_URL, Base, create_all, engine  # noqa: E402

if __name__ == "__main__":
    create_all()
    from sqlalchemy import inspect
    tables = inspect(engine).get_table_names()
    print(f"[PASS] {DATABASE_URL}")
    for name in sorted(tables):
        columns = len(inspect(engine).get_columns(name))
        print(f"   {name:<16}{columns:>3} columns")
