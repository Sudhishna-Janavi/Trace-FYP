"""Accept statement files a user chose themselves.

Why this is a separate module rather than a change to `ingest.py`

    `backend.adapters.ingest` selects an adapter by filename: a file must be
    called `chase_activity.csv` or the Chase adapter never sees it. That is
    sound for the bundled demonstration feeds, whose names this project
    controls, and it is what the "Connect bank data" path uses today. It is
    the wrong rule for a file a person picked on their phone, where the name
    is whatever their bank happened to emit.

    Rather than change that working path, this module recognises a file by
    its columns and then *stages* it under the name the existing adapter
    expects. `adapters.ingest` is then called unchanged. Detection is new;
    parsing is the code that already works.

Recognising a format

    The three exports differ unambiguously in their header row:

        Amount (USD)        Apple Card, and purchases are positive
        Debit and Credit    Capital One, exactly one of the pair is filled
        Amount              Chase, one signed column

    A file matching none of these is reported to the user rather than
    guessed at.
"""
from __future__ import annotations

import csv
import io
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

MAX_BYTES = 8 * 1024 * 1024        # per upload; the demo set is under 80 KB
MAX_FILES = 60

# column signature -> the filename adapters.ingest looks for
CHASE, CAPITALONE, APPLECARD = "chase", "capitalone", "applecard"
BANK_NAMES = {CHASE: "Chase", CAPITALONE: "Capital One", APPLECARD: "Apple Card"}


@dataclass
class StagedUpload:
    """What arrived, and what was made of it."""
    recognised: dict[str, int] = field(default_factory=dict)   # kind -> files
    rejected: list[tuple[str, str]] = field(default_factory=list)  # name, why

    @property
    def files(self) -> int:
        return sum(self.recognised.values())

    @property
    def banks(self) -> list[str]:
        return [BANK_NAMES[kind] for kind in sorted(self.recognised)]

    @property
    def banks_phrase(self) -> str:
        """"A, B and C" rather than the "A and B and C" a plain join gives."""
        names = self.banks
        if len(names) <= 1:
            return names[0] if names else ""
        return ", ".join(names[:-1]) + " and " + names[-1]


def detect(header: list[str] | None) -> str | None:
    """Which export this is, from its columns alone."""
    if not header:
        return None
    columns = {name.strip() for name in header}
    if "Amount (USD)" in columns:
        return APPLECARD
    if "Debit" in columns and "Credit" in columns:
        return CAPITALONE
    if "Amount" in columns and "Transaction Date" in columns:
        return CHASE
    return None


def _header_of(data: bytes) -> list[str] | None:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None
    try:
        return next(csv.reader(io.StringIO(text)))
    except StopIteration:
        return None


def _members(name: str, data: bytes) -> list[tuple[str, bytes]]:
    """Flatten one upload into CSVs, expanding a zip if that is what it is.

    Only the base name of a zip entry is used. An archive entry may name any
    path it likes, including one climbing out of the directory it is extracted
    into, and nothing here needs the structure anyway.
    """
    if not name.lower().endswith(".zip"):
        return [(Path(name).name, data)]
    out: list[tuple[str, bytes]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                inner = Path(info.filename).name
                if not inner.lower().endswith(".csv") or inner.startswith("."):
                    continue
                if info.file_size > MAX_BYTES:
                    continue
                out.append((inner, archive.read(info)))
    except zipfile.BadZipFile:
        return []
    return out


def stage(uploads: list[tuple[str, bytes]], directory: Path) -> StagedUpload:
    """Write recognised files into `directory` under the expected names.

    Apple Card arrives as many monthly files and its adapter globs for them,
    so each is numbered. Chase and Capital One expect exactly one file each;
    when several of either arrive they are concatenated, header kept once,
    because two exports from one bank are two parts of one statement.
    """
    result = StagedUpload()
    directory.mkdir(parents=True, exist_ok=True)
    joined: dict[str, list[bytes]] = {CHASE: [], CAPITALONE: []}

    for name, data in uploads[:MAX_FILES]:
        for inner_name, inner in _members(name, data)[:MAX_FILES]:
            if len(inner) > MAX_BYTES:
                result.rejected.append((inner_name, "larger than 8 MB"))
                continue
            kind = detect(_header_of(inner))
            if kind is None:
                result.rejected.append(
                    (inner_name, "columns match no supported bank export"))
                continue
            result.recognised[kind] = result.recognised.get(kind, 0) + 1
            if kind == APPLECARD:
                index = result.recognised[kind]
                (directory / f"applecard_{index:03d}.csv").write_bytes(inner)
            else:
                joined[kind].append(inner)

    for kind, parts in joined.items():
        if not parts:
            continue
        filename = ("chase_activity.csv" if kind == CHASE
                    else "capitalone_transactions.csv")
        if len(parts) == 1:
            (directory / filename).write_bytes(parts[0])
        else:
            merged = [parts[0].rstrip(b"\r\n")]
            for extra in parts[1:]:
                body = extra.split(b"\n", 1)
                if len(body) == 2:
                    merged.append(body[1].rstrip(b"\r\n"))
            (directory / filename).write_bytes(b"\n".join(merged) + b"\n")

    return result


def clear(directory: Path) -> None:
    shutil.rmtree(directory, ignore_errors=True)
