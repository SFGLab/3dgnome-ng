"""Job lists that several browsers share through a sync code.

A list holds the jobs of one person as the browser keeps them, each with its client route, its
label, its name and its start time. Every browser that has the code reads and writes the same
list. A write merges, it never removes. A list made without private jobs refuses them.

The code is 16 characters of Crockford base32, 80 random bits, shown in four groups. Its file is
named by a digest of the code, so the directory does not hold the codes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any

ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
GROUPS = 4
GROUP = 4
MAX_JOBS = 50
MAX_TEXT = 200
MAX_BODY = 100_000
_ROUTE = re.compile(r"/(jobs|p)/[A-Za-z0-9_-]{1,64}")
_LOOKALIKE = str.maketrans({"O": "0", "I": "1", "L": "1"})
_LOCK = threading.Lock()

Job = dict[str, Any]


def new_code() -> str:
    """A fresh sync code, K7M2-9QXD-4THP-W3RZ."""
    return "-".join("".join(secrets.choice(ALPHABET) for _ in range(GROUP)) for _ in range(GROUPS))


def normalise(text: str) -> str | None:
    """A typed code in its shown form, or None when it cannot be a code.

    Case, spaces and dashes do not matter, and O, I and L read as the digits they resemble.
    """
    bare = re.sub(r"[\s-]", "", text).upper().translate(_LOOKALIKE)
    if len(bare) != GROUPS * GROUP or any(ch not in ALPHABET for ch in bare):
        return None
    return "-".join(bare[i : i + GROUP] for i in range(0, len(bare), GROUP))


def _stamp(job: Job) -> float:
    return float(job.get("edited") or job["at"])


def clean(jobs: object, private: bool) -> list[Job]:
    """The well formed entries of a list a browser sent, newest first.

    @param jobs the entries as parsed JSON
    @param private whether the list takes private jobs
    @return entries with a job route, texts cut to MAX_TEXT, at most MAX_JOBS
    """
    out: dict[str, Job] = {}
    for j in jobs if isinstance(jobs, list) else []:
        if not isinstance(j, dict):
            continue
        url, at = j.get("url"), j.get("at")
        if not isinstance(url, str) or not _ROUTE.fullmatch(url):
            continue
        if not isinstance(at, int | float) or isinstance(at, bool):
            continue
        if url.startswith("/p/") and not private:
            continue
        job: Job = {"url": url, "label": str(j.get("label") or "")[:MAX_TEXT], "name": str(j.get("name") or "")[:MAX_TEXT], "at": at}
        edited = j.get("edited")
        if isinstance(edited, int | float) and not isinstance(edited, bool):
            job["edited"] = edited
        if url not in out or _stamp(job) > _stamp(out[url]):
            out[url] = job
    return sorted(out.values(), key=lambda j: -float(j["at"]))[:MAX_JOBS]


def merge(a: list[Job], b: list[Job]) -> list[Job]:
    """Both lists as one, newest first. A job on both keeps the entry edited last."""
    out = {j["url"]: j for j in a}
    for j in b:
        if j["url"] not in out or _stamp(j) > _stamp(out[j["url"]]):
            out[j["url"]] = j
    return sorted(out.values(), key=lambda j: -float(j["at"]))[:MAX_JOBS]


class Lists:
    """The shared lists of a site, one JSON file each under `root`."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, code: str) -> Path:
        return self.root / f"{hashlib.sha256(code.encode()).hexdigest()[:40]}.json"

    def _write(self, code: str, record: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self._path(code).with_suffix(f".{secrets.token_hex(4)}.tmp")
        tmp.write_text(json.dumps(record))
        os.replace(tmp, self._path(code))

    def create(self, jobs: object, private: bool) -> tuple[str, dict[str, Any]]:
        """Stores a new list and returns its code and its record."""
        with _LOCK:
            code = new_code()
            while self._path(code).exists():
                code = new_code()
            record = {"private": private, "jobs": clean(jobs, private), "updated": time.time()}
            self._write(code, record)
        return code, record

    def read(self, code: str) -> dict[str, Any] | None:
        """The record of a code in its shown form, or None when no list has it."""
        p = self._path(code)
        return json.loads(p.read_text()) if p.is_file() else None

    def update(self, code: str, jobs: object) -> dict[str, Any] | None:
        """Merges a browser's entries into a list and returns the record, None when no list has
        the code."""
        with _LOCK:
            record = self.read(code)
            if record is None:
                return None
            merged = merge(record["jobs"], clean(jobs, bool(record["private"])))
            if merged != record["jobs"]:
                record = {**record, "jobs": merged, "updated": time.time()}
                self._write(code, record)
        return record
