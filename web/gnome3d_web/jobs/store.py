"""The files of a job directory.

A job directory holds `request.json`, `status.json`, the engine inputs under `inputs/`, the
structures under `models/`, `marks.json`, `ep.tsv` and `engine.log`.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

# The key of a shared job, the token of a private one and the id of an upload.
KEY = re.compile(r"[A-Za-z0-9_-]{8,64}")


def find(root: Path, key: str) -> Path | None:
    """The directory of the job of a key under `root`, None for a key the site has no job of."""
    job = root / key
    if not KEY.fullmatch(key) or not ((job / "status.json").is_file() or (job / "request.json").is_file()):
        return None
    return job


def _member(p: Path) -> tuple[int, str]:
    """A structure file's member number, from the engine's `_s<k>.cif` suffix, for natural order."""
    import re

    m = re.search(r"_s(\d+)\.cif$", p.name)
    return (int(m.group(1)) if m else 0, p.name)


def _status(job: Path, **fields: object) -> None:
    path = job / "status.json"
    cur = json.loads(path.read_text()) if path.is_file() else {}
    now = time.time()
    # Each stage once, with the time it began, for the job page's status log.
    stage = fields.get("stage")
    stages = cur.get("stages", [])
    if stage and (not stages or stages[-1]["stage"] != stage):
        stages.append({"stage": stage, "at": now})
        cur["stages"] = stages
    cur.update(fields, updated=now)
    write_json(path, cur)


def write_json(path: Path, value: object) -> None:
    """Write a JSON file whole. A job page reads a job's files while the job runs and a file
    written in place is empty for a moment."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def _tail(path: Path, n: int = 30) -> str:
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:]) if path.is_file() else ""
