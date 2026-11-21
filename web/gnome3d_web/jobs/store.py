"""The files of a job directory.

A job directory holds `request.json`, `status.json`, the engine inputs under `inputs/`, the
structures under `models/`, `marks.json`, `ep.tsv` and `engine.log`.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import time
import zipfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np


# The key of a shared job and the token of a private one.
KEY = re.compile(r"[A-Za-z0-9_-]{8,64}")


def find(root: Path, key: str) -> Path | None:
    """The directory of the job of a key under `root`, None for a key the site has no job of."""
    job = root / key
    if not KEY.fullmatch(key) or not ((job / "status.json").is_file() or (job / "request.json").is_file()):
        return None
    return job


def read_json(path: Path) -> dict:
    """A JSON file of a job, empty for a file the job does not have."""
    return json.loads(path.read_text()) if path.is_file() else {}


def write_json(path: Path, value: object) -> None:
    """Write a JSON file whole. A job page reads a job's files while the job runs and a file
    written in place is empty for a moment."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def request(job: Path) -> dict:
    """The request of a job as it is on disk, with the places of its uploads."""
    return read_json(job / "request.json")


def status(job: Path) -> dict:
    """The status of a job. A job without the file waits."""
    return read_json(job / "status.json") or {"state": "queued"}


def set_status(job: Path, **fields: object) -> None:
    """Writes fields into the status of a job. A new stage is added to the stages with its time."""
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


def queued(job: Path) -> None:
    """Gives a job its first status. A restart of the site finds a waiting job by it."""
    write_json(job / "status.json", {"state": "queued", "stages": [{"stage": "queued", "at": time.time()}]})


def taken(job: Path) -> bool:
    """Whether a directory holds a job that waits, runs or is finished. A failed job leaves its key to the next request."""
    return (job / "status.json").is_file() and status(job).get("state") in ("running", "done", "queued")


def member(p: Path) -> tuple[int, str]:
    """A structure file's member number, from the engine's `_s<k>.cif` suffix, for natural order."""
    m = re.search(r"_s(\d+)\.cif$", p.name)
    return (int(m.group(1)) if m else 0, p.name)


def structures(job: Path) -> list[Path]:
    """The job's structure files in member order."""
    names = read_json(job / "marks.json").get("models", [])
    return [job / "models" / n for n in names]


def read_cif(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Positions, starts, ends and an anchor mask of a gnome3d cif, in genomic order."""
    rows = [ln.split() for ln in Path(path).read_text().splitlines() if ln.startswith("ATOM")]
    pos = np.array([[float(r[10]), float(r[11]), float(r[12])] for r in rows])
    start = np.array([int(r[16]) for r in rows])
    end = np.array([int(r[17]) for r in rows])
    anchor = np.array([r[18] == "anchor" for r in rows])
    o = np.argsort(start, kind="stable")
    return pos[o], start[o], end[o], anchor[o]


def tail(path: Path, n: int = 30) -> str:
    """The last n lines of a file."""
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:]) if path.is_file() else ""


def log_end(path: Path, n: int) -> str:
    """The last n lines of a log, read from its end so a long log costs little."""
    if not path.is_file():
        return ""
    with path.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 32_768))
        return "\n".join(f.read().decode(errors="replace").splitlines()[-n:])


def sizes(paths: Iterable[Path]) -> list[dict[str, object]]:
    """The name and the size of each file. A job writes a file under a second name and renames it,
    and that second file can be gone between the listing and the reading of its size."""
    out: list[dict[str, object]] = []
    for p in paths:
        with contextlib.suppress(FileNotFoundError):
            if p.is_file() and not p.name.endswith(".tmp"):
                out.append({"name": p.name, "size": p.stat().st_size})
    return out


def public_request(req: dict) -> dict:
    """The request without the server's paths of uploaded files."""
    out = {k: v for k, v in req.items() if k not in ("upload_vcf", "own", "expression")}
    # The source of the expression by its id. The table of an upload has a path on the server.
    if req.get("expression"):
        out["expression"] = str(req["expression"].get("source", ""))
    own = req.get("own") or {}
    if own:
        out["own"] = {
            "loops": [f for f, _ in own.get("loops", [])],
            **{k: bool(own.get(k)) for k in ("anchors", "contacts", "enhancers")},
            **{k: own[k] for k in ("loops_from", "enhancers_from") if own.get(k)},
        }
    return out


def public_log(text: str, job: Path) -> str:
    """A log with the job's directory on the server cut from every path. A private job's
    location on disk is not shown."""
    for p in {str(job.resolve()), str(job)}:
        text = text.replace(p + "/", "").replace(p, ".")
    return text


def public_status(job: Path) -> dict[str, object]:
    """The status of a job without the paths of the server in its error and its log."""
    out = status(job)
    for k in ("error", "log_tail"):
        if out.get(k):
            out[k] = public_log(str(out[k]), job)
    return out


def public_file(job: Path, name: str) -> bytes | None:
    """The request, the status or the log of a job as a download has it, without the paths of
    the server. None for every other file, which a download has as it is on disk."""
    if name == "request.json":
        return json.dumps(public_request(request(job)), indent=2).encode()
    if name == "status.json":
        return json.dumps(public_status(job), indent=2).encode()
    if name == "engine.log":
        return log_text(job).encode()
    return None


def log_text(job: Path) -> str:
    """The engine log."""
    path = job / "engine.log"
    return public_log(path.read_text(errors="replace"), job) if path.is_file() else ""


# The files of a job a visitor may download by name, the structures among them.
DOWNLOADS = re.compile(r"^(ep\.tsv|marks\.json|engine\.log|status\.json|request\.json|hicdiffusion\.bedpe|expression\.tsv|[A-Za-z0-9_.:-]+\.cif)$")


def download(job: Path, name: str) -> Path | None:
    """The file of a job a download names, None for a name that is no download or a file the job lacks."""
    if not DOWNLOADS.match(name):
        return None
    path = job / "models" / name if name.endswith(".cif") else job / name
    return path if path.is_file() else None


def zip_bytes(job: Path) -> bytes:
    """Every file of the job but the upload, its structures under models/."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(job.rglob("*")):
            rel = p.relative_to(job)
            if not p.is_file() or rel.name.startswith(("explore_", "upload.")) or rel.name.endswith(".tmp") or re.match(r"inputs/", str(rel)):
                continue
            stated = public_file(job, rel.name) if rel.parent == Path(".") else None
            if stated is not None:
                z.writestr(str(rel), stated)
                continue
            with contextlib.suppress(FileNotFoundError):
                z.write(p, str(rel))
    return buf.getvalue()


def sweep_private(results_dir: Path, retention_days: int) -> list[str]:
    """Remove private jobs and uploads older than the retention period, as the form promises
    users and return their names. Shared jobs are never removed here."""
    gone = []
    cutoff = time.time() - retention_days * 86400
    for root in (results_dir / "private", results_dir / "uploads"):
        if not root.is_dir():
            continue
        for d in root.iterdir():
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d)
                gone.append(d.name)
    return gone
