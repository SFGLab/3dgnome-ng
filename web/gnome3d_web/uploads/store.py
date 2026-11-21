"""The stored file of an upload, its record and the answer the dialog gets about it."""

from __future__ import annotations

import json
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import BinaryIO

from ..errors import Refused
from .checks import Check, records, whole
from .shapes import LoopDrawn, Meta, UploadResult


CHUNK = 1 << 20


class UploadTooLarge(ValueError):
    """An upload past the size cap."""


def store(stream: BinaryIO, name: str, kind: str, root: Path, cap: int, extra: dict[str, str] | None = None) -> Path:
    """Copies an upload to a new directory under root in chunks and writes its meta.json.
    Returns the directory. Raises UploadTooLarge past cap bytes, leaving nothing behind."""
    uid = secrets.token_urlsafe(12)
    d = root / uid
    d.mkdir(parents=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(name).name) or "upload"
    size = 0
    try:
        with (d / safe).open("wb") as out:
            while chunk := stream.read(CHUNK):
                size += len(chunk)
                if size > cap:
                    raise UploadTooLarge(f"{name} is more than {cap // (1 << 20)} MB, the largest upload this server takes")
                out.write(chunk)
    except UploadTooLarge:
        shutil.rmtree(d)
        raise
    (d / "meta.json").write_text(json.dumps({"id": uid, "kind": kind, "name": safe, "size": size, "created": time.time()} | (extra or {})))
    return d


def meta(d: Path) -> Meta:
    return json.loads((d / "meta.json").read_text())


def data_file(d: Path) -> Path:
    return d / str(meta(d)["name"])


def record_check(d: Path, c: Check) -> None:
    """Writes the outcome of the check into the upload's meta.json."""
    m = meta(d) | {"ok": c.ok}
    (d / "meta.json").write_text(json.dumps(m))


def clean_copy(d: Path, dst: Path) -> Path:
    """Writes the upload's records tab separated, without header lines or Windows line ends."""
    with dst.open("w") as out:
        for _, cols in records(data_file(d)):
            out.write("\t".join(cols) + "\n")
    return dst


def result(d: Path, c: Check) -> UploadResult:
    m = meta(d)
    # A map read by link is shown by the remote file's name and size, not the window kept.
    name, size = m.get("link_name", m["name"]), int(m.get("link_size", m["size"]))  # type: ignore[arg-type]
    return {"id": m["id"], "kind": m["kind"], "name": name, "size": size, "ok": c.ok, "summary": c.summary, "error": {"line": c.error.line, "found": c.error.found, "fix": c.error.fix} if c.error else None, "samples": c.samples, "counts": c.counts}


def preview(d: Path, region: tuple[str, int, int]) -> list[LoopDrawn]:
    """Returns the loops of a loops upload inside a region, as anchor midpoints and the count."""
    out = []
    for _, cols in records(data_file(d)):
        if len(cols) >= 7 and cols[0] == cols[3] == region[0] and whole(cols[6]) is not None:
            a, b = int(cols[1]), int(cols[5])
            if a >= region[1] and b <= region[2]:
                out.append([(a + int(cols[2])) // 2, (int(cols[4]) + b) // 2, int(cols[6])])
    return out


# The id of an upload, a random token.
ID = re.compile(r"[A-Za-z0-9_-]{8,64}")


def find(root: Path, uid: str) -> Path | None:
    """Returns the directory of an upload under `root`, None for an id without an upload."""
    d = root / uid
    return d if ID.fullmatch(uid) and (d / "meta.json").is_file() else None


def usable(root: Path, uid: object, kind: str, field: str) -> Path:
    """Returns the directory of an upload that exists, holds `kind` and passed its check.

    @raises Refused on `field` otherwise
    """
    d = find(root, str(uid))
    if d is None:
        raise Refused(f"upload {uid!r} does not exist", field)
    m = meta(d)
    if m["kind"] != kind:
        raise Refused(f"upload {uid!r} holds {m['kind']}, not {kind}", field)
    if not m.get("ok"):
        raise Refused(f"{m['name']} did not pass its check. Fix the file and upload it again.", field)
    return d


def own_data(root: Path, raw: object, cell_types: set[str]) -> dict:
    """Returns the upload directories of the own data of a request, each checked to exist, to be of
    its kind and to have passed its check.

    @raises Refused on the field own otherwise
    """
    if not isinstance(raw, dict):
        raise Refused("own is not an object of uploads", "own")
    out: dict = {}
    loops = raw.get("loops") or []
    if not isinstance(loops, list) or not all(isinstance(x, dict) for x in loops):
        raise Refused("own loops must be a list of objects with a factor and an upload", "own")
    if loops:
        out["loops"] = [(str(x.get("factor") or "loops"), usable(root, x.get("upload"), "loops", "own")) for x in loops]
    for kind in ("anchors", "contacts", "enhancers"):
        if raw.get(kind):
            out[kind] = usable(root, raw[kind], kind, "own")
    # Your own cell type may take a listed cell type's loops and enhancers in place of uploads.
    for key, allowed in (("loops_from", cell_types), ("enhancers_from", {*cell_types, "none"})):
        name = raw.get(key)
        if name:
            if name not in allowed:
                raise Refused(f"{key} names {name!r}, which is not a listed cell type", "own")
            out[key] = str(name)
    return out
