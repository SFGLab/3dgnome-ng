"""The routes of the uploads: a file, a link to a contact map and the loops of an upload in a region."""

from __future__ import annotations

import io
from urllib.parse import urlparse

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from .. import remote
from ..regions import RegionError, parse_region
from ..site import Here, Site
from . import checks, store

router = APIRouter()


def region_of(site: Site, cell: str, text: str) -> tuple[str, int, int] | None:
    """The region an upload is checked against, None when the form has none yet."""
    if not text:
        return None
    r = parse_region(text, site.base_cell(cell).genes(), site.max_span())
    return (r.chrom, r.start, r.end)


@router.post("/api/uploads")
async def api_upload(site: Here, request: Request, kind: str = Form(...), file: UploadFile = File(...), cell: str = Form(""), region: str = Form(""), factor: str = Form("")) -> JSONResponse:  # noqa: B008
    s = site.settings
    cap = s.upload_max_mb << 20
    if int(request.headers.get("content-length", "0")) > cap + (1 << 16):
        return JSONResponse({"error": f"{file.filename} is more than {s.upload_max_mb} MB, the largest upload this server takes"}, 413)
    if kind not in checks.KINDS:
        return JSONResponse({"error": f"{kind!r} is not one of {', '.join(checks.KINDS)}", "field": "kind"}, 400)
    try:
        reg = region_of(site, cell, region)
    except RegionError as e:
        return JSONResponse({"error": str(e), "field": "region"}, 400)
    try:
        d = store.store(file.file, file.filename or kind, kind, s.uploads_dir, cap, {"factor": factor} if factor else None)
    except store.UploadTooLarge as e:
        return JSONResponse({"error": str(e)}, 413)
    c = checks.check(kind, store.data_file(d), reg)
    store.record_check(d, c)
    return JSONResponse(store.result(d, c), 201)


@router.post("/api/uploads/link")
def api_upload_link(site: Here, url: str = Form(...), cell: str = Form(""), region: str = Form("")) -> JSONResponse:
    if not site.link_slots.acquire(blocking=False):
        return JSONResponse({"error": "The server is reading other links. Try again in a minute.", "fix": "Or choose Local .hic file or BEDPE of the region.", "field": "url"}, 503)
    try:
        return read_link(site, url, cell, region)
    finally:
        site.link_slots.release()


def read_link(site: Site, url: str, cell: str, region: str) -> JSONResponse:
    """Reads the region from a linked contact map and stores its contacts as an upload."""
    s = site.settings
    try:
        reg = region_of(site, cell, region)
    except RegionError as e:
        return JSONResponse({"error": str(e), "field": "region"}, 400)
    if reg is None:
        return JSONResponse({"error": "Enter the region first.", "fix": "The server reads only the bins of the region from the file.", "field": "region"}, 400)
    base = site.base_cell(cell)
    try:
        per_mb = remote.hosted_density(base.path("singletons"), reg[0])
        text, size, summary = remote.link_window(url, *reg, per_mb, base.name, s.link_allow_private)
    except remote.RemoteError as e:
        return JSONResponse({"error": str(e), "fix": e.fix, "field": "url"}, 400)
    stem = urlparse(url).path.rsplit("/", 1)[-1]
    extra = {"link": url, "link_name": stem, "link_size": str(size)}
    d = store.store(io.BytesIO(text), f"{stem}.{reg[0]}_{reg[1]}_{reg[2]}.bedpe", "contacts", s.uploads_dir, s.upload_max_mb << 20, extra)
    c = checks.Check(True, summary)
    store.record_check(d, c)
    return JSONResponse(store.result(d, c), 201)


@router.get("/api/uploads/{uid}/preview")
def api_upload_preview(site: Here, uid: str, region: str, cell: str = "") -> JSONResponse:
    d = store.find(site.settings.uploads_dir, uid)
    if d is None:
        raise HTTPException(404)
    try:
        reg = region_of(site, cell, region)
    except RegionError as e:
        return JSONResponse({"error": str(e)}, 400)
    return JSONResponse(store.preview(d, reg) if reg else [])
