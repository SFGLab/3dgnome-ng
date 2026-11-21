"""The parts of an answer that every route shares: errors, a packed JSON body and the base path."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from .errors import Refused, UnknownCell
from .site import Site


def packed_json(request: Request, value: object) -> Response:
    """Returns a JSON answer, gzipped for a client that takes gzip. For an answer of megabytes.

    Level 1 takes the contact map of 800 bins from 11 MB to under 5 MB in a fifth of a second.
    A higher level saves a tenth more and takes several times as long.
    """
    body = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    headers = {"Vary": "Accept-Encoding"}
    takes = [e.split(";")[0].strip().lower() for e in request.headers.get("accept-encoding", "").split(",")]
    if "gzip" in takes:
        body = gzip.compress(body, compresslevel=1)
        headers["Content-Encoding"] = "gzip"
    return Response(body, media_type="application/json", headers=headers)


class StripBase:
    """Removes the base path of the site from a request path that has it. The proxy in front
    may strip the base path or pass it on.
    """

    def __init__(self, app: Any, base: str) -> None:
        self.app = app
        self.base = base

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        path = scope.get("path", "")
        if scope["type"] in ("http", "websocket") and (path == self.base or path.startswith(self.base + "/")):
            raw = scope.get("raw_path") or path.encode()
            scope = {**scope, "path": path[len(self.base) :] or "/", "raw_path": raw[len(self.base.encode()) :] or b"/"}
        await self.app(scope, receive, send)


def handle_errors(app: FastAPI) -> None:
    """Answers a refused request with 400, its message and its field, and an unknown cell type with 404."""

    @app.exception_handler(Refused)
    def refused(_: Request, e: Refused) -> JSONResponse:
        return JSONResponse({"error": str(e), "field": e.field}, 400)

    @app.exception_handler(UnknownCell)
    def unknown_cell(_: Request, e: UnknownCell) -> JSONResponse:
        return JSONResponse({"error": str(e)}, 404)


def job_of(site: Site, kind: str, key: str) -> Path:
    """The directory of the job of an address, or the answer 404."""
    job = site.job(kind, key)
    if job is None:
        raise HTTPException(404)
    return job


def job_at(site: Site, url: str) -> Path:
    """The directory of the job a page names by its path, `/jobs/<key>` or `/p/<token>`, or the answer 404."""
    _, kind, key = (url.split("/", 2) + ["", ""])[:3]
    if not url.startswith("/") or kind not in ("jobs", "p"):
        raise HTTPException(404)
    return job_of(site, kind, key)
