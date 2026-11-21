"""The routes of a job: its submission, its status and its files."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from ..answers import REFUSALS, job_of
from ..context import Here
from . import store
from .shapes import JobUrl, Status
from .submit import create_job

router = APIRouter(responses=REFUSALS)

@router.post("/api/jobs", response_model=JobUrl, status_code=201)
async def api_submit(site: Here, request: Request) -> JSONResponse:
    fields = {k: v for k, v in (await request.form()).items() if isinstance(v, str)}
    return JSONResponse({"url": create_job(site, fields)}, 201)


def send(d: Path, name: str) -> Response:
    p = store.download(d, name)
    if p is None:
        raise HTTPException(404)
    stated = store.public_file(d, name)
    if stated is not None:
        kind = "application/json" if name.endswith(".json") else "text/plain; charset=utf-8"
        return Response(stated, media_type=kind, headers={"Content-Disposition": f'attachment; filename="{name}"'})
    return FileResponse(p, filename=name)


@router.get("/jobs/{key}/files/{name}", response_class=FileResponse)
def shared_file(site: Here, key: str, name: str) -> Response:
    return send(job_of(site, "jobs", key), name)


@router.get("/p/{key}/files/{name}", response_class=FileResponse)
def private_file(site: Here, key: str, name: str) -> Response:
    return send(job_of(site, "p", key), name)


@router.get("/api/jobs/{key}", response_model=Status)
def shared_status(site: Here, key: str) -> JSONResponse:
    return JSONResponse(store.public_status(job_of(site, "jobs", key)))


@router.get("/api/p/{key}", response_model=Status)
def private_status(site: Here, key: str) -> JSONResponse:
    return JSONResponse(store.public_status(job_of(site, "p", key)))
