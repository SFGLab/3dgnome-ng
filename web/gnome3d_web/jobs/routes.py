"""The routes of a job: its submission, its status and its files."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from ..http import job_of
from ..results import views
from ..site import Here
from .store import _member
from .submit import create_job

router = APIRouter()

FILES = re.compile(r"^(ep\.tsv|marks\.json|engine\.log|status\.json|request\.json|hicdiffusion\.bedpe|expression\.tsv|[A-Za-z0-9_.:-]+\.cif)$")


@router.post("/api/jobs")
async def api_submit(site: Here, request: Request, vcf: UploadFile | None = File(None)) -> JSONResponse:  # noqa: B008
    fields = {k: str(v) for k, v in (await request.form()).items() if k != "vcf"}
    url = create_job(site, fields, await vcf.read() if vcf is not None and vcf.filename else None)
    return JSONResponse({"url": url}, 201)


def files(d: Path) -> list[str]:
    out = [p.name for p in d.iterdir() if p.is_file() and FILES.match(p.name) and p.name != "upload.vcf.gz"]
    out += [p.name for p in sorted((d / "models").glob("*.cif"), key=_member)]
    return sorted(set(out))


def send(d: Path, name: str) -> Response:
    if not FILES.match(name):
        raise HTTPException(404)
    p = d / "models" / name if name.endswith(".cif") else d / name
    if not p.is_file():
        raise HTTPException(404)
    stated = views.public_file(d, name)
    if stated is not None:
        kind = "application/json" if name.endswith(".json") else "text/plain; charset=utf-8"
        return Response(stated, media_type=kind, headers={"Content-Disposition": f'attachment; filename="{name}"'})
    return FileResponse(p, filename=name)


@router.get("/jobs/{key}/files/{name}")
def shared_file(site: Here, key: str, name: str) -> Response:
    return send(job_of(site, "jobs", key), name)


@router.get("/p/{key}/files/{name}")
def private_file(site: Here, key: str, name: str) -> Response:
    return send(job_of(site, "p", key), name)


@router.get("/api/jobs/{key}")
def shared_status(site: Here, key: str) -> JSONResponse:
    return JSONResponse(views.public_status(job_of(site, "jobs", key)))


@router.get("/api/jobs/{key}/files")
def shared_files(site: Here, key: str) -> list[str]:
    return files(job_of(site, "jobs", key))


@router.get("/api/p/{key}")
def private_status(site: Here, key: str) -> JSONResponse:
    return JSONResponse(views.public_status(job_of(site, "p", key)))
