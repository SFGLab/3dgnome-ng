"""The routes of a job. They take its submission and give its status and its files."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from ..answers import REFUSALS, job_of
from ..context import Here
from . import store
from .shapes import JobState, JobStates, JobUrl, Status
from .submit import create_job

# The most jobs one request for states names. A list of a visitor is far shorter.
MAX_STATES = 500

router = APIRouter(responses=REFUSALS)

@router.post("/api/jobs", response_model=JobUrl, status_code=201)
async def api_submit(site: Here, request: Request) -> JSONResponse:
    fields = {k: v for k, v in (await request.form()).items() if isinstance(v, str)}
    return JSONResponse({"url": create_job(site, fields)}, 201)


@router.post("/api/jobs/states", response_model=JobStates)
async def api_states(site: Here, request: Request) -> JSONResponse:
    """Answers the states of the jobs of a list in one request. The body is `{"jobs": [...]}` with
    the page path of each job. A path that is no job of the site has null."""
    try:
        body = json.loads(await request.body() or b"{}")
    except ValueError:
        raise HTTPException(400) from None
    asked = body.get("jobs") if isinstance(body, dict) else None
    states: dict[str, JobState | None] = {}
    for url in (asked if isinstance(asked, list) else [])[:MAX_STATES]:
        if not isinstance(url, str):
            continue
        _, kind, key = (url.split("/", 2) + ["", ""])[:3]
        job = site.job(kind, key) if url.startswith("/") and kind in ("jobs", "p") else None
        if job is None:
            states[url] = None
            continue
        status = store.status(job)
        state: JobState = {"state": status["state"]}
        stage = status.get("stage")
        if stage:
            state["stage"] = stage
        states[url] = state
    return JSONResponse({"states": states})


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
