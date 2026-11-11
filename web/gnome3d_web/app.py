"""The web app: the form, job pages, files and a small JSON API."""

from __future__ import annotations

import json
import re
import secrets
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .catalog import FACTOR_CHOICES, Catalog
from .jobs import JobRequest, request_key
from .queue import enqueue, sweep_private
from .regions import RegionError, parse_region
from .settings import Settings

HERE = Path(__file__).resolve().parent
FILES = re.compile(r"^(ep\.tsv|marks\.json|engine\.log|status\.json|request\.json|[A-Za-z0-9_.:-]+\.cif)$")


def create_app() -> FastAPI:
    s = Settings.from_env()
    catalog = Catalog.load(s.data_dir)
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    app = FastAPI(title="3D-GNOME")
    shared = s.results_dir / "jobs"
    private = s.results_dir / "private"
    shared.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    sweep_private(s.results_dir, s.retention_days)

    def form_page(request: Request, error: str = "", status: int = 200, values: dict | None = None) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "catalog": catalog,
                "factors": FACTOR_CHOICES,
                "retention": s.retention_days,
                "hicdiffusion_missing": Settings.hicdiffusion_missing(),
                "error": error,
                "v": values or {},
            },
            status_code=status,
        )

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return form_page(request)

    @app.post("/jobs")
    async def submit(
        request: Request,
        cell_type: str = Form(...),
        region: str = Form(...),
        factors: str = Form("CTCF"),
        sv_set: str = Form(""),
        sample: str = Form(""),
        haplotype: int = Form(0),
        n_models: int = Form(10),
        hicdiffusion: str = Form(""),
        vcf: UploadFile | None = File(None),
    ):  # noqa: ANN202
        values = dict(await request.form())
        try:
            if cell_type not in catalog.cell_types:
                raise RegionError(f"unknown cell type {cell_type!r}")
            if factors not in FACTOR_CHOICES:
                raise RegionError(f"unknown factor choice {factors!r}")
            parse_region(region, catalog.cell_types[cell_type].genes())
            if hicdiffusion and Settings.hicdiffusion_missing():
                raise RegionError("HiCDiffusion is not set up on this server: " + ", ".join(Settings.hicdiffusion_missing()))
            if sv_set and sv_set not in catalog.sv_sets:
                raise RegionError(f"unknown SV set {sv_set!r}")
            if sv_set and sample not in catalog.sv_sets[sv_set].samples:
                raise RegionError(f"{sample!r} is not a sample of {sv_set}")
        except RegionError as e:
            return form_page(request, str(e), 400, values)
        req = JobRequest(
            cell_type=cell_type, region=region, factors=factors, sv_set=sv_set, sample=sample,
            haplotype=1 if haplotype == 1 else 0, n_models=max(1, min(10, n_models)), hicdiffusion=bool(hicdiffusion),
        )  # fmt: skip
        if vcf is not None and vcf.filename:
            token = secrets.token_urlsafe(18)
            job = private / token
            job.mkdir(parents=True)
            up = job / "upload.vcf.gz"
            up.write_bytes(await vcf.read())
            req.upload_vcf = str(up)
            req.sv_set = ""
            enqueue(s, req, job)
            return RedirectResponse(f"/p/{token}", status_code=303)
        key = request_key(req, catalog)
        job = shared / key
        st = job / "status.json"
        if st.is_file() and json.loads(st.read_text()).get("state") in ("running", "done", "queued"):
            return RedirectResponse(f"/jobs/{key}", status_code=303)
        job.mkdir(parents=True, exist_ok=True)
        st.write_text(json.dumps({"state": "queued"}))
        enqueue(s, req, job)
        return RedirectResponse(f"/jobs/{key}", status_code=303)

    def job_dir(kind: str, key: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", key):
            raise HTTPException(404)
        d = (shared if kind == "jobs" else private) / key
        if not (d / "status.json").is_file() and not (d / "request.json").is_file():
            raise HTTPException(404)
        return d

    def page(request: Request, kind: str, key: str) -> HTMLResponse:
        d = job_dir(kind, key)
        status = json.loads((d / "status.json").read_text()) if (d / "status.json").is_file() else {"state": "queued"}
        req = json.loads((d / "request.json").read_text()) if (d / "request.json").is_file() else {}
        cell = catalog.cell_types.get(str(req.get("cell_type", "")))
        marks = json.loads((d / "marks.json").read_text()) if (d / "marks.json").is_file() else None
        ep = (d / "ep.tsv").read_text().splitlines()[:26] if (d / "ep.tsv").is_file() else []
        return templates.TemplateResponse(
            request,
            "job.html",
            {
                "base": f"/{kind}/{key}" if kind == "jobs" else f"/p/{key}",
                "status": status,
                "req": req,
                "cell": cell,
                "marks": marks,
                "ep": [row.split("\t") for row in ep],
                "private": kind == "private",
                "retention": s.retention_days,
            },
        )

    @app.get("/jobs/{key}", response_class=HTMLResponse)
    def shared_job(request: Request, key: str) -> HTMLResponse:
        return page(request, "jobs", key)

    @app.get("/p/{key}", response_class=HTMLResponse)
    def private_job(request: Request, key: str) -> HTMLResponse:
        return page(request, "private", key)

    def files(d: Path) -> list[str]:
        out = [p.name for p in d.iterdir() if p.is_file() and FILES.match(p.name) and p.name != "upload.vcf.gz"]
        out += sorted(p.name for p in (d / "models").glob("*.cif"))
        return sorted(set(out))

    def send(d: Path, name: str) -> FileResponse:
        if not FILES.match(name):
            raise HTTPException(404)
        p = d / "models" / name if name.endswith(".cif") else d / name
        if not p.is_file():
            raise HTTPException(404)
        return FileResponse(p, filename=name)

    @app.get("/jobs/{key}/files/{name}")
    def shared_file(key: str, name: str) -> FileResponse:
        return send(job_dir("jobs", key), name)

    @app.get("/p/{key}/files/{name}")
    def private_file(key: str, name: str) -> FileResponse:
        return send(job_dir("private", key), name)

    @app.get("/api/jobs/{key}")
    def shared_status(key: str) -> JSONResponse:
        d = job_dir("jobs", key)
        return JSONResponse(json.loads((d / "status.json").read_text()))

    @app.get("/api/jobs/{key}/files")
    def shared_files(key: str) -> list[str]:
        return files(job_dir("jobs", key))

    @app.get("/api/p/{key}")
    def private_status(key: str) -> JSONResponse:
        d = job_dir("private", key)
        return JSONResponse(json.loads((d / "status.json").read_text()))

    @app.get("/api/catalog")
    def api_catalog() -> dict[str, object]:
        return {
            "cell_types": {k: {"label": c.label, "lymphoblastoid": c.lymphoblastoid, "note": c.note} for k, c in catalog.cell_types.items()},
            "sv_sets": {k: {"label": v.label, "samples": v.samples} for k, v in catalog.sv_sets.items()},
        }

    return app
