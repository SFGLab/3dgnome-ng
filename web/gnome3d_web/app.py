"""The web app: the frontend build, the form, job pages, files and a small JSON API."""

from __future__ import annotations

import json
import re
import secrets
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .catalog import BACKGROUNDS, FACTOR_CHOICES, Catalog
from .jobs import JobRequest, _member, request_key
from .queue import enqueue, sweep_private
from .regions import HICDIFFUSION_SPAN, RegionError, parse_region
from .settings import Settings
from .settings_schema import SettingError, parse_ini, schema, validate_overrides
from .viewer import viewer_payload

HERE = Path(__file__).resolve().parent
FILES = re.compile(r"^(ep\.tsv|marks\.json|engine\.log|status\.json|request\.json|[A-Za-z0-9_.:-]+\.cif)$")


def create_app() -> FastAPI:
    s = Settings.from_env()
    catalog = Catalog.load(s.data_dir)
    templates = Jinja2Templates(directory=str(HERE / "templates"))

    def pretty_region(text: str) -> str:
        """chr1:1000000-3000000 as chr1:1,000,000-3,000,000."""
        m = re.fullmatch(r"(\w+):(\d+)-(\d+)", str(text))
        return f"{m[1]}:{int(m[2]):,}-{int(m[3]):,}" if m else str(text)

    templates.env.filters["region"] = pretty_region
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
                "max_region_mb": f"{s.max_span / 1e6:g}",
                "hicdiffusion_missing": Settings.hicdiffusion_missing(),
                "error": error,
                "v": values or {},
            },
            status_code=status,
        )

    @app.get("/model", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return form_page(request)

    class JobError(ValueError):
        """A job request the server refuses, with the form field it concerns."""

        def __init__(self, message: str, field: str) -> None:
            super().__init__(message)
            self.field = field

    def create_job(fields: dict[str, str], vcf: bytes | None) -> str:
        """Checks a job request, queues the job unless the same one exists, and returns its url.

        @param fields the form fields of a job request, overrides as JSON text
        @param vcf an uploaded VCF, which makes the job private
        @raises JobError naming the field the request fails on
        """
        cell_type = fields.get("cell_type", "")
        factors = fields.get("factors", "CTCF")
        background = fields.get("background", "hic")
        sv_set, sample = fields.get("sv_set", ""), fields.get("sample", "")
        if cell_type not in catalog.cell_types:
            raise JobError(f"unknown cell type {cell_type!r}", "cell_type")
        if factors not in FACTOR_CHOICES:
            raise JobError(f"unknown factor choice {factors!r}", "factors")
        if background not in BACKGROUNDS:
            raise JobError(f"unknown contact background {background!r}", "background")
        try:
            r = parse_region(fields.get("region", ""), catalog.cell_types[cell_type].genes(), s.max_span)
        except RegionError as e:
            raise JobError(str(e), "region") from e
        if background == "hicdiffusion" and r.end - r.start > HICDIFFUSION_SPAN:
            raise JobError(
                f"{fields['region']!r} spans {r.end - r.start:,} bp. HiCDiffusion predicts up to {HICDIFFUSION_SPAN:,} bp. "
                "Pick a region of up to 2 Mb, or the Hi-C of the cell type as the contact background.",
                "background",
            )
        if background == "hicdiffusion" and Settings.hicdiffusion_missing():
            raise JobError("HiCDiffusion is not set up on this server: " + ", ".join(Settings.hicdiffusion_missing()), "background")
        if sv_set and sv_set not in catalog.sv_sets:
            raise JobError(f"unknown SV set {sv_set!r}", "sv_set")
        if sv_set and sample not in catalog.sv_sets[sv_set].samples:
            raise JobError(f"{sample!r} is not a sample of {sv_set}", "sample")
        try:
            overrides = validate_overrides(json.loads(fields.get("overrides") or "{}"))
        except (SettingError, json.JSONDecodeError, AttributeError) as e:
            raise JobError(str(e), "overrides") from e
        try:
            n_models = int(fields.get("n_models", "10"))
        except ValueError as e:
            raise JobError(f"structures must be a whole number, not {fields.get('n_models')!r}", "n_models") from e
        req = JobRequest(
            cell_type=cell_type, region=fields["region"], factors=factors, sv_set=sv_set, sample=sample,
            haplotype=1 if str(fields.get("haplotype", "0")) == "1" else 0, n_models=n_models,
            background=background, overrides=overrides,
        )  # fmt: skip
        if vcf is not None:
            token = secrets.token_urlsafe(18)
            job = private / token
            job.mkdir(parents=True)
            up = job / "upload.vcf.gz"
            up.write_bytes(vcf)
            req.upload_vcf = str(up)
            req.sv_set = ""
            enqueue(s, req, job)
            return f"/p/{token}"
        key = request_key(req, catalog)
        job = shared / key
        st = job / "status.json"
        if st.is_file() and json.loads(st.read_text()).get("state") in ("running", "done", "queued"):
            return f"/jobs/{key}"
        job.mkdir(parents=True, exist_ok=True)
        st.write_text(json.dumps({"state": "queued"}))
        enqueue(s, req, job)
        return f"/jobs/{key}"

    @app.post("/jobs")
    async def submit(request: Request, vcf: UploadFile | None = File(None)):  # noqa: ANN202, B008
        values = {k: str(v) for k, v in (await request.form()).items() if k != "vcf"}
        fields = values | {"background": "hicdiffusion" if values.get("hicdiffusion") else "hic"}
        fields["n_models"] = str(max(1, min(10, int(values.get("n_models") or 10))))
        try:
            url = create_job(fields, await vcf.read() if vcf is not None and vcf.filename else None)
        except JobError as e:
            return form_page(request, str(e), 400, values)
        return RedirectResponse(url, status_code=303)

    @app.post("/api/jobs")
    async def api_submit(request: Request, vcf: UploadFile | None = File(None)) -> JSONResponse:  # noqa: B008
        fields = {k: str(v) for k, v in (await request.form()).items() if k != "vcf"}
        try:
            url = create_job(fields, await vcf.read() if vcf is not None and vcf.filename else None)
        except JobError as e:
            return JSONResponse({"error": str(e), "field": e.field}, 400)
        return JSONResponse({"url": url}, 201)

    @app.get("/api/region")
    def api_region(cell: str, q: str) -> JSONResponse:
        if cell not in catalog.cell_types:
            return JSONResponse({"error": f"unknown cell type {cell!r}"}, 404)
        genes = catalog.cell_types[cell].genes()
        try:
            r = parse_region(q, genes, s.max_span)
        except RegionError as e:
            return JSONResponse({"error": str(e)}, 400)
        hit = genes[genes["name"].str.upper() == q.strip().upper()]
        label = f"{r.chrom}:{r.start:,}-{r.end:,}, {(r.end - r.start) / 1e6:.1f} Mb"
        body = {"chrom": r.chrom, "start": r.start, "end": r.end, "label": label}
        if len(hit):
            body["gene"] = str(hit.iloc[0]["name"])
        return JSONResponse(body)

    @app.get("/api/preview")
    def api_preview(cell: str, region: str) -> JSONResponse:
        if cell not in catalog.cell_types:
            return JSONResponse({"error": f"unknown cell type {cell!r}"}, 404)
        ct = catalog.cell_types[cell]
        try:
            r = parse_region(region, ct.genes(), s.max_span)
        except RegionError as e:
            return JSONResponse({"error": str(e)}, 400)
        loops: dict[str, list[list[int]]] = {}
        for factor, name in ct.clusters.items():
            t = pd.read_csv(ct.root / name, sep=r"\s+", header=None, usecols=range(7), comment="#")
            t = t[(t[0] == r.chrom) & (t[3] == r.chrom) & (t[1] >= r.start) & (t[5] <= r.end)]
            loops[factor] = [[int((a + b) // 2), int((c + e) // 2), int(n)] for a, b, c, e, n in zip(t[1], t[2], t[4], t[5], t[6], strict=True)]
        g = ct.genes()
        g = g[(g["chrom"] == r.chrom) & (g["tss"] >= r.start) & (g["tss"] < r.end)]
        return JSONResponse(
            {
                "loops": loops,
                "counts": {k: len(v) for k, v in loops.items()},
                "genes": [{"name": str(n), "tss": int(t)} for n, t in zip(g["name"], g["tss"], strict=True)],
            }
        )

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
        ep: list[list[str]] = []
        if (d / "ep.tsv").is_file():
            import csv

            with (d / "ep.tsv").open() as fh:
                for r in list(csv.DictReader(fh, delimiter="\t"))[:25]:
                    name = r["name"] if r["name"] and r["name"] != r["gene_id"] else r["gene_id"]
                    ep.append([name, f"{int(r['tss']):,}", f"{int(r['enhancer_centre']):,}", f"{int(r['linear_bp']) / 1000:.1f}", f"{float(r['distance']):.2f}"])
        return templates.TemplateResponse(
            request,
            "job.html",
            {
                "base": f"/{kind}/{key}" if kind == "jobs" else f"/p/{key}",
                "status": status,
                "req": req,
                "cell": cell,
                "marks": marks,
                "ep": ep,
                "private": kind == "private",
                "retention": s.retention_days,
                "max_region_mb": f"{s.max_span / 1e6:g}",
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
        out += [p.name for p in sorted((d / "models").glob("*.cif"), key=_member)]
        return sorted(set(out))

    def send(d: Path, name: str) -> FileResponse:
        if not FILES.match(name):
            raise HTTPException(404)
        p = d / "models" / name if name.endswith(".cif") else d / name
        if not p.is_file():
            raise HTTPException(404)
        return FileResponse(p, filename=name)

    def view(d: Path, name: str) -> JSONResponse:
        cif = name.removesuffix(".json") + ".cif"
        if not name.endswith(".json") or not FILES.match(cif) or not (d / "models" / cif).is_file():
            raise HTTPException(404)
        marks = json.loads((d / "marks.json").read_text()) if (d / "marks.json").is_file() else {}
        return JSONResponse(viewer_payload(d / "models" / cif, marks.get("elements", [])))

    @app.get("/jobs/{key}/view/{name}")
    def shared_view(key: str, name: str) -> JSONResponse:
        return view(job_dir("jobs", key), name)

    @app.get("/p/{key}/view/{name}")
    def private_view(key: str, name: str) -> JSONResponse:
        return view(job_dir("private", key), name)

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
            "cell_types": {
                k: {"label": c.label, "description": c.description, "data": c.data, "lymphoblastoid": c.lymphoblastoid, "note": c.note}
                for k, c in catalog.cell_types.items()
            },
            "sv_sets": {k: {"label": v.label, "samples": v.samples} for k, v in catalog.sv_sets.items()},
            "factors": list(FACTOR_CHOICES),
            "max_region_mb": s.max_span / 1e6,
            "max_structures": s.max_structures,
            "hicdiffusion": not Settings.hicdiffusion_missing(),
        }

    @app.get("/api/settings")
    def api_settings(cell: str) -> JSONResponse:
        if cell not in catalog.cell_types:
            return JSONResponse({"error": f"unknown cell type {cell!r}"}, 404)
        return JSONResponse(schema(catalog.cell_types[cell].path("config")))

    @app.post("/api/settings/ini")
    async def api_settings_ini(file: UploadFile = File(...)) -> JSONResponse:  # noqa: B008
        overrides, refused = parse_ini((await file.read(1_000_000)).decode("utf-8", "replace"))
        return JSONResponse({"overrides": overrides, "refused": refused})

    index_html = s.frontend_dist / "index.html"
    if (s.frontend_dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=s.frontend_dist / "assets"), name="assets")

    # Registered last: every path no route above answers is a client route of the frontend.
    @app.get("/{path:path}", response_class=HTMLResponse, include_in_schema=False)
    def frontend(path: str) -> HTMLResponse:
        if path.startswith(("api/", "jobs/", "p/", "assets/")):
            raise HTTPException(404)
        if not index_html.is_file():
            return HTMLResponse(f"The frontend is not built. Run npm run build in web/frontend, or set FRONTEND_DIST. Expected {index_html}.", 503)
        return HTMLResponse(index_html.read_text())

    return app
