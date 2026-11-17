"""The web app: the frontend build, the form, job pages, files and a small JSON API."""

from __future__ import annotations

import io
import json
import re
import secrets
import shutil
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import explore, lists, owndata, remote
from .catalog import BACKGROUNDS, FACTOR_CHOICES, Catalog
from .jobs import JobRequest, _member, request_key
from .queue import enqueue, queue_position, sweep_private
from .regions import HICDIFFUSION_SPAN, RegionError, parse_region
from .settings import Settings
from .settings_schema import SettingError, parse_ini, schema, validate_overrides
from .viewer import viewer_payload

HERE = Path(__file__).resolve().parent
FILES = re.compile(r"^(ep\.tsv|marks\.json|engine\.log|status\.json|request\.json|[A-Za-z0-9_.:-]+\.cif)$")


class StripBase:
    """Removes the site's base path from a request path that carries it.

    The routes are written for the root. The proxy in front may strip the base path or pass it
    on, and both reach the same route.
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


def create_app() -> FastAPI:
    s = Settings.from_env()
    catalog = Catalog.load(s.data_dir)
    templates = Jinja2Templates(directory=str(HERE / "templates"))

    def pretty_region(text: str) -> str:
        """chr1:1000000-3000000 as chr1:1,000,000-3,000,000."""
        m = re.fullmatch(r"(\w+):(\d+)-(\d+)", str(text))
        return f"{m[1]}:{int(m[2]):,}-{int(m[3]):,}" if m else str(text)

    templates.env.filters["region"] = pretty_region
    templates.env.globals["root"] = s.base_path
    app = FastAPI(title="3D-GNOME")
    if s.base_path:
        app.add_middleware(StripBase, base=s.base_path)
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

    @app.get("/model/classic", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return form_page(request)

    class JobError(ValueError):
        """A job request the server refuses, with the form field it concerns."""

        def __init__(self, message: str, field: str) -> None:
            super().__init__(message)
            self.field = field

    def create_job(fields: dict[str, str], vcf: bytes | Path | None) -> str:
        """Checks a job request, queues the job unless the same one exists, and returns its url.

        @param fields the form fields of a job request, overrides as JSON text
        @param vcf an uploaded VCF, which makes the job private. The field vcf_upload names an
            upload in its place.
        @raises JobError naming the field the request fails on
        """
        cell_type = fields.get("cell_type", "")
        factors = fields.get("factors", "CTCF")
        try:
            own_raw = json.loads(fields.get("own") or "{}")
        except json.JSONDecodeError as e:
            raise JobError(f"own is not JSON: {e}", "own") from e
        own_dirs = own_uploads(own_raw)
        if fields.get("vcf_upload"):
            vcf = owndata.data_file(checked_upload(fields["vcf_upload"], "vcf", "vcf_upload"))
        if cell_type == "own":
            if not own_dirs.get("loops") and not own_dirs.get("loops_from"):
                raise JobError("Your own data needs loops, uploaded or a listed set.", "loops")
            if "contacts" not in own_dirs:
                raise JobError("Your own data needs contacts, as .hic, .mcool or BEDPE.", "contacts")
            base = own_dirs.get("loops_from") or next(iter(catalog.cell_types))
        else:
            base = cell_type
        background = fields.get("background", "hic")
        sv_set, sample = fields.get("sv_set", ""), fields.get("sample", "")
        if base not in catalog.cell_types:
            raise JobError(f"unknown cell type {cell_type!r}", "cell_type")
        if factors not in FACTOR_CHOICES and not own_dirs.get("loops"):
            raise JobError(f"unknown factor choice {factors!r}", "factors")
        if background not in BACKGROUNDS:
            raise JobError(f"unknown contact background {background!r}", "background")
        try:
            r = parse_region(fields.get("region", ""), catalog.cell_types[base].genes(), s.max_span)
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
        if own_dirs:
            token = secrets.token_urlsafe(18)
            job = private / token
            (job / "own").mkdir(parents=True)
            own: dict[str, object] = {"loops": []}
            for i, (factor, d) in enumerate(own_dirs.get("loops", [])):
                own["loops"].append([factor, str(owndata.clean_copy(d, job / "own" / f"loops_{i}.bedpe"))])  # type: ignore[union-attr]
            for kind in ("anchors", "contacts", "enhancers"):
                if kind in own_dirs:
                    own[kind] = str(owndata.clean_copy(own_dirs[kind], job / "own" / f"{kind}.txt"))
            for key in ("loops_from", "enhancers_from"):
                if key in own_dirs:
                    own[key] = own_dirs[key]
            req.own = own
            if vcf is not None:
                req.upload_vcf = str(put_vcf(vcf, job))
                req.sv_set = ""
            enqueue(s, req, job)
            return f"/p/{token}"
        if vcf is not None:
            token = secrets.token_urlsafe(18)
            job = private / token
            job.mkdir(parents=True)
            req.upload_vcf = str(put_vcf(vcf, job))
            req.sv_set = ""
            enqueue(s, req, job)
            return f"/p/{token}"
        key = request_key(req, catalog)
        job = shared / key
        st = job / "status.json"
        if st.is_file() and json.loads(st.read_text()).get("state") in ("running", "done", "queued"):
            return f"/jobs/{key}"
        job.mkdir(parents=True, exist_ok=True)
        st.write_text(json.dumps({"state": "queued", "stages": [{"stage": "queued", "at": time.time()}]}))
        enqueue(s, req, job)
        return f"/jobs/{key}"

    def put_vcf(vcf: bytes | Path, job: Path) -> Path:
        up = job / "upload.vcf.gz"
        if isinstance(vcf, Path):
            shutil.copyfile(vcf, up)
        else:
            up.write_bytes(vcf)
        return up

    def checked_upload(uid: object, kind: str, field: str) -> Path:
        """The directory of an upload that exists, holds `kind` and passed its check.

        @raises JobError on `field` otherwise
        """
        d = uploads / str(uid)
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", str(uid)) or not (d / "meta.json").is_file():
            raise JobError(f"upload {uid!r} does not exist", field)
        m = owndata.meta(d)
        if m["kind"] != kind:
            raise JobError(f"upload {uid!r} holds {m['kind']}, not {kind}", field)
        if not m.get("ok"):
            raise JobError(f"{m['name']} did not pass its check. Fix the file and upload it again.", field)
        return d

    def own_uploads(raw: object) -> dict:
        """The upload directories a job request names, each checked to exist, to be of its kind
        and to have passed its check."""
        if not isinstance(raw, dict):
            raise JobError("own is not an object of uploads", "own")
        out: dict = {}

        def one(uid: object, kind: str) -> Path:
            return checked_upload(uid, kind, "own")

        loops = raw.get("loops") or []
        if not isinstance(loops, list) or not all(isinstance(x, dict) for x in loops):
            raise JobError("own loops must be a list of objects with a factor and an upload", "own")
        if loops:
            out["loops"] = [(str(x.get("factor") or "loops"), one(x.get("upload"), "loops")) for x in loops]
        for kind in ("anchors", "contacts", "enhancers"):
            if raw.get(kind):
                out[kind] = one(raw[kind], kind)
        # Your own cell type may take a listed cell type's loops and enhancers in place of uploads.
        for key, allowed in (("loops_from", set(catalog.cell_types)), ("enhancers_from", {*catalog.cell_types, "none"})):
            name = raw.get(key)
            if name:
                if name not in allowed:
                    raise JobError(f"{key} names {name!r}, which is not a listed cell type", "own")
                out[key] = str(name)
        return out

    @app.post("/jobs")
    async def submit(request: Request, vcf: UploadFile | None = File(None)):  # noqa: ANN202, B008
        values = {k: str(v) for k, v in (await request.form()).items() if k != "vcf"}
        fields = values | {"background": "hicdiffusion" if values.get("hicdiffusion") else "hic"}
        fields["n_models"] = str(max(1, min(10, int(values.get("n_models") or 10))))
        try:
            url = create_job(fields, await vcf.read() if vcf is not None and vcf.filename else None)
        except JobError as e:
            return form_page(request, str(e), 400, values)
        return RedirectResponse(s.base_path + url, status_code=303)

    @app.post("/api/jobs")
    async def api_submit(request: Request, vcf: UploadFile | None = File(None)) -> JSONResponse:  # noqa: B008
        fields = {k: str(v) for k, v in (await request.form()).items() if k != "vcf"}
        try:
            url = create_job(fields, await vcf.read() if vcf is not None and vcf.filename else None)
        except JobError as e:
            return JSONResponse({"error": str(e), "field": e.field}, 400)
        return JSONResponse({"url": url}, 201)

    uploads = s.results_dir / "uploads"

    def region_of(cell: str, text: str) -> tuple[str, int, int] | None:
        if not text:
            return None
        genes = catalog.cell_types[cell].genes() if cell in catalog.cell_types else next(iter(catalog.cell_types.values())).genes()
        r = parse_region(text, genes, s.max_span)
        return (r.chrom, r.start, r.end)

    @app.post("/api/uploads")
    async def api_upload(request: Request, kind: str = Form(...), file: UploadFile = File(...), cell: str = Form(""), region: str = Form(""), factor: str = Form("")) -> JSONResponse:  # noqa: B008
        cap = s.upload_max_mb << 20
        if int(request.headers.get("content-length", "0")) > cap + (1 << 16):
            return JSONResponse({"error": f"{file.filename} is more than {s.upload_max_mb} MB, the largest upload this server takes"}, 413)
        if kind not in owndata.KINDS:
            return JSONResponse({"error": f"{kind!r} is not one of {', '.join(owndata.KINDS)}", "field": "kind"}, 400)
        try:
            reg = region_of(cell, region)
        except RegionError as e:
            return JSONResponse({"error": str(e), "field": "region"}, 400)
        try:
            d = owndata.store(file.file, file.filename or kind, kind, uploads, cap, {"factor": factor} if factor else None)
        except owndata.UploadTooLarge as e:
            return JSONResponse({"error": str(e)}, 413)
        c = owndata.check(kind, owndata.data_file(d), reg)
        owndata.record_check(d, c)
        return JSONResponse(owndata.result(d, c), 201)

    # A link read holds a worker thread for up to READ_SECONDS; past the limit a read is refused
    # at once rather than queued behind the others.
    link_slots = threading.BoundedSemaphore(max(1, s.link_reads))

    @app.post("/api/uploads/link")
    def api_upload_link(url: str = Form(...), cell: str = Form(""), region: str = Form("")) -> JSONResponse:
        if not link_slots.acquire(blocking=False):
            return JSONResponse({"error": "The server is reading other links. Try again in a minute.", "fix": "Or choose Local .hic file or BEDPE of the region.", "field": "url"}, 503)
        try:
            return read_link(url, cell, region)
        finally:
            link_slots.release()

    def read_link(url: str, cell: str, region: str) -> JSONResponse:
        try:
            reg = region_of(cell, region)
        except RegionError as e:
            return JSONResponse({"error": str(e), "field": "region"}, 400)
        if reg is None:
            return JSONResponse({"error": "Enter the region first.", "fix": "The server reads only the bins of the region from the file.", "field": "region"}, 400)
        base = catalog.cell_types[cell] if cell in catalog.cell_types else next(iter(catalog.cell_types.values()))
        try:
            per_mb = remote.hosted_density(base.path("singletons"), reg[0])
            text, size, summary = remote.link_window(url, *reg, per_mb, base.name, s.link_allow_private)
        except remote.RemoteError as e:
            return JSONResponse({"error": str(e), "fix": e.fix, "field": "url"}, 400)
        stem = urlparse(url).path.rsplit("/", 1)[-1]
        extra = {"link": url, "link_name": stem, "link_size": str(size)}
        d = owndata.store(io.BytesIO(text), f"{stem}.{reg[0]}_{reg[1]}_{reg[2]}.bedpe", "contacts", uploads, s.upload_max_mb << 20, extra)
        c = owndata.Check(True, summary)
        owndata.record_check(d, c)
        return JSONResponse(owndata.result(d, c), 201)

    def upload_dir(uid: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", uid) or not (uploads / uid / "meta.json").is_file():
            raise HTTPException(404)
        return uploads / uid

    @app.get("/api/uploads/{uid}/preview")
    def api_upload_preview(uid: str, region: str, cell: str = "") -> JSONResponse:
        d = upload_dir(uid)
        try:
            reg = region_of(cell, region)
        except RegionError as e:
            return JSONResponse({"error": str(e)}, 400)
        return JSONResponse(owndata.preview(d, reg) if reg else [])

    @app.post("/api/variants/check")
    def api_variants_check(cell: str = Form(...), region: str = Form(...), factors: str = Form("CTCF"), sample: str = Form(...), haplotype: int = Form(0), upload: str = Form(""), sv_set: str = Form(""), own: str = Form("")) -> JSONResponse:  # noqa: PLR0913
        base = cell if cell in catalog.cell_types else next(iter(catalog.cell_types))
        try:
            r = parse_region(region, catalog.cell_types[base].genes(), s.max_span)
        except RegionError as e:
            return JSONResponse({"error": str(e), "field": "region"}, 400)
        if upload:
            try:
                vcf = owndata.data_file(checked_upload(upload, "vcf", "upload"))
            except JobError as e:
                return JSONResponse({"error": str(e), "field": e.field}, 400)
        elif sv_set in catalog.sv_sets:
            vcf = catalog.sv_sets[sv_set].vcf
        else:
            return JSONResponse({"error": "Choose a hosted call set or upload a VCF.", "field": "sv_set"}, 400)
        try:
            own_dirs = own_uploads(json.loads(own or "{}"))
        except (JobError, json.JSONDecodeError) as e:
            return JSONResponse({"error": str(e), "field": "own"}, 400)
        own_loops = [(f, owndata.data_file(d)) for f, d in own_dirs.get("loops", [])]
        own_anchors = owndata.data_file(own_dirs["anchors"]) if "anchors" in own_dirs else None
        try:
            out = owndata.variant_check(catalog.cell_types[base], r, factors if factors in FACTOR_CHOICES else "CTCF", vcf, sample, 1 if haplotype == 1 else 0, own_loops, own_anchors)
        except KeyError:
            return JSONResponse({"error": f"{sample!r} is not a sample of this VCF", "field": "sample"}, 400)
        return JSONResponse(out)

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
                "base": s.base_path + (f"/{kind}/{key}" if kind == "jobs" else f"/p/{key}"),
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

    @app.get("/jobs/{key}/classic", response_class=HTMLResponse)
    def shared_job(request: Request, key: str) -> HTMLResponse:
        return page(request, "jobs", key)

    @app.get("/p/{key}/classic", response_class=HTMLResponse)
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

    def explore_routes(kind: str) -> None:
        base = f"/api/{kind}/{{key}}"

        def guard(key: str):  # noqa: ANN202
            return job_dir(kind, key)

        @app.get(base + "/summary", name=f"{kind}_summary")
        def ex_summary(key: str) -> JSONResponse:
            d = guard(key)
            return JSONResponse(explore.summary(d, queue_position(s, d)))

        @app.get(base + "/structure/{i}", name=f"{kind}_structure")
        def ex_structure(key: str, i: int) -> JSONResponse:
            try:
                return JSONResponse(explore.structure(guard(key), i))
            except explore.ExploreError as e:
                return JSONResponse({"error": str(e)}, 404)

        @app.get(base + "/annotations", name=f"{kind}_annotations")
        def ex_annotations(key: str) -> JSONResponse:
            return JSONResponse(explore.annotations(guard(key)))

        @app.get(base + "/picks", name=f"{kind}_picks")
        def ex_picks(key: str) -> JSONResponse:
            return JSONResponse(explore.picks(guard(key)))

        @app.get(base + "/distance", name=f"{kind}_distance")
        def ex_distance(key: str, a: int, b: int) -> JSONResponse:
            try:
                return JSONResponse(explore.distance(guard(key), a, b))
            except explore.ExploreError as e:
                return JSONResponse({"error": str(e)}, 400)

        @app.get(base + "/genes", name=f"{kind}_genes")
        def ex_genes(key: str) -> JSONResponse:
            return JSONResponse(explore.genes(guard(key)))

        @app.get(base + "/loops", name=f"{kind}_loops")
        def ex_loops(key: str) -> JSONResponse:
            return JSONResponse(explore.loops(guard(key)))

        @app.get(base + "/contacts", name=f"{kind}_contacts")
        def ex_contacts(key: str, region: str, res: int = 25000, upper: str = "hic", lower: str = "model", norm: str = "ice") -> JSONResponse:
            d = guard(key)
            req = json.loads((d / "request.json").read_text()) if (d / "request.json").is_file() else {}
            ct = catalog.cell_types.get(str(req.get("cell_type", "")))
            # The cell type's deep map stands for the Hi-C only where the job read that cell type's
            # hosted map: not with own contacts, not with a predicted background.
            deep = ct is not None and ct.hic_url and not (req.get("own") or {}).get("contacts") and req.get("background", "hic") == "hic"
            try:
                return JSONResponse(
                    explore.contact_maps(d, region, res, upper, lower, norm, ct.hic_url if deep else "", ct.hic_name if deep else "", s.link_allow_private)  # type: ignore[union-attr]
                )
            except explore.ExploreError as e:
                return JSONResponse({"error": str(e)}, 400)

        @app.get(base + "/log", name=f"{kind}_log", response_class=PlainTextResponse)
        def ex_log(key: str) -> PlainTextResponse:
            return PlainTextResponse(explore.log_text(guard(key)))

        @app.get(base + "/files.zip", name=f"{kind}_zip")
        def ex_zip(key: str) -> Response:
            d = guard(key)
            return Response(explore.zip_bytes(d), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{d.name}.zip"'})

    explore_routes("jobs")
    explore_routes("p")

    @app.get("/api/compare")
    def api_compare(a: str, b: str) -> JSONResponse:
        dirs = []
        for url in (a, b):
            m = re.fullmatch(r"/(jobs|p)/([A-Za-z0-9_-]{8,64})", url)
            if not m:
                raise HTTPException(404)
            dirs.append(job_dir(m[1], m[2]))
        return JSONResponse(explore.compare(dirs[0], dirs[1]))

    shared_lists = lists.Lists(s.results_dir / "lists")
    BAD_CODE = "A sync code has 16 letters and digits, like K7M2-9QXD-4THP-W3RZ."
    NO_LIST = "No list has this code. Compare it with the computer that shows it."

    async def list_body(request: Request) -> dict[str, Any]:
        raw = await request.body()
        if len(raw) > lists.MAX_BODY:
            raise HTTPException(413)
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            raise HTTPException(400) from None
        return body if isinstance(body, dict) else {}

    def list_answer(code: str, record: dict[str, Any], status: int = 200) -> JSONResponse:
        return JSONResponse({"code": code, "private": record["private"], "jobs": record["jobs"]}, status)

    @app.post("/api/lists")
    async def list_create(request: Request) -> JSONResponse:
        body = await list_body(request)
        code, record = shared_lists.create(body.get("jobs"), body.get("private") is True)
        return list_answer(code, record, 201)

    @app.get("/api/lists/{code}")
    def list_read(code: str) -> JSONResponse:
        shown = lists.normalise(code)
        if shown is None:
            return JSONResponse({"error": BAD_CODE}, 400)
        record = shared_lists.read(shown)
        return list_answer(shown, record) if record else JSONResponse({"error": NO_LIST}, 404)

    @app.put("/api/lists/{code}")
    async def list_update(code: str, request: Request) -> JSONResponse:
        shown = lists.normalise(code)
        if shown is None:
            return JSONResponse({"error": BAD_CODE}, 400)
        record = shared_lists.update(shown, (await list_body(request)).get("jobs"))
        return list_answer(shown, record) if record else JSONResponse({"error": NO_LIST}, 404)

    index_html = s.frontend_dist / "index.html"
    if (s.frontend_dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=s.frontend_dist / "assets"), name="assets")

    # Registered last: every path no route above answers is a client route of the frontend.
    @app.get("/{path:path}", response_class=HTMLResponse, include_in_schema=False)
    def frontend(path: str) -> HTMLResponse:
        # A job's own page is the app's; the paths below it are the server's.
        if path.startswith(("api/", "assets/")) or re.match(r"(jobs|p)/[^/]+/", path):
            raise HTTPException(404)
        if not index_html.is_file():
            return HTMLResponse(f"The frontend is not built. Run npm run build in web/frontend, or set FRONTEND_DIST. Expected {index_html}.", 503)
        html = index_html.read_text()
        built = re.search(r'(?:src|href)="([^"]*)/assets/', html)
        if built and built[1] != s.base_path:
            return HTMLResponse(
                f"The frontend is built for the path {built[1] or '/'} and the server has BASE_PATH {s.base_path or '/'}. "
                f"Build with BASE_PATH={s.base_path or '/'} npm run build in web/frontend, or set BASE_PATH to match.",
                503,
            )
        return HTMLResponse(html)

    return app
