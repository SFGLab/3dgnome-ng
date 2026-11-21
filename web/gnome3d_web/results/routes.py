"""The routes of a job page, for a shared and for a private job, and of the comparison of two jobs."""

from __future__ import annotations


from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from ..catalog import genes
from ..answers import job_at, job_of, packed_json
from ..jobs import store
from ..context import Here
from . import compare, contact_view, views

router = APIRouter()


def job_routes(kind: str) -> APIRouter:
    """The routes of the page of one job. `kind` is the address of the job, jobs or p."""
    routes = APIRouter(prefix=f"/api/{kind}/{{key}}")

    @routes.get("/summary", name=f"{kind}_summary")
    def summary(site: Here, key: str) -> JSONResponse:
        d = job_of(site, kind, key)
        out = views.summary(d, site.queue.ahead(d))
        request: dict[str, object] = out["request"]  # type: ignore[assignment]
        out["sample_note"] = site.catalog.sample_note(str(request.get("sv_set", "")), str(request.get("sample", "")))
        return JSONResponse(out)

    @routes.get("/structure/{i}", name=f"{kind}_structure")
    def structure(site: Here, key: str, i: int) -> JSONResponse:
        try:
            return JSONResponse(views.structure(job_of(site, kind, key), i))
        except views.ExploreError as e:
            return JSONResponse({"error": str(e)}, 404)

    @routes.get("/annotations", name=f"{kind}_annotations")
    def annotations(site: Here, key: str) -> JSONResponse:
        return JSONResponse(views.annotations(job_of(site, kind, key)))

    @routes.get("/picks", name=f"{kind}_picks")
    def picks(site: Here, key: str) -> JSONResponse:
        return JSONResponse(views.picks(job_of(site, kind, key)))

    @routes.get("/distance", name=f"{kind}_distance")
    def distance(site: Here, key: str, a: int, b: int) -> JSONResponse:
        try:
            return JSONResponse(views.distance(job_of(site, kind, key), a, b))
        except views.ExploreError as e:
            return JSONResponse({"error": str(e)}, 400)

    @routes.get("/genes", name=f"{kind}_genes")
    def gene_table(site: Here, key: str) -> JSONResponse:
        return JSONResponse(views.genes(job_of(site, kind, key)))

    @routes.get("/loops", name=f"{kind}_loops")
    def loops(site: Here, key: str) -> JSONResponse:
        return JSONResponse(views.loops(job_of(site, kind, key)))

    @routes.get("/contacts", name=f"{kind}_contacts")
    def contacts(site: Here, request: Request, key: str, region: str, res: int = 25000, upper: str = "hic", lower: str = "model", norm: str = "ice") -> Response:
        d = job_of(site, kind, key)
        req = store.request(d)
        ct = site.catalog.cell_types.get(str(req.get("cell_type", "")))
        # The cell type's deep map stands for the Hi-C only where the job read that cell type's
        # hosted map: not with own contacts, not with a predicted background.
        deep = ct is not None and ct.hic_url and not (req.get("own") or {}).get("contacts") and req.get("background", "hic") == "hic"
        try:
            return packed_json(
                request,
                contact_view.contact_maps(d, region, res, upper, lower, norm, ct.hic_url if deep else "", ct.hic_name if deep else "", site.settings.link_allow_private),  # type: ignore[union-attr]
            )
        except views.ExploreError as e:
            return JSONResponse({"error": str(e)}, 400)

    @routes.get("/log", name=f"{kind}_log", response_class=PlainTextResponse)
    def log(site: Here, key: str) -> PlainTextResponse:
        return PlainTextResponse(store.log_text(job_of(site, kind, key)))

    @routes.get("/files.zip", name=f"{kind}_zip")
    def files_zip(site: Here, key: str) -> Response:
        d = job_of(site, kind, key)
        return Response(store.zip_bytes(d), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{d.name}.zip"'})

    return routes


@router.get("/api/compare")
def api_compare(site: Here, a: str, b: str, shared: int = 0) -> JSONResponse:
    return JSONResponse(compare.compare(job_at(site, a), job_at(site, b), shared == 1))


@router.get("/api/compare/gene")
def api_compare_gene(site: Here, a: str, b: str, gene: str, shared: int = 0) -> JSONResponse:
    """The distances of one gene in the structures of two jobs."""
    out = compare.compare_gene(job_at(site, a), job_at(site, b), gene, shared == 1)
    return JSONResponse(out) if out else JSONResponse({"error": f"{gene} is not a gene of both jobs"}, 404)


@router.get("/api/compare/expression")
def api_compare_expression(site: Here, a: str, b: str, shared: int = 0) -> JSONResponse:
    names = lambda: genes.names_by_id(site.catalog)  # noqa: E731
    return JSONResponse(compare.compare_expression(job_at(site, a), job_at(site, b), site.settings.contrasts_dir, shared == 1, names))
