"""The routes of a job page, for a shared and for a private job. The module also has the routes of the
comparison of two jobs."""

from __future__ import annotations


from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from ..catalog import genes
from ..answers import REFUSALS, job_at, job_of, packed_json
from ..jobs import store
from ..context import Here
from . import compare, contact_view, views
from .shapes import Annotations, ContactMaps, Distance, DistanceChanges, ExpressionChanges, GeneDistances, GeneRow, Loop, Picks, Structure, Summary

router = APIRouter(responses=REFUSALS)


def job_routes(kind: str) -> APIRouter:
    """Returns the routes of the page of one job. `kind` is the address of the job, jobs or p."""
    routes = APIRouter(prefix=f"/api/{kind}/{{key}}", responses=REFUSALS)

    @routes.get("/summary", name=f"{kind}_summary", response_model=Summary)
    def summary(site: Here, key: str) -> JSONResponse:
        d = job_of(site, kind, key)
        out = views.summary(d, site.queue.ahead(d))
        request: dict[str, object] = out["request"]  # type: ignore[assignment]
        out["sample_note"] = site.catalog.sample_note(str(request.get("sv_set", "")), str(request.get("sample", "")))
        return JSONResponse(out)

    @routes.get("/structure/{i}", name=f"{kind}_structure", response_model=Structure)
    def structure(site: Here, key: str, i: int) -> JSONResponse:
        try:
            return JSONResponse(views.structure(job_of(site, kind, key), i))
        except views.ExploreError as e:
            return JSONResponse({"error": str(e)}, 404)

    @routes.get("/annotations", name=f"{kind}_annotations", response_model=Annotations)
    def annotations(site: Here, key: str) -> JSONResponse:
        job = job_of(site, kind, key)
        return JSONResponse(views.features(job, site.base_cell(str(store.request(job).get("cell_type", ""))).genes()))

    @routes.get("/picks", name=f"{kind}_picks", response_model=Picks)
    def picks(site: Here, key: str) -> JSONResponse:
        try:
            return JSONResponse(views.picks(job_of(site, kind, key)))
        except views.ExploreError as e:
            return JSONResponse({"error": str(e)}, 404)

    @routes.get("/distance", name=f"{kind}_distance", response_model=Distance)
    def distance(site: Here, key: str, a: int, b: int) -> JSONResponse:
        try:
            return JSONResponse(views.distance(job_of(site, kind, key), a, b))
        except views.ExploreError as e:
            return JSONResponse({"error": str(e)}, 400)

    @routes.get("/genes", name=f"{kind}_genes", response_model=list[GeneRow])
    def gene_table(site: Here, key: str) -> JSONResponse:
        return JSONResponse(views.genes(job_of(site, kind, key)))

    @routes.get("/loops", name=f"{kind}_loops", response_model=list[Loop])
    def loops(site: Here, key: str) -> JSONResponse:
        return JSONResponse(views.loops(job_of(site, kind, key)))

    @routes.get("/contacts", name=f"{kind}_contacts", response_model=ContactMaps)
    def contacts(site: Here, request: Request, key: str, region: str, res: int = 25000, upper: str = "hic", lower: str = "model", norm: str = "ice") -> Response:
        d = job_of(site, kind, key)
        req = store.request(d)
        ct = site.catalog.cell_types.get(str(req.get("cell_type", "")))
        # The deep map of the cell type stands for the Hi-C only for a job that read the hosted map
        # of that cell type. A job with own contacts or a predicted background keeps its own.
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

    @routes.get("/files.zip", name=f"{kind}_zip", response_class=Response, responses={200: {"content": {"application/zip": {}}}})
    def files_zip(site: Here, key: str) -> Response:
        d = job_of(site, kind, key)
        return Response(store.zip_bytes(d), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{d.name}.zip"'})

    return routes


@router.get("/api/compare", response_model=DistanceChanges)
def api_compare(site: Here, a: str, b: str, shared: int = 0) -> JSONResponse:
    return JSONResponse(compare.compare(job_at(site, a), job_at(site, b), shared == 1))


@router.get("/api/compare/gene", response_model=GeneDistances)
def api_compare_gene(site: Here, a: str, b: str, gene: str, shared: int = 0) -> JSONResponse:
    """Returns the distances of one gene in the structures of two jobs."""
    out = compare.compare_gene(job_at(site, a), job_at(site, b), gene, shared == 1)
    return JSONResponse(out) if out else JSONResponse({"error": f"{gene} is not a gene of both jobs"}, 404)


@router.get("/api/compare/expression", response_model=ExpressionChanges)
def api_compare_expression(site: Here, a: str, b: str, shared: int = 0) -> JSONResponse:
    names = lambda: genes.names_by_id(site.catalog)  # noqa: E731
    return JSONResponse(compare.compare_expression(job_at(site, a), job_at(site, b), site.settings.contrasts_dir, shared == 1, names))
