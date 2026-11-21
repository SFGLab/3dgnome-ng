"""The routes of the form. They give the catalog, the genes, a region, its hosted
loops, the samples and the expression."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..engine import hicdiffusion
from ..regions import RegionError, parse_region
from ..answers import REFUSALS
from ..context import Here
from . import genes, preview
from .model import FACTOR_CHOICES, SAMPLE_COLUMNS
from .shapes import CatalogInfo, ExpressionOptions, GeneHit, Preview, RegionInfo, SampleTable

router = APIRouter(responses=REFUSALS)


@router.get("/api/catalog", response_model=CatalogInfo)
def api_catalog(site: Here) -> CatalogInfo:
    catalog = site.catalog
    return {
        "cell_types": {
            k: {"label": c.label, "description": c.description, "data": c.data, "lymphoblastoid": c.lymphoblastoid, "note": c.note}
            for k, c in catalog.cell_types.items()
        },
        "sv_sets": {
            k: {"label": v.label, "samples": v.samples, "sample_table": v.samples_table is not None}
            for k, v in catalog.sv_sets.items()
        },
        "factors": list(FACTOR_CHOICES),
        "max_region_mb": site.max_span() / 1e6,
        "hicdiffusion_mb": site.hicdiffusion_span() / 1e6,
        "max_structures": site.settings.max_structures,
        "hicdiffusion": not hicdiffusion.missing(),
    }


@router.get("/api/genes", response_model=list[GeneHit])
def api_genes(site: Here, cell: str, q: str, limit: int = 8) -> JSONResponse:
    """Returns the suggestions of the region field."""
    return JSONResponse(genes.find(site.cell(cell), q, limit))


@router.get("/api/region", response_model=RegionInfo)
def api_region(site: Here, cell: str, q: str) -> JSONResponse:
    table = site.cell(cell).genes()
    try:
        r = parse_region(q, table, site.max_span())
    except RegionError as e:
        return JSONResponse({"error": str(e)}, 400)
    hit = table[table["name"].str.upper() == q.strip().upper()]
    label = f"{r.shown()}, {(r.end - r.start) / 1e6:.1f} Mb"
    body = {"chrom": r.chrom, "start": r.start, "end": r.end, "label": label}
    if len(hit):
        body["gene"] = str(hit.iloc[0]["name"])
    return JSONResponse(body)


@router.get("/api/preview", response_model=Preview)
def api_preview(site: Here, cell: str, region: str) -> JSONResponse:
    ct = site.cell(cell)
    # The form draws the loops around a region as well, its own length on either side. The
    # preview takes three times the longest region a job may have.
    try:
        r = parse_region(region, ct.genes(), 3 * site.max_span())
    except RegionError as e:
        return JSONResponse({"error": str(e)}, 400)
    return JSONResponse(preview.around(ct, r))


@router.get("/api/sv_sets/{name}/samples", response_model=SampleTable)
def api_sv_samples(site: Here, name: str) -> JSONResponse:
    """Returns the sex, the population and the parents of each sample of a call set that has the table."""
    sv = site.catalog.sv_sets.get(name)
    if sv is None or sv.samples_table is None:
        return JSONResponse({"error": f"no sample table for {name!r}"}, 404)
    return JSONResponse({"columns": list(SAMPLE_COLUMNS), "rows": sv.sample_rows()})


@router.get("/api/expression", response_model=ExpressionOptions)
def api_expression(site: Here, cell: str, sv_set: str = "", sample: str = "") -> JSONResponse:
    """Returns the hosted expression that a job of this cell type and of this sample of a call set can take."""
    catalog = site.catalog
    site.cell(cell)
    known = sv_set in catalog.sv_sets and sample in catalog.sv_sets[sv_set].samples
    return JSONResponse({"options": catalog.expression_options(cell, sample if known else "", site.config.get().hidden)})
