"""The route of the variants dialog."""

from __future__ import annotations

import json

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse

from ..answers import REFUSALS
from ..catalog.model import FACTOR_CHOICES
from ..errors import Refused
from ..regions import RegionError, parse_region
from ..context import Here
from ..uploads import store as uploads
from .effect import variant_check
from .shapes import VariantCheck
from .vcf import UnknownSample

router = APIRouter(responses=REFUSALS)


@router.post("/api/variants/check", response_model=VariantCheck)
def api_variants_check(site: Here, cell: str = Form(...), region: str = Form(...), factors: str = Form("CTCF"), sample: str = Form(...), haplotype: int = Form(0), upload: str = Form(""), sv_set: str = Form(""), own: str = Form("")) -> JSONResponse:  # noqa: PLR0913
    catalog, root = site.catalog, site.settings.uploads_dir
    base = site.base_cell(cell)
    try:
        r = parse_region(region, base.genes(), site.max_span())
    except RegionError as e:
        return JSONResponse({"error": str(e), "field": "region"}, 400)
    if upload:
        vcf = uploads.data_file(uploads.usable(root, upload, "vcf", "upload"))
    elif sv_set in catalog.sv_sets:
        vcf = catalog.sv_sets[sv_set].vcf
    else:
        return JSONResponse({"error": "Choose a hosted call set or upload a VCF.", "field": "sv_set"}, 400)
    try:
        own_dirs = uploads.own_data(root, json.loads(own or "{}"), set(catalog.cell_types))
    except (Refused, json.JSONDecodeError) as e:
        return JSONResponse({"error": str(e), "field": "own"}, 400)
    own_loops = [(f, uploads.data_file(d)) for f, d in own_dirs.get("loops", [])]
    own_anchors = uploads.data_file(own_dirs["anchors"]) if "anchors" in own_dirs else None
    try:
        out = variant_check(base, r, factors if factors in FACTOR_CHOICES else "CTCF", vcf, sample, 1 if haplotype == 1 else 0, own_loops, own_anchors)
    except UnknownSample:
        return JSONResponse({"error": f"{sample!r} is not a sample of this VCF", "field": "sample"}, 400)
    return JSONResponse(out)
