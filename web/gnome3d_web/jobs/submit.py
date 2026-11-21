"""The submission of a job: a request is checked, its uploads are copied into the job and the job is queued."""

from __future__ import annotations

import json
import secrets
import shutil
from pathlib import Path

from .. import expression
from ..catalog.model import BACKGROUNDS, FACTOR_CHOICES
from ..engine import hicdiffusion
from ..engine.settings import OverrideError, validate_overrides
from ..errors import Refused
from ..regions import RegionError, parse_region
from ..context import Site
from ..uploads import store as upload_store
from . import store
from .request import JobRequest, request_key
from .workers import enqueue


def create_job(site: Site, fields: dict[str, str]) -> str:
    """Checks a job request, queues its job and returns the url of the job.

    A request on hosted data has a key. Without the field `fresh` the job of that key is
    returned when the site holds one, which is how an example opens the same job for
    everyone. With `fresh` set to 1 the request starts a job of its own.

    @param fields the form fields of a job request, overrides as JSON text. The field vcf_upload
        names an uploaded VCF, which makes the job private.
    @raises Refused with the field the request fails on
    """
    s, catalog, uploads = site.settings, site.catalog, site.settings.uploads_dir
    cell_type = fields.get("cell_type", "")
    factors = fields.get("factors", "CTCF")
    try:
        own_raw = json.loads(fields.get("own") or "{}")
    except json.JSONDecodeError as e:
        raise Refused(f"own is not JSON: {e}", "own") from e
    own_dirs = upload_store.own_data(uploads, own_raw, set(catalog.cell_types))
    vcf = None
    if fields.get("vcf_upload"):
        vcf = upload_store.data_file(upload_store.usable(uploads, fields["vcf_upload"], "vcf", "vcf_upload"))
    if cell_type == "own":
        if not own_dirs.get("loops") and not own_dirs.get("loops_from"):
            raise Refused("Your own data needs loops, uploaded or a listed set.", "loops")
        if "contacts" not in own_dirs:
            raise Refused("Your own data needs contacts, as .hic, .mcool or BEDPE.", "contacts")
        base = own_dirs.get("loops_from") or next(iter(catalog.cell_types))
    else:
        base = cell_type
    background = fields.get("background", "hic")
    sv_set, sample = fields.get("sv_set", ""), fields.get("sample", "")
    if base not in catalog.cell_types:
        raise Refused(f"unknown cell type {cell_type!r}", "cell_type")
    if factors not in FACTOR_CHOICES and not own_dirs.get("loops"):
        raise Refused(f"unknown factor choice {factors!r}", "factors")
    if background not in BACKGROUNDS:
        raise Refused(f"unknown contact background {background!r}", "background")
    try:
        r = parse_region(fields.get("region", ""), catalog.cell_types[base].genes(), site.max_span())
    except RegionError as e:
        raise Refused(str(e), "region") from e
    if background == "hicdiffusion" and r.end - r.start > site.hicdiffusion_span():
        raise Refused(
            f"{fields['region']!r} spans {r.end - r.start:,} bp. HiCDiffusion predicts up to {site.hicdiffusion_span():,} bp. "
            f"Pick a region of up to {site.hicdiffusion_span() / 1e6:g} Mb, or the Hi-C of the cell type as the contact background.",
            "background",
        )
    if background == "hicdiffusion" and hicdiffusion.missing():
        raise Refused("HiCDiffusion is not set up on this server: " + ", ".join(hicdiffusion.missing()), "background")
    if sv_set and sv_set not in catalog.sv_sets:
        raise Refused(f"unknown SV set {sv_set!r}", "sv_set")
    if sv_set and sample not in catalog.sv_sets[sv_set].samples:
        raise Refused(f"{sample!r} is not a sample of {sv_set}", "sample")
    try:
        overrides = validate_overrides(json.loads(fields.get("overrides") or "{}"))
    except (OverrideError, json.JSONDecodeError, AttributeError) as e:
        raise Refused(str(e), "overrides") from e
    try:
        n_models = int(fields.get("n_models", "10"))
    except ValueError as e:
        raise Refused(f"structures must be a whole number, not {fields.get('n_models')!r}", "n_models") from e
    req = JobRequest(
        cell_type=cell_type, region=fields["region"], factors=factors, sv_set=sv_set, sample=sample,
        haplotype=1 if str(fields.get("haplotype", "0")) == "1" else 0, n_models=n_models,
        background=background, overrides=overrides,
    )  # fmt: skip
    wanted = fields.get("expression", "")
    own_table = None
    if wanted == "own":
        own_table = own_expression(uploads, fields)
        req.expression = own_table[2]
    elif wanted:
        # A hosted table, by the id of its source. The columns are the ones of the cell type or of the sample.
        option = next((o for o in catalog.expression_options(base, sample if sv_set else "", site.config.get().hidden) if o["id"] == wanted), None)
        if option is None:
            raise Refused(f"{wanted!r} has no expression for {cell_type}" + (f" and {sample}" if sv_set else ""), "expression")
        src = catalog.expression[wanted]
        req.expression = {"source": src.name, "columns": option["columns"], "kind": src.kind, "unit": src.unit, "label": option["label"]}
    if own_dirs or vcf is not None or own_table is not None:
        return private_job(site, req, own_dirs, vcf, own_table)
    key = request_key(req, catalog)
    job = s.jobs_dir / key
    if store.taken(job):
        if fields.get("fresh") != "1":
            return f"/jobs/{key}"
        # The first job of a request has the key of the request. A repeat has a key of its own.
        key = secrets.token_hex(8)
        job = s.jobs_dir / key
    enqueue(site.queue, req, job)
    return f"/jobs/{key}"


def private_job(site: Site, req: JobRequest, own_dirs: dict, vcf: Path | None, own_table: tuple[expression.Table, list[str], dict[str, object]] | None) -> str:
    """Starts the job of a request with uploads under a random token. The table, the own data and
    the VCF are copied into the job, and the request has their places there."""
    token = secrets.token_urlsafe(18)
    job = site.settings.private_dir / token
    job.mkdir(parents=True)
    if own_table is not None:
        expression.write_job(job, own_table[0], own_table[1], own_table[2])
    if own_dirs:
        (job / "own").mkdir()
        loops = [[factor, str(upload_store.clean_copy(d, job / "own" / f"loops_{i}.bedpe"))] for i, (factor, d) in enumerate(own_dirs.get("loops", []))]
        own: dict[str, object] = {"loops": loops}
        for kind in ("anchors", "contacts", "enhancers"):
            if kind in own_dirs:
                own[kind] = str(upload_store.clean_copy(own_dirs[kind], job / "own" / f"{kind}.txt"))
        for key in ("loops_from", "enhancers_from"):
            if key in own_dirs:
                own[key] = own_dirs[key]
        req.own = own
    if vcf is not None:
        req.upload_vcf = str(put_vcf(vcf, job))
        req.sv_set = ""
    enqueue(site.queue, req, job)
    return f"/p/{token}"


def put_vcf(vcf: Path, job: Path) -> Path:
    """Copies the uploaded VCF of a request into its job."""
    up = job / "upload.vcf.gz"
    shutil.copyfile(vcf, up)
    return up


def own_expression(uploads: Path, fields: dict[str, str]) -> tuple[expression.Table, list[str], dict[str, object]]:
    """The uploaded expression table of a job request, the columns the request takes and what
    they are. The field expression_upload names the upload, expression_columns the samples
    as a JSON list, every sample when empty, and expression_kind counts or values."""
    if not fields.get("expression_upload"):
        raise Refused("Your own expression needs a table. Upload one, or choose no expression.", "expression")
    d = upload_store.usable(uploads, fields["expression_upload"], "expression", "expression")
    kind = fields.get("expression_kind", "counts")
    if kind not in expression.KINDS:
        raise Refused(f"expression_kind is {kind!r}, not one of {', '.join(expression.KINDS)}", "expression")
    name = str(upload_store.meta(d)["name"])
    try:
        columns = json.loads(fields.get("expression_columns") or "[]")
    except json.JSONDecodeError as e:
        raise Refused(f"expression_columns is not JSON: {e}", "expression") from e
    if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
        raise Refused("expression_columns must be a list of sample names", "expression")
    try:
        table = expression.read_table(upload_store.data_file(d))
        columns = columns or list(table.data.columns)
        missing = [c for c in columns if c not in table.data.columns]
        if missing:
            raise Refused(f"{name} has no sample {', '.join(missing)}", "expression")
        if kind == "counts":
            expression.check_counts(table, columns, name)
    except expression.ExpressionError as e:
        raise Refused(str(e), "expression") from e
    samples = "1 sample" if len(columns) == 1 else f"{len(columns)} samples"
    return table, columns, {"source": expression.UPLOAD, "label": f"Your own table {name}, {samples}", "kind": kind, "unit": ""}
