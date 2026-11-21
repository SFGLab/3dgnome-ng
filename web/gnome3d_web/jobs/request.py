"""The request of a job and the key of a request on hosted data.

A job on hosted inputs is keyed by a hash of the normalised request. A job with an upload is
private and keyed by a random token.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..catalog.model import Catalog
from ..regions import ANY_SPAN, parse_region


# A part of the key of every shared job. It is not the name of the package. A change gives each
# request a new key and the site stops finding the jobs it holds.
ENGINE_VERSION = "gnome3d-ng"


def _no_overrides() -> dict[str, str]:
    return {}


def _no_own() -> dict[str, object]:
    return {}


@dataclass
class JobRequest:
    cell_type: str
    region: str
    factors: str
    sv_set: str = ""
    sample: str = ""
    haplotype: int = 0
    n_models: int = 10
    hicdiffusion: bool = False
    upload_vcf: str = ""
    overrides: dict[str, str] = field(default_factory=_no_overrides)
    background: str = "hic"
    own: dict[str, object] = field(default_factory=_no_own)
    # The expression of the job by its source, its columns, the kind, the unit and the label. Empty
    # for a job without expression.
    expression: dict[str, object] = field(default_factory=_no_own)

    def __post_init__(self) -> None:
        # A request.json of the first form of the site has HiCDiffusion as a flag.
        if self.hicdiffusion:
            self.background, self.hicdiffusion = "hicdiffusion", False
        self.n_models = int(self.n_models)


def normalised(req: JobRequest, catalog: Catalog) -> dict[str, object]:
    """Returns the request with its region resolved. Equivalent requests compare equal."""
    region = parse_region(req.region, catalog.cell_types[req.cell_type].genes(), ANY_SPAN)
    d = asdict(req)
    d["region"] = str(region)
    # Keys of requests made before overrides and background existed stay the same.
    if not d["overrides"]:
        del d["overrides"]
    d["hicdiffusion"] = d["background"] == "hicdiffusion"
    if d["background"] in ("hic", "hicdiffusion"):
        del d["background"]
    # The key of a request without expression is the one it had before expression existed. A
    # table is in the key by its source, its columns and its kind. Its label is not.
    if not d["expression"]:
        del d["expression"]
    else:
        d["expression"] = {k: d["expression"].get(k) for k in ("source", "columns", "kind")}
    if not (req.sv_set or req.upload_vcf):
        d["sample"], d["haplotype"] = "", 0
    return d


def request_key(req: JobRequest, catalog: Catalog) -> str:
    d = normalised(req, catalog)
    if req.upload_vcf:
        d["upload_vcf"] = hashlib.sha256(Path(req.upload_vcf).read_bytes()).hexdigest()
    d["engine"] = ENGINE_VERSION
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
