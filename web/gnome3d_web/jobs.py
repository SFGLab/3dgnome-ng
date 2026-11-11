"""One modelling job, from a request to finished structures and their marks.

A job directory holds `request.json`, `status.json`, the engine inputs under `inputs/`, the
structures under `models/`, `marks.json`, `ep.tsv` and `engine.log`. `status.json` is rewritten
at every stage, so a page can show where a job is, and a failure records the stage and the last
lines of the log. A job on hosted inputs is keyed by a hash of the normalised request, so the
same request finds the same directory. A job with an upload is private and keyed by a random
token instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .catalog import Catalog
from .inputs import build_inputs
from .marks import ep_table, mark_beads, read_cif
from .regions import parse_region
from .svs import read_svs

ENGINE_VERSION = "gnome3d-ng"


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


def normalised(req: JobRequest, catalog: Catalog) -> dict[str, object]:
    """The request with its region resolved, so equivalent requests compare equal."""
    region = parse_region(req.region, catalog.cell_types[req.cell_type].genes())
    d = asdict(req)
    d["region"] = str(region)
    if not (req.sv_set or req.upload_vcf):
        d["sample"], d["haplotype"] = "", 0
    return d


def request_key(req: JobRequest, catalog: Catalog) -> str:
    d = normalised(req, catalog)
    if req.upload_vcf:
        d["upload_vcf"] = hashlib.sha256(Path(req.upload_vcf).read_bytes()).hexdigest()
    d["engine"] = ENGINE_VERSION
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def _status(job: Path, **fields: object) -> None:
    path = job / "status.json"
    cur = json.loads(path.read_text()) if path.is_file() else {}
    cur.update(fields, updated=time.time())
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, indent=2))
    tmp.replace(path)


def _tail(path: Path, n: int = 30) -> str:
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:]) if path.is_file() else ""


def _hicdiffusion(region_start: int, chrom: str, svs: list, fasta: Path) -> tuple[np.ndarray, int]:
    """HiCDiffusion's deterministic map of the 2 Mb window from the region's start."""
    from personal3d import hicdiffusion

    ref = hicdiffusion.Reference(fasta)
    w = hicdiffusion.haplotype_window(ref, chrom, region_start, svs)
    cond, _ = hicdiffusion.predict([w])
    return cond[0], region_start


def _lifted_marks(cell, chrom: str, svs: list):  # noqa: ANN001, ANN202
    """The cell type's genes and enhancers on `chrom` in the haplotype's coordinates, the beads'
    coordinates, those inside a deletion dropped."""
    import pandas as pd
    from personal3d.lift import Lift

    L = Lift(svs)
    genes = cell.genes()
    genes = genes[genes["chrom"] == chrom].copy()
    genes["tss"] = L.many(genes["tss"].to_numpy())
    genes = genes[genes["tss"] >= 0]
    enh = pd.read_csv(cell.path("enhancers"), sep="\t", header=None, usecols=[0, 1, 2])
    enh = enh[enh[0] == chrom].copy()
    enh[1], enh[2] = L.many(enh[1].to_numpy()), L.many(enh[2].to_numpy() - 1) + 1
    return genes, enh[(enh[1] >= 0) & (enh[2] > 0)]


def run_job(req: JobRequest, catalog: Catalog, job: Path, engine: list[str] | None = None) -> Path:
    """Run one job in `job` and return that directory. `engine` is the command that starts
    the engine, `python -m gnome3d.cli` from `ENGINE_PYTHON` by default."""
    job.mkdir(parents=True, exist_ok=True)
    (job / "request.json").write_text(json.dumps(asdict(req), indent=2))
    _status(job, state="running", stage="inputs", started=time.time())
    log = job / "engine.log"
    try:
        cell = catalog.cell_types[req.cell_type]
        region = parse_region(req.region, cell.genes())
        svs: list = []
        skipped = 0
        vcf = Path(req.upload_vcf) if req.upload_vcf else (catalog.sv_sets[req.sv_set].vcf if req.sv_set else None)
        if vcf is not None:
            haps = read_svs(vcf, req.sample, region)
            svs, skipped = haps[req.haplotype], haps[2]
        hic = None
        if req.hicdiffusion:
            _status(job, stage="hicdiffusion")
            hic = _hicdiffusion(region.start, region.chrom, svs, Path(os.environ["REFERENCE_FASTA"]))
        inputs = build_inputs(cell, region, req.factors, svs, job / "inputs", hic)
        _status(job, stage="modelling", region=inputs.region, n_svs=inputs.n_svs, skipped_svs=skipped, loops=inputs.loops)

        cmd = list(engine) if engine else [os.environ.get("ENGINE_PYTHON", "python"), "-m", "gnome3d.cli"]
        cmd += ["--config", str(inputs.config), "--region", inputs.region, "--members", f"0-{req.n_models - 1}", "--out", str(job / "models")]
        env = dict(os.environ)
        if "ENGINE_ROOT" in env:
            env["PYTHONPATH"] = env["ENGINE_ROOT"] + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        with log.open("w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=env.get("ENGINE_ROOT"), env=env).returncode
        if rc != 0:
            raise RuntimeError(f"the engine exited with {rc}")

        _status(job, stage="marks")
        cifs = sorted((job / "models").glob("*.cif"))
        models = [read_cif(c) for c in cifs]
        pos0, starts, ends, _ = models[0]
        genes, enh = _lifted_marks(cell, region.chrom, svs)
        marks = mark_beads(region.chrom, starts, ends, genes, enh)
        (job / "marks.json").write_text(json.dumps({"promoters": marks.promoters, "enhancers": marks.enhancers, "models": [c.name for c in cifs]}))
        ep_table([m[0] for m in models], marks, genes, starts, ends).to_csv(job / "ep.tsv", sep="\t", index=False)
        _status(job, state="done", stage="done", finished=time.time())
    except Exception as e:  # noqa: BLE001
        tail = _tail(log) or traceback.format_exc()
        _status(job, state="failed", error=f"{type(e).__name__}: {e}", log_tail=tail, finished=time.time())
    return job
