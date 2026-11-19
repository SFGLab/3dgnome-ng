"""One modelling job, from a request to finished structures and their marks.

A job directory holds `request.json`, `status.json`, the engine inputs under `inputs/`, the
structures under `models/`, `marks.json`, `ep.tsv` and `engine.log`. A job on hosted inputs is
keyed by a hash of the normalised request. A job with an upload is private and keyed by a
random token.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import hicdiffusion
from .catalog import Catalog
from .inputs import Own, build_inputs
from .marks import ep_table, mark_beads, read_cif
from .regions import parse_region
from .settings import Settings
from .settings_schema import apply_overrides, validate_overrides
from .svs import read_svs

ENGINE_VERSION = "gnome3d"


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

    def __post_init__(self) -> None:
        # The old form sent HiCDiffusion as a flag; request.json files of that form still read.
        if self.hicdiffusion:
            self.background, self.hicdiffusion = "hicdiffusion", False
        self.n_models = max(1, min(Settings.from_env().max_structures, int(self.n_models)))


def normalised(req: JobRequest, catalog: Catalog) -> dict[str, object]:
    """The request with its region resolved. Equivalent requests compare equal."""
    region = parse_region(req.region, catalog.cell_types[req.cell_type].genes(), Settings.from_env().max_span)
    d = asdict(req)
    d["region"] = str(region)
    # Keys of requests made before overrides and background existed stay the same.
    if not d["overrides"]:
        del d["overrides"]
    d["hicdiffusion"] = d["background"] == "hicdiffusion"
    if d["background"] in ("hic", "hicdiffusion"):
        del d["background"]
    if not (req.sv_set or req.upload_vcf):
        d["sample"], d["haplotype"] = "", 0
    return d


def request_key(req: JobRequest, catalog: Catalog) -> str:
    d = normalised(req, catalog)
    if req.upload_vcf:
        d["upload_vcf"] = hashlib.sha256(Path(req.upload_vcf).read_bytes()).hexdigest()
    d["engine"] = ENGINE_VERSION
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def _member(p: Path) -> tuple[int, str]:
    """A structure file's member number, from the engine's `_s<k>.cif` suffix, for natural order."""
    import re

    m = re.search(r"_s(\d+)\.cif$", p.name)
    return (int(m.group(1)) if m else 0, p.name)


def _status(job: Path, **fields: object) -> None:
    path = job / "status.json"
    cur = json.loads(path.read_text()) if path.is_file() else {}
    now = time.time()
    # Each stage once, with the time it began, for the job page's status log.
    stage = fields.get("stage")
    stages = cur.get("stages", [])
    if stage and (not stages or stages[-1]["stage"] != stage):
        stages.append({"stage": stage, "at": now})
        cur["stages"] = stages
    cur.update(fields, updated=now)
    write_json(path, cur)


def write_json(path: Path, value: object) -> None:
    """Write a JSON file whole. A job page reads a job's files while the job runs and a file
    written in place is empty for a moment."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


# After the name of a gene's second copy in a tandem duplication.
COPY = ", second copy"


def _tail(path: Path, n: int = 30) -> str:
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:]) if path.is_file() else ""


def _lifted_marks(cell, chrom: str, svs: list, enhancers: Path | None = None, cell_enhancers: bool = True):  # noqa: ANN001, ANN202
    """The genes and enhancers of the cell type on `chrom` in the coordinates of the haplotype.

    Those inside a deletion are dropped. Those inside a duplication are there twice and the second
    copy of a gene has `COPY` after its name. An enhancers upload replaces the cell type's.
    """
    import pandas as pd
    from personal3d.lift import Lift

    L = Lift(svs)
    genes = cell.genes()
    genes = genes[genes["chrom"] == chrom].copy()
    twin = genes.copy()
    twin["tss"] = L.copies(genes["tss"].to_numpy())
    twin = twin[twin["tss"] >= 0]
    twin["name"] = twin["name"] + COPY
    twin["gene_id"] = twin["gene_id"].astype(str) + ".copy2"
    genes["tss"] = L.many(genes["tss"].to_numpy())
    genes = pd.concat([genes[genes["tss"] >= 0], twin], ignore_index=True)
    if enhancers is None and not cell_enhancers:
        return genes, pd.DataFrame({0: pd.Series(dtype=str), 1: pd.Series(dtype=int), 2: pd.Series(dtype=int)})
    enh = pd.read_csv(enhancers or cell.path("enhancers"), sep="\t", header=None, usecols=[0, 1, 2])
    enh = enh[enh[0] == chrom].copy()
    start, end = enh[1].to_numpy(), enh[2].to_numpy()
    second = enh.copy()
    second[1], second[2] = L.copies(start), L.copies(end - 1) + 1
    second = second[(second[1] >= 0) & (second[2] > 0)]
    enh[1], enh[2] = L.many(start), L.many(end - 1) + 1
    return genes, pd.concat([enh[(enh[1] >= 0) & (enh[2] > 0)], second], ignore_index=True)


def run_job(
    req: JobRequest, catalog: Catalog, job: Path, engine: list[str] | None = None, env_extra: dict[str, str] | None = None
) -> Path:
    """Run one job in `job` and return that directory. `engine` is the command that starts
    the engine, `python -m gnome3d.cli` from `ENGINE_PYTHON` by default. `env_extra` is added to
    the engine's environment, the GPU of a queue slot for example."""
    job.mkdir(parents=True, exist_ok=True)
    write_json(job / "request.json", asdict(req))
    _status(job, state="running", stage="inputs", started=time.time())
    log = job / "engine.log"
    try:
        o = req.own
        if req.cell_type != "own":
            cell = catalog.cell_types[req.cell_type]
        else:
            cell = catalog.cell_types[str(o["loops_from"])] if o and o.get("loops_from") else next(iter(catalog.cell_types.values()))
        own = Own(
            loops=[(str(f), Path(p)) for f, p in o.get("loops", [])],  # type: ignore[union-attr]
            anchors=Path(str(o["anchors"])) if o.get("anchors") else None,
            contacts=Path(str(o["contacts"])) if o.get("contacts") else None,
            enhancers=Path(str(o["enhancers"])) if o.get("enhancers") else None,
        ) if o else None
        region = parse_region(req.region, cell.genes(), Settings.from_env().max_span)
        svs: list = []
        skipped = 0
        vcf = Path(req.upload_vcf) if req.upload_vcf else (catalog.sv_sets[req.sv_set].vcf if req.sv_set else None)
        if vcf is not None:
            haps = read_svs(vcf, req.sample, region)
            svs, skipped = haps[req.haplotype], haps[2]
        hic = None
        if req.background == "hicdiffusion":
            _status(job, stage="hicdiffusion")
            hic = hicdiffusion.predict(region.chrom, region.start, svs, log, env_extra), region.start
        inputs = build_inputs(cell, region, req.factors, svs, job / "inputs", hic, req.background, own)
        apply_overrides(inputs.config, validate_overrides(req.overrides))
        _status(job, stage="modelling", region=inputs.region, n_svs=inputs.n_svs, skipped_svs=skipped, loops=inputs.loops)

        cmd = list(engine) if engine else [os.environ.get("ENGINE_PYTHON", "python"), "-m", "gnome3d.cli"]
        cmd += ["--config", str(inputs.config), "--region", inputs.region, "--members", f"0-{req.n_models - 1}", "--out", str(job / "models")]
        env = dict(os.environ) | (env_extra or {})
        if "ENGINE_ROOT" in env:
            env["PYTHONPATH"] = env["ENGINE_ROOT"] + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        # The engine skips a structure whose file exists. A job run again after a restart goes on
        # where it stopped and its log is kept.
        kept = len(list((job / "models").glob("*.cif"))) if (job / "models").is_dir() else 0
        with log.open("a" if hic is not None or kept else "w") as fh:
            if kept:
                fh.write(f"\nThe job was started again and goes on after {kept} of {req.n_models} structures.\n")
                fh.flush()
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=env.get("ENGINE_ROOT"), env=env).returncode
        if rc != 0:
            raise RuntimeError(f"the engine exited with {rc}")

        _status(job, stage="marks")
        cifs = sorted((job / "models").glob("*.cif"), key=_member)
        models = [read_cif(c) for c in cifs]
        pos0, starts, ends, _ = models[0]
        borrowed = str(o.get("enhancers_from") or "none") if o else "none"
        enh_file = own.enhancers if own and own.enhancers else None
        if enh_file is None and req.cell_type == "own" and borrowed != "none":
            enh_file = catalog.cell_types[borrowed].path("enhancers")
        genes, enh = _lifted_marks(cell, region.chrom, svs, enh_file, cell_enhancers=req.cell_type != "own")
        inputs_lo, inputs_hi = (int(x) for x in inputs.region.split(":")[1].split("-"))
        marks = mark_beads(region.chrom, starts, ends, genes, enh)
        # read_cif sorts beads by start; the viewer selects atoms in file order
        order = np.argsort([int(r.split()[16]) for r in cifs[0].read_text().splitlines() if r.startswith("ATOM")], kind="stable")
        (job / "marks.json").write_text(
            json.dumps(
                {
                    "promoters": marks.promoters,
                    "enhancers": marks.enhancers,
                    "viewer": {
                        "promoters": sorted(int(order[k]) for k in marks.promoters.values()),
                        "enhancers": sorted(int(order[k]) for k in marks.enhancers),
                    },
                    "models": [c.name for c in cifs],
                    # haplotype coordinates; a promoter is its TSS and a kilobase either side
                    "elements": [
                        {"kind": "promoter", "name": str(g.name), "start": int(g.tss) - 1000, "end": int(g.tss) + 1000}
                        for g in genes.itertuples(index=False)
                        if inputs_lo <= int(g.tss) < inputs_hi
                    ]
                    + [
                        {"kind": "enhancer", "name": "", "start": int(a), "end": int(b)}
                        for a, b in zip(enh[1], enh[2], strict=True)
                        if inputs_lo <= int(a) and int(b) <= inputs_hi
                    ],
                }
            )
        )
        ep_table([m[0] for m in models], marks, genes, starts, ends).to_csv(job / "ep.tsv", sep="\t", index=False)
        _status(job, state="done", stage="done", finished=time.time())
    except Exception as e:  # noqa: BLE001
        tail = _tail(log) or traceback.format_exc()
        _status(job, state="failed", error=f"{type(e).__name__}: {e}", log_tail=tail, finished=time.time())
    return job
