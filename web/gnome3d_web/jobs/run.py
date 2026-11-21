"""One modelling job, from a request to finished structures and their marks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from personal3d.lift import Lift

from .. import expression
from ..catalog.model import Catalog
from ..engine import hicdiffusion
from ..engine.inputs import Own, build_inputs
from ..engine.settings import apply_overrides, validate_overrides
from ..regions import ANY_SPAN, parse_region
from ..variants.vcf import read_svs
from .marks import ep_table, mark_beads
from .request import JobRequest
from .store import member, read_cif, set_status, tail, write_json


# After the name of a gene's second copy in a tandem duplication.
COPY = ", second copy"


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


# Starts a command on a set of cores: the cores as the first argument, the command after it. The
# process takes the cores and becomes the command, and every thread of the command has them.
_ON_CORES = "import os, sys; os.sched_setaffinity(0, {int(c) for c in sys.argv[1].split(',')}); os.execvp(sys.argv[2], sys.argv[2:])"


def on_cores(cmd: list[str], cores: list[int] | None) -> list[str]:
    """Returns the command that runs `cmd` on `cores` alone, `cmd` itself for no cores or on a
    system that cannot pin a process."""
    if not cores or not hasattr(os, "sched_setaffinity"):
        return cmd
    return [sys.executable, "-c", _ON_CORES, ",".join(str(c) for c in cores), *cmd]


def run_job(
    req: JobRequest,
    catalog: Catalog,
    job: Path,
    engine: list[str] | None = None,
    env_extra: dict[str, str] | None = None,
    cores: list[int] | None = None,
) -> Path:
    """Run one job in `job` and return that directory. `engine` is the command that starts
    the engine, `python -m gnome3d.cli` from `ENGINE_PYTHON` by default. `env_extra` is added to
    the engine's environment, the GPU of a queue slot for example. `cores` are the cores the
    engine runs on, every core of this process when None."""
    job.mkdir(parents=True, exist_ok=True)
    write_json(job / "request.json", asdict(req))
    set_status(job, state="running", stage="inputs", started=time.time())
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
        region = parse_region(req.region, cell.genes(), ANY_SPAN)
        svs: list = []
        skipped = 0
        vcf = Path(req.upload_vcf) if req.upload_vcf else (catalog.sv_sets[req.sv_set].vcf if req.sv_set else None)
        if vcf is not None:
            haps = read_svs(vcf, req.sample, region)
            svs, skipped = haps[req.haplotype], haps[2]
        hic = None
        if req.background == "hicdiffusion":
            set_status(job, stage="hicdiffusion")
            cond = hicdiffusion.predict(region.chrom, region.start, svs, log, env_extra)
            hicdiffusion.write_map(job / hicdiffusion.MAP_FILE, region.chrom, Lift(svs).edge(region.start), cond)
            hic = cond, region.start
        inputs = build_inputs(cell, region, req.factors, svs, job / "inputs", hic, req.background, own)
        # The table of an upload is in the job since the request. A hosted table is read here,
        # from the place the catalog gives its source.
        if req.expression and req.expression.get("source") != expression.UPLOAD:
            e = req.expression
            source = catalog.expression.get(str(e.get("source")))
            if source is None:
                raise ValueError(f"the site has no expression table {e.get('source')!r}")
            columns = [str(c) for c in e["columns"]]  # type: ignore[union-attr]
            table = expression.read_table(source.path, columns)
            if e["kind"] == "counts":
                expression.check_counts(table, columns, str(e["label"]))
            expression.write_job(job, table, columns, {k: e.get(k, "") for k in ("source", "label", "kind", "unit")})
        apply_overrides(inputs.config, validate_overrides(req.overrides))
        set_status(job, stage="modelling", region=inputs.region, n_svs=inputs.n_svs, skipped_svs=skipped, loops=inputs.loops)

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
            rc = subprocess.run(on_cores(cmd, cores), stdout=fh, stderr=subprocess.STDOUT, cwd=env.get("ENGINE_ROOT"), env=env).returncode
        if rc != 0:
            raise RuntimeError(f"the engine exited with {rc}")

        set_status(job, stage="marks")
        cifs = sorted((job / "models").glob("*.cif"), key=member)
        models = [read_cif(c) for c in cifs]
        pos0, starts, ends, _ = models[0]
        borrowed = str(o.get("enhancers_from") or "none") if o else "none"
        enh_file = own.enhancers if own and own.enhancers else None
        if enh_file is None and req.cell_type == "own" and borrowed != "none":
            enh_file = catalog.cell_types[borrowed].path("enhancers")
        genes, enh = _lifted_marks(cell, region.chrom, svs, enh_file, cell_enhancers=req.cell_type != "own")
        inputs_lo, inputs_hi = (int(x) for x in inputs.region.split(":")[1].split("-"))
        marks = mark_beads(region.chrom, starts, ends, genes, enh)
        (job / "marks.json").write_text(
            json.dumps(
                {
                    "promoters": marks.promoters,
                    "enhancers": marks.enhancers,
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
        set_status(job, state="done", stage="done", finished=time.time())
    except Exception as e:  # noqa: BLE001
        end = tail(log) or traceback.format_exc()
        set_status(job, state="failed", error=f"{type(e).__name__}: {e}", log_tail=end, finished=time.time())
    return job
